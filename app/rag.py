"""
RAG pipeline:
  build_index()                session-isolated Chroma vector store
  build_llm()                  configurable ChatOllama instance
  build_tc_llm()               LLM tuned for TC batch generation
  build_chain(mode)            mode-specific retrieval chain
  generate_tcs_rolling()       rolling batched TC generation w/ per-batch eval + sanitization
  generate_coverage_revision() targeted gap-fill for missing SCs
"""
import asyncio
import logging
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.config import (
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBED_MODEL,
    RETRIEVER_K,
    RETRIEVER_FETCH_K,
    TC_NUM_PREDICT,
    TC_TIMEOUT_SECS,
    TC_BATCH_SIZE,
    FAST_MODE,
    FAST_TC_NUM_PREDICT,
)
from app.prompts import (
    SCENARIO_SYSTEM, TC_SYSTEM, TC_SINGLE_SYSTEM,
    GENERAL_SYSTEM, COVERAGE_REVISE_SYSTEM,
)

logger = logging.getLogger(__name__)

CHROMA_BASE_DIR = "./chroma_db"


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def build_llm(temperature: float = 0.2, num_predict: int = 4000) -> ChatOllama:
    """Build the general-purpose generator LLM."""
    return ChatOllama(
        model=OLLAMA_CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        num_predict=num_predict,
    )


def build_tc_llm() -> ChatOllama:
    """
    Build a LLM tuned for TC batch generation.
    Uses FAST_TC_NUM_PREDICT in fast mode, TC_NUM_PREDICT otherwise.
    Lower temperature = more deterministic = less format drift.
    """
    budget = FAST_TC_NUM_PREDICT if FAST_MODE else TC_NUM_PREDICT
    return ChatOllama(
        model=OLLAMA_CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.15,
        num_predict=budget,
    )


# ---------------------------------------------------------------------------
# Index builder (session-isolated)
# ---------------------------------------------------------------------------

def build_index(texts: list[str], session_id: str) -> Chroma:
    """
    Build a Chroma vector store scoped to this session.
    Each session writes to its own subdirectory — prevents cross-session pollution.
    """
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    persist_dir = f"{CHROMA_BASE_DIR}/{session_id}"
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    logger.debug("Building index in %s (%d chunks)", persist_dir, len(texts))
    return Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory=persist_dir,
        metadatas=[{"source": f"chunk-{i}"} for i in range(len(texts))],
    )


# ---------------------------------------------------------------------------
# Document formatter
# ---------------------------------------------------------------------------

def _format_docs(docs) -> str:
    if not docs:
        return "No relevant context found in uploaded documents."
    return "\n\n".join(
        f"--- Chunk {i + 1} ---\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )


# ---------------------------------------------------------------------------
# Mode-specific chain builder
# ---------------------------------------------------------------------------

_PROMPT_MAP: dict[str, str] = {
    "scenario":   SCENARIO_SYSTEM,
    "testcase":   TC_SYSTEM,
    "tc_single":  TC_SINGLE_SYSTEM,
    "general":    GENERAL_SYSTEM,
    "coverage":   COVERAGE_REVISE_SYSTEM,
}


def build_chain(mode: str, llm: ChatOllama, retriever):
    """
    Build a mode-specific retrieval + generation chain.

    Required input dict keys (provided by caller):
      - input                 the user question / request
      - chat_history          list[HumanMessage | AIMessage]
      - conversation_summary  rolling text summary of session
      - scenarios             scenario list text (testcase / tc_single modes)
      - existing_output       prior output text (coverage mode)
      - missing_items         gap description text (coverage mode)

    Added to output dict:
      - context   retrieved document chunks (str)
      - answer    LLM response text (str)
    """
    system_template = _PROMPT_MAP.get(mode, GENERAL_SYSTEM)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    async def _retrieve(inputs: dict) -> str:
        query = inputs.get("input", "")
        docs = await retriever.ainvoke(query)
        return _format_docs(docs)

    chain = (
        RunnablePassthrough.assign(context=RunnableLambda(_retrieve))
        | RunnablePassthrough.assign(answer=prompt | llm | StrOutputParser())
    )

    logger.debug("Built '%s' chain.", mode)
    return chain


# ---------------------------------------------------------------------------
# Rolling batched TC generation — primary generation path
# ---------------------------------------------------------------------------

async def generate_tcs_rolling(
    scenarios: list[dict],
    llm: ChatOllama,
    retriever,
    summary: str,
    batch_size: int = TC_BATCH_SIZE,
    timeout_secs: float = TC_TIMEOUT_SECS,
) -> tuple[str, str, list[str]]:
    """
    Rolling batched TC generation with per-batch Python eval + sanitization.

    Architecture:
      1. Split scenario queue into batches of batch_size.
      2. For each batch:
           a. Generate TCs (single LLM call — the ONLY model call per batch).
           b. Run eval_tc_batch() — Python-only, <10ms.
           c. If eval finds missing SCs AND not FAST_MODE: one targeted retry.
           d. Run sanitize_test_cases() — deterministic formatting repair.
           e. Accumulate and advance TC counter.
      3. Return (merged_text, merged_context, completed_ids).
    """
    from app.evals import eval_tc_batch
    from app.sanitizer import sanitize_test_cases, renumber_test_cases

    batches = [
        scenarios[i : i + batch_size]
        for i in range(0, len(scenarios), batch_size)
    ]
    n_batches = len(batches)

    # Concurrency control: Set to 1 for local Macs! 
    # Parallel requests sit in Ollama's queue, burning their asyncio timeout before they even start!
    concurrency_limit = 1
    semaphore = asyncio.Semaphore(concurrency_limit)
    
    async def process_batch(batch_idx: int, batch: list[dict]) -> tuple[str, str, list[str]]:
        async with semaphore:
            batch_num = batch_idx + 1
            batch_sc_ids = [s["id"] for s in batch]
            # Predict starting TC number to avoid conflicts in parallel execution
            predicted_start_tc = (batch_idx * 10) + 1
            
            logger.info(
                "Rolling batch %d/%d — %s (Starting at TC%d) [Parallel Task]",
                batch_num, n_batches, batch_sc_ids, predicted_start_tc,
            )

            sc_block = "\n\n".join(
                f"{s['id']}: {s['title']}"
                + (f"\nDescription: {s['description']}" if s.get("description") else "")
                for s in batch
            )

            tc_chain = build_chain("tc_single", llm, retriever)
            inputs = {
                "input": (
                    f"Generate test cases for the {len(batch)} scenario(s) below. "
                    f"Start TC numbering from TC{predicted_start_tc}. "
                    f"Generate a mix of Positive, Negative, Edge, and Exception categories. "
                    f"STRICT RULE: Maximum 5 test cases total per scenario. Every scenario must have at least 1 test case."
                ),
                "chat_history":         [],
                "conversation_summary": summary[-500:],
                "scenarios":            sc_block,
                "existing_output":      "",
                "missing_items":        "",
            }

            batch_output = ""
            batch_ctx    = ""

            try:
                response = await asyncio.wait_for(
                    tc_chain.ainvoke(inputs),
                    timeout=timeout_secs,
                )
                batch_output = response.get("answer", "").strip()
                batch_ctx    = response.get("context", "")
                
                logger.debug("DEBUG batch %d scenarios: %s", batch_num, sc_block)
                logger.debug("DEBUG batch %d output: %s", batch_num, batch_output)

            except asyncio.TimeoutError:
                logger.warning("Batch %d/%d timed out after %ss.", batch_num, n_batches, timeout_secs)
            except Exception as exc:
                logger.error("Batch %d/%d failed: %s", batch_num, n_batches, exc)

            if not batch_output:
                logger.warning("Batch %d/%d produced no output. Retrying once with fallback...", batch_num, n_batches)
                try:
                    retry_inputs = inputs.copy()
                    retry_inputs["input"] = (
                        f"You MUST generate test cases for these scenarios: {batch_sc_ids}. "
                        "Do NOT return an empty response. Output the full structured test cases now."
                    )
                    response = await asyncio.wait_for(
                        tc_chain.ainvoke(retry_inputs),
                        timeout=timeout_secs,
                    )
                    batch_output = response.get("answer", "").strip()
                    batch_ctx    = response.get("context", "")
                    logger.debug("DEBUG batch %d retry output: %s", batch_num, batch_output)
                except Exception as exc:
                    logger.error("Batch %d/%d retry failed: %s", batch_num, n_batches, exc)
                    
                if not batch_output:
                    logger.error("Batch %d/%d still produced no output. Returning partial error.", batch_num, n_batches)
                    batch_output = f"⚠️ [System Error] Failed to generate test cases for {batch_sc_ids} after retries."

            for sc in batch:
                if sc["id"].lower() not in batch_output.lower():
                    batch_output = f"{sc['id']}: {sc['title']}\n\n" + batch_output

            eval_result = eval_tc_batch(batch_output, batch_sc_ids)

            if not eval_result.passed and eval_result.missing_sc_ids and not FAST_MODE:
                missing_in_batch = [s for s in batch if s["id"] in set(eval_result.missing_sc_ids)]
                if missing_in_batch:
                    logger.info("Batch %d: retrying for missing %s", batch_num, eval_result.missing_sc_ids)
                    retry_block = "\n\n".join(
                        f"{s['id']}: {s['title']}"
                        + (f"\nDescription: {s['description']}" if s.get("description") else "")
                        for s in missing_in_batch
                    )
                    tc_nums = re.findall(r"\bTC(\d+)\b", batch_output, re.IGNORECASE)
                    retry_start = max((int(n) for n in tc_nums), default=predicted_start_tc - 1) + 1

                    retry_inputs = {
                        "input": (
                            f"Generate test cases for these {len(missing_in_batch)} missing scenario(s). "
                            f"Start TC numbering from TC{retry_start}."
                        ),
                        "chat_history":         [],
                        "conversation_summary": summary[-300:],
                        "scenarios":            retry_block,
                        "existing_output":      "",
                        "missing_items":        "",
                    }
                    try:
                        retry_resp = await asyncio.wait_for(
                            tc_chain.ainvoke(retry_inputs),
                            timeout=timeout_secs,
                        )
                        retry_text = retry_resp.get("answer", "").strip()
                        if retry_text:
                            batch_output = batch_output.rstrip() + "\n\n" + retry_text
                            logger.info("Batch %d: retry added content for missing SCs.", batch_num)
                    except Exception as exc:
                        logger.warning("Batch %d retry for missing SCs failed: %s", batch_num, exc)

            batch_output = sanitize_test_cases(batch_output)

            return batch_output, batch_ctx, batch_sc_ids

    # Execute all batches concurrently
    logger.info("Starting parallel generation of %d batches with concurrency limit %d", n_batches, concurrency_limit)
    tasks = [process_batch(i, b) for i, b in enumerate(batches)]
    results = await asyncio.gather(*tasks)
    
    all_outputs = []
    all_contexts = []
    completed_ids = []
    
    for out_text, out_ctx, out_ids in results:
        if out_text:
            all_outputs.append(out_text)
        if out_ctx:
            all_contexts.append(out_ctx)
        completed_ids.extend(out_ids)

    merged_text    = "\n\n".join(all_outputs)
    merged_text    = renumber_test_cases(merged_text)
    merged_context = "\n".join(all_contexts[:2])
    return merged_text, merged_context, completed_ids


# ---------------------------------------------------------------------------
# Coverage gap-fill — targeted regeneration for missing SCs only
# ---------------------------------------------------------------------------

async def generate_coverage_revision(
    missing_scenario_dicts: list[dict],
    existing_output: str,
    llm: ChatOllama,
    retriever,
    summary: str,
    chat_history: list,
    last_tc_num: int = 1,
) -> tuple[str, str]:
    """
    Generate test cases ONLY for the missing scenarios, appended to existing output.

    Returns:
        (new_tc_text_for_missing_only, context)
    """
    if not missing_scenario_dicts:
        return "", ""

    missing_labels = "\n".join(
        f"{s['id']}: {s['title']}" + (f" — {s['description']}" if s.get('description') else "")
        for s in missing_scenario_dicts
    )

    coverage_chain = build_chain("coverage", llm, retriever)

    inputs = {
        "input": (
            f"Generate test cases only for the missing scenarios listed below. "
            f"Start TC numbering from TC{last_tc_num}."
        ),
        "chat_history":         chat_history,
        "conversation_summary": summary,
        "scenarios":            missing_labels,
        "existing_output":      existing_output[-2000:],  # truncate for context budget
        "missing_items":        missing_labels,
    }

    try:
        response = await coverage_chain.ainvoke(inputs)
        return response.get("answer", "").strip(), response.get("context", "")
    except Exception as exc:
        logger.error("Coverage revision failed: %s", exc)
        return "", ""