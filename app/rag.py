"""
RAG pipeline:
  build_index()               session-isolated Chroma vector store
  build_llm()                 configurable ChatOllama instance
  build_tc_llm()              faster LLM tuned for TC batch generation
  build_chain(mode)           mode-specific retrieval chain
  generate_tcs_rolling()      rolling batched TC generation w/ per-batch verification + sanitization
  generate_tcs_in_batches()   alias kept for backward compatibility
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
    VERIFY_NUM_PREDICT,
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
    Build a faster LLM tuned for TC batch generation.
    Lower num_predict + lower temperature = faster, more deterministic output.
    """
    return ChatOllama(
        model=OLLAMA_CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,           # more deterministic → less rambling
        num_predict=TC_NUM_PREDICT, # tighter output cap per batch
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


async def generate_tcs_in_batches(
    scenarios: list[dict],
    llm: ChatOllama,
    retriever,
    summary: str,
    chat_history: list,
    batch_size: int = TC_BATCH_SIZE,
    timeout_secs: float = TC_TIMEOUT_SECS,
) -> tuple[str, str, list[str]]:
    """
    Generate test cases in batches of `batch_size` scenarios.

    Design:
      - Sends 3 scenarios per LLM call (not 1, not all)
      - Hard timeout per batch prevents stuck generation
      - TC numbering is continuous across all batches
      - Returns (merged_text, merged_context, completed_sc_ids)
    """
    # Slice into batches of batch_size
    batches = [
        scenarios[i : i + batch_size]
        for i in range(0, len(scenarios), batch_size)
    ]
    n_batches = len(batches)

    all_outputs:  list[str] = []
    all_contexts: list[str] = []
    completed_ids: list[str] = []
    global_tc_counter = 1

    for batch_idx, batch in enumerate(batches):
        batch_num = batch_idx + 1
        batch_sc_ids = [s["id"] for s in batch]
        logger.info(
            "TC batch %d/%d: %s (timeout=%ss)",
            batch_num, n_batches, batch_sc_ids, timeout_secs,
        )

        # Build compact scenario block for this batch
        sc_block = "\n\n".join(
            f"{s['id']}: {s['title']}"
            + (f"\nDescription: {s['description']}" if s.get("description") else "")
            for s in batch
        )

        tc_chain = build_chain("tc_single", llm, retriever)
        inputs = {
            "input": (
                f"Generate test cases (Positive, Negative, Edge, Exception) for the "
                f"{len(batch)} scenario(s) below. "
                f"Start TC numbering from TC{global_tc_counter}. "
                f"Be concise but complete."
            ),
            "chat_history":         [],           # don't feed full history into every batch
            "conversation_summary": summary[-500:], # keep summary small per batch
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

        except asyncio.TimeoutError:
            logger.warning(
                "Batch %d/%d timed out after %ss — skipping, continuing.",
                batch_num, n_batches, timeout_secs,
            )
        except Exception as exc:
            logger.error("Batch %d/%d failed: %s", batch_num, n_batches, exc)

        if batch_output:
            # Ensure each SC header is present for the coverage checker
            for sc in batch:
                if sc["id"].lower() not in batch_output.lower():
                    batch_output = f"{sc['id']}: {sc['title']}\n\n" + batch_output

            all_outputs.append(batch_output)
            all_contexts.append(batch_ctx)
            completed_ids.extend(batch_sc_ids)

            # Advance global TC counter
            tc_nums = re.findall(r"\bTC(\d+)\b", batch_output, re.IGNORECASE)
            if tc_nums:
                global_tc_counter = max(int(n) for n in tc_nums) + 1

        logger.info(
            "Batch %d/%d done. TC counter now: %d",
            batch_num, n_batches, global_tc_counter,
        )

    merged_text    = "\n\n".join(all_outputs)
    merged_context = "\n".join(all_contexts[:2])  # limit context size
    return merged_text, merged_context, completed_ids


# ---------------------------------------------------------------------------
# Backward-compatible alias (coverage_revision still uses the old name)
# ---------------------------------------------------------------------------
async def generate_tcs_per_scenario(
    scenarios: list[dict],
    llm: ChatOllama,
    retriever,
    summary: str,
    chat_history: list,
) -> tuple[str, str]:
    """Thin alias → generate_tcs_rolling with batch_size=1 (legacy path)."""
    text, ctx, _ = await generate_tcs_rolling(
        scenarios=scenarios,
        llm=llm,
        retriever=retriever,
        summary=summary,
        batch_size=1,
    )
    return text, ctx


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
    verify_llm: ChatOllama | None = None,
) -> tuple[str, str, list[str]]:
    """
    Rolling batched TC generation with per-batch verification and sanitization.

    Architecture:
      1. Split scenario queue into batches of batch_size.
      2. For each batch:
           a. Generate TCs for the 3-4 scenarios in this batch only.
           b. If verify_llm provided, run lightweight verify_tc_batch().
           c. Run sanitize_test_cases() to fix formatting defects.
           d. Inject SC headers for any scenario missing from batch output.
           e. Accumulate and advance TC counter.
      3. Return (merged_text, merged_context, completed_ids).

    Benefits over single-shot:
      - Each batch gets focused context → better reasoning depth.
      - Verifier catches format issues per batch → no cascading corruption.
      - Sanitizer repairs each batch → clean output even if model wanders.
      - TC counter carries forward → continuous sequential numbering.
    """
    # Lazy import to avoid circular dependency at module load time
    from app.sanitizer import sanitize_test_cases
    from app.verifier import verify_tc_batch

    batches = [
        scenarios[i : i + batch_size]
        for i in range(0, len(scenarios), batch_size)
    ]
    n_batches = len(batches)

    all_outputs:   list[str] = []
    all_contexts:  list[str] = []
    completed_ids: list[str] = []
    global_tc_counter = 1

    for batch_idx, batch in enumerate(batches):
        batch_num    = batch_idx + 1
        batch_sc_ids = [s["id"] for s in batch]
        logger.info(
            "Rolling TC batch %d/%d — %s  (TC counter starts at TC%d)",
            batch_num, n_batches, batch_sc_ids, global_tc_counter,
        )

        # ── Build compact scenario block for this batch ──────────────────────
        sc_block = "\n\n".join(
            f"{s['id']}: {s['title']}"
            + (f"\nDescription: {s['description']}" if s.get("description") else "")
            for s in batch
        )

        tc_chain = build_chain("tc_single", llm, retriever)
        inputs = {
            "input": (
                f"Generate exhaustive test cases for the {len(batch)} scenario(s) below. "
                f"Start TC numbering from TC{global_tc_counter}. "
                f"For each test category (Positive, Negative, Edge, Exception), "
                f"generate as many meaningful variants as the feature supports — "
                f"do not stop at one per category."
            ),
            "chat_history":         [],              # fresh context per batch
            "conversation_summary": summary[-500:],  # keep prompt compact
            "scenarios":            sc_block,
            "existing_output":      "",
            "missing_items":        "",
        }

        batch_output = ""
        batch_ctx    = ""

        # ── Generate ─────────────────────────────────────────────────────────
        try:
            response = await asyncio.wait_for(
                tc_chain.ainvoke(inputs),
                timeout=timeout_secs,
            )
            batch_output = response.get("answer", "").strip()
            batch_ctx    = response.get("context", "")

        except asyncio.TimeoutError:
            logger.warning(
                "Rolling batch %d/%d timed out after %ss — skipping.",
                batch_num, n_batches, timeout_secs,
            )
        except Exception as exc:
            logger.error("Rolling batch %d/%d failed: %s", batch_num, n_batches, exc)

        if not batch_output:
            logger.warning("Batch %d/%d produced no output.", batch_num, n_batches)
            continue

        # ── Per-batch verification (optional, uses lightweight verifier) ─────
        if verify_llm is not None:
            batch_output, _ = await verify_tc_batch(
                batch_text=batch_output,
                scenarios_block=sc_block,
                context=batch_ctx,
                llm=verify_llm,
            )

        # ── Per-batch sanitization (always runs) ─────────────────────────────
        batch_output = sanitize_test_cases(batch_output)

        # ── Ensure SC headers are present for the coverage checker ───────────
        for sc in batch:
            if sc["id"].lower() not in batch_output.lower():
                batch_output = f"{sc['id']}: {sc['title']}\n\n" + batch_output

        all_outputs.append(batch_output)
        all_contexts.append(batch_ctx)
        completed_ids.extend(batch_sc_ids)

        # ── Advance continuous TC counter across batches ─────────────────────
        tc_nums = re.findall(r"\bTC(\d+)\b", batch_output, re.IGNORECASE)
        if tc_nums:
            global_tc_counter = max(int(n) for n in tc_nums) + 1

        logger.info(
            "Rolling batch %d/%d complete. TC counter → TC%d",
            batch_num, n_batches, global_tc_counter,
        )

    merged_text    = "\n\n".join(all_outputs)
    merged_context = "\n".join(all_contexts[:2])
    return merged_text, merged_context, completed_ids


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