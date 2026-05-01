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
    SCENARIO_SYSTEM, TC_SYSTEM, TC_TITLE_SYSTEM, TC_EXPAND_SYSTEM,
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
    "tc_title":   TC_TITLE_SYSTEM,
    "tc_expand":  TC_EXPAND_SYSTEM,
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
# Batched TC Title generation (Pass 1)
# ---------------------------------------------------------------------------

async def generate_tcs_rolling(
    scenarios: list[dict],
    llm: ChatOllama,
    retriever,
    summary: str,
    batch_size: int = 3,  # Small batches to preserve strict SC grouping
    timeout_secs: float = TC_TIMEOUT_SECS,
) -> tuple[str, str, list[str]]:
    """
    Sequential batched TC Title generation.
    Runs batches one at a time so TC numbering is always correct.
    """
    batches = [
        scenarios[i : i + batch_size]
        for i in range(0, len(scenarios), batch_size)
    ]
    n_batches = len(batches)

    all_outputs: list[str]   = []
    all_contexts: list[str]  = []
    completed_ids: list[str] = []
    next_tc_num = 1  # Global TC counter maintained across batches

    for batch_idx, batch in enumerate(batches):
        batch_num    = batch_idx + 1
        batch_sc_ids = [s["id"] for s in batch]

        logger.info(
            "Title batch %d/%d — %s (starting at TC%d)",
            batch_num, n_batches, batch_sc_ids, next_tc_num,
        )

        sc_block = "\n\n".join(
            f"{s['id']}: {s['title']}"
            + (f"\nDescription: {s['description']}" if s.get("description") else "")
            for s in batch
        )

        tc_chain = build_chain("tc_title", llm, retriever)
        inputs = {
            "input": (
                f"List test case index entries. "
                f"Output only Title, Type, and Goal per TC. "
                f"Start TC numbering from TC{next_tc_num}."
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
        except Exception as exc:
            logger.error("Batch %d/%d failed: %s", batch_num, n_batches, exc)

        if not batch_output:
            batch_output = f"⚠️ Failed to generate index for {batch_sc_ids}."
        else:
            # Count actual TC lines in this output to advance the global counter
            tc_hits = re.findall(r"(?:^|\n)TC\s*\d+\s*:", batch_output, re.IGNORECASE)
            next_tc_num += max(len(tc_hits), 1)

        all_outputs.append(batch_output)
        if batch_ctx:
            all_contexts.append(batch_ctx)
        completed_ids.extend(batch_sc_ids)

    merged_text    = "\n\n".join(all_outputs)
    # NOTE: renumber_test_cases is intentionally NOT called here.
    # It does not understand SC-grouping and would corrupt the grouped structure.
    merged_context = "\n".join(all_contexts[:2])
    return merged_text, merged_context, completed_ids


# ---------------------------------------------------------------------------
# Test Case Expansion
# ---------------------------------------------------------------------------

async def expand_test_case(
    tc_title: str,
    tc_type: str,
    tc_goal: str,
    scenario_context: str,
    llm: ChatOllama,
    retriever,
    summary: str,
) -> tuple[str, str]:
    """
    Expands a single test case outline into full steps.
    """
    expand_chain = build_chain("tc_expand", llm, retriever)
    inputs = {
        "input": "Generate the steps and expected outcomes for this test case.",
        "chat_history": [],
        "conversation_summary": summary[-500:],
        "scenarios": scenario_context,
        "existing_output": "",
        "missing_items": "",
    }
    
    # We inject the TC variables dynamically into the prompt
    # ChatPromptTemplate handles this if we pass them in inputs, but we need to modify build_chain or just replace here.
    # It's easier to inject them into the input text directly since our build_chain uses a generic ChatPromptTemplate.
    # Wait, the prompt TC_EXPAND_SYSTEM uses {tc_title}, {tc_type}, {tc_goal}.
    # We must pass these to the chain invoke.
    inputs["tc_title"] = tc_title
    inputs["tc_type"] = tc_type
    inputs["tc_goal"] = tc_goal

    try:
        response = await expand_chain.ainvoke(inputs)
        return response.get("answer", "").strip(), response.get("context", "")
    except Exception as exc:
        logger.error("Expansion failed for %s: %s", tc_title, exc)
        return f"⚠️ Failed to generate steps: {exc}", ""




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