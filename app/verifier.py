"""
Verifier layer: runs a second LLM pass to validate format + grounding,
then a deterministic Python coverage check for completeness.

Returns (final_text, verdict_label).
One LLM revision maximum — no infinite loops.
Coverage gaps are fixed by the caller (targeted regeneration), not here.
"""
import asyncio
import logging
import re

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from app.config import OLLAMA_BASE_URL, OLLAMA_VERIFY_MODEL, VERIFY_NUM_PREDICT
from app.coverage import (
    CoverageReport,
    check_scenario_coverage,
    check_scenario_numbering,
    check_tc_coverage,
    extract_scenario_ids,
)
from app.prompts import SCENARIO_VERIFIER_SYSTEM, TC_VERIFIER_SYSTEM

logger = logging.getLogger(__name__)

_VERDICT_RE = re.compile(r"VERDICT:\s*(PASS|REVISE)", re.IGNORECASE)
_REVISED_RE = re.compile(r"REVISED OUTPUT:\s*\n([\s\S]+)", re.IGNORECASE)


def build_verifier_llm() -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_VERIFY_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        num_predict=VERIFY_NUM_PREDICT,  # verdicts are short — no need for 4000 tokens
    )



def _parse_verifier_response(response_text: str) -> tuple[str, str]:
    """
    Parse verifier LLM output.
    Returns (verdict, revised_text_or_empty_string).
    """
    m = _VERDICT_RE.search(response_text)
    verdict = m.group(1).upper() if m else "PASS"

    revised = ""
    if verdict == "REVISE":
        rm = _REVISED_RE.search(response_text)
        revised = rm.group(1).strip() if rm else ""

    return verdict, revised


# ---------------------------------------------------------------------------
# Scenario verifier
# ---------------------------------------------------------------------------

async def verify_scenarios(
    generated: str,
    context: str,
    llm: ChatOllama | None = None,
) -> tuple[str, str, CoverageReport]:
    """
    Verify scenario output in two passes:
      1. LLM verifier: format validity + document grounding
      2. Python coverage checker: ensures all 5 coverage buckets are present

    Returns:
        (final_text, label, coverage_report)
        label: "PASS", "REVISED", or "COVERAGE_INCOMPLETE"
        On verifier failure: returns (generated, "PASS", report) — fail safe.
    """
    if llm is None:
        llm = build_verifier_llm()

    # Pass 1 — LLM: format + grounding
    final_text = generated
    label = "PASS"

    prompt = SCENARIO_VERIFIER_SYSTEM.format(
        context=context[:2000],
        generated=generated,
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        verdict, revised = _parse_verifier_response(response.content)

        if verdict == "REVISE" and revised:
            logger.info("Verifier (scenario): format/grounding revision requested.")
            final_text = revised
            label = "REVISED"

    except Exception as exc:
        logger.warning("Scenario LLM verifier failed (%s) — using generator output.", exc)

    # Pass 2 — Python: coverage completeness (deterministic, always runs)
    coverage = check_scenario_coverage(final_text)

    # Pass 3 — Python: numbering integrity (always runs after renumbering)
    num_report = check_scenario_numbering(final_text)
    if not num_report.is_complete:
        logger.warning(
            "Scenario numbering still broken after renumbering: %s",
            num_report.issues,
        )
        # Merge issues into coverage report so caller sees both
        coverage.issues.extend(num_report.issues)
        if coverage.is_complete:  # numbering alone failed
            coverage.is_complete = False
            if label == "PASS":
                label = "COVERAGE_INCOMPLETE"

    if not coverage.is_complete:
        logger.info(
            "Scenario coverage incomplete. Missing buckets: %s",
            coverage.missing_buckets,
        )
        # Label as incomplete so caller can act — we do NOT attempt LLM re-revision here
        # to avoid infinite loops. The caller decides how to handle the gap.
        if label == "PASS":
            label = "COVERAGE_INCOMPLETE"

    return final_text, label, coverage


# ---------------------------------------------------------------------------
# Test case verifier
# ---------------------------------------------------------------------------

async def verify_test_cases(
    generated: str,
    scenarios: str,
    context: str,
    llm: ChatOllama | None = None,
) -> tuple[str, str, CoverageReport]:
    """
    Verify test case output in two passes:
      1. LLM verifier: format validity + grounding
      2. Python coverage checker: ensures every input scenario has TCs

    Returns:
        (final_text, label, coverage_report)
        label: "PASS", "REVISED", or "COVERAGE_INCOMPLETE"
        On verifier failure: returns (generated, "PASS", report) — fail safe.
    """
    if llm is None:
        llm = build_verifier_llm()

    final_text = generated
    label = "PASS"

    # Pass 1 — LLM: format + grounding
    prompt = TC_VERIFIER_SYSTEM.format(
        scenarios=scenarios,
        context=context[:2000],
        generated=generated,
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        verdict, revised = _parse_verifier_response(response.content)

        if verdict == "REVISE" and revised:
            logger.info("Verifier (TC): format/grounding revision requested.")
            final_text = revised
            label = "REVISED"

    except Exception as exc:
        logger.warning("TC LLM verifier failed (%s) — using generator output.", exc)

    # Pass 2 — Python: cross-check TC output covers every input scenario
    expected_ids = extract_scenario_ids(scenarios)
    coverage = check_tc_coverage(final_text, expected_ids)

    if not coverage.is_complete:
        logger.info(
            "TC coverage incomplete. Missing scenarios: %s",
            coverage.missing_scenario_ids,
        )
        if label == "PASS":
            label = "COVERAGE_INCOMPLETE"

    return final_text, label, coverage


async def verify_tc_batch(
    batch_text: str,
    scenarios_block: str,
    context: str,
    llm: ChatOllama,
    timeout_secs: float = 30.0,
) -> tuple[str, str]:
    """
    Lightweight per-batch TC verifier used by the rolling generation loop.

    Checks:
      - Format validity (every TC has Title, Type, Steps)
      - All scenarios in the batch have at least one TC

    Returns:
        (final_text, label)  where label is "PASS" or "REVISED"
        Fail-safe: returns (batch_text, "PASS") on any error or timeout.
    """
    prompt = TC_VERIFIER_SYSTEM.format(
        scenarios=scenarios_block,
        context=context[:800],     # keep batch verifier prompt small
        generated=batch_text,
    )

    try:
        response = await asyncio.wait_for(
            llm.ainvoke([HumanMessage(content=prompt)]),
            timeout=timeout_secs,
        )
        verdict, revised = _parse_verifier_response(response.content)
        if verdict == "REVISE" and revised:
            logger.info("Batch verifier: revision applied.")
            return revised, "REVISED"

    except asyncio.TimeoutError:
        logger.warning("Batch verifier timed out — keeping original batch output.")
    except Exception as exc:
        logger.warning("Batch verifier error (%s) — keeping original.", exc)

    return batch_text, "PASS"
