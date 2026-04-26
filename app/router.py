"""
Intent router: classifies user input into one of three modes.
Uses phrase-scoring + regex heuristics. No LLM call required.
"""
import re
from typing import Literal

Intent = Literal["scenario", "testcase", "testcase_no_scenarios", "coverage_revise", "general"]

# ---------------------------------------------------------------------------
# Keyword / phrase sets — include common typos and abbreviations
# ---------------------------------------------------------------------------

_SCENARIO_PHRASES: frozenset[str] = frozenset([
    "scenario", "scenarios", "scanrio", "scanrios", "scneario", "scnearios",
    "scenaro", "scen",
    "test scenario", "test scenarios",
    "what to test", "what should we test", "what shud we test",
    "what should i test", "what can we test",
    "test coverage", "coverage", "testing areas", "testing area",
    "what are the scenarios", "list scenarios",
    "generate scenarios", "gen scenarios", "create scenarios",
    "give scenarios", "show scenarios", "identify scenarios",
    "identify test scenarios", "testing scope", "scope of testing",
    "areas to test", "coverage areas", "test scope",
])

_TC_PHRASES: frozenset[str] = frozenset([
    "test case", "test cases", "testcase", "testcases",
    "tc", "generate tc", "gen tc",
    "positive test", "negative test", "edge test", "exception test",
    "nagative", "nagtive", "negatve", "negitive",
    "edgecase", "edge case", "edge cases",
    "write test", "create test case", "create test cases",
    "detailed test", "all test cases", "exhaustive test",
    "full test", "give test cases", "list test cases",
    "generate test cases", "test case generation",
    "cases for", "cases from", "all cases",
    "generate cases", "gen cases",
])

# Phrases that indicate the user is pointing out a coverage gap
_COVERAGE_REVISE_PHRASES: frozenset[str] = frozenset([
    "you missed", "u missed", "missed scenarios", "missing scenarios",
    "missed test cases", "missing test cases", "missed cases",
    "add remaining", "generate missing", "remaining scenarios",
    "incomplete", "not covered", "you skipped", "skipped scenarios",
    "skipped test", "add missing", "generate remaining",
    "you forgot", "forgotten", "left out", "not generated",
    "only generated", "didnt generate", "didn t generate",
    "cover remaining", "cover missing", "generate for remaining",
    "fill the gap", "fill gap", "complete the",
])

# Patterns that suggest "what should we test" intent → route to scenario
_SCENARIO_PATTERNS: list[str] = [
    r"what\s+(should|shud|shall|can)\s+(we|i|you)\s+test",
    r"how\s+(should|shud|shall|can)\s+(we|i|you)\s+test",
    r"(testing|test)\s+plan",
    r"(check|verify|validate).{0,20}feature",
    r"suggest.{0,20}test",
]


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _score(normalized: str, phrase_set: frozenset[str]) -> float:
    """
    Return cumulative score: multi-word phrases score higher than single words.
    Longer phrase match = more confident signal.
    """
    score = 0.0
    for phrase in phrase_set:
        if phrase in normalized:
            score += len(phrase.split()) * 1.5
    return score


def classify_intent(user_message: str, has_scenarios: bool = False) -> Intent:
    """
    Classify intent from raw user message.

    Args:
        user_message:  Raw user input string.
        has_scenarios: True if the session already has stored scenarios.

    Returns:
        "scenario"              — user wants scenario generation
        "testcase"              — user wants test cases (scenarios exist)
        "testcase_no_scenarios" — user wants test cases but no scenarios in session
        "coverage_revise"       — user is reporting missing coverage to be filled
        "general"               — general QA question
    """
    normalized = _normalize(user_message)

    # Coverage correction must be checked FIRST — it overrides other signals
    # e.g. "you missed three scenarios" scores on _SCENARIO_PHRASES too
    cr_score = _score(normalized, _COVERAGE_REVISE_PHRASES)
    if cr_score >= 1.5:  # at least one multi-word phrase matched
        return "coverage_revise"

    sc_score = _score(normalized, _SCENARIO_PHRASES)
    tc_score = _score(normalized, _TC_PHRASES)

    # Regex patterns for ambiguous scenario-style questions
    if sc_score == 0 and tc_score == 0:
        for pat in _SCENARIO_PATTERNS:
            if re.search(pat, normalized):
                sc_score += 1.0
                break

    # Fully ambiguous → general QA
    if sc_score == 0 and tc_score == 0:
        return "general"

    if tc_score > sc_score:
        return "testcase" if has_scenarios else "testcase_no_scenarios"

    return "scenario"
