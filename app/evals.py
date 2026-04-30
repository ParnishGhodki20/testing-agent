"""
Eval layer — deterministic, rule-based validation replacing the LLM verifier.

No LLM calls. All checks are regex/string-based and run in <100ms total.

Public API:
  eval_scenarios(text)                → EvalResult
  eval_tc_batch(text, expected_ids)   → EvalResult
  eval_tc_output(text, expected_ids)  → EvalResult  (full merged output)
"""
import logging
import re
from dataclasses import dataclass, field

from app.coverage import (
    CoverageReport,
    check_scenario_coverage,
    check_scenario_numbering,
    check_tc_coverage,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Outcome of a deterministic eval pass."""
    passed: bool
    issues: list[str] = field(default_factory=list)
    missing_sc_ids: list[str] = field(default_factory=list)
    coverage_report: CoverageReport | None = None

    def summary(self) -> str:
        if self.passed:
            return "✅ Eval passed."
        parts = []
        if self.issues:
            parts.append(f"{len(self.issues)} issue(s)")
        if self.missing_sc_ids:
            parts.append(f"Missing SCs: {', '.join(self.missing_sc_ids)}")
        return "⚠️ " + " | ".join(parts)


# ---------------------------------------------------------------------------
# Regex patterns for structural checks
# ---------------------------------------------------------------------------

# SC header at start of line
_SC_HEADER_RE = re.compile(r"^\s*SC\s*\d+\s*:", re.IGNORECASE | re.MULTILINE)

# TC header at start of line
_TC_HEADER_RE = re.compile(r"^\s*TC\s*\d+\s*[:\.]", re.IGNORECASE | re.MULTILINE)

# Action: or numbered step line (1. Action: ...)
_ACTION_RE = re.compile(
    r"^\s*(?:\d+[\.\)]\s*)?Action\s*:", re.IGNORECASE | re.MULTILINE
)

# Expected Result: line
_ER_RE = re.compile(
    r"^\s*Expected\s+(?:Result|Outcome)\s*:", re.IGNORECASE | re.MULTILINE
)

# Placeholder titles that should never reach output
_PLACEHOLDER_RE = re.compile(
    r"\[Test\s+Case\s+Title\]|\[TC\s+Title\]|\[Title\]|\[Enter\s+title\]"
    r"|\[Scenario\s+Title\]",
    re.IGNORECASE,
)

# Verifier metadata that should never reach output
_VERIFIER_META_RE = re.compile(
    r"^\s*(?:VERDICT|Verifier)\s*:\s*(?:PASS|REVISE|REVISED|COVERAGE_INCOMPLETE)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# TC labels that should NOT appear in scenario output
_TC_IN_SCENARIO_RE = re.compile(
    r"(?:^|\n)\s*(?:TC\s*\d+\s*[:\.]|Action\s*:|Expected\s+(?:Result|Outcome)\s*:"
    r"|Type\s*:\s*(?:Positive|Negative|Edge|Exception|Boundary)"
    r"|Preconditions?\s*:)",
    re.IGNORECASE,
)

# Numbered step pattern: "1.", "2)", "1. Action:", etc.
_NUMBERED_STEP_RE = re.compile(
    r"^\s*(\d+)\s*[\.\)]\s*", re.MULTILINE
)


# ---------------------------------------------------------------------------
# Scenario evals
# ---------------------------------------------------------------------------

def eval_scenarios(scenario_text: str) -> EvalResult:
    """
    Evaluate scenario output — all checks are deterministic.

    Checks:
      1. SC numbering is sequential (no gaps, no duplicates)
      2. Required coverage buckets are present (functional, negative, edge, etc.)
      3. No test case content mixed in (no Action/ER/TC lines)
      4. No placeholder titles
      5. No verifier metadata
    """
    issues: list[str] = []

    # 1. Numbering integrity
    num_report = check_scenario_numbering(scenario_text)
    if not num_report.is_complete:
        issues.extend(num_report.issues)

    # 2. Coverage buckets
    cov_report = check_scenario_coverage(scenario_text)
    if not cov_report.is_complete:
        issues.extend(cov_report.issues)

    # 3. No TC content mixed in
    tc_matches = _TC_IN_SCENARIO_RE.findall(scenario_text)
    if tc_matches:
        issues.append(
            f"Test case content found in scenario output "
            f"({len(tc_matches)} occurrence(s)). Scenarios should be titles + descriptions only."
        )

    # 4. No placeholders
    placeholders = _PLACEHOLDER_RE.findall(scenario_text)
    if placeholders:
        issues.append(f"Placeholder titles detected: {placeholders[:3]}")

    # 5. No verifier metadata
    if _VERIFIER_META_RE.search(scenario_text):
        issues.append("Verifier metadata found in output — should be stripped.")

    passed = len(issues) == 0

    if issues:
        logger.info("Scenario eval: %d issue(s) — %s", len(issues), issues[:3])

    return EvalResult(
        passed=passed,
        issues=issues,
        coverage_report=cov_report,
    )


# ---------------------------------------------------------------------------
# Test case evals
# ---------------------------------------------------------------------------

def _check_step_er_pairs(tc_text: str) -> list[str]:
    """
    Verify that every numbered step (1., 2., 3.) has a corresponding
    Expected Result within the same TC block.

    Returns a list of issue descriptions (empty = all good).
    """
    issues: list[str] = []

    # Split by TC headers to check each TC individually
    tc_blocks = re.split(r"(?=^\s*TC\s*\d+\s*[:\.])", tc_text, flags=re.IGNORECASE | re.MULTILINE)

    for block in tc_blocks:
        if not block.strip():
            continue

        # Get TC identifier
        tc_match = _TC_HEADER_RE.search(block)
        if not tc_match:
            continue
        tc_id = tc_match.group(0).strip().rstrip(":.")

        # Count numbered steps and Expected Result lines
        steps = _NUMBERED_STEP_RE.findall(block)
        er_count = len(_ER_RE.findall(block))

        if len(steps) > 0 and er_count == 0:
            issues.append(f"{tc_id}: {len(steps)} step(s) but 0 Expected Results")
        elif len(steps) > er_count + 1:
            # Allow some tolerance (off by 1 is format noise), flag if >1 gap
            issues.append(
                f"{tc_id}: {len(steps)} step(s) but only {er_count} Expected Result(s)"
            )

    return issues


def eval_tc_batch(
    tc_text: str,
    expected_sc_ids: list[str],
) -> EvalResult:
    """
    Evaluate a single batch of TC output — lightweight, per-batch check.

    Checks:
      1. All expected SCs in this batch have at least one TC
      2. No placeholder titles
      3. Step / Expected Result pairing
      4. No verifier metadata
    """
    issues: list[str] = []
    missing_ids: list[str] = []

    # 1. SC coverage for this batch
    cov = check_tc_coverage(tc_text, expected_sc_ids)
    if not cov.is_complete:
        missing_ids = cov.missing_scenario_ids
        issues.extend(cov.issues)

    # 2. Placeholders
    placeholders = _PLACEHOLDER_RE.findall(tc_text)
    if placeholders:
        issues.append(f"Placeholder titles detected: {placeholders[:3]}")

    # 3. Step/ER pairing
    pair_issues = _check_step_er_pairs(tc_text)
    if pair_issues:
        issues.extend(pair_issues[:5])  # cap to avoid noise

    # 4. Verifier metadata
    if _VERIFIER_META_RE.search(tc_text):
        issues.append("Verifier metadata found in batch output.")

    passed = len(issues) == 0
    if issues:
        logger.info("TC batch eval: %d issue(s) — %s", len(issues), issues[:3])

    return EvalResult(
        passed=passed,
        issues=issues,
        missing_sc_ids=missing_ids,
        coverage_report=cov,
    )


def eval_tc_output(
    tc_text: str,
    expected_sc_ids: list[str],
) -> EvalResult:
    """
    Evaluate final merged TC output — full cross-check.

    Checks:
      1. Every expected SC has at least one TC
      2. No placeholder titles
      3. Step / Expected Result pairing
      4. No verifier metadata
      5. TC headers exist (output is not empty/garbage)
    """
    issues: list[str] = []
    missing_ids: list[str] = []

    # 1. SC coverage
    cov = check_tc_coverage(tc_text, expected_sc_ids)
    if not cov.is_complete:
        missing_ids = cov.missing_scenario_ids
        issues.extend(cov.issues)

    # 2. Placeholders
    placeholders = _PLACEHOLDER_RE.findall(tc_text)
    if placeholders:
        issues.append(f"Placeholder titles in final output: {placeholders[:3]}")

    # 3. Step/ER pairing
    pair_issues = _check_step_er_pairs(tc_text)
    if pair_issues:
        issues.extend(pair_issues[:10])

    # 4. Verifier metadata
    if _VERIFIER_META_RE.search(tc_text):
        issues.append("Verifier metadata in final output.")

    # 5. TC headers exist
    tc_count = len(_TC_HEADER_RE.findall(tc_text))
    if tc_count == 0:
        issues.append("No TC headers found in output — generation may have failed.")

    passed = len(issues) == 0 and len(missing_ids) == 0
    if issues:
        logger.info("TC output eval: %d issue(s), %d missing SCs", len(issues), len(missing_ids))

    return EvalResult(
        passed=passed,
        issues=issues,
        missing_sc_ids=missing_ids,
        coverage_report=cov,
    )


# ---------------------------------------------------------------------------
# Coverage map builder — SC → [TC1, TC2, ...]
# ---------------------------------------------------------------------------

# SC header regex for splitting
_SC_SPLIT_RE = re.compile(r"(?=^\s*SC\s*\d+\s*:)", re.IGNORECASE | re.MULTILINE)
_SC_ID_EXTRACT_RE = re.compile(r"^\s*SC\s*(\d+)\s*:", re.IGNORECASE)
_TC_ID_EXTRACT_RE = re.compile(r"^\s*TC\s*(\d+)\s*[:\.]", re.IGNORECASE | re.MULTILINE)


def build_coverage_map(
    tc_text: str,
    expected_sc_ids: list[str],
) -> dict[str, list[str]]:
    """
    Build a mapping of SC → list of TC ids found under that SC header.

    Example return:
        {"SC1": ["TC1", "TC2"], "SC2": ["TC3"], "SC3": []}

    Any SC with an empty list = missing coverage.
    """
    # Initialize all expected SCs with empty lists
    coverage: dict[str, list[str]] = {sid: [] for sid in expected_sc_ids}

    # Split by SC headers
    blocks = _SC_SPLIT_RE.split(tc_text)

    for block in blocks:
        if not block.strip():
            continue

        # Find which SC this block belongs to
        sc_match = _SC_ID_EXTRACT_RE.search(block)
        if not sc_match:
            continue
        sc_id = f"SC{sc_match.group(1)}"

        # Find all TC ids within this SC block
        tc_ids = [f"TC{m.group(1)}" for m in _TC_ID_EXTRACT_RE.finditer(block)]

        if sc_id in coverage:
            coverage[sc_id].extend(tc_ids)
        else:
            coverage[sc_id] = tc_ids

    return coverage


def format_coverage_map(coverage: dict[str, list[str]]) -> str:
    """
    Format coverage map for user display.

    Example output:
        SC1 → TC1, TC2, TC3
        SC2 → TC4, TC5
        SC3 → ⚠️ No test cases
    """
    lines: list[str] = []
    for sc_id, tc_ids in coverage.items():
        if tc_ids:
            lines.append(f"{sc_id} → {', '.join(tc_ids)}")
        else:
            lines.append(f"{sc_id} → ⚠️ No test cases")
    return "\n".join(lines)

