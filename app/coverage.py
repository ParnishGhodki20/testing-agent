"""
Coverage completeness checker — deterministic, no LLM call.

Provides:
  check_scenario_coverage()   — checks 5 required coverage buckets
  check_scenario_numbering()  — validates SC numbering is sequential (no gaps/duplicates)
  check_tc_coverage()         — cross-checks TC output covers every input SC
  extract_scenario_ids()      — extracts SC IDs from scenario text
"""
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# Matches SC1:, SC 1:, sc2: etc.
_SC_ID_RE = re.compile(r"\bSC\s*(\d+)\s*[:\-\.]?", re.IGNORECASE)

# Matches TC1:, TC 01., TC001: etc.
_TC_ID_RE = re.compile(r"\bTC\s*(\d+)\s*[:\.]", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Coverage bucket definitions
# Keywords that signal each bucket is represented in scenario output
# ---------------------------------------------------------------------------
_BUCKET_KEYWORDS: dict[str, list[str]] = {
    "functional": [
        "valid", "success", "creat", "login", "submit", "save",
        "positive", "standard", "normal", "basic", "core", "happy",
        "complet", "workflow", "process",
    ],
    "invalid_negative": [
        "invalid", "missing", "empty", "null", "blank", "error",
        "negative", "incorrect", "wrong", "fail", "reject", "unauthor",
        "without", "no ", "not provided",
    ],
    "boundary_edge": [
        "boundary", "limit", "max", "min", "edge", "exceed", "zero",
        "large", "special character", "long", "short", "exact",
        "threshold", "overflow",
    ],
    "dependency_security": [
        "depend", "integrat", "security", "access", "permission",
        "role", "link", "relation", "auth", "privilege", "session",
        "concurrent", "lock",
    ],
    "regression": [
        "regression", "existing", "backward", "previous", "break",
        "change", "impact", "affected", "retain", "preserv",
    ],
}


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class CoverageReport:
    is_complete: bool
    missing_scenario_ids: list[str] = field(default_factory=list)
    missing_buckets: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.is_complete:
            return "✅ Coverage complete."
        parts = []
        if self.missing_buckets:
            parts.append(f"Missing coverage buckets: {', '.join(self.missing_buckets)}")
        if self.missing_scenario_ids:
            parts.append(f"No test cases for: {', '.join(self.missing_scenario_ids)}")
        return "⚠️ " + " | ".join(parts)


# ---------------------------------------------------------------------------
# Scenario coverage check
# ---------------------------------------------------------------------------

def check_scenario_coverage(scenario_text: str) -> CoverageReport:
    """
    Check that scenario output contains keywords representing all 5 coverage buckets.

    Returns CoverageReport with is_complete=True only when all buckets are present.
    """
    lowered = scenario_text.lower()
    missing: list[str] = []

    for bucket, keywords in _BUCKET_KEYWORDS.items():
        if not any(kw in lowered for kw in keywords):
            missing.append(bucket)

    issues = [f"Missing coverage bucket: {b}" for b in missing]
    return CoverageReport(
        is_complete=len(missing) == 0,
        missing_buckets=missing,
        issues=issues,
    )


# ---------------------------------------------------------------------------
# Scenario numbering integrity check
# ---------------------------------------------------------------------------

# Only matches SC headers at the start of a line (avoids inline SC references)
_SC_HEADER_NUM_RE = re.compile(r"^\s*SC\s*(\d+)\s*:", re.IGNORECASE | re.MULTILINE)


def check_scenario_numbering(scenario_text: str) -> CoverageReport:
    """
    Validate that SC numbering is strictly sequential with no gaps or duplicates.

    Expected: SC1, SC2, SC3, SC4 ...
    Invalid:  SC1, SC5, SC5, SC10   (skips + duplicate + jump)

    Returns CoverageReport(is_complete=True) only when all numbers are
    sequential from 1 with no repeats.
    """
    nums = [int(m.group(1)) for m in _SC_HEADER_NUM_RE.finditer(scenario_text)]

    if not nums:
        return CoverageReport(
            is_complete=False,
            issues=["No scenario headers (SCn:) found in output"],
        )

    issues: list[str] = []
    seen: set[int] = set()

    for i, n in enumerate(nums):
        expected = i + 1
        if n in seen:
            issues.append(f"Duplicate SC{n} at position {i + 1}")
        elif n != expected:
            issues.append(f"Numbering break at position {i + 1}: expected SC{expected}, got SC{n}")
        seen.add(n)

    return CoverageReport(
        is_complete=len(issues) == 0,
        issues=issues,
    )


# ---------------------------------------------------------------------------
# TC coverage check
# ---------------------------------------------------------------------------

def extract_scenario_ids(text: str) -> list[str]:
    """
    Extract SC IDs (sorted, de-duplicated) from scenario or TC text.
    Returns e.g. ["SC1", "SC2", "SC3"]
    """
    nums = sorted({int(m.group(1)) for m in _SC_ID_RE.finditer(text)})
    return [f"SC{n}" for n in nums]


def check_tc_coverage(tc_text: str, expected_ids: list[str]) -> CoverageReport:
    """
    Cross-check that TC output contains a section for every expected scenario ID.

    Args:
        tc_text:       Raw LLM TC output text.
        expected_ids:  List of scenario IDs that must be covered, e.g. ["SC1","SC2","SC3"]

    Returns CoverageReport with missing_scenario_ids populated for any gaps.
    """
    covered = set(extract_scenario_ids(tc_text))
    missing = [sid for sid in expected_ids if sid not in covered]
    issues  = [f"No test cases found for: {sid}" for sid in missing]

    return CoverageReport(
        is_complete=len(missing) == 0,
        missing_scenario_ids=missing,
        issues=issues,
    )


def has_tcs_for_scenario(tc_text: str, scenario_id: str) -> bool:
    """
    Returns True if the TC text contains at least one TC under the given scenario.
    E.g. scenario_id = "SC3"
    """
    # Find SC3: header then look for at least one TC after it before next SC
    pattern = re.compile(
        rf"SC\s*{re.escape(scenario_id.lstrip('SC'))}\s*:.*?(?=SC\s*\d+\s*:|$)",
        re.IGNORECASE | re.DOTALL,
    )
    m = pattern.search(tc_text)
    if not m:
        return False
    return bool(_TC_ID_RE.search(m.group(0)))
