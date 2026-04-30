"""
Output sanitizer — post-processes LLM output before it reaches the user.

Functions:
  sanitize_scenarios()    clean scenario text (strip verifier noise)
  sanitize_test_cases()   clean TC text (fix formatting, strip metadata, repair placeholders)
"""
import re

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Verifier footer as appended by main.py:  "---\n*Verifier: REVISED*"
_VERIFIER_FOOTER_RE = re.compile(
    r"\n+---\n\*?Verifier\s*:\s*\S+\*?\s*$",
    re.IGNORECASE,
)

# Any line that is just a verdict/verifier status marker
_VERDICT_LINE_RE = re.compile(
    r"^\s*(VERDICT|Verifier)\s*:\s*(PASS|REVISE|REVISED|COVERAGE_INCOMPLETE)\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# TC header where the title is still a template placeholder
# Matches: TC1: [Test Case Title]  /  TC2: [Title]  /  TC3: [Enter title here]
_TC_PLACEHOLDER_TITLE_RE = re.compile(
    r"^(TC\s*\d+\s*[:\.])\s*\[[\w\s]+\]\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# Inline mention of [Test Case Title] anywhere (e.g. inside a line)
_INLINE_PLACEHOLDER_RE = re.compile(
    r"\[Test\s+Case\s+Title\]|\[TC\s+Title\]|\[Title\]|\[Enter\s+title\]",
    re.IGNORECASE,
)

# Detects Action: and Expected Result: merged on the same line.
# Pattern: line contains "Action:" or a numbered step, AND "Expected Result:" later in same line.
_MERGED_AR_RE = re.compile(
    r"((?:\d+[\.\)]\s*)?(?:Action|Step)\s*:\s+.+?)\s{1,}(Expected\s+(?:Result|Outcome)\s*:)",
    re.IGNORECASE,
)

# Standardise "Type:" labels — value must be one of the known categories
_VALID_TYPE_VALUES = frozenset(["positive", "negative", "edge", "exception", "boundary"])
_TYPE_LINE_RE = re.compile(
    r"^(\s*Type\s*:\s*)(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fix_merged_action_er(text: str) -> str:
    """
    Split lines where Action/step and Expected Result/Outcome appear concatenated
    on the same line. Preserves indentation.

    Handles:
      1. Action: Click submit  Expected Result: Modal closes
      2. 1. Do something  Expected Outcome: Result appears
      3. Action: Click button Expected Outcome: Page refreshes

    After:
      1. Action: Click submit
         Expected Result: Modal closes

      1. Do something
         Expected Outcome: Result appears
    """
    # Match Expected Result: or Expected Outcome: mid-line
    _er_mid_re = re.compile(
        r"Expected\s+(?:Result|Outcome)\s*:", re.IGNORECASE
    )
    # Match Action: or numbered step at start
    _act_start_re = re.compile(
        r"(?:\d+[\.\)]\s*)?(?:Action|Step)\s*:", re.IGNORECASE
    )
    # Match plain numbered step: "1. Do something"
    _num_start_re = re.compile(r"\d+[\.\)]\s+")

    lines = text.splitlines()
    result: list[str] = []

    for line in lines:
        er_m = _er_mid_re.search(line)
        if not er_m:
            result.append(line)
            continue

        # Check if there's an action/step or numbered step BEFORE the ER on the same line
        prefix = line[:er_m.start()]
        act_m = _act_start_re.search(prefix)
        num_m = _num_start_re.match(prefix.lstrip())

        # Only split if there's meaningful content before ER
        if (act_m or num_m) and len(prefix.strip()) > 3:
            indent = " " * (len(line) - len(line.lstrip()))
            action_part = prefix.rstrip()
            er_part     = line[er_m.start():]
            result.append(action_part)
            result.append(indent + er_part.lstrip())
        else:
            result.append(line)

    return "\n".join(result)



def _repair_placeholder_titles(text: str) -> str:
    """
    Replace placeholder TC titles with a generic sensible label.
    TC3: [Test Case Title]  →  TC3: Test Scenario Verification
    """
    def _replacer(m: re.Match) -> str:
        return m.group(1) + " Test Scenario Verification"

    text = _TC_PLACEHOLDER_TITLE_RE.sub(_replacer, text)
    text = _INLINE_PLACEHOLDER_RE.sub("Test Scenario Verification", text)
    return text


def _standardise_type_labels(text: str) -> str:
    """
    Normalise Type: label values to Title Case.
    Removes any Type: line whose value isn't a recognised category.
    """
    def _norm(m: re.Match) -> str:
        raw = m.group(2).strip().lower()
        if raw in _VALID_TYPE_VALUES:
            return m.group(1) + raw.title()
        # Unknown value — keep it but normalise to title case
        return m.group(1) + m.group(2).strip().title()

    return _TYPE_LINE_RE.sub(_norm, text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sanitize_scenarios(text: str) -> str:
    """
    Clean scenario output for user display.

    - Strips verifier footers / verdict lines
    - Collapses excessive blank lines
    """
    text = _VERDICT_LINE_RE.sub("", text)
    text = _VERIFIER_FOOTER_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sanitize_test_cases(text: str) -> str:
    """
    Clean TC output for user display.

    Repairs:
      1. Verifier verdict lines / footer
      2. Placeholder TC titles ([Test Case Title])
      3. Action / Expected Result merged on same line
      4. Type: label normalisation
      5. Excessive blank lines
    """
    # 1. Strip verifier metadata
    text = _VERDICT_LINE_RE.sub("", text)
    text = _VERIFIER_FOOTER_RE.sub("", text)

    # 2. Fix merged Action/Expected Result lines
    text = _fix_merged_action_er(text)

    # 3. Repair placeholder titles
    text = _repair_placeholder_titles(text)

    # 4. Standardise Type: labels
    text = _standardise_type_labels(text)

    # 4.5. Remove empty lines and indentation before Expected Outcome and Action steps
    text = re.sub(r"([^\n])\n+\s*(Expected Outcome:)", r"\1\n\2", text, flags=re.IGNORECASE)
    text = re.sub(r"([^\n])\n+\s*(\d+\.\s*Action:)", r"\1\n\2", text, flags=re.IGNORECASE)

    # 5. Collapse blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def renumber_test_cases(text: str) -> str:
    """
    Deterministically renumbers all 'TCn:' headings from 1 to N sequentially,
    removing any gaps caused by parallel batch generation or LLM skipped numbers.
    """
    counter = 1
    
    def replacer(m: re.Match) -> str:
        nonlocal counter
        replacement = f"TC{counter}:"
        if m.group(1):
            replacement = m.group(1) + replacement
        counter += 1
        return replacement

    # Matches "**TC1:**" or "TC 2:" or "TC03:"
    # Using re.IGNORECASE to catch "tc1:" or "Tc1:"
    return re.sub(r"(^|\n)(?:\*\*)?TC\s*\d+\s*:?(?:\*\*)?", replacer, text, flags=re.IGNORECASE)
