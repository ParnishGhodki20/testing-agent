"""
Output sanitizer — post-processes LLM output before it reaches the user.

Functions:
  sanitize_scenarios()    clean scenario text
  sanitize_test_cases()   clean TC outline text
"""
import re

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

_VERIFIER_FOOTER_RE = re.compile(
    r"\n+---\n\*?Verifier\s*:\s*\S+\*?\s*$",
    re.IGNORECASE,
)

_VERDICT_LINE_RE = re.compile(
    r"^\s*(VERDICT|Verifier)\s*:\s*(PASS|REVISE|REVISED|COVERAGE_INCOMPLETE)\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sanitize_scenarios(text: str) -> str:
    """Clean scenario output for user display."""
    text = _VERDICT_LINE_RE.sub("", text)
    text = _VERIFIER_FOOTER_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Matches LLM preamble lines like "Here is the list of..." / "Below is..." / "Sure! Here are..."
_PREAMBLE_RE = re.compile(
    r"^\s*(?:here\s+(?:is|are)|below\s+(?:is|are)|sure[!,]?|certainly[!,]?|of\s+course[!,]?)"
    r"[^\n]*\n",
    re.IGNORECASE | re.MULTILINE,
)

def sanitize_test_cases(text: str) -> str:
    """
    Clean TC outline output for user display.
    - Strips verifier noise
    - Strips LLM preamble / intro lines
    - Strips any hallucinated steps, actions, or preconditions
    - Enforces [Generate Steps](#expand:TCn) on every TC that is missing it
    - Collapses excess blank lines
    """
    text = _VERDICT_LINE_RE.sub("", text)
    text = _VERIFIER_FOOTER_RE.sub("", text)

    # Strip LLM intro lines ("Here is the list...", "Sure, here are...")
    text = _PREAMBLE_RE.sub("", text)

    # Backend enforcement: strip hallucinated steps / preconditions / actions
    hallucination_start = r"(?i)(?:^|\n)\s*(?:Preconditions?|Actions?|Expected\s+(?:Outcome|Result)s?|(?:\d+\.\s*)?Steps?)[:\n]"
    next_header = r"(?=(?:\n\s*TC\d+\s*:|\n\s*SC\d+\s*:|\Z))"
    text = re.sub(hallucination_start + r".*?" + next_header, "", text, flags=re.DOTALL)

    # Enforce [Generate Steps](#expand:TCn) on every TC entry that is missing it.
    # A TC block is: a line starting with TC<n>: ... followed by Type: and Goal: lines.
    # After the Goal: line, if the expand link is absent, we inject it.
    def _inject_expand(m: re.Match) -> str:
        tc_num = m.group(1)
        block  = m.group(0)
        link   = f"[Generate Steps](#expand:TC{tc_num})"
        if link not in block:
            # Append after the Goal: line (last non-empty line in block before next header)
            block = block.rstrip() + f"\n{link}"
        return block

    # Match a TC block: TC<n>: ... up to but not including the next TC/SC or end
    text = re.sub(
        r"TC(\d+)\s*:.*?(?=\n(?:TC|SC)\s*\d+\s*:|\Z)",
        _inject_expand,
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Collapse excess blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def renumber_test_cases(text: str) -> str:
    """
    Deterministically renumbers all 'TCn:' headings from 1 to N sequentially.
    """
    counter = 1
    
    def replacer(m: re.Match) -> str:
        nonlocal counter
        replacement = f"TC{counter}:"
        if m.group(1):
            replacement = m.group(1) + replacement
        counter += 1
        return replacement

    return re.sub(r"(^|\n)(?:\*\*)?TC\s*\d+\s*:?(?:\*\*)?", replacer, text, flags=re.IGNORECASE)


def sanitize_expansion(text: str) -> str:
    """
    Post-process an expanded test case to enforce compact, aligned step formatting.

    Rules enforced:
      - Strip any LLM preamble / intro lines
      - Merge orphaned step numbers back onto the Action line
        e.g.  "1.\nAction: foo"  →  "1. Action: foo"
      - Remove blank lines between steps (lines starting with a digit or "Expected Outcome")
      - Collapse 3+ consecutive blank lines to a single blank line elsewhere
    """
    # Strip LLM intro lines ("Here are the steps...", "Sure! Below is...")
    text = _PREAMBLE_RE.sub("", text)

    # Merge "N.\n   Action:" or "N.\nAction:" onto one line
    # Handles: "1.\n   Action: foo" → "1. Action: foo"
    text = re.sub(
        r"(\d+)\.\s*\n\s*(Action\s*:)",
        r"\1. \2",
        text,
        flags=re.IGNORECASE,
    )

    # Remove blank lines that appear between a step's Action line and its Expected Outcome line
    # i.e. "Action: ...\n\nExpected Outcome:" → "Action: ...\nExpected Outcome:"
    text = re.sub(
        r"(Action\s*:[^\n]*)\n{2,}(\s*Expected\s+Outcome\s*:)",
        r"\1\n\2",
        text,
        flags=re.IGNORECASE,
    )

    # Remove blank lines between "Expected Outcome: ..." and the next numbered step
    text = re.sub(
        r"(Expected\s+Outcome\s*:[^\n]*)\n{2,}(\s*\d+\.\s*Action\s*:)",
        r"\1\n\2",
        text,
        flags=re.IGNORECASE,
    )

    # Collapse any remaining 3+ blank lines to one blank line
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

