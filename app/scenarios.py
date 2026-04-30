"""
Scenario and test-case parsing utilities.

parse_scenario_titles()     : extract SC lines from LLM scenario output
renumber_scenarios()        : deterministic sequential renumbering (fixes gaps/duplicates)
parse_test_cases()          : extract structured TC records from LLM TC output
generate_expected_outcomes(): call Ollama to fill missing expected outcomes
                              (bounded concurrency — max 4 parallel calls)
extract_scenarios()         : combine parse + outcome generation (for ADO push)
"""
import asyncio
import logging
import re

import requests

from app.config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL

logger = logging.getLogger(__name__)

# Semaphore: prevent Ollama saturation when generating expected outcomes
_OUTCOME_SEMAPHORE = asyncio.Semaphore(4)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches: SC1: title  /  sc 1: title  /  **SC1:** title (case-insensitive)
_SC_RE = re.compile(r"\bSC\s*(\d+)\s*[:\-\.]?\s*(.+)$", re.IGNORECASE)

# Matches: TC1: title  /  TC 01: title  /  TC001: title
_TC_RE = re.compile(r"^TC\s*(\d+)\s*[:\.]?\s*(.+)$", re.IGNORECASE)

# Matches: "1. Action: ..." or "1) Action: ..." or just "1. ..."
_STEP_RE = re.compile(r"^\d+[\.\)]\s*(?:Action\s*:\s*)?(.+)$", re.IGNORECASE)

# Matches: "Expected Result: ..." or "Expected Outcome: ..."
_ER_RE = re.compile(r"^(?:Expected\s+(?:Result|Outcome)\s*:\s*)(.+)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Scenario title parser
# ---------------------------------------------------------------------------

def parse_scenario_titles(text: str) -> list[dict]:
    """
    Extract scenario records from LLM scenario output.

    Returns list of {"id": "SC1", "title": "...", "priority": "...", "description": "..."}
    Priority defaults to "Medium" if not found.
    """
    results: list[dict] = []
    current: dict | None = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        m = _SC_RE.search(line)
        if m:
            if current:
                results.append(current)
            current = {
                "id": f"SC{m.group(1)}",
                "title": m.group(2).strip(),
                "priority": "Medium",
                "description": "",
            }
            continue

        if current and line.lower().startswith("priority:"):
            pval = line[len("priority:"):].strip().title()
            if pval in ("Critical", "High", "Medium", "Low"):
                current["priority"] = pval
            continue

        if current and line.lower().startswith("description:"):
            current["description"] = line[len("description:"):].strip()

    if current:
        results.append(current)

    return results



# ---------------------------------------------------------------------------
# Scenario renumbering — deterministic post-processing
# ---------------------------------------------------------------------------

# Matches an SC header at the START of a line: SC1:, SC 2:, sc10: etc.
_SC_HEADER_LINE_RE = re.compile(r"^(\s*SC\s*)\d+(\s*:)", re.IGNORECASE | re.MULTILINE)


def renumber_scenarios(text: str) -> str:
    """
    Renumber SC headers sequentially by order of appearance.

    Replaces whatever numbers the LLM produced (which may have gaps,
    duplicates, or jumps) with a clean SC1, SC2, SC3... sequence.

    Only affects lines that START with SC<n>: — inline mentions of
    scenario numbers in descriptions are left untouched.

    Returns the corrected text.
    """
    counter = [0]

    def _replacer(m: re.Match) -> str:
        counter[0] += 1
        # Preserve original whitespace/casing prefix and suffix (e.g. "SC " and ":")
        return f"{m.group(1)}{counter[0]}{m.group(2)}"

    return _SC_HEADER_LINE_RE.sub(_replacer, text)


# ---------------------------------------------------------------------------
# Test case parser
# ---------------------------------------------------------------------------

def parse_test_cases(text: str) -> list[dict]:
    """
    Parse LLM test case output into structured records.

    Returns list of:
    {
        "id":           "TC1",
        "title":        "...",
        "type":         "Positive|Negative|Edge|Exception",
        "preconditions": ["..."],
        "steps":        [{"action": "...", "expected": "..."}],
        "scenario_id":  "SC1"  (populated if SC header found before TC)
    }
    """
    results: list[dict] = []
    current_tc: dict | None = None
    current_scenario_id = ""
    in_preconditions = False
    last_step: dict | None = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        # Scenario header — track which scenario TCs belong to
        sc = _SC_RE.match(line)
        if sc:
            current_scenario_id = f"SC{sc.group(1)}"
            in_preconditions = False
            continue

        # TC header
        tc = _TC_RE.match(line)
        if tc:
            if current_tc:
                results.append(current_tc)
            current_tc = {
                "id": f"TC{int(tc.group(1)):03d}",
                "title": tc.group(2).strip(),
                "type": "",
                "preconditions": [],
                "steps": [],
                "scenario_id": current_scenario_id,
            }
            last_step = None
            in_preconditions = False
            continue

        if current_tc is None:
            continue

        low = line.lower()

        # Type line
        if low.startswith("type:"):
            current_tc["type"] = line[5:].strip()
            continue

        # Preconditions block
        if low.startswith("precondition"):
            in_preconditions = True
            continue
        if in_preconditions and line.startswith("-"):
            current_tc["preconditions"].append(line[1:].strip())
            continue

        # Numbered step / action
        step_m = _STEP_RE.match(line)
        if step_m:
            in_preconditions = False
            last_step = {"action": step_m.group(1).strip(), "expected": ""}
            current_tc["steps"].append(last_step)
            continue

        # Expected result — attach to last step
        er_m = _ER_RE.match(line)
        if er_m and last_step is not None:
            last_step["expected"] = er_m.group(1).strip()
            continue

        # Continuation lines for expected result
        if last_step is not None and last_step["expected"]:
            last_step["expected"] += " " + line

    if current_tc:
        results.append(current_tc)

    return results


# ---------------------------------------------------------------------------
# Expected outcome generator (for ADO push — fills blank expected fields)
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, model: str) -> str:
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=30,
        )
        return r.json().get("response", "").strip()
    except Exception as exc:
        logger.warning("Ollama call failed: %s", exc)
        return "Expected result not available."


async def _generate_one_outcome(step: str, model: str) -> str:
    async with _OUTCOME_SEMAPHORE:
        prompt = f"Generate a concise expected outcome for this test step:\n\nStep: {step}"
        return await asyncio.to_thread(_call_ollama, prompt, model)


async def generate_expected_outcomes(
    steps: list[str],
    model: str = OLLAMA_CHAT_MODEL,
) -> list[str]:
    """
    Generate expected outcomes for a list of action steps.
    Bounded to 4 concurrent Ollama calls via semaphore.
    """
    return list(
        await asyncio.gather(*[_generate_one_outcome(s, model) for s in steps])
    )


# ---------------------------------------------------------------------------
# High-level: extract scenarios (for ADO push flow)
# ---------------------------------------------------------------------------

async def extract_scenarios(
    input_text: str,
    model: str = OLLAMA_CHAT_MODEL,
) -> list[dict]:
    """
    Parse raw scenario text and return enriched records.
    Used by the ADO push flow.
    """
    return parse_scenario_titles(input_text)