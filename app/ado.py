"""
Azure DevOps REST client.
Creates Test Case work items with full field population:
  - Title, Steps XML
  - AreaPath, IterationPath
  - Priority, Regression flag
  - Tags, AssignedTo
  - Parent Feature link
"""
import base64
import logging
from xml.sax.saxutils import escape

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Steps XML builder
# ---------------------------------------------------------------------------

def _build_steps_xml(steps: list[str], outcomes: list[str]) -> str:
    """
    Build ADO-compatible steps XML.
    Each step gets an Action and an Expected Result parameterized string.
    """
    xml = f'<steps id="0" last="{len(steps)}">'
    for i, step in enumerate(steps):
        expected = outcomes[i] if i < len(outcomes) else "Verify the action completes successfully."
        xml += (
            f'<step id="{i + 1}" type="ValidationStep">'
            f'<parameterizedString isformatted="true">{escape(step)}</parameterizedString>'
            f'<parameterizedString isformatted="true">{escape(expected)}</parameterizedString>'
            f'</step>'
        )
    xml += "</steps>"
    return xml


# ---------------------------------------------------------------------------
# Authorization header
# ---------------------------------------------------------------------------

def _auth_header(pat: str) -> str:
    encoded = base64.b64encode(f":{pat}".encode()).decode()
    return f"Basic {encoded}"


# ---------------------------------------------------------------------------
# Create Test Case work item
# ---------------------------------------------------------------------------

def create_test_case(
    title: str,
    precondition: str,
    steps: list[str],
    outcomes: list[str],
    priority: int,
    regression: bool,
    area_path: str,
    iteration_path: str,
    base_url: str,
    project: str,
    feature_id: str,
    assigned_to: str,
    tag: str,
    pat: str,
) -> str:
    """
    Create a single Test Case work item in ADO.

    Returns a human-readable result string ("Created ID: <id>" or "Error <code>: <msg>").
    """
    url = (
        f"{base_url}/{project}"
        f"/_apis/wit/workitems/$Test%20Case"
        f"?api-version=7.0"
    )

    headers = {
        "Content-Type": "application/json-patch+json",
        "Authorization": _auth_header(pat),
    }

    steps_xml = _build_steps_xml(steps, outcomes)

    payload = [
        {"op": "add", "path": "/fields/System.Title",                       "value": title},
        {"op": "add", "path": "/fields/Microsoft.VSTS.TCM.Steps",           "value": steps_xml},
        {"op": "add", "path": "/fields/Microsoft.VSTS.TCM.LocalDataSource", "value": ""},
    ]

    # Area and Iteration
    if area_path:
        payload.append({"op": "add", "path": "/fields/System.AreaPath",      "value": area_path})
    if iteration_path:
        payload.append({"op": "add", "path": "/fields/System.IterationPath", "value": iteration_path})

    # Priority (1–4)
    payload.append({"op": "add", "path": "/fields/Microsoft.VSTS.Common.Priority", "value": priority})

    # Precondition (stored as description)
    if precondition:
        payload.append({"op": "add", "path": "/fields/System.Description", "value": precondition})

    # Regression tag
    if regression:
        payload.append({
            "op": "add",
            "path": "/fields/Microsoft.VSTS.Common.AutomationStatus",
            "value": "Planned",
        })

    # Tags
    if tag:
        payload.append({"op": "add", "path": "/fields/System.Tags", "value": tag})

    # Assigned To
    if assigned_to:
        payload.append({"op": "add", "path": "/fields/System.AssignedTo", "value": assigned_to})

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
    except requests.RequestException as exc:
        logger.error("ADO request failed: %s", exc)
        return f"Error: {exc}"

    if r.status_code not in (200, 201):
        logger.error("ADO API error %s: %s", r.status_code, r.text[:300])
        return f"Error {r.status_code}: {r.text[:200]}"

    work_item = r.json()
    work_item_id = work_item["id"]

    # Link to parent Feature if provided
    if feature_id:
        _link_to_feature(work_item_id, feature_id, base_url, project, pat)

    logger.info("Created ADO Test Case ID: %s", work_item_id)
    return f"Created ID: {work_item_id}"


# ---------------------------------------------------------------------------
# Feature link helper
# ---------------------------------------------------------------------------

def _link_to_feature(
    work_item_id: int,
    feature_id: str,
    base_url: str,
    project: str,
    pat: str,
) -> None:
    """Add a parent-child relation link from the test case to the feature."""
    url = (
        f"{base_url}/{project}"
        f"/_apis/wit/workitems/{work_item_id}"
        f"?api-version=7.0"
    )
    headers = {
        "Content-Type": "application/json-patch+json",
        "Authorization": _auth_header(pat),
    }
    payload = [{
        "op": "add",
        "path": "/relations/-",
        "value": {
            "rel": "System.LinkTypes.Hierarchy-Reverse",
            "url": (
                f"{base_url}/{project}"
                f"/_apis/wit/workitems/{feature_id}"
            ),
            "attributes": {"comment": "Linked by Testing Copilot"},
        },
    }]
    try:
        r = requests.patch(url, headers=headers, json=payload, timeout=30)
        if r.status_code not in (200, 201):
            logger.warning("Feature link failed %s: %s", r.status_code, r.text[:200])
    except requests.RequestException as exc:
        logger.warning("Feature link request failed: %s", exc)