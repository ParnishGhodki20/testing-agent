import asyncio
import logging
import re
import requests

from app.config import (
OLLAMA_BASE_URL,
OLLAMA_CHAT_MODEL
)

logger = logging.getLogger(__name__)


def _call_ollama(
prompt,
model
):

    url=f"{OLLAMA_BASE_URL}/api/generate"

    response=requests.post(
        url,
        json={
            "model":model,
            "prompt":prompt,
            "stream":False
        },
        timeout=30
    )

    return (
        response.json()
        .get("response","")
        .strip()
    )


async def generate_expected_outcomes(
steps,
model
):

    async def _outcome(step):

        prompt=f"""
Generate expected outcome:

Step:
{step}
"""

        return await asyncio.to_thread(
            _call_ollama,
            prompt,
            model
        )

    return list(
        await asyncio.gather(
            *[
             _outcome(s)
             for s in steps
            ]
        )
    )


def _parse_scenarios(
input_text
):

    results=[]

    tc_re = re.compile(
        r".*TC(\d{3}):\s*(.*)$"
    )

    current_title=""
    current_id=""
    steps=[]

    for raw in input_text.splitlines():

        line=raw.strip()

        tc=tc_re.match(line)

        if tc:

            if current_title and steps:

                results.append(
                    (
                        current_id,
                        current_title,
                        "",
                        steps
                    )
                )

            current_id=f"TC{tc.group(1)}"

            current_title=tc.group(2)

            steps=[]

            continue

        if line and line[0].isdigit():

            steps.append(line)


    if current_title and steps:

        results.append(
            (
                current_id,
                current_title,
                "",
                steps
            )
        )

    return results


async def extract_scenarios(
input_text,
model=OLLAMA_CHAT_MODEL
):

    raw = _parse_scenarios(
        input_text
    )

    scenarios=[]

    for tc_id,title,pre,steps in raw:

        scenarios.append(
            {
             "ScenarioID":tc_id,
             "Title":title,
             "Precondition":pre,
             "Steps":steps
            }
        )

    return scenarios