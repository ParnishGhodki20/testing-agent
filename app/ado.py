import base64
import requests
from xml.sax.saxutils import escape


def _build_steps_xml(
steps,
outcomes
):

    xml=f'<steps id="0" last="{len(steps)}">'

    for i,step in enumerate(steps):

        expected=(
            outcomes[i]
            if i < len(outcomes)
            else "Expected result"
        )

        xml += (
        f'<step id="{i+1}" '
        f'type="ValidationStep">'
        )

        xml += (
        f'<parameterizedString '
        f'isformatted="true">'
        f'{escape(step)}'
        f'</parameterizedString>'
        )

        xml += (
        f'<parameterizedString '
        f'isformatted="true">'
        f'{escape(expected)}'
        f'</parameterizedString>'
        )

        xml+="</step>"

    xml+="</steps>"

    return xml



def create_test_case(
title,
precondition,
steps,
outcomes,
priority,
regression,
area_path,
iteration_path,
base_url,
project,
feature_id,
assigned_to,
tag,
pat
):

    url=(
      f"{base_url}/{project}"
      f"/_apis/wit/workitems/"
      f"$Test%20Case"
      f"?api-version=7.0"
    )


    headers={

      "Content-Type":
      "application/json-patch+json",

      "Authorization":
      f"Basic {
      base64.b64encode(
      f':{pat}'.encode()
      ).decode()
      }"
    }


    payload=[
      {
       "op":"add",
       "path":
       "/fields/System.Title",
       "value":title
      },
      {
       "op":"add",
       "path":
       "/fields/Microsoft.VSTS.TCM.Steps",
       "value":
       _build_steps_xml(
           steps,
           outcomes
       )
      }
    ]


    r=requests.post(
       url,
       headers=headers,
       json=payload
    )

    if r.status_code in [200,201]:

       return (
         f"Created ID:"
         f"{r.json()['id']}"
       )

    return (
      f"Error "
      f"{r.status_code}"
    )