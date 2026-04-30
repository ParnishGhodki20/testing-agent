"""
Icertis Testing Copilot — Chainlit entry point.

Architecture (v2 — eval-based):
  on_chat_start  → upload → extract → index → ready
  on_message     → classify_intent → mode chain
                 → eval layer (Python, deterministic)
                 → one retry if eval fails on completeness
                 → sanitizer / formatter
                 → final output
                 → [optional] ADO push flow for test cases
"""
import asyncio
import logging
import re
import uuid

import chainlit as cl
from langchain_core.messages import AIMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config
from app.ado import create_test_case
from app.config import (
    ADO_AREA_PATH, ADO_ASSIGNED_TO, ADO_BASE_URL,
    ADO_FEATURE_ID, ADO_ITERATION_PATH, ADO_PAT,
    ADO_PROJECT, ADO_TAG,
    TC_BATCH_SIZE,
    FAST_MODE,
)
from app.coverage import check_tc_coverage
from app.evals import eval_scenarios, eval_tc_output, build_coverage_map, format_coverage_map
from app.extractors import extract_text
from app.rag import (
    build_chain, build_index, build_llm, build_tc_llm,
    generate_tcs_rolling, generate_coverage_revision,
)
from app.sanitizer import sanitize_scenarios, sanitize_test_cases
from app.router import classify_intent
from app.scenarios import (
    generate_expected_outcomes,
    parse_scenario_titles,
    parse_test_cases,
    renumber_scenarios,
)

logger = logging.getLogger(__name__)
_AUTHOR = "icertis-testing-copilot"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trim_summary(summary: str, max_chars: int = 4000) -> str:
    """Keep most-recent portion of rolling summary, never cut mid-line."""
    if len(summary) <= max_chars:
        return summary
    trimmed = summary[-max_chars:]
    nl = trimmed.find("\n")
    return trimmed[nl + 1:] if nl != -1 else trimmed


def _last_tc_num(tc_text: str) -> int:
    """Return the highest TC number seen in tc_text, or 1 if none found."""
    nums = re.findall(r"\bTC(\d+)\b", tc_text, re.IGNORECASE)
    return max((int(n) for n in nums), default=0) + 1


def _update_memory(
    user_input: str,
    final_answer: str,
    chat_history: list,
    summary: str,
) -> tuple[list, str]:
    """Append turn to chat history and rolling summary; return updated copies."""
    chat_history = chat_history + [
        HumanMessage(content=user_input),
        AIMessage(content=final_answer),
    ]
    summary += f"\nUser: {user_input}\nAssistant: {final_answer[:800]}\n"
    return chat_history[-12:], _trim_summary(summary)


# ---------------------------------------------------------------------------
# on_chat_start
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)

    # ── File upload ──────────────────────────────────────────────────────────
    files = await cl.AskFileMessage(
        content=(
            "**Icertis Testing Copilot** ready.\n\n"
            "Upload one or more feature / requirement documents to begin.\n"
            "Supported: PDF, DOCX, PPTX, XLSX, CSV, TXT, JSON, XML, and more."
        ),
        accept=config.ACCEPT_FILE_TYPES,
        max_size_mb=config.MAX_SIZE_MB,
        max_files=config.MAX_FILES,
        timeout=86400,
        raise_on_timeout=False,
    ).send()

    if not files:
        await cl.Message(
            content="No files uploaded. Please refresh and upload your documents.",
            author=_AUTHOR,
        ).send()
        return

    # ── Text extraction ──────────────────────────────────────────────────────
    all_chunks: list[str] = []
    processed: list[str] = []
    skipped: list[str] = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    async with cl.Step(name="📄 Extracting document text…") as step:
        for file in files:
            ext = file.name.rsplit(".", 1)[-1].lower()
            text = await extract_text(file.path, ext)
            if text:
                chunks = splitter.split_text(text)
                all_chunks.extend(chunks)
                processed.append(file.name)
                logger.info("Extracted %d chunks from %s", len(chunks), file.name)
            else:
                skipped.append(file.name)
                logger.warning("Could not extract text from: %s", file.name)

        step.output = (
            f"✅ Processed: {', '.join(processed)}"
            + (f"\n⚠️ Skipped (unsupported/empty): {', '.join(skipped)}" if skipped else "")
        )

    if not all_chunks:
        await cl.Message(
            content="⚠️ Could not extract text from any uploaded file. Please check file formats.",
            author=_AUTHOR,
        ).send()
        return

    # ── Build index ──────────────────────────────────────────────────────────
    async with cl.Step(name="🔍 Building knowledge index…"):
        docsearch = await asyncio.to_thread(build_index, all_chunks, session_id)

    retriever = docsearch.as_retriever(
        search_type="mmr",
        search_kwargs={"k": config.RETRIEVER_K, "fetch_k": config.RETRIEVER_FETCH_K},
    )

    # ── Build LLMs & chains (single model — no verifier) ────────────────────
    gen_llm = build_llm(temperature=0.2, num_predict=4000)
    tc_llm  = build_tc_llm()

    scenario_chain = build_chain("scenario", gen_llm, retriever)
    general_chain  = build_chain("general",  gen_llm, retriever)

    # ── Session state ────────────────────────────────────────────────────────
    cl.user_session.set("scenario_chain",   scenario_chain)
    cl.user_session.set("general_chain",    general_chain)
    cl.user_session.set("gen_llm",          gen_llm)
    cl.user_session.set("tc_llm",           tc_llm)
    cl.user_session.set("retriever",        retriever)
    cl.user_session.set("chat_history",     [])
    cl.user_session.set("conversation_summary", "")
    cl.user_session.set("last_scenarios",   "")
    cl.user_session.set("last_scenarios_parsed", [])
    cl.user_session.set("last_tc_output",   "")
    cl.user_session.set("last_context",     "")

    doc_count   = len(processed)
    chunk_count = len(all_chunks)
    mode_label = "⚡ Fast" if FAST_MODE else "🔬 Standard"
    await cl.Message(
        content=(
            f"✅ Ready! Indexed **{chunk_count} chunks** from **{doc_count} document(s)**.\n"
            f"Mode: {mode_label}\n\n"
            "**What you can do:**\n"
            "- 💬 Ask any question about the documents\n"
            "- 📋 Say *'Generate scenarios'* for high-level test scenarios\n"
            "- 🧪 Say *'Generate test cases'* after scenarios are ready\n"
            "- 🚀 After test cases, I'll offer to push them to Azure DevOps"
        ),
        author=_AUTHOR,
    ).send()


# ---------------------------------------------------------------------------
# on_message
# ---------------------------------------------------------------------------

@cl.on_message
async def on_message(message: cl.Message):
    # ── Guard ────────────────────────────────────────────────────────────────
    scenario_chain = cl.user_session.get("scenario_chain")
    if not scenario_chain:
        await cl.Message(
            content="⚠️ Session not initialised. Please refresh and upload your documents.",
            author=_AUTHOR,
        ).send()
        return

    general_chain         = cl.user_session.get("general_chain")
    gen_llm               = cl.user_session.get("gen_llm")
    tc_llm                = cl.user_session.get("tc_llm")
    retriever             = cl.user_session.get("retriever")
    chat_history          = cl.user_session.get("chat_history", [])
    summary               = cl.user_session.get("conversation_summary", "")
    last_scenarios        = cl.user_session.get("last_scenarios", "")
    last_scenarios_parsed = cl.user_session.get("last_scenarios_parsed", [])
    last_tc_output        = cl.user_session.get("last_tc_output", "")
    last_context          = cl.user_session.get("last_context", "")
    pending_pass2         = cl.user_session.get("pending_pass2", [])

    user_input = message.content.strip()

    # ── Intent classification ────────────────────────────────────────────────
    intent = classify_intent(user_input, has_scenarios=bool(last_scenarios), has_pending_pass2=bool(pending_pass2))
    logger.info("Intent: '%s' | has_scenarios: %s", intent, bool(last_scenarios))

    # ════════════════════════════════════════════════════════════════════════
    # Branch: No scenarios yet for TC request
    # ════════════════════════════════════════════════════════════════════════
    if intent == "testcase_no_scenarios":
        await cl.Message(
            content=(
                "⚠️ No scenarios found in this session yet.\n\n"
                "Please generate scenarios first:\n"
                "*'Generate scenarios'* or *'What should we test?'*\n\n"
                "Once scenarios are ready, request test cases."
            ),
            author=_AUTHOR,
        ).send()
        return

    # ════════════════════════════════════════════════════════════════════════
    # Branch: Coverage correction — user reports missing items
    # ════════════════════════════════════════════════════════════════════════
    if intent == "coverage_revise":
        await _handle_coverage_revise(
            user_input=user_input,
            last_scenarios_parsed=last_scenarios_parsed,
            last_tc_output=last_tc_output,
            last_context=last_context,
            gen_llm=gen_llm,
            retriever=retriever,
            chat_history=chat_history,
            summary=summary,
        )
        return

    # ════════════════════════════════════════════════════════════════════════
    # Branch: General QA
    # ════════════════════════════════════════════════════════════════════════
    if intent == "general":
        base_inputs = {
            "input":                user_input,
            "chat_history":         chat_history,
            "conversation_summary": summary,
            "scenarios":            "",
            "existing_output":      "",
            "missing_items":        "",
        }
        try:
            async with cl.Step(name="💬 Searching documents…"):
                response = await general_chain.ainvoke(base_inputs)
        except Exception as exc:
            logger.exception("General chain failed.")
            await cl.Message(content=f"❌ Error: {exc}", author=_AUTHOR).send()
            return

        final_answer = response.get("answer", "No response generated.")
        new_context  = response.get("context", last_context)

        chat_history, summary = _update_memory(user_input, final_answer, chat_history, summary)
        cl.user_session.set("chat_history", chat_history)
        cl.user_session.set("conversation_summary", summary)
        if new_context:
            cl.user_session.set("last_context", new_context)

        await cl.Message(content=final_answer, author=_AUTHOR).send()
        return

    # ════════════════════════════════════════════════════════════════════════
    # Branch: Scenario generation
    # ════════════════════════════════════════════════════════════════════════
    if intent == "scenario":
        base_inputs = {
            "input":                user_input,
            "chat_history":         chat_history,
            "conversation_summary": summary,
            "scenarios":            "",
            "existing_output":      "",
            "missing_items":        "",
        }

        try:
            async with cl.Step(name="📋 Generating scenarios…"):
                response = await scenario_chain.ainvoke(base_inputs)
        except Exception as exc:
            logger.exception("Scenario chain failed.")
            await cl.Message(content=f"❌ Generation error: {exc}", author=_AUTHOR).send()
            return

        raw_answer  = response.get("answer", "")
        new_context = response.get("context", last_context)

        # ── Deterministic renumbering ────────────────────────────────────────
        raw_answer = renumber_scenarios(raw_answer)

        # ── Sanitize ────────────────────────────────────────────────────────
        raw_answer = sanitize_scenarios(raw_answer)

        # ── Eval (Python-only, <10ms) ────────────────────────────────────────
        async with cl.Step(name="✅ Evaluating scenario quality…") as estep:
            eval_result = eval_scenarios(raw_answer)
            estep.output = eval_result.summary()

        final_answer = raw_answer

        # ── If eval found missing coverage buckets → one auto-supplement ─────
        if (eval_result.coverage_report
                and not eval_result.coverage_report.is_complete
                and not FAST_MODE):
            missing_desc = ", ".join(eval_result.coverage_report.missing_buckets)
            async with cl.Step(name=f"🔄 Adding missing coverage: {missing_desc}…") as rstep:
                revise_inputs = {
                    "input": (
                        f"The following coverage areas are missing from the scenarios: {missing_desc}. "
                        f"Add new scenarios covering only these missing areas. "
                        f"Do NOT re-generate existing scenarios. "
                        f"Continue SC numbering from where the list left off:\n\n{final_answer}"
                    ),
                    "chat_history":         chat_history,
                    "conversation_summary": summary,
                    "scenarios":            "",
                    "existing_output":      "",
                    "missing_items":        "",
                }
                try:
                    rev_response = await scenario_chain.ainvoke(revise_inputs)
                    supplement   = rev_response.get("answer", "").strip()
                    if supplement:
                        supplement = renumber_scenarios(supplement)
                        supplement = sanitize_scenarios(supplement)
                        # If first pass was a refusal (no SCs), replace it entirely
                        if not re.search(r"(?i)\bSC\s*\d+\s*[:\-\.]?", final_answer):
                            final_answer = supplement
                        else:
                            final_answer = final_answer.rstrip() + "\n\n" + supplement
                    rstep.output = "✅ Additional scenarios added."

                except Exception as exc:
                    logger.warning("Coverage supplement failed: %s", exc)
                    rstep.output = f"⚠️ Could not add missing coverage: {exc}"

        # ── Store parsed scenarios ───────────────────────────────────────────
        parsed = parse_scenario_titles(final_answer)
        if parsed:
            clean = "\n".join(
                f"{s['id']}: {s['title']}\nPriority: {s['priority']}\nDescription: {s['description']}"
                for s in parsed
            )
            cl.user_session.set("last_scenarios", clean)
            cl.user_session.set("last_scenarios_parsed", parsed)
            logger.info("Stored %d scenarios in session.", len(parsed))
        else:
            cl.user_session.set("last_scenarios", final_answer)
            cl.user_session.set("last_scenarios_parsed", [])

        if new_context:
            cl.user_session.set("last_context", new_context)

        chat_history, summary = _update_memory(user_input, final_answer, chat_history, summary)
        cl.user_session.set("chat_history", chat_history)
        cl.user_session.set("conversation_summary", summary)

        await cl.Message(content=final_answer, author=_AUTHOR).send()
        return

    # ════════════════════════════════════════════════════════════════════════
    # Branch: Test case generation (prioritised multi-pass + rolling batches)
    # ════════════════════════════════════════════════════════════════════════
    if intent == "testcase":
        scenarios_to_process = last_scenarios_parsed or parse_scenario_titles(last_scenarios)

        if not scenarios_to_process:
            await cl.Message(
                content=(
                    "⚠️ Could not find structured scenarios in session. "
                    "Please regenerate scenarios first."
                ),
                author=_AUTHOR,
            ).send()
            return

        sc_ids = [s["id"] for s in scenarios_to_process]

        # ── Split scenarios by priority into two passes ───────────────────────
        pass1 = [s for s in scenarios_to_process if s.get("priority", "Medium") in ("Critical", "High")]
        pass2 = [s for s in scenarios_to_process if s.get("priority", "Medium") in ("Medium", "Low")]
        
        # Save Medium/Low to session for conditional step
        cl.user_session.set("pending_pass2", pass2)
        
        if pass1:
            p1_ids = [s["id"] for s in pass1]
            n1 = (len(pass1) + TC_BATCH_SIZE - 1) // TC_BATCH_SIZE
            p1_priorities = sorted({s.get("priority", "Medium") for s in pass1})
            
            async with cl.Step(
                name=(
                    f"🔴 Pass 1 — {', '.join(p1_priorities)} priority: "
                    f"{len(pass1)} scenario(s) in {n1} batch(es)…"
                )
            ) as gstep:
                tc_text, ctx, completed = await generate_tcs_rolling(
                    scenarios=pass1,
                    llm=tc_llm,
                    retriever=retriever,
                    summary=summary,
                )
                missed = [s for s in p1_ids if s not in completed]
                gstep.output = (
                    f"✅ Covered: {', '.join(completed)}"
                    + (f" | ⚠️ Missed: {', '.join(missed)}" if missed else "")
                )
            
            if not tc_text.strip():
                await cl.Message(
                    content="⚠️ TC generation returned empty output. Please try again.",
                    author=_AUTHOR,
                ).send()
                return

            final_tc = sanitize_test_cases(tc_text)
            
            # Update memory
            cl.user_session.set("last_tc_output", final_tc)
            cl.user_session.set("last_context", ctx)
            
            # Post generation evaluation
            final_eval = eval_tc_output(final_tc, p1_ids)
            coverage_map = build_coverage_map(final_tc, p1_ids)
            map_display  = format_coverage_map(coverage_map)
            
            async with cl.Step(name="📊 Scenario → Test Case Coverage Map") as mstep:
                mstep.output = map_display
                
            chat_history, summary = _update_memory(user_input, final_tc, chat_history, summary)
            cl.user_session.set("chat_history", chat_history)
            cl.user_session.set("conversation_summary", summary)
            
            await cl.Message(content=final_tc, author=_AUTHOR).send()
            
            if pass2:
                await cl.Message(
                    content="**Test cases for Critical and High priority scenarios are ready.**\n\nDo you want me to generate test cases for Medium and Low priority scenarios as well?",
                    author=_AUTHOR
                ).send()
            elif ADO_PAT:
                await _offer_ado_push(final_tc, last_scenarios, ctx)
        else:
            await cl.Message(
                content="No Critical or High priority scenarios found. Type 'generate' to start Medium/Low.",
                author=_AUTHOR
            ).send()
        return

    # ════════════════════════════════════════════════════════════════════════
    # Branch: Test Case Continuation (Pass 2)
    # ════════════════════════════════════════════════════════════════════════
    if intent == "testcase_continue":
        pass2 = cl.user_session.get("pending_pass2", [])
        if not pass2:
            return
            
        p2_ids = [s["id"] for s in pass2]
        n2 = (len(pass2) + TC_BATCH_SIZE - 1) // TC_BATCH_SIZE
        p2_priorities = sorted({s.get("priority", "Medium") for s in pass2})
        
        async with cl.Step(
            name=(
                f"🟡 Pass 2 — {', '.join(p2_priorities)} priority: "
                f"{len(pass2)} scenario(s) in {n2} batch(es)…"
            )
        ) as gstep:
            tc_text, ctx, completed = await generate_tcs_rolling(
                scenarios=pass2,
                llm=tc_llm,
                retriever=retriever,
                summary=summary,
            )
            missed = [s for s in p2_ids if s not in completed]
            gstep.output = (
                f"✅ Covered: {', '.join(completed)}"
                + (f" | ⚠️ Missed: {', '.join(missed)}" if missed else "")
            )
            
        cl.user_session.set("pending_pass2", []) # Clear queue
        
        if not tc_text.strip():
            await cl.Message(content="⚠️ Generation returned empty output.", author=_AUTHOR).send()
            return
            
        final_tc = sanitize_test_cases(tc_text)
        
        # Append to existing
        existing_tc = cl.user_session.get("last_tc_output", "")
        combined_tc = existing_tc + "\n\n" + final_tc if existing_tc else final_tc
        cl.user_session.set("last_tc_output", combined_tc)
        
        final_eval = eval_tc_output(final_tc, p2_ids)
        coverage_map = build_coverage_map(final_tc, p2_ids)
        map_display  = format_coverage_map(coverage_map)
        
        async with cl.Step(name="📊 Scenario → Test Case Coverage Map") as mstep:
            mstep.output = map_display
            
        await cl.Message(content=final_tc, author=_AUTHOR).send()
        
        if ADO_PAT:
            await _offer_ado_push(combined_tc, last_scenarios, ctx)
        return


# ---------------------------------------------------------------------------
# Coverage revision handler
# ---------------------------------------------------------------------------

async def _handle_coverage_revise(
    user_input: str,
    last_scenarios_parsed: list[dict],
    last_tc_output: str,
    last_context: str,
    gen_llm,
    retriever,
    chat_history: list,
    summary: str,
) -> None:
    """
    Handle user follow-up: 'you missed scenarios' / 'add missing test cases'.
    Detects what is actually missing and generates ONLY the gap.
    """
    if last_tc_output and last_scenarios_parsed:
        sc_ids = [s["id"] for s in last_scenarios_parsed]
        cov = check_tc_coverage(last_tc_output, sc_ids)

        if cov.is_complete:
            await cl.Message(
                content=(
                    "✅ All scenarios already have test cases.\n\n"
                    f"Covered scenarios: {', '.join(sc_ids)}\n\n"
                    "If you believe specific test cases are missing, please describe which "
                    "scenario or test type you'd like expanded."
                ),
                author=_AUTHOR,
            ).send()
            return

        missing_ids   = set(cov.missing_scenario_ids)
        missing_dicts = [s for s in last_scenarios_parsed if s["id"] in missing_ids]

        await cl.Message(
            content=(
                f"🔍 Detected missing test cases for: **{', '.join(cov.missing_scenario_ids)}**\n"
                "Generating now…"
            ),
            author=_AUTHOR,
        ).send()

        async with cl.Step(
            name=f"🔄 Generating missing TCs for: {', '.join(cov.missing_scenario_ids)}…"
        ) as rstep:
            last_tc_num = _last_tc_num(last_tc_output)
            gap_text, _ = await generate_coverage_revision(
                missing_scenario_dicts=missing_dicts,
                existing_output=last_tc_output,
                llm=gen_llm,
                retriever=retriever,
                summary=summary,
                chat_history=chat_history,
                last_tc_num=last_tc_num,
            )
            rstep.output = (
                f"✅ Generated TCs for {', '.join(cov.missing_scenario_ids)}"
                if gap_text else "⚠️ Gap-fill returned empty."
            )

        if gap_text:
            gap_text = sanitize_test_cases(gap_text)
            updated_tc = last_tc_output.rstrip() + "\n\n" + gap_text
            cl.user_session.set("last_tc_output", updated_tc)

            chat_history_new, summary_new = _update_memory(user_input, gap_text, chat_history, summary)
            cl.user_session.set("chat_history", chat_history_new)
            cl.user_session.set("conversation_summary", summary_new)

            await cl.Message(content=gap_text, author=_AUTHOR).send()
        else:
            await cl.Message(
                content="⚠️ Could not generate missing test cases. Please try again.",
                author=_AUTHOR,
            ).send()

    elif last_scenarios_parsed:
        sc_ids = [s["id"] for s in last_scenarios_parsed]
        await cl.Message(
            content=(
                f"ℹ️ Your current scenarios are: **{', '.join(sc_ids)}**\n\n"
                "If you believe scenarios are missing, please describe what you expect "
                "and I'll generate additional scenarios.\n\n"
                "Or say *'Generate test cases'* to generate test cases for the current list."
            ),
            author=_AUTHOR,
        ).send()
    else:
        await cl.Message(
            content=(
                "⚠️ No prior scenarios or test cases found in this session.\n\n"
                "Please start by generating scenarios:\n"
                "*'Generate scenarios'*"
            ),
            author=_AUTHOR,
        ).send()


# ---------------------------------------------------------------------------
# ADO push flow
# ---------------------------------------------------------------------------

async def _offer_ado_push(tc_text: str, scenarios: str, context: str) -> None:
    """Multi-step ADO push conversation."""
    push_action = await cl.AskActionMessage(
        content="🚀 Test cases generated. Push to Azure DevOps?",
        actions=[
            cl.Action(name="yes", value="yes", label="✅ Yes, push to ADO"),
            cl.Action(name="no",  value="no",  label="❌ No, skip"),
        ],
    ).send()

    if not push_action or push_action.get("value") != "yes":
        await cl.Message(content="ADO push skipped.", author=_AUTHOR).send()
        return

    test_cases = parse_test_cases(tc_text)
    if not test_cases:
        await cl.Message(
            content="⚠️ Could not parse any test cases from the output. ADO push skipped.",
            author=_AUTHOR,
        ).send()
        return

    # Priority
    priority_msg = await cl.AskUserMessage(
        content="Priority for all test cases? Enter 1 (High), 2 (Medium), or 3 (Low):",
        timeout=60,
    ).send()
    try:
        priority = int((priority_msg or {}).get("output", "2"))
        if priority not in (1, 2, 3, 4):
            priority = 2
    except ValueError:
        priority = 2

    # Regression
    regression_action = await cl.AskActionMessage(
        content="Mark as Regression Tests?",
        actions=[
            cl.Action(name="yes", value="yes", label="✅ Yes"),
            cl.Action(name="no",  value="no",  label="❌ No"),
        ],
    ).send()
    regression = (regression_action or {}).get("value") == "yes"

    results: list[str] = []
    async with cl.Step(name=f"🚀 Pushing {len(test_cases)} test cases to ADO…"):
        for tc in test_cases:
            steps   = [s["action"]   for s in tc["steps"]]
            raw_exp = [s["expected"] for s in tc["steps"]]

            outcomes = []
            for i, exp in enumerate(raw_exp):
                if exp:
                    outcomes.append(exp)
                else:
                    filled = await generate_expected_outcomes([steps[i]])
                    outcomes.append(filled[0] if filled else "")

            result = await asyncio.to_thread(
                create_test_case,
                title          = f"[{tc['scenario_id']}] {tc['title']}",
                precondition   = "\n".join(tc.get("preconditions", [])),
                steps          = steps,
                outcomes       = outcomes,
                priority       = priority,
                regression     = regression,
                area_path      = ADO_AREA_PATH,
                iteration_path = ADO_ITERATION_PATH,
                base_url       = ADO_BASE_URL,
                project        = ADO_PROJECT,
                feature_id     = ADO_FEATURE_ID,
                assigned_to    = ADO_ASSIGNED_TO,
                tag            = ADO_TAG,
                pat            = ADO_PAT,
            )
            results.append(f"**{tc['id']}** — {tc['title']}: {result}")

    await cl.Message(
        content=f"✅ ADO push complete:\n\n" + "\n".join(results),
        author=_AUTHOR,
    ).send()