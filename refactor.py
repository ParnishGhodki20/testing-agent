import re

def refactor_main():
    with open('main.py', 'r') as f:
        content = f.read()

    # The block we are replacing starts at:
    #         # ── Split scenarios by priority into two passes ───────────────────────
    # And ends at:
    #         if ADO_PAT:
    #             await _offer_ado_push(final_tc, last_scenarios, merged_context or last_context)
    #         return
    
    start_str = "        # ── Split scenarios by priority into two passes ───────────────────────"
    end_str = "        if ADO_PAT:\n            await _offer_ado_push(final_tc, last_scenarios, merged_context or last_context)\n        return"
    
    if start_str not in content or end_str not in content:
        print("Could not find start or end string!")
        return

    start_idx = content.find(start_str)
    end_idx = content.find(end_str) + len(end_str)

    new_block = """        # ── Split scenarios by priority into two passes ───────────────────────
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
                    content="**Test cases for Critical and High priority scenarios are ready.**\\n\\nDo you want me to generate test cases for Medium and Low priority scenarios as well?",
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
        combined_tc = existing_tc + "\\n\\n" + final_tc if existing_tc else final_tc
        cl.user_session.set("last_tc_output", combined_tc)
        
        final_eval = eval_tc_output(final_tc, p2_ids)
        coverage_map = build_coverage_map(final_tc, p2_ids)
        map_display  = format_coverage_map(coverage_map)
        
        async with cl.Step(name="📊 Scenario → Test Case Coverage Map") as mstep:
            mstep.output = map_display
            
        await cl.Message(content=final_tc, author=_AUTHOR).send()
        
        if ADO_PAT:
            await _offer_ado_push(combined_tc, last_scenarios, ctx)
        return"""

    new_content = content[:start_idx] + new_block + content[end_idx:]
    
    with open('main.py', 'w') as f:
        f.write(new_content)
        
    print("Successfully replaced block!")

if __name__ == "__main__":
    refactor_main()
