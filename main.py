import asyncio
import logging
import chainlit as cl
from langchain_core.messages import AIMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app import config
from app.extractors import extract_text
from app.rag import build_index, build_llm, build_retrieval_chain

logger = logging.getLogger(__name__)
_AUTHOR = "icertis-testing-copilot"

@cl.on_chat_start
async def on_chat_start():
    files = await cl.AskFileMessage(
        content=f"Upload Word, PDF, or TXT feature files",
        accept=config.ACCEPT_FILE_TYPES,
        max_size_mb=config.MAX_SIZE_MB,
        max_files=config.MAX_FILES,
        timeout=86400,
        raise_on_timeout=False
    ).send()

    if not files:
        await cl.Message(content="No files uploaded.", author=_AUTHOR).send()
        return

    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )

    for file in files:
        ext = file.name.rsplit(".", 1)[-1].lower()
        text = await extract_text(file.path, ext)
        if text:
            all_chunks.extend(splitter.split_text(text))

    docsearch = await asyncio.to_thread(build_index, all_chunks)
    retriever = docsearch.as_retriever(search_type="mmr", search_kwargs={"k":5, "fetch_k":10})

    chain = build_retrieval_chain(build_llm(), retriever)
    cl.user_session.set("chain", chain)
    cl.user_session.set("chat_history", [])
    cl.user_session.set("conversation_summary", "")
    cl.user_session.set("last_scenarios", "")

    await cl.Message(content="Ready. Ask for 'Scenarios' to begin.", author=_AUTHOR).send()

@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")
    if not chain:
        await cl.Message(content="Session expired. Re-upload documents.", author=_AUTHOR).send()
        return

    chat_history = cl.user_session.get("chat_history", [])
    summary = cl.user_session.get("conversation_summary", "")
    last_scenarios = cl.user_session.get("last_scenarios", "")

    user_input = message.content.strip()
    lowered = user_input.lower()

    # Trigger exhaustive TC generation if scenarios exist in memory
    if "test case" in lowered and last_scenarios:
        enhanced_input = f"Using these scenarios:\n{last_scenarios}\n\nGenerate all possible test cases (Positive, Negative, Exception, Edge) for each scenario. User Request: {user_input}"
    else:
        enhanced_input = user_input

    try:
        response = await chain.ainvoke({
            "input": enhanced_input,
            "chat_history": chat_history,
            "conversation_summary": summary
        })
        answer = response.get("answer", "No response generated.")
    except Exception as e:
        logger.exception("LLM failure")
        await cl.Message(content=f"Error: {e}", author=_AUTHOR).send()
        return

    # Store scenario titles to memory for subsequent TC requests
    if "SC1:" in answer:
        cl.user_session.set("last_scenarios", answer)

    chat_history.extend([HumanMessage(content=user_input), AIMessage(content=answer)])
    cl.user_session.set("chat_history", chat_history[-12:])

    summary += f"\nUser: {user_input}\nAssistant: {answer}\n"
    cl.user_session.set("conversation_summary", summary[-3000:])

    await cl.Message(content=answer, author=_AUTHOR).send()