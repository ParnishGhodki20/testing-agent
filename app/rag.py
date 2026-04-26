import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.config import (
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBED_MODEL
)

logger = logging.getLogger(__name__)

CHROMA_DIR = "./chroma_db"

def build_llm():
    return ChatOllama(
        model=OLLAMA_CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
        num_predict=3000 # Increased for exhaustive test case generation
    )

def build_index(texts):
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    Path(CHROMA_DIR).mkdir(exist_ok=True)
    return Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        metadatas=[{"source": f"chunk-{i}"} for i in range(len(texts))]
    )

def _format_docs(docs):
    if not docs:
        return "No relevant context found."
    return "\n\n".join(
        f"--- Document Chunk ---\n{doc.page_content}"
        for doc in docs
    )

def build_retrieval_chain(llm, retriever):
    condense_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite latest user question as standalone question. Do not answer."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    condense_chain = condense_prompt | llm | StrOutputParser()

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are an intelligent AI assistant with expertise in QA and software testing for Icertis ICI.

========================================
STEP 1 — Scenario Generation
========================================
If user asks for scenarios:
Generate a numbered list of ONLY high-level scenario titles.
Format:
SC1: [Scenario Name]
SC2: [Scenario Name]

Rules:
- DO NOT generate test cases, steps, or outcomes yet.
- Output ONLY the list of scenario titles.

========================================
STEP 2 — Test Case Generation
========================================
If user asks for test cases:
Use the previously generated scenarios stored in the conversation summary.
For EACH scenario, generate ALL possible test cases (Positive, Negative, Exception, and Edge).

For EACH test case use EXACT structure:

TC<number>

Title:
[Test case title]

Description:
[What is validated]

Type:
Positive / Negative / Exception / Edge

Preconditions:
- [Condition]

Steps:
1. Action:
[Action step]
Expected Outcome:
[Expected result for this step]

2. Action:
[Next action]
Expected Outcome:
[Expected result for this step]

Rules:
- Sequential numbering TC1, TC2, TC3...
- Every step MUST have an Expected Outcome.
- Generate multiple test cases per scenario to ensure full coverage.

Conversation so far:
{conversation_summary}

Document Context:
{context}
"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    async def _route_question(inputs):
        if inputs.get("chat_history"):
            return await (condense_chain.ainvoke(inputs))
        return inputs["input"]

    async def _retrieve_context(inputs):
        docs = await retriever.ainvoke(inputs["standalone_question"])
        return _format_docs(docs)

    return (
        RunnablePassthrough.assign(standalone_question=RunnableLambda(_route_question))
        | RunnablePassthrough.assign(context=RunnableLambda(_retrieve_context))
        | RunnablePassthrough.assign(answer=qa_prompt | llm | StrOutputParser())
    )