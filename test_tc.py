import asyncio
import os
from langchain_ollama import ChatOllama
from app.rag import build_tc_llm, generate_tcs_rolling

async def run_test():
    llm = build_tc_llm()
    
    # Mock scenarios matching our parser's expected input
    scenarios = [
        {"id": "SC1", "title": "Admin Login", "priority": "Critical", "description": "Admin logs into system"},
        {"id": "SC2", "title": "Upload Document", "priority": "High", "description": "User uploads PDF"}
    ]
    
    # Mock retriever
    class MockRetriever:
        async def ainvoke(self, query):
            class MockDoc:
                page_content = "This is a mock document about admin login and file uploads."
            return [MockDoc()]
            
    print("Starting generation...")
    tc_text, ctx, completed = await generate_tcs_rolling(
        scenarios=scenarios,
        llm=llm,
        retriever=MockRetriever(),
        summary="Test summary",
        batch_size=1,
        timeout_secs=300
    )
    
    print("\n--- RESULTS ---")
    print("COMPLETED:", completed)
    print("TC TEXT:\n", tc_text)

if __name__ == "__main__":
    asyncio.run(run_test())
