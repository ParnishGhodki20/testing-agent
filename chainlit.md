# Testing Copilot

Upload your requirement or specification documents and ask questions about them. The assistant will answer using only the content in your uploaded files.

If the response contains test scenarios, you will be prompted to push them directly to Azure DevOps as Test Cases.

**Supported file types**

`.pdf` `.docx` `.pptx` `.xlsx` `.csv` `.xml` `.json` `.txt` `.sql` `.ts` `.java` `.ps1` `.py` `.cs` `.jpg` `.jpeg` `.png` `.gif` `.html` `.eml` `.zip`

**How to run**

```bash
chainlit run main.py
```

**Required environment variables** (set in `.env`)

| Variable | Purpose |
|---|---|
| `OLLAMA_BASE_URL` | Ollama server URL (default: `http://localhost:11434`) |
| `OLLAMA_CHAT_MODEL` | Chat model (default: `llama3.2-vision`) |
| `OLLAMA_EMBED_MODEL` | Embedding model (default: `snowflake-arctic-embed2`) |
| `ADO_PAT` | Azure DevOps personal access token |
| `ADO_BASE_URL` | ADO organization URL |
| `ADO_PROJECT` | ADO project name |
| `ADO_AREA_PATH` | Area path for test cases |
| `ADO_ITERATION_PATH` | Iteration path for test cases |
