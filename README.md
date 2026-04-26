# Testing Copilot

A Chainlit-based chat application that lets you upload requirement or specification documents, ask questions about them using RAG (Retrieval-Augmented Generation), and optionally push generated test scenarios directly to Azure DevOps as Test Cases.

## How It Works

```
Upload files → Extract text → Embed + Index (Chroma) → Chat (RAG)
                                                            ↓
                                              Detect test scenarios
                                                            ↓
                                              Push to Azure DevOps (optional)
```

## Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com) running locally
- `tesseract` installed for OCR (optional, for image/PDF image extraction)

### Required Ollama Models

Pull these before starting the app:

```bash
ollama pull snowflake-arctic-embed2   # embeddings
ollama pull llama3.2-vision           # chat + scenario inference
```

For code-heavy documents, optionally pull:

```bash
ollama pull codellama
```

## Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r req.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your values
```

## Configuration

All configuration is done via `.env`. See `.env.example` for all available options.

| Variable | Required | Default | Description |
|---|---|---|---|
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_CHAT_MODEL` | No | `llama3.2-vision` | Model used for Q&A and scenario inference |
| `OLLAMA_EMBED_MODEL` | No | `snowflake-arctic-embed2` | Embedding model for indexing |
| `OLLAMA_CODE_MODEL` | No | `codellama` | Model used for code file inference |
| `ADO_PAT` | For ADO push | — | Azure DevOps personal access token |
| `ADO_BASE_URL` | For ADO push | `https://dev.azure.com/icertisvsts` | ADO organization URL |
| `ADO_PROJECT` | For ADO push | — | ADO project name |
| `ADO_AREA_PATH` | For ADO push | — | Area path for test cases |
| `ADO_ITERATION_PATH` | For ADO push | — | Iteration path for test cases |
| `ADO_FEATURE_ID` | No | — | Link test cases to this feature work item |
| `ADO_ASSIGNED_TO` | No | — | Assign created test cases to this user |
| `ADO_TAG` | No | `GeneratedByCopilot` | Tag applied to created test cases |
| `TEXT_SPLITTER_CHUNK_SIZE` | No | `1500` | RAG chunk size in characters |
| `TEXT_SPLITTER_CHUNK_OVERLAP` | No | `20` | Overlap between chunks |
| `OCR_TIMEOUT` | No | `5` | Seconds before OCR times out per image |
| `HEADLESS` | No | `false` | Set `true` to prevent auto browser open |
| `DEBUG` | No | `false` | Enable debug logging |

## Running

```bash
source .venv/bin/activate
chainlit run main.py --headless
```

App available at: `http://localhost:8000`

## Supported File Types

| Category | Extensions |
|---|---|
| Documents | `.pdf` `.docx` `.pptx` |
| Spreadsheets | `.xlsx` `.csv` |
| Code | `.xml` `.json` `.sql` `.ts` `.java` `.ps1` `.py` `.cs` |
| Text | `.txt` `.html` `.eml` |
| Images | `.jpg` `.jpeg` `.png` `.gif` |
| Archives | `.zip` |

Images embedded inside PDF, DOCX, and PPTX files are also OCR-processed.

## Project Structure

```
main.py              # Chainlit entry point — handlers only
app/
├── config.py        # All env vars and constants
├── extractors.py    # File-to-text extraction per format
├── rag.py           # Chroma index + LangChain retrieval chain
├── scenarios.py     # Test scenario parsing + LLM inference
└── ado.py           # Azure DevOps REST client
chainlit.md          # UI welcome/sidebar content
req.txt              # Python dependencies
.env.example         # Environment variable template
```

## ADO Test Case Flow

When the model's answer contains test scenarios (detected by `TC###:` pattern or markdown headers), the app prompts:

1. Push to ADO? `[Yes / No]`
2. Which scenarios? `all` or `TC001,TC002`
3. Priority for all selected? `[1 / 2 / 3]`
4. Mark as Regression Test? `[Yes / No]`

Expected outcomes for each step are generated automatically via Ollama before pushing.
