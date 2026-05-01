# Icertis Testing Copilot

A high-performance, Chainlit-based AI application featuring a custom-built React frontend. It allows users to upload requirement or specification documents, query them using RAG (Retrieval-Augmented Generation), and automatically generate priority-driven Test Cases with strict formatting.

## Key Features

- **Custom Premium UI:** A fully custom React frontend (bypassing default Chainlit) featuring dynamic document-based chat naming, sidebar history search, dark mode aesthetics, and an embedded Icertis logo.
- **Progressive Generation Engine:** To prevent timeouts and manage local LLM resources efficiently, test case generation operates in a two-pass system:
  - **Phase 1:** Automatically generates test cases for *Critical* and *High* priority scenarios.
  - **Phase 2:** Pauses and prompts the user before generating *Medium* and *Low* priority scenarios.
- **Strict Formatting Control:** Custom sanitization pipelines enforce zero-gap, continuous formatting between test steps and expected outcomes for a clean, professional display.
- **Export to Markdown:** Easily share generated test cases by downloading a clean `.md` file directly from the chat UI (excluding welcome prompts).
- **Ollama Powered:** Fully offline inference utilizing `llama3.1:8b` for structured, high-accuracy reasoning.

## How It Works

```
Upload files → Extract text → Embed + Index (Chroma) → Chat (RAG)
                                                            ↓
                                        Detect & Prioritize Test Scenarios
                                                            ↓
                                        Generate Critical/High TCs (Phase 1)
                                                            ↓
                                        Prompt User for Medium/Low (Phase 2)
```

## Prerequisites

- Python 3.12+
- Node.js & npm (for the custom React frontend)
- [Ollama](https://ollama.com) running locally
- `tesseract` installed for OCR (optional, for image/PDF image extraction)

### Required Ollama Models

Pull these before starting the app:

```bash
ollama pull snowflake-arctic-embed2   # embeddings
ollama pull llama3.1:8b               # chat + strict test case generation
```

## Setup & Running

This project uses a dual-stack configuration: a Python backend and a React custom frontend.

### 1. Backend Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r req.txt

# Copy and configure environment
cp .env.example .env
```

### 2. Frontend Setup (One-time)

Because this app uses a custom React frontend mounted by Chainlit, you must build the frontend first.

```bash
cd frontend
npm install
npm run build
cd ..
```

*Note: Chainlit is configured (via `.chainlit/config.toml`) to serve the UI directly from `frontend/dist`. Assets like logos are served natively from the root `/public` directory.*

### 3. Start the Server

Run the backend in headless mode (it will serve the compiled React frontend automatically):

```bash
chainlit run main.py --headless
```

App available at: `http://localhost:8000`

## Configuration

Core tuning is done via `.env`. See `.env.example` for all available options.

| Variable | Required | Default | Description |
|---|---|---|---|
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_CHAT_MODEL` | No | `llama3.1:8b` | Model used for Q&A and scenario inference |
| `OLLAMA_EMBED_MODEL` | No | `snowflake-arctic-embed2` | Embedding model for indexing |
| `TC_BATCH_SIZE` | No | `1` | Strictly `1` to prevent Ollama queue timeouts |
| `TC_TIMEOUT_SECS` | No | `400` | Extends timeout for complex generation |

## Azure DevOps (ADO) Integration

The Azure DevOps push flow is completely intact and operational. When the model generates test scenarios and test cases, the system will prompt you in the chat:

1. Push to ADO? `[Yes / No]`
2. Which scenarios? `all` or `TC001,TC002`
3. Priority for all selected? `[1 / 2 / 3]`
4. Mark as Regression Test? `[Yes / No]`

The application will leverage `app/ado.py` and the `ADO_*` credentials in your `.env` file to directly map the strictly formatted steps and expected outcomes directly into Azure DevOps Test Plans.

## Project Structure

```
main.py              # Chainlit entry point & Intent Routing
app/
├── config.py        # All env vars and constants
├── extractors.py    # File-to-text extraction per format
├── rag.py           # Core generation pipeline (Progressive Engine)
├── router.py        # Intent classification (detects 'testcase_continue')
├── sanitizer.py     # Regex formatters for strict UI layout
├── ado.py           # Azure DevOps REST client push logic
└── prompts.py       # Strict system prompts & constraints
frontend/            # Custom React UI built with Vite
├── src/App.tsx      # Core logic (upload, chat naming, share, search)
└── src/index.css    # Premium styling & spacing tokens
public/              # Static assets served by Chainlit (e.g., logo.png)
.chainlit/           # Chainlit config (points to frontend/dist)
```
