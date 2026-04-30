"""
Central configuration — all values from environment variables with safe defaults.
"""
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

if Path(".env").exists():
    load_dotenv(override=True)

_debug = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

logging.basicConfig(
    stream=sys.stdout,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    level=logging.DEBUG if _debug else logging.INFO,
)

_logger = logging.getLogger(__name__)


def _get_env(name: str, default, cast_type=int):
    try:
        return cast_type(os.getenv(name, default))
    except (ValueError, TypeError):
        _logger.warning("Invalid value for %s, using default %s.", name, default)
        return default


# ---------------------------------------------------------------------------
# Ollama — single generator model (no separate verifier model)
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Generator model — used for ALL generation (scenarios, TCs, general QA)
# llama3.1:8b is optimal for strict formatting and complex RAG reasoning
OLLAMA_CHAT_MODEL: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")

# Embedding model
OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "snowflake-arctic-embed2")


# ---------------------------------------------------------------------------
# Azure DevOps
# ---------------------------------------------------------------------------

ADO_PAT: str           = os.getenv("ADO_PAT", "")
ADO_BASE_URL: str      = os.getenv("ADO_BASE_URL", "https://dev.azure.com/icertisvsts")
ADO_PROJECT: str       = os.getenv("ADO_PROJECT", "")
ADO_AREA_PATH: str     = os.getenv("ADO_AREA_PATH", "")
ADO_ITERATION_PATH: str = os.getenv("ADO_ITERATION_PATH", "")
ADO_FEATURE_ID: str    = os.getenv("ADO_FEATURE_ID", "")
ADO_ASSIGNED_TO: str   = os.getenv("ADO_ASSIGNED_TO", "")
ADO_TAG: str           = os.getenv("ADO_TAG", "GeneratedByCopilot")


# ---------------------------------------------------------------------------
# Chainlit / upload
# ---------------------------------------------------------------------------

MAX_SIZE_MB: int = _get_env("CHAINLIT_MAX_SIZE_MB", 500)
MAX_FILES: int   = _get_env("CHAINLIT_MAX_FILES", 1000)


# ---------------------------------------------------------------------------
# Text splitting (RAG)
# ---------------------------------------------------------------------------

# 1500 chars keeps full requirement paragraphs intact.
# 200 overlap (~13%) preserves sentence boundaries across chunk edges.
CHUNK_SIZE: int    = _get_env("TEXT_SPLITTER_CHUNK_SIZE", 1500)
CHUNK_OVERLAP: int = _get_env("TEXT_SPLITTER_CHUNK_OVERLAP", 200)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

# Fewer but higher-quality chunks: 6 × 1500 = 9000 chars of focused context
RETRIEVER_K: int       = _get_env("RETRIEVER_K", 6)
RETRIEVER_FETCH_K: int = _get_env("RETRIEVER_FETCH_K", 20)


# ---------------------------------------------------------------------------
# Test case generation performance (Parallel Granular Architecture)
# ---------------------------------------------------------------------------

# Processing 1 scenario per batch ensures maximum format adherence
TC_BATCH_SIZE: int      = _get_env("TC_BATCH_SIZE", 1)

# Hard timeout per generation
TC_TIMEOUT_SECS: float  = _get_env("TC_TIMEOUT_SECS", 300, float)

# Token allocation per scenario (1500 is generous for 1 scenario)
TC_NUM_PREDICT: int     = _get_env("TC_NUM_PREDICT", 1500)


# ---------------------------------------------------------------------------
# Fast mode (FAST_MODE=true in .env for demo / quick iteration)
# ---------------------------------------------------------------------------
# When True:
#   - eval retries are SKIPPED (Python checks still run but don't trigger retry)
#   - uses FAST_TC_NUM_PREDICT tokens (shorter, faster output)
# When False (default):
#   - one targeted retry per batch if eval detects missing SC coverage

_fast_raw = os.getenv("FAST_MODE", "false").lower()
FAST_MODE: bool          = _fast_raw in ("true", "1", "t", "yes")
FAST_TC_NUM_PREDICT: int = _get_env("FAST_TC_NUM_PREDICT", 1200)


# ---------------------------------------------------------------------------
# Other
# ---------------------------------------------------------------------------

OCR_TIMEOUT: float = _get_env("OCR_TIMEOUT", 5, float)


# ---------------------------------------------------------------------------
# File type config
# ---------------------------------------------------------------------------

CODE_EXTENSIONS: frozenset[str] = frozenset({
    ".xml", ".json", ".sql", ".ts", ".js", ".java", ".ps1", ".py", ".cs", ".md",
})

SUPPORTED_EXTENSIONS: list[str] = [
    ".pdf", ".docx", ".pptx", ".xlsx", ".csv",
    ".xml", ".json", ".txt", ".sql", ".ts", ".js", ".java", ".ps1", ".py", ".cs",
    ".jpg", ".jpeg", ".png", ".gif", ".html", ".eml", ".zip", ".md",
]

# AskFileMessage requires MIME types (browser uses these for file-picker filtering)
ACCEPT_FILE_TYPES: dict[str, list[str]] = {
    "application/pdf":                                                                     [".pdf"],
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document":            [".docx"],
    "application/vnd.openxmlformats-officedocument.presentationml.presentation":          [".pptx"],
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":                  [".xlsx"],
    "text/csv":                                                                            [".csv"],
    "application/xml":                                                                     [".xml"],
    "application/json":                                                                    [".json"],
    "text/plain":              [".txt", ".sql", ".ts", ".js", ".java", ".ps1", ".py", ".cs", ".md"],
    "image/jpeg":                                                                   [".jpg", ".jpeg"],
    "image/png":                                                                           [".png"],
    "image/gif":                                                                           [".gif"],
    "text/html":                                                                          [".html"],
    "message/rfc822":                                                                      [".eml"],
    "application/zip":                                                                     [".zip"],
    "application/octet-stream":                                              [".docx", ".pptx", ".xlsx"],
}
