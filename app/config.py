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
# Ollama
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Generator model — used for scenario / TC / general generation
OLLAMA_CHAT_MODEL: str = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")

# Verifier model — second-pass validation (can be same or larger model)
OLLAMA_VERIFY_MODEL: str = os.getenv("OLLAMA_VERIFY_MODEL", "qwen2.5:7b")

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
# Text splitting
# ---------------------------------------------------------------------------

CHUNK_SIZE: int    = _get_env("TEXT_SPLITTER_CHUNK_SIZE", 1500)   # aligned with .env.example
CHUNK_OVERLAP: int = _get_env("TEXT_SPLITTER_CHUNK_OVERLAP", 150)  # 10% overlap is best practice


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

RETRIEVER_K: int       = _get_env("RETRIEVER_K", 8)        # fetch 8 chunks per query
RETRIEVER_FETCH_K: int = _get_env("RETRIEVER_FETCH_K", 20) # MMR candidate pool


# ---------------------------------------------------------------------------
# Test case generation performance
# ---------------------------------------------------------------------------

# Scenarios per batch (3 = sweet spot for local 3b–7b models)
TC_BATCH_SIZE: int      = _get_env("TC_BATCH_SIZE", 3)

# Hard timeout per batch in seconds — prevents stuck generation
TC_TIMEOUT_SECS: float  = _get_env("TC_TIMEOUT_SECS", 90, float)

# Max tokens the TC generator may produce per batch
# 2500 allows ~4-6 TCs per scenario with full steps; raise to 3500 for very deep coverage
TC_NUM_PREDICT: int     = _get_env("TC_NUM_PREDICT", 2500)

# Max tokens the verifier may produce (only needs a short verdict)
VERIFY_NUM_PREDICT: int = _get_env("VERIFY_NUM_PREDICT", 1000)



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
