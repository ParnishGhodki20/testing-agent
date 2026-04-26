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


OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL: str = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:3b")
OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "snowflake-arctic-embed2")


ADO_PAT: str = os.getenv("ADO_PAT", "")
ADO_BASE_URL: str = os.getenv("ADO_BASE_URL", "https://dev.azure.com/icertisvsts")
ADO_PROJECT: str = os.getenv("ADO_PROJECT", "")
ADO_AREA_PATH: str = os.getenv("ADO_AREA_PATH", "")
ADO_ITERATION_PATH: str = os.getenv("ADO_ITERATION_PATH", "")
ADO_FEATURE_ID: str = os.getenv("ADO_FEATURE_ID", "")
ADO_ASSIGNED_TO: str = os.getenv("ADO_ASSIGNED_TO", "")
ADO_TAG: str = os.getenv("ADO_TAG", "GeneratedByCopilot")

MAX_SIZE_MB: int = _get_env("CHAINLIT_MAX_SIZE_MB", 500)
MAX_FILES: int = _get_env("CHAINLIT_MAX_FILES", 1000)
CHUNK_SIZE: int = _get_env("TEXT_SPLITTER_CHUNK_SIZE", 800)
CHUNK_OVERLAP: int = _get_env("TEXT_SPLITTER_CHUNK_OVERLAP", 20)
OCR_TIMEOUT: float = _get_env("OCR_TIMEOUT", 5, float)

CODE_EXTENSIONS: frozenset[str] = frozenset({
    ".xml", ".json", ".sql", ".ts", ".java", ".ps1", ".py", ".cs"
})

SUPPORTED_EXTENSIONS: list[str] = [
    ".pdf", ".docx", ".pptx", ".xlsx", ".csv",
    ".xml", ".json", ".txt", ".sql", ".ts", ".java", ".ps1", ".py", ".cs",
    ".jpg", ".jpeg", ".png", ".gif", ".html", ".eml", ".zip",
]

# AskFileMessage requires MIME types, not raw extensions.
# Dict format: {"mime/type": [".ext"]} gives browser both type + extension hints.
ACCEPT_FILE_TYPES: dict[str, list[str]] = {
    "application/pdf": [".pdf"],
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": [".pptx"],
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
    "text/csv": [".csv"],
    "application/xml": [".xml"],
    "application/json": [".json"],
    "text/plain": [".txt", ".sql", ".ts", ".java", ".ps1", ".py", ".cs"],
    "image/jpeg": [".jpg", ".jpeg"],
    "image/png": [".png"],
    "image/gif": [".gif"],
    "text/html": [".html"],
    "message/rfc822": [".eml"],
    "application/zip": [".zip"],
    "application/octet-stream": [".docx", ".pptx", ".xlsx"],  # fallback for strict browsers
}
