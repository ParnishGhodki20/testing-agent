import asyncio
import io
from pathlib import Path
import pandas as pd
import pypdf
from docx import Document as DocxDocument

async def _process_pdf(path):
    parts = []
    reader = pypdf.PdfReader(path)
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "".join(parts)

async def _process_docx(path):
    doc = DocxDocument(path)
    return "\n".join(p.text for p in doc.paragraphs)

_SYNC_PROCESSORS = {
    "xlsx": lambda p: pd.read_excel(p).to_string(),
    "csv": lambda p: pd.read_csv(p).to_string(),
    "txt": lambda p: Path(p).read_text(encoding="utf-8", errors="ignore"),
    "feature": lambda p: Path(p).read_text(encoding="utf-8", errors="ignore")
}

_ASYNC_PROCESSORS = {
    "pdf": _process_pdf,
    "docx": _process_docx
}

async def extract_text(file_path, extension):
    ext = extension.lstrip(".").lower()
    if ext in _ASYNC_PROCESSORS:
        return await _ASYNC_PROCESSORS[ext](file_path)
    if ext in _SYNC_PROCESSORS:
        return _SYNC_PROCESSORS[ext](file_path)
    return ""