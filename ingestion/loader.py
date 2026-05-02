"""
ingestion/loader.py
-------------------
PDF ingestion: load → clean → return raw text.

Improvements over v1:
- Preserves paragraph structure (double-newlines) for the chunker
- Better OCR artifact removal
- Exposes page-level metadata hooks for future use
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

import pdfplumber

from config.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """
    Remove common PDF extraction noise while keeping paragraph structure.
    """
    # OCR artifacts like (cid:123)
    text = re.sub(r"\(cid:\d+\)", "", text)
    # Invisible unicode
    text = re.sub(r"[\u200b\ufeff\u00ad]", "", text)
    # Non-ASCII characters → space
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    # Excessive punctuation / noise symbols (keep basic sentence punctuation)
    text = re.sub(r"[^\w\s.,;:!?'\"\-\(\)\n]", " ", text)
    # Collapse runs of spaces within a line (but keep newlines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Collapse 3+ consecutive newlines → 2 (paragraph separator)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_pdf(file_path: str | Path) -> str:
    """
    Extract and clean text from *file_path*.

    Returns a single string with paragraph boundaries preserved as ``\\n\\n``.
    Raises ``FileNotFoundError`` if the file does not exist.
    Raises ``ValueError`` if no text could be extracted.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    log.info("Loading PDF", extra={"path": str(path)})
    pages: List[str] = []

    with pdfplumber.open(str(path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(page_text)
            else:
                log.warning("Empty page", extra={"page": page_num, "file": path.name})

    if not pages:
        raise ValueError(f"No extractable text in {path.name}")

    raw = "\n\n".join(pages)
    cleaned = _clean_text(raw)
    log.info(
        "PDF loaded",
        extra={"file": path.name, "pages": len(pages), "chars": len(cleaned)},
    )
    return cleaned
