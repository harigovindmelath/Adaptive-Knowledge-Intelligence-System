"""
ingestion/chunker.py
--------------------
Semantic chunker with configurable size and overlap.

Improvements over v1:
- Overlap window to avoid context truncation at boundaries
- Sentence-level splitting fallback
- Metadata includes position (chunk_index, total_chunks)
"""
from __future__ import annotations

import re
import uuid
from typing import Dict, List

from config.loader import get as cfg
from config.logger import get_logger

log = get_logger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_into_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def chunk_text(
    text: str,
    source_file: str,
    target_words: int | None = None,
    overlap_words: int | None = None,
) -> List[Dict]:
    """
    Split *text* into overlapping chunks.

    Each chunk dict contains:
        chunk_id    : UUID string
        text        : chunk content
        source_file : originating filename
        chunk_index : position in document (0-based)
    """
    target_words = target_words or int(cfg("ingestion", "chunk_size_words", 350))
    overlap_words = overlap_words or int(cfg("ingestion", "chunk_overlap_words", 50))

    # Split by paragraphs first
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        log.warning("No paragraphs found; falling back to full text.", extra={"source": source_file})
        paragraphs = [text]

    # Build word-level sliding window across paragraph stream
    words: List[str] = []
    for para in paragraphs:
        words.extend(para.split())
        words.append("\n\n")  # paragraph marker (removed later)

    chunks: List[Dict] = []
    start = 0
    step = max(1, target_words - overlap_words)

    while start < len(words):
        end = min(start + target_words, len(words))
        window = words[start:end]
        chunk_text_str = " ".join(w for w in window if w != "\n\n").strip()
        if chunk_text_str:
            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "text": chunk_text_str,
                    "source_file": source_file,
                    "chunk_index": len(chunks),
                }
            )
        start += step

    # Backfill total_chunks
    total = len(chunks)
    for c in chunks:
        c["total_chunks"] = total

    log.info(
        "Chunking complete",
        extra={"source": source_file, "chunks": total, "target_words": target_words},
    )
    return chunks
