"""
retrieval/bm25_retriever.py
---------------------------
BM25 keyword retrieval using rank_bm25.
CPU-native, no GPU needed. Serves as the sparse leg of hybrid retrieval.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from config.logger import get_logger

log = get_logger(__name__)

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
    log.warning("rank_bm25 not installed. BM25 retrieval disabled. Install with: pip install rank-bm25")


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokeniser with stop-word-lite removal."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    stop = {"the", "a", "an", "is", "in", "on", "of", "and", "or", "to", "it", "for", "with"}
    return [t for t in tokens if t not in stop]


class BM25Index:
    """
    Wraps rank_bm25.BM25Okapi to match the VectorStore.search() interface.
    """

    def __init__(self) -> None:
        self._bm25: "BM25Okapi | None" = None
        self.data: List[Tuple[str, Dict]] = []  # (text, metadata)

    def add_chunks(self, chunks: List[Dict]) -> None:
        if not _BM25_AVAILABLE or not chunks:
            return
        self.data = [(c["text"], c) for c in chunks]
        tokenized = [_tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        log.info("BM25 index built", extra={"docs": len(chunks)})

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, Dict, float]]:
        """
        Return top-k (text, metadata, normalised_bm25_score) tuples.
        Scores are min-max normalised to [0, 1].
        """
        if not _BM25_AVAILABLE or self._bm25 is None or not self.data:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)  # numpy array

        # Normalise
        max_score = scores.max() if scores.max() > 0 else 1.0
        norm_scores = scores / max_score

        top_indices = scores.argsort()[::-1][:top_k]
        results: List[Tuple[str, Dict, float]] = []
        for idx in top_indices:
            text, meta = self.data[idx]
            results.append((text, meta, float(norm_scores[idx])))
        return results
