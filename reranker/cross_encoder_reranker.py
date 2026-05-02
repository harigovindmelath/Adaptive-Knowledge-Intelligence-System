"""
reranker/cross_encoder_reranker.py
-----------------------------------
Lightweight cross-encoder reranker using a tiny MS-MARCO model.
Falls back gracefully if the model is unavailable (returns input unchanged).

Why cross-encoder?
  Bi-encoders (FAISS) trade accuracy for speed.
  A cross-encoder scores (query, doc) jointly for better relevance —
  but is too slow to run over thousands of docs.
  Here we re-rank only the top-10 from hybrid retrieval (fast enough on CPU).
"""
from __future__ import annotations

from typing import Dict, List

from config.loader import get as cfg
from config.logger import get_logger

log = get_logger(__name__)

_RERANKER_AVAILABLE = False
_cross_encoder = None


def _load_cross_encoder():
    global _cross_encoder, _RERANKER_AVAILABLE
    if _cross_encoder is not None:
        return _cross_encoder
    try:
        from sentence_transformers import CrossEncoder  # type: ignore

        model_name = cfg("reranker", "model", "cross-encoder/ms-marco-MiniLM-L-2-v2")
        log.info("Loading cross-encoder", extra={"model": model_name})
        _cross_encoder = CrossEncoder(model_name, max_length=512)
        _RERANKER_AVAILABLE = True
        return _cross_encoder
    except Exception as e:
        log.warning(
            "Cross-encoder unavailable; skipping reranker.",
            extra={"error": str(e)},
        )
        _RERANKER_AVAILABLE = False
        return None


def rerank(query: str, chunks: List[Dict], top_k: int | None = None) -> List[Dict]:
    """
    Re-rank *chunks* against *query* using a cross-encoder.
    Returns the top-k chunks sorted by cross-encoder relevance score (descending).
    Appends ``rerank_score`` to each chunk dict.

    If the reranker is disabled or fails, returns the original list unchanged.
    """
    enabled = cfg("reranker", "enabled", True)
    if not enabled:
        return chunks

    top_k = top_k or int(cfg("reranker", "top_k", 5))
    model = _load_cross_encoder()
    if model is None:
        return chunks

    try:
        pairs = [(query, c["text"]) for c in chunks]
        scores = model.predict(pairs, show_progress_bar=False)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = round(float(score), 4)

        reranked = sorted(chunks, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        log.info("Reranking done", extra={"query": query[:60], "returned": top_k})
        return reranked[:top_k]
    except Exception as e:
        log.error("Reranker failed; returning original order.", extra={"error": str(e)})
        return chunks
