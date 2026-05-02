"""
retrieval/hybrid_retriever.py
------------------------------
Hybrid retrieval: FAISS (dense) + BM25 (sparse) with weighted score fusion.

Pipeline:
  1. Run FAISS search  → [(text, meta, score_dense)]
  2. Run BM25 search   → [(text, meta, score_sparse)]
  3. Normalise both score distributions to [0, 1]
  4. Fuse:  final_score = w_faiss * score_dense + w_bm25 * score_sparse
  5. Deduplicate by chunk_id, keep top-k
  6. Return with retrieval_confidence = mean of top-3 fused scores
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from config.loader import get as cfg
from config.logger import get_logger
from retrieval.bm25_retriever import BM25Index
from retrieval.vector_store import VectorStore

log = get_logger(__name__)


def _normalise(scores: List[float]) -> List[float]:
    if not scores:
        return scores
    min_s, max_s = min(scores), max(scores)
    rng = max_s - min_s if max_s != min_s else 1.0
    return [(s - min_s) / rng for s in scores]


class HybridRetriever:
    """
    Unified retriever over a VectorStore + BM25Index.
    Call `.index(chunks)` once after ingestion, then `.retrieve(query)` per query.
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        bm25_index: BM25Index | None = None,
    ) -> None:
        self.vector_store = vector_store or VectorStore()
        self.bm25_index = bm25_index or BM25Index()

        self._faiss_weight: float = float(cfg("retrieval", "faiss_weight", 0.6))
        self._bm25_weight: float = float(cfg("retrieval", "bm25_weight", 0.4))
        self._faiss_top_k: int = int(cfg("retrieval", "faiss_top_k", 10))
        self._bm25_top_k: int = int(cfg("retrieval", "bm25_top_k", 10))
        self._final_top_k: int = int(cfg("retrieval", "final_top_k", 5))
        self._confidence_threshold: float = float(cfg("retrieval", "confidence_threshold", 0.45))

    # ------------------------------------------------------------------

    def index(self, chunks: List[Dict]) -> None:
        """Index chunks into both FAISS and BM25."""
        log.info("Indexing chunks", extra={"count": len(chunks)})
        self.vector_store.add_chunks(chunks)
        self.bm25_index.add_chunks(chunks)

    # ------------------------------------------------------------------

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> Tuple[List[Dict], float]:
        """
        Retrieve relevant chunks and return ``(chunks, retrieval_confidence)``.

        *chunks* is a list of dicts with keys:
            chunk_id, text, source_file, chunk_index, total_chunks, fused_score

        *retrieval_confidence* ∈ [0, 1] — mean fused score of top-3 results.
        """
        final_k = top_k or self._final_top_k

        # --- Dense retrieval
        dense_results = self.vector_store.search(query, top_k=self._faiss_top_k)
        # --- Sparse retrieval
        sparse_results = self.bm25_index.search(query, top_k=self._bm25_top_k)

        # Map: chunk_id → {"meta": Dict, "dense": float, "sparse": float}
        combined: Dict[str, Dict] = {}

        dense_scores = [r[2] for r in dense_results]
        sparse_scores = [r[2] for r in sparse_results]
        norm_dense = _normalise(dense_scores)
        norm_sparse = _normalise(sparse_scores)

        for (text, meta, _), score in zip(dense_results, norm_dense):
            cid = meta.get("chunk_id", text[:20])
            combined.setdefault(cid, {"meta": meta, "dense": 0.0, "sparse": 0.0})
            combined[cid]["dense"] = score

        for (text, meta, _), score in zip(sparse_results, norm_sparse):
            cid = meta.get("chunk_id", text[:20])
            combined.setdefault(cid, {"meta": meta, "dense": 0.0, "sparse": 0.0})
            combined[cid]["sparse"] = score

        # Fuse
        fused: List[Tuple[str, float]] = []
        for cid, entry in combined.items():
            fused_score = (
                self._faiss_weight * entry["dense"]
                + self._bm25_weight * entry["sparse"]
            )
            fused.append((cid, fused_score))

        fused.sort(key=lambda x: x[1], reverse=True)

        # Build output
        output: List[Dict] = []
        for cid, score in fused[:final_k]:
            entry = combined[cid]
            meta = dict(entry["meta"])
            meta["fused_score"] = round(score, 4)
            output.append(meta)

        # Retrieval confidence = mean of top-3 fused scores
        top3_scores = [s for _, s in fused[:3]]
        retrieval_confidence = (sum(top3_scores) / len(top3_scores)) if top3_scores else 0.0

        log.info(
            "Retrieval complete",
            extra={
                "query": query[:80],
                "returned": len(output),
                "confidence": round(retrieval_confidence, 3),
            },
        )
        return output, retrieval_confidence
