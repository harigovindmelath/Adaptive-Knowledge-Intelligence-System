"""
retrieval/vector_store.py
-------------------------
FAISS-backed vector store with on-disk embedding cache.

Improvements over v1:
- Disk-based embedding cache (avoids re-encoding on restart)
- Returns normalised cosine distances converted to similarity scores
- Thread-safe add / search
"""
from __future__ import annotations

import hashlib
import pickle
import threading
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config.loader import get as cfg
from config.logger import get_logger

log = get_logger(__name__)


class VectorStore:
    """
    FAISS IndexFlatIP (inner product on L2-normalised vecs == cosine similarity).
    """

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or cfg("embedding", "model", "BAAI/bge-small-en-v1.5")
        self._cache_dir = Path(cfg("embedding", "cache_dir", ".cache/embeddings"))
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        self._model: SentenceTransformer | None = None
        self.index: faiss.Index | None = None
        self.data: List[Tuple[str, Dict]] = []  # (text, metadata)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            log.info("Loading embedding model", extra={"model": self._model_name})
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts, using per-text disk cache where available."""
        use_cache = cfg("cache", "embedding_cache", True)
        vectors: List[np.ndarray | None] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        if use_cache:
            for i, text in enumerate(texts):
                cache_file = self._cache_dir / f"{self._cache_key(text)}.pkl"
                if cache_file.exists():
                    try:
                        with open(cache_file, "rb") as f:
                            vectors[i] = pickle.load(f)
                    except Exception:
                        pass
                if vectors[i] is None:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts

        if uncached_texts:
            model = self._get_model()
            new_vecs = model.encode(
                uncached_texts,
                show_progress_bar=len(uncached_texts) > 10,
                normalize_embeddings=True,
            )
            for idx, vec in zip(uncached_indices, new_vecs):
                vectors[idx] = vec
                if use_cache:
                    cache_file = self._cache_dir / f"{self._cache_key(texts[idx])}.pkl"
                    try:
                        with open(cache_file, "wb") as f:
                            pickle.dump(vec, f)
                    except Exception:
                        pass

        return np.vstack(vectors).astype("float32")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: List[Dict]) -> None:
        """Index a list of chunk dicts (must contain 'text' key)."""
        if not chunks:
            return
        with self._lock:
            texts = [c["text"] for c in chunks]
            embeddings = self._encode(texts)

            if self.index is None:
                dim = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dim)  # cosine via normalised vecs
                log.info("FAISS index created", extra={"dim": dim})

            self.index.add(embeddings)
            self.data.extend([(c["text"], c) for c in chunks])
            log.info("Chunks indexed", extra={"added": len(chunks), "total": len(self.data)})

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict, float]]:
        """
        Return top-k (text, metadata, similarity_score) tuples.
        similarity_score ∈ [0, 1] (higher = more similar).
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        with self._lock:
            q_emb = self._encode([query])
            k = min(top_k, self.index.ntotal)
            scores, indices = self.index.search(q_emb, k)
            results: List[Tuple[str, Dict, float]] = []
            for idx, score in zip(indices[0], scores[0]):
                if 0 <= idx < len(self.data):
                    text, meta = self.data[idx]
                    results.append((text, meta, float(score)))
            return results
