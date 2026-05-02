"""
evaluator/evaluator.py
-----------------------
Answer evaluator: hallucination detection + grounding verification.

Outputs structured JSON-ready dict:
  {
    "grounded": bool,
    "confidence": float,          # 0–100
    "explanation": str,
    "claims": [
        {
          "claim": str,
          "supported": bool,
          "confidence": float,    # 0–1
          "source_chunk_id": str | None,
          "support_level": "strong" | "weak" | "none"
        }
    ]
  }

Improvements over v1:
  - Semantic cosine similarity (not substring match)
  - Structured output contract (not plain int)
  - Explanation field for explainability
  - Configurable thresholds from YAML
"""
from __future__ import annotations

import re
import threading
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config.loader import get as cfg
from config.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Singleton model (shared with reranker to save RAM)
# ---------------------------------------------------------------------------

_model: Optional[SentenceTransformer] = None
_model_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                model_name = cfg("evaluator", "model", "all-MiniLM-L6-v2")
                log.info("Loading evaluator model", extra={"model": model_name})
                _model = SentenceTransformer(model_name)
    return _model


# ---------------------------------------------------------------------------
# Claim splitting
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_claims(text: str) -> List[str]:
    raw = _SENTENCE_SPLIT.split(text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 10]


# ---------------------------------------------------------------------------
# Semantic verification
# ---------------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _verify_claims(
    claims: List[str],
    chunks: List[Dict],
    model: SentenceTransformer,
    strong_thr: float,
    weak_thr: float,
) -> List[Dict]:
    if not claims or not chunks:
        return [
            {
                "claim": c,
                "supported": False,
                "confidence": 0.0,
                "source_chunk_id": None,
                "support_level": "none",
            }
            for c in claims
        ]

    chunk_texts = [c.get("text", "") for c in chunks]
    chunk_ids = [c.get("chunk_id", "?") for c in chunks]

    claim_embs = model.encode(claims, normalize_embeddings=True, show_progress_bar=False)
    chunk_embs = model.encode(chunk_texts, normalize_embeddings=True, show_progress_bar=False)

    results: List[Dict] = []
    for i, claim in enumerate(claims):
        c_emb = claim_embs[i]
        sims = [_cosine(c_emb, ch_emb) for ch_emb in chunk_embs]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        if best_sim >= strong_thr:
            support_level = "strong"
            supported = True
        elif best_sim >= weak_thr:
            support_level = "weak"
            supported = True
        else:
            support_level = "none"
            supported = False

        results.append(
            {
                "claim": claim,
                "supported": supported,
                "confidence": round(best_sim, 3),
                "source_chunk_id": chunk_ids[best_idx] if supported else None,
                "support_level": support_level,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class Evaluator:
    """Evaluates an answer for hallucinations and context grounding."""

    def __init__(self) -> None:
        self._strong_thr: float = float(
            cfg("evaluator", "similarity_threshold_strong", 0.65)
        )
        self._weak_thr: float = float(
            cfg("evaluator", "similarity_threshold_weak", 0.55)
        )
        self._min_confidence: float = float(
            cfg("evaluator", "min_grounded_confidence", 50.0)
        )

    def evaluate(self, answer: str, chunks: List[Dict]) -> Dict:
        """
        Evaluate *answer* against *chunks*.

        Returns a structured evaluation dict (see module docstring).
        """
        log.info("Evaluating answer", extra={"answer_len": len(answer), "chunks": len(chunks)})

        if not answer or not chunks:
            return {
                "grounded": False,
                "confidence": 0.0,
                "explanation": "No answer or no context provided.",
                "claims": [],
            }

        # Detect obvious uncertainty / insufficiency markers
        uncertainty_phrases = [
            "insufficient information",
            "not in the context",
            "cannot answer",
            "i don't know",
            "i do not know",
        ]
        if any(p in answer.lower() for p in uncertainty_phrases):
            return {
                "grounded": False,
                "confidence": 0.0,
                "explanation": "Answer indicates insufficient context.",
                "claims": [],
            }

        claims = _split_claims(answer)
        if not claims:
            return {
                "grounded": False,
                "confidence": 0.0,
                "explanation": "No verifiable claims extracted from answer.",
                "claims": [],
            }

        model = _get_model()
        verified = _verify_claims(
            claims, chunks, model, self._strong_thr, self._weak_thr
        )

        n_supported = sum(1 for v in verified if v["supported"])
        confidence = (n_supported / len(verified)) * 100.0

        grounded = confidence >= self._min_confidence
        strong_count = sum(1 for v in verified if v["support_level"] == "strong")
        weak_count = sum(1 for v in verified if v["support_level"] == "weak")
        unsupported_count = len(verified) - n_supported

        explanation = (
            f"{n_supported}/{len(verified)} claims supported "
            f"({strong_count} strong, {weak_count} weak, {unsupported_count} unsupported). "
            f"Confidence: {confidence:.1f}%."
        )

        log.info(
            "Evaluation complete",
            extra={
                "grounded": grounded,
                "confidence": round(confidence, 1),
                "claims_total": len(verified),
            },
        )

        return {
            "grounded": grounded,
            "confidence": round(confidence, 1),
            "explanation": explanation,
            "claims": verified,
        }
