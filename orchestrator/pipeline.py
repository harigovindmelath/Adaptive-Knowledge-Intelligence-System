"""
orchestrator/pipeline.py
------------------------
Central orchestrator for AKIS-v2.

Full execution flow:
  1. Receive query
  2. Hybrid retrieval (FAISS + BM25)
  3. Check retrieval confidence
     └─ Low? → Rewrite query → Retry retrieval (up to max_rewrites)
  4. Re-rank retrieved chunks
  5. Generate answer
     └─ LLM unavailable? → Uncertainty response
  6. Evaluate answer (hallucination / grounding check)
     └─ Low grounding? → Uncertainty response
  7. Return structured PipelineResult

Each stage emits a structured trace log (stage, input, output_summary).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config.loader import get as cfg
from config.logger import get_logger
from evaluator.evaluator import Evaluator
from generator.generator import Generator
from reranker.cross_encoder_reranker import rerank
from retrieval.hybrid_retriever import HybridRetriever

log = get_logger(__name__)

_FALLBACK_MSG: str = cfg(
    "self_healing",
    "fallback_message",
    "I cannot answer this question with sufficient confidence based on the provided documents.",
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Complete output of a single pipeline run."""

    trace_id: str
    query: str
    rewritten_query: Optional[str]
    answer: str
    grounded: bool
    confidence: float  # 0–100
    explanation: str
    provider_used: str
    status: str  # SUCCESS | LOW_CONFIDENCE | GENERATION_FAILED | NO_CONTEXT
    retries: int
    claims: List[Dict]
    sources: List[Dict]
    latency_ms: float

    def to_dict(self) -> Dict:
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "rewritten_query": self.rewritten_query,
            "answer": self.answer,
            "grounded": self.grounded,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "provider_used": self.provider_used,
            "status": self.status,
            "retries": self.retries,
            "claims": self.claims,
            "sources": [
                {
                    "chunk_id": s.get("chunk_id", "?"),
                    "text": s.get("text", ""),
                    "source_file": s.get("source_file", "?"),
                    "fused_score": s.get("fused_score", 0.0),
                }
                for s in self.sources
            ],
            "latency_ms": self.latency_ms,
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Pipeline:
    """
    Stateful pipeline: call `index(chunks)` once, then `run(query)` per query.
    """

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        generator: Generator | None = None,
        evaluator: Evaluator | None = None,
    ) -> None:
        self.retriever = retriever or HybridRetriever()
        self.generator = generator or Generator()
        self.evaluator = evaluator or Evaluator()

        self._conf_threshold: float = float(cfg("retrieval", "confidence_threshold", 0.45))
        self._max_rewrites: int = int(cfg("self_healing", "max_rewrites", 2))
        self._max_retries: int = int(cfg("generator", "max_retries", 2))

    # ------------------------------------------------------------------

    def index(self, chunks: List[Dict]) -> None:
        """Index document chunks. Must be called before any `run()` calls."""
        self.retriever.index(chunks)
        log.info("Pipeline indexed", extra={"chunks": len(chunks)})

    # ------------------------------------------------------------------

    def run(self, query: str) -> PipelineResult:
        """Execute the full pipeline for *query* and return a PipelineResult."""
        t0 = time.time()
        trace_id = str(uuid.uuid4())[:8]
        retries = 0
        rewritten_query: Optional[str] = None
        active_query = query

        log.info("Pipeline start", extra={"trace_id": trace_id, "query": query[:80]})

        # ── Stage 1–3: Retrieval + self-healing loop ──────────────────────
        chunks, confidence = self.retriever.retrieve(active_query)

        for attempt in range(self._max_rewrites):
            if confidence >= self._conf_threshold:
                break
            if not chunks and attempt == 0:
                log.warning(
                    "No documents retrieved; attempting rewrite.",
                    extra={"trace_id": trace_id},
                )
            else:
                log.info(
                    "Low retrieval confidence — rewriting query.",
                    extra={"trace_id": trace_id, "confidence": round(confidence, 3)},
                )

            new_query = self.generator.rewrite_query(active_query)
            if not new_query or new_query.lower() == active_query.lower():
                log.info(
                    "Query rewrite produced no change; stopping retry.",
                    extra={"trace_id": trace_id},
                )
                break

            rewritten_query = new_query
            active_query = new_query
            retries += 1
            chunks, confidence = self.retriever.retrieve(active_query)
            log.info(
                "Retry retrieval",
                extra={
                    "trace_id": trace_id,
                    "attempt": attempt + 1,
                    "new_confidence": round(confidence, 3),
                    "rewritten": new_query[:80],
                },
            )

        # ── No context at all ─────────────────────────────────────────────
        if not chunks:
            return self._build_result(
                trace_id, query, rewritten_query,
                answer=_FALLBACK_MSG,
                grounded=False, confidence=0.0,
                explanation="No relevant documents found after retrieval retries.",
                provider="none", status="NO_CONTEXT",
                retries=retries, claims=[], sources=[],
                t0=t0,
            )

        # ── Stage 4: Rerank ───────────────────────────────────────────────
        chunks = rerank(active_query, chunks)

        # ── Stage 5: Generate ─────────────────────────────────────────────
        answer, provider = self.generator.generate(chunks, active_query)

        if not answer:
            return self._build_result(
                trace_id, query, rewritten_query,
                answer=_FALLBACK_MSG,
                grounded=False, confidence=0.0,
                explanation="LLM generation failed (no response from any provider).",
                provider="none", status="GENERATION_FAILED",
                retries=retries, claims=[], sources=chunks,
                t0=t0,
            )

        # ── Stage 6: Evaluate ─────────────────────────────────────────────
        eval_result = self.evaluator.evaluate(answer, chunks)
        grounded = eval_result["grounded"]
        conf = eval_result["confidence"]
        explanation = eval_result["explanation"]
        claims = eval_result["claims"]

        if not grounded:
            # Return uncertainty-aware response rather than hallucination
            final_answer = (
                f"{_FALLBACK_MSG}\n\n"
                f"(Partial answer — low grounding confidence {conf:.0f}%): {answer}"
            )
            status = "LOW_CONFIDENCE"
        else:
            final_answer = answer
            status = "SUCCESS"

        log.info(
            "Pipeline complete",
            extra={
                "trace_id": trace_id,
                "status": status,
                "confidence": conf,
                "retries": retries,
                "provider": provider,
                "latency_ms": round((time.time() - t0) * 1000, 1),
            },
        )

        return self._build_result(
            trace_id, query, rewritten_query,
            answer=final_answer,
            grounded=grounded, confidence=conf,
            explanation=explanation,
            provider=provider, status=status,
            retries=retries, claims=claims, sources=chunks,
            t0=t0,
        )

    # ------------------------------------------------------------------
    # Internal builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_result(
        trace_id, query, rewritten_query, answer, grounded,
        confidence, explanation, provider, status,
        retries, claims, sources, t0,
    ) -> PipelineResult:
        return PipelineResult(
            trace_id=trace_id,
            query=query,
            rewritten_query=rewritten_query,
            answer=answer,
            grounded=grounded,
            confidence=confidence,
            explanation=explanation,
            provider_used=provider,
            status=status,
            retries=retries,
            claims=claims,
            sources=sources,
            latency_ms=round((time.time() - t0) * 1000, 1),
        )
