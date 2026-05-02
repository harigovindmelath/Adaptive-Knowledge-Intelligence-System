"""
api/main.py
-----------
FastAPI interface for AKIS-v2.

Endpoints:
  GET  /              → health check
  POST /query         → run pipeline on a query
  POST /ingest        → ingest a new PDF document
  GET  /metrics       → simple query-count and avg-latency stats

Design:
  - Pipeline is built at startup with the default PDF (if set)
  - A RWLock-style pattern prevents reads during re-indexing
  - All responses are JSON (Pydantic models)
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel

from config.loader import CONFIG, get as cfg
from config.logger import get_logger
from ingestion.chunker import chunk_text
from ingestion.loader import load_pdf
from orchestrator.pipeline import Pipeline

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# App & global state
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AKIS-v2 — Adaptive Knowledge Intelligence System",
    version="2.0.0",
    description="Production-grade RAG pipeline with self-healing and hallucination detection.",
)

_pipeline: Optional[Pipeline] = None
_pipeline_lock = threading.Lock()

# Simple in-memory metrics
_metrics: Dict = {"queries": 0, "total_latency_ms": 0.0, "errors": 0}


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None


class ClaimOut(BaseModel):
    claim: str
    supported: bool
    confidence: float
    source_chunk_id: Optional[str] = None
    support_level: str


class SourceOut(BaseModel):
    chunk_id: str
    text: str
    source_file: str
    fused_score: float


class QueryResponse(BaseModel):
    trace_id: str
    query: str
    rewritten_query: Optional[str]
    answer: str
    grounded: bool
    confidence: float
    explanation: str
    provider_used: str
    status: str
    retries: int
    claims: List[ClaimOut]
    sources: List[SourceOut]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    version: str


class MetricsResponse(BaseModel):
    total_queries: int
    avg_latency_ms: float
    errors: int


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def _build_pipeline_from_pdf(pdf_path: str) -> Pipeline:
    text = load_pdf(pdf_path)
    chunks = chunk_text(text, source_file=Path(pdf_path).name)
    pipeline = Pipeline()
    pipeline.index(chunks)
    return pipeline


@app.on_event("startup")
def startup_event() -> None:
    global _pipeline
    default_pdf = os.getenv("AKIS_DEFAULT_PDF", "")
    if default_pdf and Path(default_pdf).exists():
        log.info("Building pipeline from default PDF.", extra={"pdf": default_pdf})
        try:
            with _pipeline_lock:
                _pipeline = _build_pipeline_from_pdf(default_pdf)
            log.info("Pipeline ready.")
        except Exception as e:
            log.error("Pipeline init failed.", extra={"error": str(e)})
    else:
        log.info("No default PDF set. POST /ingest to load a document.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="running",
        pipeline_ready=_pipeline is not None,
        version="2.0.0",
    )


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest) -> QueryResponse:
    global _pipeline, _metrics
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialised. POST a PDF to /ingest first.",
        )
    try:
        result = _pipeline.run(req.query)
        _metrics["queries"] += 1
        _metrics["total_latency_ms"] += result.latency_ms
    except Exception as e:
        _metrics["errors"] += 1
        log.error("Query execution error", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    d = result.to_dict()
    return QueryResponse(
        trace_id=d["trace_id"],
        query=d["query"],
        rewritten_query=d.get("rewritten_query"),
        answer=d["answer"],
        grounded=d["grounded"],
        confidence=d["confidence"],
        explanation=d["explanation"],
        provider_used=d["provider_used"],
        status=d["status"],
        retries=d["retries"],
        claims=[ClaimOut(**c) for c in d["claims"]],
        sources=[SourceOut(**s) for s in d["sources"]],
        latency_ms=d["latency_ms"],
    )


@app.post("/ingest")
async def ingest_endpoint(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> Dict:
    """
    Upload a PDF and rebuild the pipeline in the background.
    Returns immediately; check GET / to confirm pipeline_ready.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    tmp_path = Path("/tmp") / file.filename
    content = await file.read()
    tmp_path.write_bytes(content)
    log.info("PDF uploaded", extra={"file": file.filename, "bytes": len(content)})

    def _rebuild():
        global _pipeline
        try:
            new_pipeline = _build_pipeline_from_pdf(str(tmp_path))
            with _pipeline_lock:
                _pipeline = new_pipeline
            log.info("Pipeline rebuilt from uploaded PDF.", extra={"file": file.filename})
        except Exception as e:
            log.error("Pipeline rebuild failed.", extra={"error": str(e)})

    background_tasks.add_task(_rebuild)
    return {"message": f"Ingesting {file.filename}. Pipeline will be ready shortly."}


@app.get("/metrics", response_model=MetricsResponse)
def metrics_endpoint() -> MetricsResponse:
    q = _metrics["queries"]
    avg = (_metrics["total_latency_ms"] / q) if q > 0 else 0.0
    return MetricsResponse(
        total_queries=q,
        avg_latency_ms=round(avg, 1),
        errors=_metrics["errors"],
    )


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=cfg("api", "host", "0.0.0.0"),
        port=int(cfg("api", "port", 8000)),
        reload=False,
    )
