# AKIS-v2 — Adaptive Knowledge Intelligence System

**Production-grade RAG pipeline · Self-healing · Hallucination detection · CPU-only**

---

## Project Structure

```
AKIS_v2/
├── config/
│   ├── config.yaml          ← Central config (all tunables in one place)
│   ├── loader.py            ← YAML loader + env-var overrides
│   └── logger.py            ← Structured JSON logger (all modules)
│
├── ingestion/
│   ├── loader.py            ← PDF ingestion + cleaning
│   └── chunker.py           ← Semantic chunker with overlap
│
├── retrieval/
│   ├── vector_store.py      ← FAISS IndexFlatIP + embedding cache
│   ├── bm25_retriever.py    ← BM25 keyword retrieval (rank_bm25)
│   └── hybrid_retriever.py  ← Weighted fusion (FAISS + BM25) + confidence score
│
├── reranker/
│   └── cross_encoder_reranker.py  ← ms-marco-MiniLM-L-2 cross-encoder (CPU)
│
├── generator/
│   └── generator.py         ← Ollama + Gemini backends, query rewriting
│
├── evaluator/
│   └── evaluator.py         ← Semantic grounding checker, structured JSON output
│
├── orchestrator/
│   └── pipeline.py          ← Central orchestrator + self-healing loop
│
├── api/
│   └── main.py              ← FastAPI: /query, /ingest, /metrics
│
├── ui/
│   └── app.py               ← Streamlit dashboard
│
├── eval/
│   └── benchmark.py         ← Evaluation framework (grounding accuracy, answer sim)
│
├── tests/
│   └── test_pipeline.py     ← Unit + integration tests (pytest, mocked LLM)
│
├── data/                    ← Place your PDFs here
├── logs/                    ← akis.log (auto-created, structured JSON)
├── requirements.txt
└── AKIS_v2_Colab.ipynb      ← Step-by-step Colab execution notebook
```

---

## Execution Flow (Step-by-Step)

```
User query
    │
    ▼
[1] HybridRetriever.retrieve(query)
    ├─ FAISS dense search     → top-10 (cosine similarity)
    ├─ BM25 sparse search     → top-10 (keyword relevance)
    ├─ Score normalisation    → [0, 1] for each system
    ├─ Weighted fusion        → 0.6 × dense + 0.4 × sparse
    └─ Returns (chunks, retrieval_confidence)
    │
    ▼
[2] Confidence check
    ├─ confidence ≥ 0.45? → proceed
    └─ confidence < 0.45? → Generator.rewrite_query()
                             └─ retry (up to 2 times)
    │
    ▼
[3] CrossEncoder.rerank(query, chunks)
    └─ Re-scores top-10 via ms-marco-MiniLM → returns top-5
    │
    ▼
[4] Generator.generate(chunks, query)
    ├─ Ollama (local)   → primary
    └─ Gemini (API)     → fallback if Ollama fails
    │
    ▼
[5] Evaluator.evaluate(answer, chunks)
    ├─ Split answer into claims
    ├─ Encode claims + chunks with MiniLM
    ├─ Cosine similarity per claim → best chunk match
    ├─ strong ≥ 0.65, weak ≥ 0.55, else none
    └─ Returns {grounded, confidence, explanation, claims[]}
    │
    ▼
[6] PipelineResult
    ├─ status: SUCCESS | LOW_CONFIDENCE | GENERATION_FAILED | NO_CONTEXT
    ├─ answer (or fallback message)
    ├─ confidence %
    ├─ claim-level grounding details
    └─ source chunks
```

---

## Changelog (v1 → v2)

### Added
- `retrieval/hybrid_retriever.py` — Weighted FAISS+BM25 fusion with confidence score
- `retrieval/bm25_retriever.py` — BM25 keyword retrieval (was missing in v1)
- `reranker/cross_encoder_reranker.py` — CPU cross-encoder reranking
- `ingestion/chunker.py` — Overlap-aware chunking (was fixed 350-word no overlap)
- `config/config.yaml` + `config/loader.py` — Centralised configuration
- `config/logger.py` — Structured JSON logging for every stage
- `orchestrator/pipeline.py` — Central pipeline orchestrator with full trace
- `evaluator/evaluator.py` — Structured `{grounded, confidence, explanation, claims[]}` output
- `api/main.py` — `/ingest` endpoint, `/metrics` endpoint, background rebuild
- `eval/benchmark.py` — Benchmark dataset + grounding accuracy + answer similarity metrics
- `tests/test_pipeline.py` — pytest unit tests with mocked LLM

### Improved
- **Retrieval confidence scoring**: now computed from fused scores, used to trigger self-healing
- **Query rewriting**: integrated into orchestrator self-healing loop (was stub in v1)
- **Evaluator output**: structured JSON vs plain integer confidence in v1
- **Embedding caching**: disk-level per-text MD5 cache (avoids re-encoding on restart)
- **Context building**: token-budget-aware with chunk_id anchors in prompt
- **Fallback**: uncertainty-aware response with partial answer flagged rather than silent failure

### Replaced
- `llm_router.py` (monolithic, hardcoded Gemini key in source) → `generator/generator.py` (clean, config-driven)
- `validation/verifier.py` (substring match) → `evaluator/evaluator.py` (semantic cosine)
- `validation/scorer.py` (int %) → embedded in `Evaluator.evaluate()` returning float + explanation
- `retrieval/multi_query.py` (heuristic alt queries) → self-healing loop with real LLM rewrite

---

## Component Verification Checklist

| Module | File | Status |
|--------|------|--------|
| Ingestion | `ingestion/loader.py` | ✅ |
| Chunking | `ingestion/chunker.py` | ✅ |
| FAISS store | `retrieval/vector_store.py` | ✅ |
| BM25 | `retrieval/bm25_retriever.py` | ✅ |
| Hybrid fusion | `retrieval/hybrid_retriever.py` | ✅ |
| Reranker | `reranker/cross_encoder_reranker.py` | ✅ |
| Generator | `generator/generator.py` | ✅ |
| Evaluator | `evaluator/evaluator.py` | ✅ |
| Orchestrator | `orchestrator/pipeline.py` | ✅ |
| API | `api/main.py` | ✅ |
| UI | `ui/app.py` | ✅ |
| Config | `config/config.yaml` | ✅ |
| Logging | `config/logger.py` | ✅ |
| Benchmark | `eval/benchmark.py` | ✅ |
| Tests | `tests/test_pipeline.py` | ✅ |

---

## Test Cases

### TC-1: Normal query with relevant document
- **Input**: "What are the main skills mentioned?"
- **Expected**: retrieval_confidence ≥ 0.45, no rewrite, answer generated, grounded=True
- **Evaluator**: ≥ 50% claims supported → SUCCESS

### TC-2: Vague query (triggers rewrite)
- **Input**: "Tell me about stuff"
- **Expected**: confidence < 0.45 on first pass → query rewritten → retry retrieval
- **Evaluator**: depends on rewrite quality; status = LOW_CONFIDENCE possible

### TC-3: Out-of-domain query
- **Input**: "xkcd purple unicorn marmalade"
- **Expected**: no relevant chunks OR all chunks score near-zero → NO_CONTEXT or LOW_CONFIDENCE
- **Answer**: fallback message, grounded=False

### TC-4: LLM unavailable
- **Input**: Any valid query when Ollama is down and no Gemini key
- **Expected**: `generate()` returns None → status = GENERATION_FAILED
- **Answer**: fallback message; sources still returned (retrieval succeeded)

### TC-5: Conflicting documents
- **Input**: Query with two contradictory chunks
- **Expected**: Answer generated; evaluator detects weak/no support for contradicted claims
- **Status**: LOW_CONFIDENCE; explanation notes unsupported claims

---

## Failure Cases & Debugging

### Common failure points

| Failure | Root cause | How to debug |
|---------|------------|--------------|
| Empty FAISS index | PDF extracted no text | Check `logs/akis.log` for "Empty page" warnings |
| All claims unsupported | Evaluator model not downloaded | Check for HuggingFace download errors in logs |
| LOW_CONFIDENCE on everything | Threshold too high | Lower `evaluator.similarity_threshold_strong` in config.yaml |
| Query rewrite loops | LLM returns same query | Rewrite stops after max_rewrites; check generator logs |
| GENERATION_FAILED | Ollama not running + no Gemini key | Set `GEMINI_API_KEY` or start Ollama |
| Slow first run | Models downloading | Normal; cached after first run |

### Logs to inspect
```bash
cat logs/akis.log | python -m json.tool | grep '"level": "ERROR"'
cat logs/akis.log | python -m json.tool | grep '"msg": "Retrieval complete"'
```

---

## Limitations

1. **No GPU acceleration** — all embedding and inference is CPU; expect 5–30s per query
2. **LLM dependency** — generation requires Ollama locally or a Gemini API key; without either, GENERATION_FAILED is returned (retrieval still works)
3. **Evaluator is approximate** — cosine similarity ≠ true entailment; short claims may get false positives
4. **Single-document pipeline** — `/ingest` replaces the current index rather than appending
5. **No streaming** — LLM response is returned only after full generation
6. **In-memory metrics** — metrics reset on server restart; no persistent dashboard
7. **BM25 tokeniser is naive** — basic word tokeniser; domain-specific stop words not tuned

---

## Next Improvements (Path to Industry-Level)

1. **Multi-document index** — append to FAISS/BM25 rather than replace; add document registry
2. **LLM streaming** — stream Ollama tokens to Streamlit for perceived speed
3. **Proper NLI evaluator** — replace cosine similarity with a small NLI model (e.g. DeBERTa-small)
4. **Persistent metrics** — write to SQLite or push to Prometheus
5. **HyDE** — Hypothetical Document Embeddings for better dense retrieval
6. **Async FastAPI** — switch pipeline.run() to async for concurrent requests
7. **Auth** — add API key header for the FastAPI interface

---

## Resume Bullets

```
• Architected a production-grade self-healing RAG pipeline (AKIS-v2) with modular
  ingestion, hybrid FAISS+BM25 retrieval, cross-encoder reranking, and LLM generation;
  deployed on CPU with full observability via structured JSON logging.

• Implemented an automated confidence-scoring and query-rewriting loop that detects low
  retrieval quality, rewrites the query via the LLM, and retries before falling back to
  an uncertainty-aware response — reducing hallucination risk without human intervention.

• Built a semantic hallucination detection layer that splits generated answers into atomic
  claims and verifies each claim against retrieved context using cosine similarity, emitting
  structured {grounded, confidence, explanation, claims[]} JSON for downstream auditability.

• Exposed the system as a FastAPI service with /query, /ingest, and /metrics endpoints,
  enabling background PDF re-indexing, live query serving, and lightweight performance
  monitoring — demonstrating real-world deployment readiness on constrained hardware.
```

---

## Quick Start (Colab)

```python
# 1. Install
!pip install -q faiss-cpu sentence-transformers rank-bm25 pdfplumber PyYAML pydantic

# 2. Set path
import sys; sys.path.insert(0, 'AKIS_v2')

# 3. Ingest
from ingestion.loader import load_pdf
from ingestion.chunker import chunk_text
from orchestrator.pipeline import Pipeline

text = load_pdf('your_doc.pdf')
chunks = chunk_text(text, 'your_doc.pdf')
pipeline = Pipeline()
pipeline.index(chunks)

# 4. Query
result = pipeline.run('What is this document about?')
print(result.status, result.confidence, result.answer)
```
