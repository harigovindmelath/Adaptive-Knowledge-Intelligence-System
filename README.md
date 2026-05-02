
# Adaptive Knowledge Intelligence System (AKIS-v2)

A production-oriented Retrieval-Augmented Generation (RAG) system designed to reduce hallucinations through confidence-aware retrieval, query rewriting, and post-generation grounding verification. The system is built to run on CPU-only environments while maintaining a modular, extensible architecture.

---

## Overview

Most RAG pipelines assume that retrieval is reliable and proceed directly to generation. In practice, weak retrieval leads to confident hallucinations.

AKIS-v2 addresses this by introducing a **self-healing pipeline**:

* Retrieval quality is explicitly measured
* Low-confidence queries are rewritten and retried
* Generated answers are validated against retrieved context
* The system returns structured confidence and grounding signals

The goal is not just to generate answers, but to **reason about when those answers should be trusted**.

---

## System Design

The system is organized into independent modules coordinated by a central orchestrator:

```text
Query
  → Hybrid Retrieval (FAISS + BM25)
  → Confidence Scoring
      → (if low) Query Rewrite + Retry
  → Cross-Encoder Reranking
  → LLM Generation
  → Grounding Evaluation
  → Structured Response
```

### Key design choices

* **Hybrid retrieval**
  Combines dense embeddings (MiniLM + FAISS) with sparse keyword search (BM25) to improve recall across both semantic and lexical queries.

* **Confidence-aware control flow**
  Retrieval produces a normalized confidence score used to decide whether generation should proceed or the query should be rewritten.

* **Self-healing loop**
  Queries that underperform are rewritten using the LLM and re-executed, avoiding generation on weak context.

* **Post-generation validation**
  Answers are decomposed into claims and verified against retrieved chunks using semantic similarity.

* **CPU-first design**
  All components are selected and configured to run within constrained environments (e.g., Colab CPU, 8GB RAM).

---

## Project Structure

```text
AKIS_v2/
├── ingestion/        # PDF loading and semantic chunking
├── retrieval/        # FAISS vector store + BM25 + fusion logic
├── reranker/         # Cross-encoder reranking (MiniLM)
├── generator/        # LLM interface + query rewriting
├── evaluator/        # Claim-level grounding verification
├── orchestrator/     # Pipeline control and self-healing loop
├── api/              # FastAPI service layer
├── ui/               # Streamlit interface
├── eval/             # Evaluation and benchmarking
├── tests/            # Unit and integration tests
├── config/           # Central configuration and logging
```

---

## Execution Flow

1. Retrieve top-k candidates using FAISS and BM25
2. Normalize and fuse scores to compute retrieval confidence
3. If confidence is below threshold:

   * Rewrite query
   * Retry retrieval (bounded attempts)
4. Rerank results using a cross-encoder
5. Generate answer using local LLM (with API fallback)
6. Evaluate grounding:

   * Split answer into claims
   * Match each claim against retrieved context
7. Return structured result:

   * answer
   * confidence score
   * grounding status
   * supporting context

---

## Evaluation and Observability

The system includes basic evaluation and tracing capabilities:

* Retrieval confidence tracking
* Claim-level grounding analysis
* Structured JSON logs for each pipeline stage
* Benchmark script for:

  * retrieval recall
  * answer similarity
  * grounding accuracy

This allows inspection of failure modes rather than treating the pipeline as a black box.

---

## Failure Handling

The system explicitly models failure cases:

* Low-quality queries → rewritten and retried
* No relevant context → fallback response with uncertainty
* Conflicting documents → partial grounding with explanation
* LLM unavailable → generation failure surfaced cleanly

The pipeline avoids silent failure and instead returns **interpretable outputs**.

---

## Limitations

* CPU-only inference results in higher latency (5–30s per query)
* Grounding uses cosine similarity, not full entailment modeling
* Single-document indexing (no multi-document aggregation yet)
* No streaming responses
* Metrics are not persisted across runs

---

## Future Work

* Multi-document indexing and retrieval
* NLI-based grounding evaluator
* Streaming generation
* Persistent metrics (database or monitoring integration)
* Async request handling for concurrency

---

## Running the System (Colab)

```python
!pip install -q faiss-cpu sentence-transformers rank-bm25 pdfplumber PyYAML pydantic

import sys
sys.path.insert(0, 'AKIS_v2')

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_text
from orchestrator.pipeline import Pipeline

text = load_pdf('your_doc.pdf')
chunks = chunk_text(text, 'your_doc.pdf')

pipeline = Pipeline()
pipeline.index(chunks)

result = pipeline.run("What is this document about?")
print(result.answer)
```

---
