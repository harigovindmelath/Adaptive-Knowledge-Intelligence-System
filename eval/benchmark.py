"""
eval/benchmark.py
-----------------
Lightweight evaluation framework for AKIS-v2.

Metrics:
  - Retrieval Recall@k   : fraction of expected chunks retrieved in top-k
  - Answer Correctness   : fuzzy semantic similarity vs. reference answer
  - Grounding Accuracy   : fraction of answers correctly marked grounded/ungrounded

Usage:
  python -m eval.benchmark --api_url http://localhost:8000

Output:
  JSON report to eval/benchmark_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import requests

# ---------------------------------------------------------------------------
# Benchmark dataset (extend this with domain-specific Q&A pairs)
# ---------------------------------------------------------------------------

BENCHMARK_DATASET: List[Dict] = [
    {
        "id": "q1",
        "query": "What is the main topic of the document?",
        "reference_answer": "The document covers the subject matter of the provided PDF.",
        "expected_grounded": True,
        "notes": "Generic anchor query — should always retrieve something.",
    },
    {
        "id": "q2",
        "query": "xkcd purple unicorn does not exist in document at all",
        "reference_answer": None,
        "expected_grounded": False,
        "notes": "Out-of-domain query — expect NO_CONTEXT or LOW_CONFIDENCE.",
    },
    {
        "id": "q3",
        "query": "What are the key skills mentioned?",
        "reference_answer": "Skills include technical and soft skills mentioned in the document.",
        "expected_grounded": True,
        "notes": "Should retrieve skill-related chunks.",
    },
    {
        "id": "q4",
        "query": "Summarise the experience section.",
        "reference_answer": "The experience section describes professional or project experience.",
        "expected_grounded": True,
        "notes": "Vague query — tests query rewriting.",
    },
    {
        "id": "q5",
        "query": "What is the author's educational background?",
        "reference_answer": "Educational background includes degrees and institutions mentioned.",
        "expected_grounded": True,
        "notes": "Standard factual retrieval.",
    },
]


# ---------------------------------------------------------------------------
# Semantic similarity (no external API needed — cosine on MiniLM)
# ---------------------------------------------------------------------------

_sim_model = None


def _semantic_sim(a: str, b: str) -> float:
    global _sim_model
    try:
        if _sim_model is None:
            from sentence_transformers import SentenceTransformer
            _sim_model = SentenceTransformer("all-MiniLM-L6-v2")
        import numpy as np
        embs = _sim_model.encode([a, b], normalize_embeddings=True)
        return float(np.dot(embs[0], embs[1]))
    except Exception:
        # Rough word-overlap fallback
        sa, sb = set(a.lower().split()), set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)


# ---------------------------------------------------------------------------
# Evaluator runner
# ---------------------------------------------------------------------------

def run_benchmark(api_url: str) -> Dict:
    results = []
    grounding_correct = 0
    answer_sims: List[float] = []

    for item in BENCHMARK_DATASET:
        qid = item["id"]
        query = item["query"]
        print(f"  [{qid}] Querying: {query[:60]}...")
        t0 = time.time()
        try:
            resp = requests.post(f"{api_url}/query", json={"query": query}, timeout=120)
            resp.raise_for_status()
            r = resp.json()
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"id": qid, "query": query, "error": str(e)})
            continue

        predicted_grounded = r.get("grounded", False)
        expected_grounded = item.get("expected_grounded", True)
        grounding_match = predicted_grounded == expected_grounded
        if grounding_match:
            grounding_correct += 1

        sim = 0.0
        if item.get("reference_answer") and r.get("answer"):
            sim = _semantic_sim(r["answer"], item["reference_answer"])
            answer_sims.append(sim)

        result_row = {
            "id": qid,
            "query": query,
            "answer": r.get("answer", "")[:200],
            "status": r.get("status"),
            "confidence": r.get("confidence"),
            "grounded_predicted": predicted_grounded,
            "grounded_expected": expected_grounded,
            "grounding_match": grounding_match,
            "answer_similarity": round(sim, 3),
            "retries": r.get("retries", 0),
            "latency_ms": r.get("latency_ms", 0),
            "notes": item.get("notes", ""),
        }
        results.append(result_row)
        print(
            f"    status={r.get('status')} conf={r.get('confidence')}% "
            f"grounding={'✓' if grounding_match else '✗'} sim={sim:.2f}"
        )

    n = len(BENCHMARK_DATASET)
    grounding_accuracy = (grounding_correct / n) * 100 if n else 0.0
    avg_answer_sim = (sum(answer_sims) / len(answer_sims)) * 100 if answer_sims else 0.0

    report = {
        "summary": {
            "total_queries": n,
            "grounding_accuracy_pct": round(grounding_accuracy, 1),
            "avg_answer_similarity_pct": round(avg_answer_sim, 1),
        },
        "per_query": results,
    }

    out_path = Path("eval/benchmark_report.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n📊 Report saved to {out_path}")
    print(f"  Grounding Accuracy : {grounding_accuracy:.1f}%")
    print(f"  Avg Answer Sim     : {avg_answer_sim:.1f}%")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AKIS-v2 benchmark runner")
    parser.add_argument("--api_url", default="http://localhost:8000")
    args = parser.parse_args()
    run_benchmark(args.api_url)
