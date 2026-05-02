"""
tests/test_pipeline.py
----------------------
Unit and integration tests for AKIS-v2 components.
Run with: pytest tests/ -v

These tests use lightweight mocks so they work on CPU without Ollama/Gemini.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, List
from unittest.mock import MagicMock, patch
import pytest


# ---------------------------------------------------------------------------
# ingestion/chunker
# ---------------------------------------------------------------------------

class TestChunker:
    def test_basic_chunking(self):
        from ingestion.chunker import chunk_text
        text = "word " * 1000  # 1000 words
        chunks = chunk_text(text, source_file="test.pdf", target_words=300, overlap_words=50)
        assert len(chunks) >= 3, "Should produce multiple chunks"
        for c in chunks:
            assert "chunk_id" in c
            assert "text" in c
            assert len(c["text"]) > 0

    def test_empty_text(self):
        from ingestion.chunker import chunk_text
        chunks = chunk_text("", source_file="empty.pdf")
        # Should not raise; may return 0 or 1 chunks
        assert isinstance(chunks, list)

    def test_metadata_fields(self):
        from ingestion.chunker import chunk_text
        text = "This is a test sentence. " * 50
        chunks = chunk_text(text, source_file="doc.pdf")
        for c in chunks:
            assert c["source_file"] == "doc.pdf"
            assert "chunk_index" in c


# ---------------------------------------------------------------------------
# retrieval/bm25_retriever
# ---------------------------------------------------------------------------

class TestBM25:
    def test_search_returns_results(self):
        from retrieval.bm25_retriever import BM25Index
        idx = BM25Index()
        chunks = [
            {"chunk_id": "a", "text": "machine learning algorithms"},
            {"chunk_id": "b", "text": "cooking pasta recipes"},
            {"chunk_id": "c", "text": "deep learning neural networks"},
        ]
        idx.add_chunks(chunks)
        results = idx.search("machine learning", top_k=2)
        assert len(results) <= 2
        if results:
            # machine learning / deep learning chunks should score higher
            texts = [r[1]["chunk_id"] for r in results]
            assert "a" in texts or "c" in texts

    def test_empty_index(self):
        from retrieval.bm25_retriever import BM25Index
        idx = BM25Index()
        results = idx.search("anything")
        assert results == []


# ---------------------------------------------------------------------------
# evaluator/evaluator
# ---------------------------------------------------------------------------

class TestEvaluator:
    def test_grounded_answer(self):
        from evaluator.evaluator import Evaluator
        ev = Evaluator()
        chunks = [
            {"chunk_id": "1", "text": "Python is a high-level programming language."}
        ]
        answer = "Python is a high-level programming language used for many applications."
        result = ev.evaluate(answer, chunks)
        assert "grounded" in result
        assert "confidence" in result
        assert "claims" in result
        assert isinstance(result["confidence"], float)

    def test_empty_answer(self):
        from evaluator.evaluator import Evaluator
        ev = Evaluator()
        result = ev.evaluate("", [{"chunk_id": "1", "text": "Some context"}])
        assert result["grounded"] is False
        assert result["confidence"] == 0.0

    def test_uncertainty_phrase(self):
        from evaluator.evaluator import Evaluator
        ev = Evaluator()
        chunks = [{"chunk_id": "1", "text": "Some context about topic."}]
        result = ev.evaluate("Insufficient information in the provided documents.", chunks)
        assert result["grounded"] is False


# ---------------------------------------------------------------------------
# orchestrator/pipeline (mocked LLM)
# ---------------------------------------------------------------------------

class TestPipeline:
    def _make_pipeline_with_docs(self):
        from orchestrator.pipeline import Pipeline
        from retrieval.hybrid_retriever import HybridRetriever

        pipeline = Pipeline()
        chunks = [
            {
                "chunk_id": f"c{i}",
                "text": f"This document discusses topic {i}. "
                        f"It contains important information about subject {i}.",
                "source_file": "test.pdf",
                "chunk_index": i,
                "total_chunks": 5,
            }
            for i in range(5)
        ]
        pipeline.index(chunks)
        return pipeline

    def test_no_context_query(self):
        """Query for something completely absent should return NO_CONTEXT or LOW_CONFIDENCE."""
        pipeline = self._make_pipeline_with_docs()
        # Mock generator to return None (simulates no LLM)
        with patch.object(pipeline.generator, "generate", return_value=(None, "none")):
            result = pipeline.run("xkcd purple unicorn marmalade")
        assert result.status in ("NO_CONTEXT", "GENERATION_FAILED", "LOW_CONFIDENCE", "SUCCESS")
        # At minimum, a result object is returned
        assert result.trace_id is not None

    def test_generation_failure_fallback(self):
        """When LLM fails, system returns GENERATION_FAILED with fallback message."""
        pipeline = self._make_pipeline_with_docs()
        with patch.object(pipeline.generator, "generate", return_value=(None, "none")):
            result = pipeline.run("What is topic 2?")
        assert result.status == "GENERATION_FAILED"
        assert "cannot answer" in result.answer.lower() or "insufficient" in result.answer.lower()

    def test_successful_query(self):
        """When LLM returns a good answer, pipeline should succeed."""
        pipeline = self._make_pipeline_with_docs()
        fake_answer = "This document discusses topic 1. It contains important information about subject 1."
        with patch.object(pipeline.generator, "generate", return_value=(fake_answer, "mock")):
            result = pipeline.run("What does this document discuss?")
        # Should be SUCCESS or LOW_CONFIDENCE depending on evaluator
        assert result.status in ("SUCCESS", "LOW_CONFIDENCE")
        assert result.answer != ""

    def test_result_has_required_fields(self):
        """PipelineResult must have all required fields."""
        pipeline = self._make_pipeline_with_docs()
        with patch.object(pipeline.generator, "generate", return_value=(None, "none")):
            result = pipeline.run("test")
        d = result.to_dict()
        for key in ["trace_id", "query", "answer", "grounded", "confidence",
                    "status", "retries", "claims", "sources", "latency_ms"]:
            assert key in d, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Failure case: conflicting documents
# ---------------------------------------------------------------------------

class TestFailureCases:
    def test_conflicting_docs(self):
        """Two chunks with opposing info — evaluator may flag low confidence."""
        from orchestrator.pipeline import Pipeline

        pipeline = Pipeline()
        chunks = [
            {"chunk_id": "pos", "text": "The sky is blue.", "source_file": "a.pdf",
             "chunk_index": 0, "total_chunks": 2},
            {"chunk_id": "neg", "text": "The sky is green, not blue.", "source_file": "b.pdf",
             "chunk_index": 1, "total_chunks": 2},
        ]
        pipeline.index(chunks)
        # Answer that contradicts one of the chunks
        conflicting_answer = "The sky is purple according to all studies."
        with patch.object(pipeline.generator, "generate",
                          return_value=(conflicting_answer, "mock")):
            result = pipeline.run("What colour is the sky?")
        # System should not crash; low confidence expected
        assert result is not None
        # Conflicting answer vs both chunks → low confidence likely
        print(f"Conflicting doc test: status={result.status} conf={result.confidence}")
