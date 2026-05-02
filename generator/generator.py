"""
generator/generator.py
-----------------------
LLM answer generation with:
  - Ollama (local, default)
  - Gemini (API fallback)
  - Query rewriting via the same LLM endpoint (no separate model)

Design decisions:
  - Single class; provider selected from config
  - All LLM calls go through _call_llm() for uniform error handling
  - Context window hard-capped at config max_context_tokens
"""
from __future__ import annotations

import os
import textwrap
from typing import Dict, List, Optional, Tuple

import requests

from config.loader import get as cfg
from config.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_ANSWER_PROMPT = textwrap.dedent("""\
    You are a precise assistant. Answer the question using ONLY the provided context.
    If the answer is not present in the context, respond with exactly:
    "Insufficient information in the provided documents."

    Context:
    {context}

    Question: {query}
    Answer:""")

_REWRITE_PROMPT = textwrap.dedent("""\
    The following search query returned low-quality results.
    Rewrite it to be more specific and informative.
    Return ONLY the rewritten query, nothing else.

    Original query: {query}
    Rewritten query:""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_context(chunks: List[Dict], max_tokens: int) -> str:
    """Concatenate chunk texts up to the token budget (approx words ≈ tokens)."""
    lines: List[str] = []
    word_count = 0
    for c in chunks:
        text = c.get("text", "")
        words = text.split()
        if word_count + len(words) > max_tokens:
            # Include partial chunk if some budget remains
            remaining = max_tokens - word_count
            if remaining > 20:
                lines.append(f"[{c.get('chunk_id', '?')}] " + " ".join(words[:remaining]))
            break
        lines.append(f"[{c.get('chunk_id', '?')}] {text}")
        word_count += len(words)
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str) -> Optional[str]:
    url = cfg("generator", "ollama_url", "http://localhost:11434/api/generate")
    model = cfg("generator", "ollama_model", "mistral")
    try:
        resp = requests.post(
            url,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        # Ollama non-stream: {"response": "..."}
        return resp.json().get("response", "").strip() or None
    except Exception as e:
        log.error("Ollama call failed", extra={"error": str(e)})
        return None


def _call_gemini(prompt: str) -> Optional[str]:
    api_key_env = cfg("generator", "gemini_api_key_env", "GEMINI_API_KEY")
    api_key = os.getenv(api_key_env)
    if not api_key:
        log.warning("GEMINI_API_KEY not set; skipping Gemini.")
        return None
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-pro:generateContent?key={api_key}"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        candidates = resp.json().get("candidates", [])
        if candidates:
            return candidates[0]["content"]["parts"][0]["text"].strip() or None
        return None
    except Exception as e:
        log.error("Gemini call failed", extra={"error": str(e)})
        return None


# ---------------------------------------------------------------------------
# Main Generator class
# ---------------------------------------------------------------------------

class Generator:
    """
    Generates answers from retrieved context.
    Provider priority: ollama → gemini → None (uncertainty response returned).
    """

    def __init__(self) -> None:
        self._provider: str = cfg("generator", "provider", "ollama")
        self._max_context_tokens: int = int(cfg("generator", "max_context_tokens", 2000))

    def _call_llm(self, prompt: str) -> Optional[str]:
        answer = None
        if self._provider in ("ollama", "both"):
            answer = _call_ollama(prompt)
        if answer is None:
            log.info("Falling back to Gemini.")
            answer = _call_gemini(prompt)
        return answer

    def generate(
        self,
        chunks: List[Dict],
        query: str,
    ) -> Tuple[Optional[str], str]:
        """
        Generate an answer for *query* given *chunks*.

        Returns ``(answer, provider_used)`` where *provider_used* ∈
        {"ollama", "gemini", "none"}.
        """
        context = _build_context(chunks, self._max_context_tokens)
        prompt = _ANSWER_PROMPT.format(context=context, query=query)
        log.info("Generating answer", extra={"query": query[:80], "context_words": len(context.split())})
        answer = self._call_llm(prompt)
        if answer:
            return answer, self._provider
        return None, "none"

    def rewrite_query(self, query: str) -> Optional[str]:
        """
        Ask the LLM to rewrite a low-quality query.
        Returns the rewritten query string, or None on failure.
        """
        prompt = _REWRITE_PROMPT.format(query=query)
        log.info("Rewriting query", extra={"original": query[:80]})
        rewritten = self._call_llm(prompt)
        if rewritten:
            # Strip quotes/newlines the model might add
            rewritten = rewritten.strip().strip('"').strip("'").strip()
            log.info("Query rewritten", extra={"rewritten": rewritten[:80]})
        return rewritten
