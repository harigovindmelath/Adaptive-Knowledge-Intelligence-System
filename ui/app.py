"""
ui/app.py
---------
Streamlit dashboard for AKIS-v2.
Runs against the FastAPI backend at API_URL.

Features:
  - Query input + submit
  - Status badge (SUCCESS / LOW_CONFIDENCE / etc.)
  - Confidence colour gauge
  - Claim-by-claim grounding table
  - Collapsible source chunks
  - Live metrics sidebar
  - PDF upload
"""
from __future__ import annotations

import io
from typing import Any, Dict, List

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="AKIS-v2 Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _api_get(path: str) -> Dict:
    try:
        r = requests.get(f"{API_URL}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _api_post(path: str, payload: Dict) -> Dict:
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _colour(confidence: float) -> str:
    if confidence >= 80:
        return "green"
    elif confidence >= 50:
        return "orange"
    return "red"


def _status_colour(status: str) -> str:
    return {"SUCCESS": "green", "LOW_CONFIDENCE": "orange"}.get(status, "red")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ System")
    health = _api_get("/")
    ready_icon = "✅" if health.get("pipeline_ready") else "❌"
    st.markdown(f"**Pipeline:** {ready_icon} {'Ready' if health.get('pipeline_ready') else 'Not initialised'}")

    st.markdown("---")
    st.subheader("📊 Metrics")
    metrics = _api_get("/metrics")
    if "error" not in metrics:
        st.metric("Total Queries", metrics.get("total_queries", 0))
        st.metric("Avg Latency (ms)", f"{metrics.get('avg_latency_ms', 0):.0f}")
        st.metric("Errors", metrics.get("errors", 0))
    else:
        st.warning("Cannot reach API.")

    st.markdown("---")
    st.subheader("📄 Upload PDF")
    uploaded = st.file_uploader("Upload a PDF to ingest", type=["pdf"])
    if uploaded and st.button("Ingest PDF"):
        with st.spinner("Uploading…"):
            files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
            try:
                resp = requests.post(f"{API_URL}/ingest", files=files, timeout=30)
                st.success(resp.json().get("message", "Ingested."))
            except Exception as e:
                st.error(str(e))

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.title("🔍 AKIS-v2 — Adaptive Knowledge Intelligence System")
st.caption("Hybrid RAG · Self-Healing · Hallucination Detection")

query = st.text_area("Enter your question:", height=80)

col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("Ask", type="primary", use_container_width=True)

if submit and query.strip():
    with st.spinner("Running pipeline…"):
        result = _api_post("/query", {"query": query})

    if "error" in result:
        st.error(f"API Error: {result['error']}")
    else:
        # ── Answer ─────────────────────────────────────────────────────
        status = result.get("status", "?")
        sc = _status_colour(status)
        conf = result.get("confidence", 0.0)
        cc = _colour(conf)

        st.markdown("---")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Status", status)
        col_b.metric("Confidence", f"{conf:.0f}%")
        col_c.metric("Retries", result.get("retries", 0))
        col_d.metric("Latency", f"{result.get('latency_ms', 0):.0f} ms")

        if result.get("rewritten_query"):
            st.info(f"🔄 Query rewritten to: *{result['rewritten_query']}*")

        st.subheader("📝 Answer")
        st.markdown(
            f"<div style='background:#f0f2f6;padding:1em;border-radius:6px;font-size:1.05em'>"
            f"{result.get('answer','')}</div>",
            unsafe_allow_html=True,
        )
        st.caption(result.get("explanation", ""))

        # ── Claims ─────────────────────────────────────────────────────
        claims = result.get("claims", [])
        if claims:
            st.subheader("🔬 Claim Analysis")
            for c in claims:
                icon = "✅" if c.get("supported") else "❌"
                lvl = c.get("support_level", "none")
                conf_c = c.get("confidence", 0.0)
                src = c.get("source_chunk_id") or "—"
                st.markdown(
                    f"{icon} **{c.get('claim','')}**  "
                    f"<span style='color:gray;font-size:0.85em'>"
                    f"[{lvl}] conf={conf_c:.2f} · src={src}</span>",
                    unsafe_allow_html=True,
                )

        # ── Sources ────────────────────────────────────────────────────
        sources = result.get("sources", [])
        if sources:
            st.subheader("📚 Retrieved Sources")
            for s in sources:
                label = (
                    f"Chunk {s.get('chunk_id','?')[:8]}… | "
                    f"{s.get('source_file','?')} | "
                    f"score={s.get('fused_score', 0):.3f}"
                )
                with st.expander(label):
                    st.write(s.get("text", ""))

elif submit:
    st.warning("Please enter a question.")
