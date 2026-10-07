"""Streamlit frontend for the Multi-Agent Financial Research Assistant."""
from __future__ import annotations

import json

import httpx
import plotly.graph_objects as go
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Financial Research Assistant",
    page_icon="📊",
    layout="wide",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    api_url = st.text_input("API URL", value=API_URL)

    st.markdown("---")
    st.subheader("📥 Ingest SEC Filings")
    ticker_input = st.text_input("Ticker", placeholder="AAPL, MSFT, TSLA…").upper()
    filing_types = st.multiselect("Filing types", ["10-K", "10-Q"], default=["10-K", "10-Q"])
    num_filings = st.slider("Number of filings per type", 1, 10, 4)

    if st.button("Ingest", use_container_width=True):
        if not ticker_input:
            st.warning("Enter a ticker first.")
        else:
            with st.spinner(f"Downloading and indexing {ticker_input}…"):
                try:
                    resp = httpx.post(
                        f"{api_url}/ingest",
                        json={"ticker": ticker_input, "filing_types": filing_types, "num_filings": num_filings},
                        timeout=300,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(data["message"])
                    else:
                        st.error(f"Error: {resp.json().get('detail', resp.text)}")
                except Exception as e:
                    st.error(f"Could not reach API: {e}")

    st.markdown("---")
    st.subheader("📊 Run Evaluation")
    if st.button("Evaluate Pipeline", use_container_width=True):
        with st.spinner("Running RAGAS evaluation…"):
            try:
                resp = httpx.post(f"{api_url}/evaluate", timeout=600)
                if resp.status_code == 200:
                    metrics = resp.json()
                    st.json(metrics)
                else:
                    st.error(resp.text)
            except Exception as e:
                st.error(str(e))

# ── Main panel ─────────────────────────────────────────────────────────────────
st.title("📊 Multi-Agent Financial Research Assistant")
st.caption("Planner → Retriever → Analyst → Verifier — powered by LangGraph + RAG")

query = st.text_area(
    "Ask a financial research question",
    placeholder="Compare Apple's revenue growth and key risk factors over 2022–2023.",
    height=100,
)

if st.button("🔍 Research", use_container_width=True, type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Agents at work… (Planner → Retriever → Analyst → Verifier)"):
            try:
                resp = httpx.post(
                    f"{api_url}/query",
                    json={"query": query},
                    timeout=180,
                )
            except Exception as e:
                st.error(f"Could not reach API: {e}")
                st.stop()

        if resp.status_code != 200:
            st.error(f"API error: {resp.json().get('detail', resp.text)}")
            st.stop()

        data = resp.json()

        # ── Verification badge ─────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        grounded = data["is_grounded"]
        col1.metric("Verification", "✅ PASS" if grounded else "❌ FAIL")
        col2.metric("Sub-queries planned", len(data["sub_queries"]))
        col3.metric("Sources retrieved", len(data["sources"]))

        # ── Answer ────────────────────────────────────────────────────────
        st.markdown("### Answer")
        st.markdown(data["final_answer"])

        # ── Unsupported claims warning ─────────────────────────────────────
        if data["unsupported_claims"]:
            with st.expander("⚠️ Unsupported / potentially hallucinated claims", expanded=True):
                for claim in data["unsupported_claims"]:
                    st.markdown(f"- {claim}")

        # ── Planner sub-queries ────────────────────────────────────────────
        with st.expander("🗂️ Planner sub-queries"):
            for i, sq in enumerate(data["sub_queries"], 1):
                st.markdown(f"**{i}.** {sq}")

        # ── Sources ───────────────────────────────────────────────────────
        with st.expander("📄 Retrieved sources"):
            for i, src in enumerate(data["sources"], 1):
                st.markdown(f"**Source {i}** — `{src['source']}`")
                st.text(src["content"][:400] + ("…" if len(src["content"]) > 400 else ""))
                st.markdown("---")

        # ── Market data ───────────────────────────────────────────────────
        if data.get("market_data"):
            with st.expander("📈 Live market data (yfinance)"):
                mkt = data["market_data"]
                tickers = list(mkt.keys())
                last_prices = [mkt[t].get("last_price") or 0 for t in tickers]
                fig = go.Figure(go.Bar(x=tickers, y=last_prices, marker_color="#1f77b4"))
                fig.update_layout(title="Last Price ($)", xaxis_title="Ticker", yaxis_title="Price")
                st.plotly_chart(fig, use_container_width=True)
                st.json(mkt)

        # ── Verifier detail ───────────────────────────────────────────────
        with st.expander("🔍 Verifier explanation"):
            st.markdown(data["verification_result"] or "No detail available.")
