"""FastAPI application — exposes query, ingest, and evaluation endpoints."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.agents.graph import run_query
from src.api.schemas import (
    EvalResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceDoc,
)
from src.config import get_settings

logging.basicConfig(level=get_settings().log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Financial Research Assistant API starting up…")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Multi-Agent Financial Research Assistant",
    description="Agentic RAG over SEC filings with hallucination verification.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Run the full multi-agent pipeline and return a grounded answer."""
    try:
        state = run_query(req.query)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(exc))

    sources = [
        SourceDoc(
            content=doc.page_content[:500],
            source=doc.metadata.get("source", "unknown"),
        )
        for doc in state.get("retrieved_docs", [])
    ]

    return QueryResponse(
        query=state["query"],
        final_answer=state["final_answer"],
        is_grounded=state["is_grounded"],
        unsupported_claims=state["unsupported_claims"],
        verification_result=state["verification_result"],
        sub_queries=state["sub_queries"],
        sources=sources,
        market_data=state.get("market_data", {}),
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """Download SEC filings for a ticker, chunk them, and add to the vector store."""
    from src.data_ingestion.document_processor import load_documents, split_documents
    from src.data_ingestion.sec_downloader import download_filings
    from src.rag.vector_store import build_vector_store, load_vector_store

    try:
        raw_dir = download_filings(req.ticker, req.filing_types, req.num_filings)
        docs = load_documents(raw_dir)
        if not docs:
            raise HTTPException(status_code=404, detail=f"No documents found for {req.ticker}")
        chunks = split_documents(docs)

        # Try to append to existing store; rebuild if missing
        try:
            store = load_vector_store()
            store.add_documents(chunks)
        except FileNotFoundError:
            store = build_vector_store(chunks)

        return IngestResponse(
            ticker=req.ticker,
            chunks_indexed=len(chunks),
            message=f"Successfully indexed {len(chunks)} chunks for {req.ticker}.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Ingestion error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/evaluate", response_model=EvalResponse)
async def evaluate_pipeline():
    """Run RAGAS evaluation over the built-in eval dataset."""
    from src.evaluation.ragas_eval import hallucination_report, run_evaluation

    try:
        df = run_evaluation(output_path=Path("data/eval_results.csv"))
        report = hallucination_report(df)
        return EvalResponse(**report)
    except Exception as exc:
        logger.exception("Evaluation error")
        raise HTTPException(status_code=500, detail=str(exc))
