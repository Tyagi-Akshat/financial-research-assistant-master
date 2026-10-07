from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=2000, description="Financial research question")


class SourceDoc(BaseModel):
    content: str
    source: str


class QueryResponse(BaseModel):
    query: str
    final_answer: str
    is_grounded: bool
    unsupported_claims: list[str]
    verification_result: str
    sub_queries: list[str]
    sources: list[SourceDoc]
    market_data: dict


class IngestRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol, e.g. AAPL")
    filing_types: list[str] = Field(default=["10-K", "10-Q"])
    num_filings: int = Field(default=4, ge=1, le=20)


class IngestResponse(BaseModel):
    ticker: str
    chunks_indexed: int
    message: str


class EvalResponse(BaseModel):
    total_questions: int
    hallucination_rate_pct: float
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_context_recall: float
