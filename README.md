# Multi-Agent Financial Research Assistant

An agentic RAG system that answers complex questions over SEC financial filings by coordinating
four specialised agents, with a built-in reliability evaluation layer, deployed on AWS.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────┐     decomposes     ┌──────────────────────────┐
│   Planner   │ ─────────────────► │  sub-queries [ 2–4 ]     │
└─────────────┘                    └──────────┬───────────────┘
                                              │
                                              ▼
                                   ┌─────────────────────┐
                                   │  Retriever Agent    │  ◄─ FAISS / Chroma vector store
                                   │  (MMR RAG, k=6)     │     sentence-transformers embeddings
                                   └──────────┬──────────┘
                                              │ retrieved docs
                                              ▼
                                   ┌─────────────────────┐
                                   │  Analyst Agent      │  ◄─ yfinance live market data
                                   │  (synthesis + calc) │
                                   └──────────┬──────────┘
                                              │ draft answer
                                              ▼
                                   ┌─────────────────────┐
                                   │  Verifier / Critic  │  ← THE RELIABILITY STAR
                                   │  (groundedness check│    checks every claim vs sources
                                   │   + hallucination   │    PASS → finalise
                                   │   detection)        │    FAIL → re-route to analyst (×1)
                                   └──────────┬──────────┘
                                              │
                                              ▼
                                        Final Answer
                               (with verification badge + source citations)
```

## Stack

| Layer | Technology |
|---|---|
| Agent orchestration | **LangGraph** (StateGraph with conditional edges) |
| RAG | LangChain + **FAISS** (or Chroma) + `sentence-transformers` |
| LLM | **Ollama / llama3.1** (local, zero cost) · OpenAI · Anthropic |
| Financial data | SEC EDGAR filings (free) · **yfinance** (live prices) |
| Backend | **FastAPI** + Uvicorn |
| Frontend | **Streamlit** + Plotly |
| Evaluation | **RAGAS** (faithfulness, relevancy, precision, recall) + LLM-as-judge |
| Deployment | **Docker** · **AWS** (EC2, ECR, S3) · **Terraform** IaC |
| CI/CD | **GitHub Actions** (test → build → push ECR → deploy via SSM) |

---

## Quick Start (local)

### 1. Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai) installed and running: `ollama pull llama3.1`

### 2. Install

```bash
git clone Tyagi-Akshat/financial-research-assistant-master
cd financial-research-assistant
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
cp .env.example .env          # edit if needed
```

### 3. Ingest SEC filings

```bash
python scripts/ingest.py --tickers AAPL MSFT TSLA --filings 10-K 10-Q --num 4
```

This downloads filings from SEC EDGAR, chunks them, and builds the FAISS vector store under `data/vectorstore/`.

### 4. Start the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

### 5. Start the frontend

```bash
streamlit run frontend/app.py
```

Open http://localhost:8501 — ask a question, watch the four agents collaborate.

---

## Docker Compose (full stack)

```bash
cp .env.example .env
docker-compose up --build
```

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |
| Ollama | http://localhost:11434 |

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/query` | POST | Run the full agent pipeline |
| `/ingest` | POST | Download + index filings for a ticker |
| `/evaluate` | POST | Run RAGAS evaluation harness |

**Example query:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare Apple and Microsoft revenue growth from 2022 to 2023 and highlight key risk factors."}'
```

---

## Evaluation

Run the RAGAS harness over the built-in QA dataset:

```bash
python -m src.evaluation.ragas_eval
```

Or via the API:

```bash
curl -X POST http://localhost:8000/evaluate
```

### Metrics measured

| Metric | What it measures |
|---|---|
| **Faithfulness** | Are all answer claims entailed by the retrieved context? |
| **Answer Relevancy** | Is the answer relevant to the question? |
| **Context Precision** | Are the retrieved chunks actually useful? |
| **Context Recall** | Does the retrieved set cover the ground-truth answer? |
| **Verifier Grounding Rate** | % of answers passing the in-pipeline verifier agent |
| **Hallucination Rate** | % of answers containing unverifiable claims |

Results are saved to `data/eval_results.csv`. Fill in your actual before/after numbers:

| | Without Verifier | With Verifier |
|---|---|---|
| Hallucination rate | [X]% | [Y]% |
| Avg faithfulness | [A] | [B] |

---

## AWS Deployment

### Prerequisites
- AWS CLI configured
- Terraform ≥ 1.5
- An S3 bucket for Terraform state (update `terraform/main.tf`)

```bash
cd terraform
terraform init
terraform apply
```

Terraform provisions: VPC, EC2 (t3.medium), ECR repositories, S3 bucket for artefacts, IAM roles.

### CI/CD Secrets required (GitHub → Settings → Secrets)

| Secret | Value |
|---|---|
| `AWS_DEPLOY_ROLE_ARN` | IAM role ARN for GitHub OIDC |
| `EC2_INSTANCE_ID` | e.g. `i-0abc123` |
| `ECR_REGISTRY` | `<account>.dkr.ecr.us-east-1.amazonaws.com` |

Push to `main` → tests run → images built & pushed to ECR → deployed to EC2 via SSM.

---

## Project Structure

```
financial-research-assistant/
├── src/
│   ├── agents/
│   │   ├── state.py          # LangGraph AgentState schema
│   │   ├── planner.py        # Decomposes query → sub-queries
│   │   ├── retriever_agent.py# RAG retrieval over vector store
│   │   ├── analyst.py        # Synthesises answer + yfinance data
│   │   ├── verifier.py       # Groundedness check + finaliser
│   │   └── graph.py          # LangGraph workflow + retry logic
│   ├── rag/
│   │   ├── embeddings.py     # sentence-transformers (local)
│   │   ├── vector_store.py   # FAISS / Chroma build + load
│   │   └── retriever.py      # MMR retrieval + formatting
│   ├── data_ingestion/
│   │   ├── sec_downloader.py # SEC EDGAR downloader
│   │   └── document_processor.py # Load + chunk documents
│   ├── evaluation/
│   │   ├── eval_dataset.py   # Ground-truth QA pairs
│   │   ├── ragas_eval.py     # RAGAS pipeline evaluation
│   │   └── llm_judge.py      # LLM-as-judge fallback
│   ├── api/
│   │   ├── main.py           # FastAPI app
│   │   └── schemas.py        # Pydantic request/response models
│   ├── config.py             # Settings (pydantic, dotenv)
│   └── llm_factory.py        # Provider-agnostic LLM factory
├── frontend/
│   └── app.py                # Streamlit UI
├── scripts/
│   └── ingest.py             # CLI ingestion script
├── tests/
│   ├── test_agents.py        # Agent unit tests (mocked LLM)
│   └── test_api.py           # API integration tests
├── terraform/
│   └── main.tf               # AWS infrastructure (EC2, ECR, S3)
├── .github/workflows/
│   └── ci-cd.yml             # Test → Build → Deploy pipeline
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.frontend
└── requirements.txt
```
