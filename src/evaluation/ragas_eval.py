"""RAGAS-based evaluation: faithfulness, answer relevancy, context precision/recall."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.agents.graph import run_query
from src.evaluation.eval_dataset import EVAL_DATASET

logger = logging.getLogger(__name__)


def _build_ragas_dataset(results: list[dict]) -> Dataset:
    rows = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    for r in results:
        rows["question"].append(r["question"])
        rows["answer"].append(r["answer"])
        rows["contexts"].append(r["contexts"])
        rows["ground_truth"].append(r["ground_truth"])
    return Dataset.from_dict(rows)


def run_evaluation(output_path: Path | None = None) -> pd.DataFrame:
    """Run the full agent pipeline over the eval dataset, then score with RAGAS."""
    results = []
    for item in EVAL_DATASET:
        logger.info("Evaluating: %s", item["question"][:80])
        state = run_query(item["question"])
        results.append({
            "question": item["question"],
            "answer": state["final_answer"],
            "contexts": [d.page_content for d in state.get("retrieved_docs", [])],
            "ground_truth": item["ground_truth"],
            "is_grounded": state.get("is_grounded", False),
            "unsupported_claims": state.get("unsupported_claims", []),
        })

    dataset = _build_ragas_dataset(results)
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    df = scores.to_pandas()

    # Append grounding flags
    df["verifier_grounded"] = [r["is_grounded"] for r in results]
    df["unsupported_claim_count"] = [len(r["unsupported_claims"]) for r in results]

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Evaluation results saved to %s", output_path)

    return df


def hallucination_report(df: pd.DataFrame) -> dict:
    """Summarise key reliability metrics for the README before/after numbers."""
    total = len(df)
    hallucinated = int((~df["verifier_grounded"]).sum())
    return {
        "total_questions": total,
        "hallucinated_answers": hallucinated,
        "hallucination_rate_pct": round(hallucinated / total * 100, 1),
        "avg_faithfulness": round(df["faithfulness"].mean(), 3),
        "avg_answer_relevancy": round(df["answer_relevancy"].mean(), 3),
        "avg_context_precision": round(df["context_precision"].mean(), 3),
        "avg_context_recall": round(df["context_recall"].mean(), 3),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results_path = Path("data/eval_results.csv")
    df = run_evaluation(output_path=results_path)
    report = hallucination_report(df)
    print(json.dumps(report, indent=2))
