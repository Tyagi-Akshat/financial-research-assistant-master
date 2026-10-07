"""Sample evaluation dataset — ground-truth QA pairs for RAGAS."""
from __future__ import annotations

# Each entry: question, ground_truth answer, list of relevant source snippets.
# Replace / extend with your actual filings content after ingestion.
EVAL_DATASET = [
    {
        "question": "What was Apple's total net revenue in fiscal year 2023?",
        "ground_truth": "Apple's total net revenue in fiscal year 2023 was $383.3 billion.",
        "relevant_contexts": [
            "Apple Inc. reported total net revenue of $383.3 billion for fiscal year 2023, "
            "a decrease of approximately 3% from $394.3 billion in fiscal year 2022."
        ],
    },
    {
        "question": "What are the primary risk factors Apple disclosed in its 2023 10-K?",
        "ground_truth": (
            "Apple's 2023 10-K highlighted risks including global economic conditions, "
            "supply chain disruptions, intense competition, regulatory changes, and "
            "dependence on third-party manufacturers."
        ),
        "relevant_contexts": [
            "The Company's business can be impacted by political events, trade and other "
            "international disputes, war, terrorism, natural disasters, public health issues, "
            "industrial accidents and other business interruptions."
        ],
    },
    {
        "question": "How did Microsoft's cloud revenue grow between 2022 and 2023?",
        "ground_truth": (
            "Microsoft's Intelligent Cloud segment revenue grew from $75.3 billion in fiscal "
            "2022 to $87.9 billion in fiscal 2023, a year-over-year increase of approximately 17%."
        ),
        "relevant_contexts": [
            "Intelligent Cloud revenue was $87.9 billion for fiscal year 2023, compared to "
            "$75.3 billion for fiscal year 2022."
        ],
    },
    {
        "question": "What is Tesla's gross margin trend from 2021 to 2023?",
        "ground_truth": (
            "Tesla's gross margin peaked at approximately 29% in 2021, declined to around 25% in 2022, "
            "and fell further to about 18% in 2023 due to aggressive price cuts."
        ),
        "relevant_contexts": [
            "Our gross profit in 2023 was $4.0 billion on revenues of $97.7 billion, resulting in a "
            "gross margin of approximately 18.2%, compared to approximately 25.6% in 2022."
        ],
    },
]
