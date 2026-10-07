#!/usr/bin/env python
"""CLI: download SEC filings for one or more tickers and build the vector store.

Usage:
    python scripts/ingest.py --tickers AAPL MSFT TSLA --filings 10-K 10-Q --num 4
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_ingestion.document_processor import load_documents, split_documents
from src.data_ingestion.sec_downloader import download_filings
from src.rag.vector_store import build_vector_store, load_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest SEC filings into the vector store")
    parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols")
    parser.add_argument("--filings", nargs="+", default=["10-K", "10-Q"])
    parser.add_argument("--num", type=int, default=4, help="Filings per type per ticker")
    args = parser.parse_args()

    all_chunks = []
    for ticker in args.tickers:
        logger.info("=== Processing %s ===", ticker)
        raw_dir = download_filings(ticker, args.filings, args.num)
        docs = load_documents(raw_dir)
        if not docs:
            logger.warning("No documents found for %s — skipping", ticker)
            continue
        chunks = split_documents(docs)
        all_chunks.extend(chunks)
        logger.info("  %d chunks from %s", len(chunks), ticker)

    if not all_chunks:
        logger.error("No chunks to index. Exiting.")
        sys.exit(1)

    logger.info("Building vector store with %d total chunks…", len(all_chunks))
    try:
        store = load_vector_store()
        store.add_documents(all_chunks)
        logger.info("Appended to existing vector store.")
    except FileNotFoundError:
        build_vector_store(all_chunks)
        logger.info("New vector store created.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
