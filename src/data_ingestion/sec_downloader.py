"""Download SEC 10-K / 10-Q filings via sec-edgar-downloader."""
from __future__ import annotations

import logging
from pathlib import Path

from sec_edgar_downloader import Downloader

from src.config import get_settings

logger = logging.getLogger(__name__)


def download_filings(
    ticker: str,
    filing_types: list[str] | None = None,
    num_filings: int = 4,
    output_dir: Path | None = None,
) -> Path:
    """Download filings for *ticker* and return the directory they were saved in."""
    cfg = get_settings()
    filing_types = filing_types or ["10-K", "10-Q"]
    output_dir = output_dir or (Path(cfg.vector_store_path).parent / "raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    dl = Downloader(company_name="FinancialResearchBot", email_address=cfg.sec_user_agent, save_dir=str(output_dir))

    for ftype in filing_types:
        logger.info("Downloading %s %s filings for %s", num_filings, ftype, ticker)
        try:
            dl.get(ftype, ticker, limit=num_filings)
        except Exception as exc:
            logger.warning("Could not download %s for %s: %s", ftype, ticker, exc)

    return output_dir
