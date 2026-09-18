"""
Institutional Historical Market Data Ingestion Pipeline: CME E-mini Nasdaq-100 (NQ)
Author: Senior Data Engineer
Target: Continuous 1-Minute OHLCV for Strategy Backtesting & Pessimistic Rule Validation

Specifications:
  - Provider: Databento Historical API
  - Dataset: GLBX.MDP3 (CME Globex MDP 3.0)
  - Schema: ohlcv-1m (1-minute bars)
  - Stype In: parent (Continuous rolled front-month contract)
  - Symbol: NQ
  - Timezone: US/Eastern (converted from UTC)
  - Storage: Parquet with Snappy compression (data/nq_1m_continuous.parquet)
"""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import sys

# Support UTF-8 encoding on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("DatabentoPipeline")

# Locate project root and load environment variables from .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)


def get_api_key() -> str:
    """Securely fetches Databento API key from environment."""
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key or api_key.strip() == "":
        raise ValueError(
            f"DATABENTO_API_KEY is not set or empty. Please set it in {ENV_PATH}"
        )
    # Mask API key for secure logging
    masked = api_key[:5] + "..." + api_key[-4:] if len(api_key) > 9 else "***"
    logger.info(f"Loaded Databento API Key: {masked}")
    return api_key.strip()


def fetch_and_save_continuous(
    symbol: str = "NQ.FUT",
    start_date: str = "2018-01-01",
    end_date: str | None = None,
    output_filename: str | None = None,
    estimate_cost_only: bool = False
) -> Path:
    """
    Connects to Databento Historical API, requests continuous CME 1m OHLCV data (NQ or ES),
    normalizes the timezone to US/Eastern, and saves as compressed Parquet.
    """
    try:
        # pyrefly: ignore [missing-import]
        import databento as db
    except ImportError as e:
        logger.critical(
            "databento package is not installed in the current environment.\n"
            "Run: pip install databento"
        )
        raise e

    api_key = get_api_key()
    client = db.Historical(key=api_key)

    if end_date is None:
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if output_filename is None:
        prefix = "es" if "ES" in symbol.upper() else "nq"
        output_filename = f"{prefix}_1m_continuous.parquet"

    dataset = "GLBX.MDP3"
    schema = "ohlcv-1m"
    stype_in = "parent"

    logger.info("=" * 80)
    logger.info("DATABENTO HISTORICAL DATA INGESTION REQUEST")
    logger.info(f"Dataset      : {dataset}")
    logger.info(f"Schema       : {schema}")
    logger.info(f"Symbology In : {stype_in} (Continuous contract)")
    logger.info(f"Symbol       : {symbol}")
    logger.info(f"Date Range   : {start_date} -> {end_date}")
    logger.info("=" * 80)

    # Pre-flight: Estimate query cost and billable size
    try:
        cost_usd = client.metadata.get_cost(
            dataset=dataset,
            symbols=symbol,
            schema=schema,
            start=start_date,
            end=end_date,
            stype_in=stype_in
        )
        logger.info(f"Estimated Databento Query Cost: ${cost_usd:.4f} USD")
    except Exception as e:
        logger.warning(f"Could not retrieve cost estimate from metadata API: {e}")

    try:
        billable_bytes = client.metadata.get_billable_size(
            dataset=dataset,
            symbols=symbol,
            schema=schema,
            start=start_date,
            end=end_date,
            stype_in=stype_in
        )
        billable_mb = billable_bytes / (1024 * 1024)
        logger.info(f"Estimated Compressed Payload: {billable_mb:.2f} MB ({billable_bytes:,} bytes)")
    except Exception as e:
        logger.warning(f"Could not retrieve billable size from metadata API: {e}")

    if estimate_cost_only:
        logger.info("[ESTIMATE-ONLY] Exiting without downloading payload.")
        return Path()

    # Step 3: Stream and download historical range
    logger.info("Downloading historical OHLCV-1m bars via Databento API...")
    data_bento = client.timeseries.get_range(
        dataset=dataset,
        symbols=symbol,
        schema=schema,
        start=start_date,
        end=end_date,
        stype_in=stype_in
    )

    # Step 4: Convert to Pandas DataFrame & Timezone conversion
    logger.info("Converting raw DBN stream to Pandas DataFrame...")
    df: pd.DataFrame = data_bento.to_df()
    logger.info(f"Raw DataFrame retrieved with {len(df):,} rows.")

    # Databento ohlcv indices are UTC timestamps
    logger.info("Converting timestamps to 'US/Eastern' (CME Market Session Time)...")
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
        else:
            df.index = df.index.tz_convert("US/Eastern")
    elif "ts_event" in df.columns:
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True).dt.tz_convert("US/Eastern")
        df.set_index("ts_event", inplace=True)
    else:
        logger.warning("No DatetimeIndex or 'ts_event' column identified. Timestamp index preserved as-is.")

    # Column standardization (ensure standard OHLCV capitalization for backtester parity)
    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Sort index chronologically and remove duplicate bars if any
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="first")]

    # Ensure output directory exists
    output_dir = PROJECT_ROOT / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename

    # Save as compressed Parquet
    logger.info(f"Saving {len(df):,} bars to compressed Parquet: {output_path} (snappy)...")
    df.to_parquet(output_path, engine="pyarrow", compression="snappy")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        f"[SUCCESS] Dataset saved to {output_path} ({file_size_mb:.2f} MB on disk)."
    )
    logger.info(f"Data Range: {df.index.min()} to {df.index.max()}")
    logger.info(f"Available Columns: {list(df.columns)}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Institutional Databento Historical Data Ingestion Pipeline for CME Futures (NQ / ES)"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="NQ.FUT",
        help="CME Globex parent contract symbol: NQ.FUT or ES.FUT (default: NQ.FUT)"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2018-01-01",
        help="Start date in YYYY-MM-DD format (default: 2018-01-01 for 8+ years)"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date in YYYY-MM-DD format (default: current UTC date)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Target parquet filename inside data/ (default: nq_1m_continuous.parquet or es_1m_continuous.parquet)"
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Estimate query cost and payload size without downloading"
    )

    args = parser.parse_args()

    try:
        fetch_and_save_continuous(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            output_filename=args.output,
            estimate_cost_only=args.estimate
        )
    except Exception as e:
        logger.error(f"Pipeline execution aborted: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
