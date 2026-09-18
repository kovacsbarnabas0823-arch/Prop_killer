"""
Core Data Loader: Institutional CME Futures Parquet Ingestion & Vectorized Masking
Author: Senior Quantitative Developer

Features:
  - Task 1: High-performance ingestion of continuous Parquet data using PyArrow
  - Task 2: Vectorized RTH time-filtering mask (09:30 - 11:30 US/Eastern)
  - Timezone normalization, memory profiling, and chronological bar validation
"""

from __future__ import annotations
import logging
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("DataLoader")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_continuous_nq_parquet(
    filepath: str | Path = "data/nq_1m_continuous.parquet",
    filter_rth: bool = True,
    rth_start: str = "09:30",
    rth_end: str = "11:30",
    target_timezone: str = "US/Eastern"
) -> pd.DataFrame:
    """
    Task 1 & Task 2:
    Loads 1-minute CME NQ continuous parquet dataset via PyArrow and applies
    an institutional vectorized time mask to filter for strategy operational hours.

    Parameters:
        filepath: Relative or absolute path to Parquet file.
        filter_rth: If True, filters strictly to [rth_start, rth_end].
        rth_start: Start of ORB formation window (default: "09:30").
        rth_end: Invalidation / hard close cutoff (default: "11:30").
        target_timezone: Expected session timezone (default: "US/Eastern").
    """
    full_path = PROJECT_ROOT / filepath if not Path(filepath).is_absolute() else Path(filepath)
    if not full_path.exists():
        raise FileNotFoundError(
            f"Parquet dataset not found at {full_path}. Ensure databento_pipeline.py has run."
        )

    logger.info(f"Loading Parquet dataset from: {full_path}...")
    # Task 1: High-throughput ingestion with pyarrow
    df: pd.DataFrame = pd.read_parquet(full_path, engine="pyarrow")
    raw_rows = len(df)
    raw_mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    # Validate DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if "ts_event" in df.columns:
            df["ts_event"] = pd.to_datetime(df["ts_event"])
            df.set_index("ts_event", inplace=True)
        elif "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
        else:
            raise ValueError("Parquet file must contain a DatetimeIndex or timestamp column.")

    # Timezone normalization
    if df.index.tz is None:
        logger.info(f"Localizing naive timestamps to UTC and converting to {target_timezone}...")
        df.index = df.index.tz_localize("UTC").tz_convert(target_timezone)
    elif str(df.index.tz) != target_timezone:
        logger.info(f"Converting timezone from {df.index.tz} to {target_timezone}...")
        df.index = df.index.tz_convert(target_timezone)

    # Standardize column naming
    col_map = {
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume",
        "OPEN": "Open", "HIGH": "High", "LOW": "Low", "CLOSE": "Close", "VOLUME": "Volume"
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Dataset is missing required OHLCV columns: {missing}")

    # If Databento parent multi-symbol data is present, resolve to continuous front-month
    if "symbol" in df.columns and df["symbol"].nunique() > 1:
        logger.info("Resolving Databento parent symbols into continuous front-month contract...")
        # 1. Filter out calendar spreads (symbols containing '-')
        outrights_mask = ~df["symbol"].str.contains("-", na=False)
        df_out = df[outrights_mask]

        # 2. Determine the highest-volume active front-month contract per day
        daily_vols = df_out.groupby([df_out.index.date, "symbol"])["Volume"].sum().reset_index()
        active_per_day = daily_vols.sort_values("Volume", ascending=False).groupby("level_0").first()
        active_map = active_per_day["symbol"].to_dict()

        # 3. Filter DataFrame for the daily active front-month contract
        is_front_month = [
            sym == active_map.get(d)
            for d, sym in zip(df_out.index.date, df_out["symbol"])
        ]
        df = df_out[is_front_month].copy()
        logger.info(f"Resolved to continuous front-month series: {len(df):,} bars across {len(active_map)} trading days.")

    # Sort index chronologically and remove any residual duplicate timestamps
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="first")]

    # Task 2: Vectorized RTH time-filtering mask
    if filter_rth:
        logger.info(f"Applying vectorized RTH window mask: [{rth_start} - {rth_end} {target_timezone}]...")
        df_filtered = df.between_time(rth_start, rth_end).copy()
        filtered_rows = len(df_filtered)
        filtered_mem_mb = df_filtered.memory_usage(deep=True).sum() / (1024 * 1024)
        reduction_pct = ((raw_rows - filtered_rows) / raw_rows) * 100.0

        logger.info(
            f"RTH Filter complete: {filtered_rows:,} / {raw_rows:,} rows retained "
            f"({reduction_pct:.1f}% reduction). Memory: {filtered_mem_mb:.2f} MB."
        )
        return df_filtered

    return df


def extract_daily_trading_sessions(
    df: pd.DataFrame,
    rth_start: str = "09:30",
    rth_end: str = "15:45"
) -> list[dict]:
    """
    Task 3: Splits continuous 24h market data into structured daily sessions in O(N) time.
    Uses exchange timestamps (ts_event) to establish the true exchange opening price,
    paired with prior overnight bars for Volume Profile calculation.

    Returns a list of dicts for each RTH date:
      - 'date': Current date (datetime.date)
      - 'overnight_bars': DataFrame from 18:00 prior trading day to 09:29:59
      - 'rth_bars': DataFrame from 09:30:00 to rth_end
      - 'exchange_open_price': First trade Open at or after 09:30:00 EST
      - 'daily_high': Day's high
      - 'daily_low': Day's low
      - 'daily_close': Day's close
      - 'noise': min(|Open - Low|, |Open - High|) for the RTH session
    """
    # Group by calendar date in O(N)
    date_groups = dict(tuple(df.groupby(df.index.date)))
    dates = sorted(list(date_groups.keys()))

    sessions = []

    for i, curr_date in enumerate(dates):
        grp = date_groups[curr_date]
        curr_rth_bars = grp.between_time(rth_start, rth_end)
        if curr_rth_bars.empty:
            continue

        # Extract true exchange opening price at 09:30:00
        first_rth_bar = curr_rth_bars.iloc[0]
        exchange_open = float(first_rth_bar["Open"])

        # Overnight session: prior day 18:00-23:59 + current day 00:00-09:29
        on_parts = []
        if i > 0:
            prior_grp = date_groups[dates[i - 1]]
            on_parts.append(prior_grp.between_time("18:00", "23:59"))
        on_parts.append(grp.between_time("00:00", "09:29"))
        overnight_bars = pd.concat(on_parts) if on_parts else pd.DataFrame()

        # Calculate daily Noise = min(|Open - Low|, |Open - High|)
        rth_high = float(curr_rth_bars["High"].max())
        rth_low = float(curr_rth_bars["Low"].min())
        rth_close = float(curr_rth_bars["Close"].iloc[-1])
        noise = min(abs(exchange_open - rth_low), abs(exchange_open - rth_high))
        # Opening Range Volume (09:30 to 09:59) for 10:00 RVOL arbitration
        or_vol = float(curr_rth_bars.between_time("09:30", "09:59")["Volume"].sum())

        sessions.append({
            "date": curr_date,
            "overnight_bars": overnight_bars,
            "rth_bars": curr_rth_bars,
            "exchange_open_price": exchange_open,
            "daily_high": rth_high,
            "daily_low": rth_low,
            "daily_close": rth_close,
            "noise": noise,
            "or_volume_1000": or_vol
        })

    # Compute rolling Daily 20 EMA strictly lagged by 1 day (zero lookahead bias)
    if sessions:
        daily_closes = pd.Series([s["daily_close"] for s in sessions])
        ema_20_series = daily_closes.ewm(span=20, adjust=False).mean()
        or_vols = [s["or_volume_1000"] for s in sessions]

        for i, s in enumerate(sessions):
            if i > 0:
                s["daily_20_ema"] = float(ema_20_series.iloc[i - 1])
            else:
                s["daily_20_ema"] = float(s["exchange_open_price"])

            # Compute 14-day rolling average OR volume strictly lagged by 1 day
            if i < 14:
                avg_vol = np.mean(or_vols[:max(1, i)]) if i > 0 else 1.0
            else:
                avg_vol = np.mean(or_vols[i - 14:i])
            s["rvol_1000"] = round(s["or_volume_1000"] / max(avg_vol, 1.0), 2)

    logger.info(f"Extracted {len(sessions)} structured daily sessions with paired overnight/RTH windows, lagged Daily 20 EMA, and 10:00 RVOL.")
    return sessions


def load_continuous_es_parquet(
    filepath: str | Path = "data/es_1m_continuous.parquet",
    filter_rth: bool = False,
    rth_start: str = "09:30",
    rth_end: str = "15:45",
    target_timezone: str = "US/Eastern"
) -> pd.DataFrame:
    """
    Task 1 (Agent 2):
    Ingests continuous 1-minute CME E-mini S&P 500 (ES/MES) futures dataset
    using the identical institutional pipeline and timezone standardization.
    """
    return load_continuous_nq_parquet(
        filepath=filepath,
        filter_rth=filter_rth,
        rth_start=rth_start,
        rth_end=rth_end,
        target_timezone=target_timezone
    )


if __name__ == "__main__":
    df = load_continuous_nq_parquet(filter_rth=False)
    sessions = extract_daily_trading_sessions(df)
    print("DataLoader verification:")
    print(f"Total sessions: {len(sessions)}")
    print("Sample session 10:", {k: (len(v) if isinstance(v, pd.DataFrame) else v) for k, v in sessions[10].items()})
