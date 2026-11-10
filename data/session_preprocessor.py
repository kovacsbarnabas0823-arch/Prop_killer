"""
Timezone-Aware Session Preprocessor & Continuous NQ Ingestion
Author: Principal Quantitative Architect & Systems Engineer

Handles:
  - High-throughput PyArrow ingestion of CME NQ/MNQ 1-minute continuous data.
  - Multi-symbol parent resolution with automatic fast-cache for sub-second reloading.
  - DST-safe dual timezone conversion (America/New_York and Europe/London).
  - Causal 14-day Daily ATR calculation (shifted by 1 session, zero lookahead).
  - Session slicing: London Session (08:00 LON - 09:30 NY) and NY RTH (09:30 - 16:00 NY).
  - Vectorized Anchored VWAP and Intraday Rolling Features.
"""

from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import pytz

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("SessionPreprocessor")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SessionDataPreprocessor:
    def __init__(
        self,
        raw_filepath: str | Path = "data/nq_1m_continuous.parquet",
        cache_filepath: str | Path = "data/nq_1m_resolved_cache.parquet"
    ):
        self.raw_path = PROJECT_ROOT / raw_filepath if not Path(raw_filepath).is_absolute() else Path(raw_filepath)
        self.cache_path = PROJECT_ROOT / cache_filepath if not Path(cache_filepath).is_absolute() else Path(cache_filepath)
        self.tz_ny = pytz.timezone("America/New_York")
        self.tz_lon = pytz.timezone("Europe/London")
        self.df: Optional[pd.DataFrame] = None
        self.daily_atr_map: Dict[Any, float] = {}
        self.prev_day_close_map: Dict[Any, float] = {}
        self.prev_day_high_map: Dict[Any, float] = {}
        self.prev_day_low_map: Dict[Any, float] = {}
        self.prev_day_range_map: Dict[Any, float] = {}
        self.prev2_day_high_map: Dict[Any, float] = {}
        self.prev2_day_low_map: Dict[Any, float] = {}
        self.noise_band_map: Dict[Any, Dict[str, Tuple[float, float]]] = {}
        self.daily_dict: Dict[Any, pd.DataFrame] = {}

    def load_data(self, force_recompute: bool = False) -> pd.DataFrame:
        """
        Loads continuous 1m front-month NQ data from cache or resolves raw Databento parquet.
        """
        if not force_recompute and self.cache_path.exists():
            logger.info(f"Loading cached continuous NQ dataset from: {self.cache_path}...")
            df = pd.read_parquet(self.cache_path, engine="pyarrow")
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC").tz_convert(self.tz_ny)
            elif str(df.index.tz) != "America/New_York":
                df.index = df.index.tz_convert(self.tz_ny)
            self.df = df
            self._compute_daily_features()
            logger.info(f"Loaded {len(df):,} bars from cache successfully.")
            return self.df

        logger.info(f"Reading raw dataset from {self.raw_path}...")
        df_raw = pd.read_parquet(self.raw_path, engine="pyarrow")

        # Standardize DatetimeIndex
        if not isinstance(df_raw.index, pd.DatetimeIndex):
            if "ts_event" in df_raw.columns:
                df_raw["ts_event"] = pd.to_datetime(df_raw["ts_event"])
                df_raw.set_index("ts_event", inplace=True)
            elif "timestamp" in df_raw.columns:
                df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
                df_raw.set_index("timestamp", inplace=True)

        if df_raw.index.tz is None:
            df_raw.index = df_raw.index.tz_localize("UTC").tz_convert(self.tz_ny)
        else:
            df_raw.index = df_raw.index.tz_convert(self.tz_ny)

        # Standardize column naming
        col_map = {
            "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume",
            "OPEN": "Open", "HIGH": "High", "LOW": "Low", "CLOSE": "Close", "VOLUME": "Volume"
        }
        df_raw.rename(columns={k: v for k, v in col_map.items() if k in df_raw.columns}, inplace=True)

        # Resolve Databento parent multi-symbol data into continuous front-month contract
        if "symbol" in df_raw.columns and df_raw["symbol"].nunique() > 1:
            logger.info("Resolving Databento outright contracts by daily maximum volume...")
            outrights_mask = ~df_raw["symbol"].str.contains("-", na=False)
            df_out = df_raw[outrights_mask]

            daily_vols = df_out.groupby([df_out.index.date, "symbol"])["Volume"].sum().reset_index()
            active_per_day = daily_vols.sort_values("Volume", ascending=False).groupby("level_0").first()
            active_map = active_per_day["symbol"].to_dict()

            is_front_month = [
                sym == active_map.get(d)
                for d, sym in zip(df_out.index.date, df_out["symbol"])
            ]
            df = df_out[is_front_month].copy()
        else:
            df = df_raw.copy()

        # Sort and deduplicate
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="first")]

        # Retain necessary OHLCV columns
        cols_to_keep = ["Open", "High", "Low", "Close", "Volume"]
        df = df[cols_to_keep]

        # Cache to disk for instant subsequent loads
        logger.info(f"Saving resolved continuous dataset to cache at {self.cache_path}...")
        df.to_parquet(self.cache_path, engine="pyarrow", compression="snappy")

        self.df = df
        self._compute_daily_features()
        logger.info(f"Dataset preparation complete: {len(df):,} bars spanning {len(self.daily_atr_map)} days.")
        return self.df

    def _compute_daily_features(self):
        """
        Computes causal Daily ATR_14 and previous day's 16:00 close.
        Strictly shifted by 1 day to ensure 0 lookahead bias.
        """
        df = self.df
        # Slicing RTH sessions (09:30 to 16:00 NY)
        rth = df.between_time("09:30", "16:00")
        daily_bars = rth.groupby(rth.index.date).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last"
        })

        # Calculate True Range
        prev_close = daily_bars["Close"].shift(1)
        tr1 = daily_bars["High"] - daily_bars["Low"]
        tr2 = (daily_bars["High"] - prev_close).abs()
        tr3 = (daily_bars["Low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 14-day Wilder / SMA ATR, shifted by 1 day so today's session only uses prior 14 days
        daily_atr = tr.rolling(window=14).mean().shift(1)

        prev_high = daily_bars["High"].shift(1)
        prev_low = daily_bars["Low"].shift(1)
        prev_range = (daily_bars["High"] - daily_bars["Low"]).shift(1)
        prev2_high = daily_bars["High"].shift(2)
        prev2_low = daily_bars["Low"].shift(2)

        self.daily_atr_map = daily_atr.to_dict()
        self.prev_day_close_map = prev_close.to_dict()
        self.prev_day_high_map = prev_high.to_dict()
        self.prev_day_low_map = prev_low.to_dict()
        self.prev_day_range_map = prev_range.to_dict()
        self.prev2_day_high_map = prev2_high.to_dict()
        self.prev2_day_low_map = prev2_low.to_dict()

        logger.info("Pre-grouping sessions into daily lookup table...")
        self.daily_dict = {d: group for d, group in df.groupby(df.index.date)}

        logger.info("Precomputing Zarattini-Baltussen time-of-day noise envelopes (Multi-Horizon 10:00-14:00)...")
        checkpoints = [
            "10:00", "10:15", "10:30", "10:45", "11:00", "11:15", "11:30", "11:45",
            "12:00", "12:15", "12:30", "12:45", "13:00", "13:15", "13:30", "13:45", "14:00"
        ]
        dates_list = sorted(list(self.daily_dict.keys()))

        # For each checkpoint, extract Close across all days
        rth_1m = df.between_time("09:30", "14:00")
        opens_0930 = rth_1m.between_time("09:30", "09:30").groupby(rth_1m.between_time("09:30", "09:30").index.date)["Open"].first()

        moves_by_cp = {}
        for cp in checkpoints:
            closes_cp = rth_1m.between_time(cp, cp).groupby(rth_1m.between_time(cp, cp).index.date)["Close"].last()
            df_cp = pd.DataFrame({"Open_0930": opens_0930, "Close_cp": closes_cp}).dropna()
            df_cp["move"] = (df_cp["Close_cp"] - df_cp["Open_0930"]).abs()
            # 14-day rolling mean shifted by 1 day
            sigma_cp = df_cp["move"].rolling(window=14).mean().shift(1)
            moves_by_cp[cp] = sigma_cp.to_dict()

        # Build noise_band_map with Regime-Adaptive Multiplier (Kd)
        for d in dates_list:
            if d not in self.prev_day_close_map or pd.isna(self.prev_day_close_map[d]):
                continue
            p_close = self.prev_day_close_map[d]
            day_df = self.daily_dict.get(d)
            if day_df is None or day_df.empty:
                continue
            rth_sub = day_df.between_time("09:30", "09:30")
            if rth_sub.empty:
                continue
            d_open = float(rth_sub.iloc[0]["Open"])
            ref_high = max(d_open, p_close)
            ref_low = min(d_open, p_close)

            # Compute prior day's RTH range ratio (comp_ratio = PriorDay_RTH_Range / ATR_14)
            prev_rng = self.prev_day_range_map.get(d)
            atr_val = self.daily_atr_map.get(d)
            if prev_rng is not None and not pd.isna(prev_rng) and atr_val is not None and not pd.isna(atr_val) and atr_val > 0:
                comp_ratio = prev_rng / atr_val
            else:
                comp_ratio = 1.0

            if comp_ratio <= 0.85:
                kd = 1.12
            elif comp_ratio <= 1.20:
                kd = 1.25
            else:
                kd = 1.35

            self.noise_band_map[d] = {}
            for cp in checkpoints:
                sig = moves_by_cp[cp].get(d)
                if sig is not None and not pd.isna(sig) and sig > 0:
                    upper = ref_high + (kd * sig)
                    lower = ref_low - (kd * sig)
                    self.noise_band_map[d][cp] = (upper, lower)

    def get_session_bars(self, session_date) -> pd.DataFrame:
        """Returns all 1-minute bars for a specific date in America/New_York timezone via instant O(1) hash lookup."""
        return self.daily_dict.get(session_date, pd.DataFrame())

    def get_noise_band(self, session_date, time_str: str) -> Optional[Tuple[float, float]]:
        """Returns (UpperBand, LowerBand) for the given date and active time checkpoint."""
        day_bands = self.noise_band_map.get(session_date)
        if not day_bands:
            return None
        # Find latest checkpoint <= time_str
        valid_cps = [cp for cp in day_bands.keys() if cp <= time_str]
        if not valid_cps:
            return None
        latest_cp = max(valid_cps)
        return day_bands[latest_cp]

    def get_london_range(self, session_date, df_day_full: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Computes London session High, Low, Range:
        Strictly defined from 08:00:00 Europe/London to 09:29:59 America/New_York.
        Handles DST differences seamlessly.
        """
        # Convert the full day's data (including overnight) to Europe/London
        df_lon = df_day_full.tz_convert(self.tz_lon)
        # Slicing 08:00 London to 09:30 NY (14:30 London typically, but exact by converting 09:30 NY to London time)
        # 08:00 Europe/London start
        # End is the bar immediately before 09:30:00 America/New_York
        # Let's filter by exact timestamp bounds
        ny_start = self.tz_ny.localize(pd.Timestamp(f"{session_date} 09:30:00"))
        lon_start = pd.Timestamp(f"{session_date} 08:00:00", tz=self.tz_lon)

        # Ensure lon_start is strictly before ny_start
        lon_bars = df_day_full[(df_day_full.index >= lon_start) & (df_day_full.index < ny_start)]
        if len(lon_bars) < 30:
            return None, None, None

        lon_h = float(lon_bars["High"].max())
        lon_l = float(lon_bars["Low"].min())
        lon_w = lon_h - lon_l
        return lon_h, lon_l, lon_w


# Global helper instance
_preprocessor = None

def get_preprocessor() -> SessionDataPreprocessor:
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = SessionDataPreprocessor()
        _preprocessor.load_data()
    return _preprocessor
