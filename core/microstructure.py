"""
Microstructure Engine: Overnight Volume Profile (VAH/VAL) & Rolling VWAP Z-Score
Author: Senior Quantitative Developer
Target: CME E-mini Nasdaq-100 (NQ/MNQ) Futures

Components:
  1. Overnight Session Volume Profile:
     - Scopes 18:00:00 (prior day) to 09:29:59 (current day) EST.
     - Discretizes traded volume into CME tick bins (0.25 index points).
     - Identifies Point of Control (POC), Value Area High (VAH), Value Area Low (VAL).
  2. Rolling Session VWAP & Volatility Z-Score:
     - Volume-weighted mean price and standard deviation from 09:30:00 EST.
     - Normalized Z-Score to filter statistical overextension at breakout.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def calculate_overnight_volume_profile(
    overnight_bars: pd.DataFrame,
    tick_size: float = 0.25,
    value_area_pct: float = 0.70
) -> Dict[str, float]:
    """
    Calculates the Overnight Volume Profile, POC, VAH, and VAL.

    Parameters:
        overnight_bars: 1-minute OHLCV DataFrame from 18:00 (prior) to 09:29 (current).
        tick_size: CME NQ minimum tick size (0.25 pts).
        value_area_pct: Percentage of volume inside Value Area (typically 0.68 - 0.70).

    Returns:
        Dict with 'poc', 'vah', 'val', 'total_volume'.
    """
    if overnight_bars.empty or "Volume" not in overnight_bars.columns:
        return {"poc": np.nan, "vah": np.nan, "val": np.nan, "total_volume": 0}

    total_vol = overnight_bars["Volume"].sum()
    if total_vol <= 0:
        return {"poc": np.nan, "vah": np.nan, "val": np.nan, "total_volume": 0}

    # Use Typical Price (H+L+C)/3 discretized to tick_size for accurate volume distribution
    typical_prices = (overnight_bars["High"] + overnight_bars["Low"] + overnight_bars["Close"]) / 3.0
    binned_prices = np.round(typical_prices / tick_size) * tick_size

    # Aggregate volume by price level
    vol_by_price = overnight_bars["Volume"].groupby(binned_prices).sum()
    sorted_prices = np.sort(vol_by_price.index.values)
    vol_array = np.array([vol_by_price[p] for p in sorted_prices])

    # Point of Control (POC): Price with the single highest volume
    poc_idx = int(np.argmax(vol_array))
    poc_price = float(sorted_prices[poc_idx])

    # Value Area Expansion: Expand from POC up and down until target volume is reached
    target_vol = total_vol * value_area_pct
    current_vol = vol_array[poc_idx]

    up_idx = poc_idx
    down_idx = poc_idx
    n_prices = len(sorted_prices)

    while current_vol < target_vol and (up_idx < n_prices - 1 or down_idx > 0):
        # Look at next two ticks above and below
        vol_up = vol_array[up_idx + 1] if up_idx < n_prices - 1 else 0
        vol_down = vol_array[down_idx - 1] if down_idx > 0 else 0

        if vol_up >= vol_down and up_idx < n_prices - 1:
            up_idx += 1
            current_vol += vol_up
        elif down_idx > 0:
            down_idx -= 1
            current_vol += vol_down
        else:
            up_idx += 1
            current_vol += vol_up

    val_price = float(sorted_prices[down_idx])
    vah_price = float(sorted_prices[up_idx])

    return {
        "poc": poc_price,
        "vah": vah_price,
        "val": val_price,
        "total_volume": int(total_vol)
    }


def compute_session_vwap_and_zscore(
    session_bars: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes cumulative session VWAP, rolling standard deviation, and Z-Score
    for an intraday RTH session starting at 09:30 EST.

    Returns:
        DataFrame with added columns: ['vwap', 'vwap_std', 'vwap_zscore'].
    """
    df = session_bars.copy()
    if df.empty:
        df["vwap"] = np.nan
        df["vwap_std"] = np.nan
        df["vwap_zscore"] = np.nan
        return df

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = np.maximum(df["Volume"].values, 1.0)

    cum_vol = np.cumsum(vol)
    cum_pv = np.cumsum(typical_price.values * vol)
    vwap = cum_pv / cum_vol

    # Volume-weighted variance = sum(V * (P - VWAP)^2) / sum(V)
    # Using parallel running variance: sum(V * P^2)/sum(V) - VWAP^2
    cum_pv2 = np.cumsum((typical_price.values ** 2) * vol)
    variance = np.maximum((cum_pv2 / cum_vol) - (vwap ** 2), 0.0)
    std_dev = np.sqrt(variance)

    # Standard deviation floor of 1.0 point to prevent division by near-zero at session start
    std_dev = np.maximum(std_dev, 1.0)
    zscore = (df["Close"].values - vwap) / std_dev

    df["vwap"] = np.round(vwap, 2)
    df["vwap_std"] = np.round(std_dev, 2)
    df["vwap_zscore"] = np.round(zscore, 2)

    return df
