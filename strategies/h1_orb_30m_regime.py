"""
Module 1: Upgraded 30m Filtered ORB (Phase 2 High-Conviction Asymmetric 1:2.5R)
Author: Principal Quantitative Architect

Strategy Rules:
  - Tightened Opening Range Runway: 0.20 * ATR_14 <= OR_W <= 0.55 * ATR_14.
  - Macro/Volatility Compression & PDH/PDL Structural Filter:
      * Condition A (Prior-Day Compression): PriorDay_Range <= 0.85 * ATR_14 OR Inside Day.
      * Condition B (Clean Structural Space): Long requires (Close_5m > PDH OR PDH - Close_5m >= 2.0 * OR_W).
        Short requires (Close_5m < PDL OR Close_5m - PDL >= 2.0 * OR_W).
  - Prime Institutional Execution Window: 10:00:00 to 11:15:00 NY.
      * Full 5m candle body close outside OR_H/OR_L.
      * Aligned with Anchored RTH VWAP.
      * Volume_5m >= 1.20 * SMA(Volume_5m, 6).
  - Asymmetric 100% Position Geometry (No Premature 1R Scale-out!):
      * SL_dist = min(55.0, max(0.50 * OR_W, 0.22 * ATR_14)).
      * scale_out_tp1 = False.
      * Delayed Breakeven (+1.5R): Move stop to Entry +/- 2.0 pts only after +1.5R.
      * Full Target: +2.5R (Entry +/- 2.5 * SL_dist) or 15:44:00 NY EOD flatten.
"""

from __future__ import annotations
import logging
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, TradeSignal

logger = logging.getLogger("H1_ORB_30M")


class H1ORB30mRegime(BaseStrategy):
    def __init__(
        self,
        min_atr_mult: float = 0.20,
        max_atr_mult: float = 0.55,          # Tightened from 0.75 to 0.55
        volume_factor: float = 1.20,         # Upgraded to 1.20x
        volume_sma_bars: int = 6,
        target_r_multiple: float = 2.5,      # Asymmetric 2.5R target
        be_trigger_r: float = 1.5,           # Delayed breakeven trigger
        max_sl_pts: float = 55.0
    ):
        super().__init__(name="H1_Filtered_30M_ORB_v2", strategy_id="H1")
        self.min_atr_mult = min_atr_mult
        self.max_atr_mult = max_atr_mult
        self.volume_factor = volume_factor
        self.volume_sma_bars = volume_sma_bars
        self.target_r_multiple = target_r_multiple
        self.be_trigger_r = be_trigger_r
        self.max_sl_pts = max_sl_pts

    def evaluate_session(
        self,
        session_date,
        df_session_1m: pd.DataFrame,
        daily_atr: float,
        prev_close: Optional[float] = None,
        prev_high: Optional[float] = None,
        prev_low: Optional[float] = None,
        prev_range: Optional[float] = None,
        prev2_high: Optional[float] = None,
        prev2_low: Optional[float] = None,
        **kwargs
    ) -> List[TradeSignal]:
        if daily_atr is None or np.isnan(daily_atr) or daily_atr <= 0:
            return []

        # 1. Opening Range: 09:30:00 to 09:59:59 NY
        or_bars = df_session_1m.between_time("09:30", "09:59")
        if len(or_bars) < 25:
            return []

        or_h = float(or_bars["High"].max())
        or_l = float(or_bars["Low"].min())
        or_w = or_h - or_l

        # Tightened Opening Range Runway: 0.20 * ATR_14 <= OR_W <= 0.55 * ATR_14
        if not (self.min_atr_mult * daily_atr <= or_w <= self.max_atr_mult * daily_atr):
            return []

        # 2. Check Macro/Volatility Compression Filter
        is_prior_day_compressed = False
        if prev_range is not None and not np.isnan(prev_range) and prev_range <= 0.85 * daily_atr:
            is_prior_day_compressed = True
        elif (prev_high is not None and prev_low is not None and prev2_high is not None and prev2_low is not None):
            if prev_high < prev2_high and prev_low > prev2_low:
                is_prior_day_compressed = True  # Inside Day

        # Anchored RTH VWAP
        rth_bars = df_session_1m.between_time("09:30", "15:44")
        if rth_bars.empty:
            return []

        tp = (rth_bars["High"] + rth_bars["Low"] + rth_bars["Close"]) / 3.0
        cum_vol = rth_bars["Volume"].cumsum()
        cum_tp_vol = (tp * rth_bars["Volume"]).cumsum()
        vwap_series = cum_tp_vol / cum_vol.replace(0, np.nan)

        # 3. Resample 5m bars for entry evaluation
        resampled_5m = rth_bars.resample("5min", closed="left", label="left").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }).dropna()

        vol_sma = resampled_5m["Volume"].rolling(window=self.volume_sma_bars).mean()

        # Prime Institutional Window: 10:00:00 to 11:15:00 NY
        trade_window_5m = resampled_5m.between_time("10:00", "11:15")
        if trade_window_5m.empty:
            return []

        # Structural Stop Distance: max(0.50 * OR_W, 0.22 * ATR_14) capped at 55.0 pts
        sl_dist = min(self.max_sl_pts, max(0.50 * or_w, 0.22 * daily_atr))

        for bar_start, bar_5m in trade_window_5m.iterrows():
            bar_end_time = bar_start + pd.Timedelta(minutes=5)
            exec_time = bar_end_time

            if bar_start not in vol_sma.index or pd.isna(vol_sma.loc[bar_start]):
                continue

            current_vol_sma = vol_sma.loc[bar_start]
            if bar_5m["Volume"] < self.volume_factor * current_vol_sma:
                continue

            prior_1m_mask = vwap_series.index < exec_time
            if not prior_1m_mask.any():
                continue
            current_vwap = float(vwap_series.loc[prior_1m_mask].iloc[-1])

            close_5m = float(bar_5m["Close"])

            # Check Long Breakout
            if close_5m > or_h and close_5m > current_vwap:
                # Structural Space Check vs PDH
                clean_space = True
                if prev_high is not None and not np.isnan(prev_high):
                    if close_5m <= prev_high and (prev_high - close_5m) < (2.0 * or_w):
                        clean_space = False

                if not (is_prior_day_compressed or clean_space):
                    continue

                future_bars = df_session_1m.loc[df_session_1m.index >= exec_time]
                if future_bars.empty:
                    continue
                open_1m = float(future_bars.iloc[0]["Open"])
                entry_price = open_1m + 0.50
                tp_target = entry_price + (self.target_r_multiple * sl_dist)

                return [TradeSignal(
                    strategy_id="H1",
                    strategy_name="H1_Filtered_30M_ORB_v2",
                    timestamp=bar_end_time,
                    execution_time=exec_time,
                    direction=1,
                    entry_price=entry_price,
                    stop_loss_price=entry_price - sl_dist,
                    tp1_price=tp_target,
                    sl_dist=sl_dist,
                    scale_out_tp1=False,
                    be_trigger_r=self.be_trigger_r,
                    hard_time_exit="15:44:00",
                    metadata={"or_w": or_w, "daily_atr": daily_atr, "compression": is_prior_day_compressed}
                )]

            # Check Short Breakout
            elif close_5m < or_l and close_5m < current_vwap:
                # Structural Space Check vs PDL
                clean_space = True
                if prev_low is not None and not np.isnan(prev_low):
                    if close_5m >= prev_low and (close_5m - prev_low) < (2.0 * or_w):
                        clean_space = False

                if not (is_prior_day_compressed or clean_space):
                    continue

                future_bars = df_session_1m.loc[df_session_1m.index >= exec_time]
                if future_bars.empty:
                    continue
                open_1m = float(future_bars.iloc[0]["Open"])
                entry_price = open_1m - 0.50
                tp_target = entry_price - (self.target_r_multiple * sl_dist)

                return [TradeSignal(
                    strategy_id="H1",
                    strategy_name="H1_Filtered_30M_ORB_v2",
                    timestamp=bar_end_time,
                    execution_time=exec_time,
                    direction=-1,
                    entry_price=entry_price,
                    stop_loss_price=entry_price + sl_dist,
                    tp1_price=tp_target,
                    sl_dist=sl_dist,
                    scale_out_tp1=False,
                    be_trigger_r=self.be_trigger_r,
                    hard_time_exit="15:44:00",
                    metadata={"or_w": or_w, "daily_atr": daily_atr, "compression": is_prior_day_compressed}
                )]

        return []
