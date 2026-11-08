"""
Module 1: Upgraded 30m Filtered ORB (h1_orb_30m_regime.py)
Author: Principal Quantitative Architect

Strategy Rules:
  - Opening Range (09:30:00 - 09:59:59 NY): OR_H, OR_L, OR_W = OR_H - OR_L.
  - Trailing 14-day RTH ATR_14 (shifted 1 session, zero lookahead).
  - Regime Filter: Arm ONLY if 0.20 * ATR_14 <= OR_W <= 0.75 * ATR_14.
  - Entry Trigger (10:00:00 - 13:30:00 NY):
      * 5-minute candle body close outside the range:
          Close_5m > OR_H for Long; Close_5m < OR_L for Short.
      * Anchored RTH VWAP alignment: Close_5m > VWAP for Long; Close_5m < VWAP for Short.
      * Breakout 5m Volume >= 1.15 * SMA(Volume_5m, 6).
      * Fill at next 1m Open + 0.50 pt slippage. Max 1 trade/day.
  - Stop & Multi-Stage Targets:
      * SL_dist = min(OR_W, max(0.55 * OR_W, 0.25 * ATR_14)).
      * TP1 (50%): Entry +/- 1.0 * OR_W. Upon hit, SL moves to Entry +/- 2.0 pts (BE+).
      * TP2 (50%): Entry +/- 2.0 * OR_W or 15m 12-EMA close.
      * Hard flatten at 15:44:00 NY.
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
        max_atr_mult: float = 0.75,
        volume_factor: float = 1.15,
        volume_sma_bars: int = 6
    ):
        super().__init__(name="H1_Filtered_30M_ORB", strategy_id="H1")
        self.min_atr_mult = min_atr_mult
        self.max_atr_mult = max_atr_mult
        self.volume_factor = volume_factor
        self.volume_sma_bars = volume_sma_bars

    def evaluate_session(
        self,
        session_date,
        df_session_1m: pd.DataFrame,
        daily_atr: float,
        prev_close: Optional[float] = None,
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

        # Regime Filter: 0.20 * ATR_14 <= OR_W <= 0.75 * ATR_14
        if not (self.min_atr_mult * daily_atr <= or_w <= self.max_atr_mult * daily_atr):
            return []

        # Calculate Anchored RTH VWAP
        rth_bars = df_session_1m.between_time("09:30", "15:44")
        if rth_bars.empty:
            return []

        typical_price = (rth_bars["High"] + rth_bars["Low"] + rth_bars["Close"]) / 3.0
        cum_vol = rth_bars["Volume"].cumsum()
        cum_tp_vol = (typical_price * rth_bars["Volume"]).cumsum()
        vwap_series = cum_tp_vol / cum_vol.replace(0, np.nan)

        # 2. Resample to 5-minute bars for trigger evaluation
        # We resample strictly within RTH starting 09:30
        resampled_5m = rth_bars.resample("5min", closed="left", label="left").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }).dropna()

        # Compute 5m Volume SMA
        vol_sma = resampled_5m["Volume"].rolling(window=self.volume_sma_bars).mean()

        # Evaluation Window: 10:00:00 to 13:30:00 NY
        trade_window_5m = resampled_5m.between_time("10:00", "13:30")
        if trade_window_5m.empty:
            return []

        # Calculate 15m 12-EMA for runner trail
        resampled_15m = rth_bars.resample("15min", closed="left", label="left").agg({"Close": "last"}).dropna()
        ema12_15m = resampled_15m["Close"].ewm(span=12, adjust=False).mean()

        # Compute SL distance: min(OR_W, max(0.55 * OR_W, 0.25 * ATR_14))
        sl_dist = min(or_w, max(0.55 * or_w, 0.25 * daily_atr))

        signals: List[TradeSignal] = []

        for bar_start, bar_5m in trade_window_5m.iterrows():
            # Bar finishes at bar_start + 5 minutes
            bar_end_time = bar_start + pd.Timedelta(minutes=5)
            # The next 1m bar open for execution is bar_end_time
            exec_time = bar_end_time

            # Verify that bar_5m has volume data and sma
            if bar_start not in vol_sma.index or pd.isna(vol_sma.loc[bar_start]):
                continue

            current_vol_sma = vol_sma.loc[bar_start]
            if bar_5m["Volume"] < self.volume_factor * current_vol_sma:
                continue

            # VWAP at bar end
            # find nearest 1m bar up to bar_end_time - 1min
            prior_1m_mask = vwap_series.index < exec_time
            if not prior_1m_mask.any():
                continue
            current_vwap = float(vwap_series.loc[prior_1m_mask].iloc[-1])

            close_5m = float(bar_5m["Close"])

            # Check Long Breakout
            if close_5m > or_h and close_5m > current_vwap:
                # Find execution 1m bar
                future_bars = df_session_1m.loc[df_session_1m.index >= exec_time]
                if future_bars.empty:
                    continue
                open_1m = float(future_bars.iloc[0]["Open"])
                entry_price = open_1m + 0.50

                sig = TradeSignal(
                    strategy_id="H1",
                    strategy_name="H1_Filtered_30M_ORB",
                    timestamp=bar_end_time,
                    execution_time=exec_time,
                    direction=1,
                    entry_price=entry_price,
                    stop_loss_price=entry_price - sl_dist,
                    tp1_price=entry_price + (1.0 * or_w),
                    tp2_price=entry_price + (2.0 * or_w),
                    sl_dist=sl_dist,
                    tp1_dist=1.0 * or_w,
                    tp2_dist=2.0 * or_w,
                    breakeven_trigger_price=entry_price + 2.0,
                    trail_rule="EMA15_12",
                    hard_time_exit="15:44:00",
                    metadata={"or_w": or_w, "daily_atr": daily_atr, "ratio": or_w / daily_atr}
                )
                signals.append(sig)
                break  # Max 1 trade per session

            # Check Short Breakout
            elif close_5m < or_l and close_5m < current_vwap:
                future_bars = df_session_1m.loc[df_session_1m.index >= exec_time]
                if future_bars.empty:
                    continue
                open_1m = float(future_bars.iloc[0]["Open"])
                entry_price = open_1m - 0.50

                sig = TradeSignal(
                    strategy_id="H1",
                    strategy_name="H1_Filtered_30M_ORB",
                    timestamp=bar_end_time,
                    execution_time=exec_time,
                    direction=-1,
                    entry_price=entry_price,
                    stop_loss_price=entry_price + sl_dist,
                    tp1_price=entry_price - (1.0 * or_w),
                    tp2_price=entry_price - (2.0 * or_w),
                    sl_dist=sl_dist,
                    tp1_dist=1.0 * or_w,
                    tp2_dist=2.0 * or_w,
                    breakeven_trigger_price=entry_price - 2.0,
                    trail_rule="EMA15_12",
                    hard_time_exit="15:44:00",
                    metadata={"or_w": or_w, "daily_atr": daily_atr, "ratio": or_w / daily_atr}
                )
                signals.append(sig)
                break  # Max 1 trade per session

        return signals
