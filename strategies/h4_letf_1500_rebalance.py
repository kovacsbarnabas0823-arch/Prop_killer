"""
Module 4: Late-Day Leveraged ETF Flow & 0DTE Gamma Pin Module (strategies/h4_letf_1500_rebalance.py)
Author: Principal Quantitative Architect & Financial Engineer

Empirical Foundations:
  - Gao, Han, Li, & Zhou (2018, JFE): "Market Intraday Momentum"
  - Baltussen, Swinkels, & van Vliet (2021, JFE): "Predictable Rebalancing Flows"
  - Zarattini (2024, SSRN 4824172): "An Effective Intraday Momentum Strategy"

Phase 4 Upgraded Architecture (PF >= 1.50 Institutional Specification):
  - Execution Point: Strictly at 15:00:00 NY (the start of the final hour where institutional LETF rebalance orders hit the order book).
  - High-Conviction Displacement Threshold:
      * abs(Close_1500 - PrevClose_1600) / PrevClose_1600 >= 0.0085 (or abs(Close_1500 - Open_0930) / ATR_14 >= 0.70).
  - Triple Trend Alignment:
      1. First-Half-Hour Return r_1 (09:30-10:00) matches r_ROD direction.
      2. Midday Continuation: sign(Close_1500 - Close_1330) == sign(r_ROD) (confirming institutional TWAP/hedging acceleration).
      3. Extreme Session Position: RangePos_1500 >= 0.80 (Long) or <= 0.20 (Short), with Close_1500 on trend side of VWAP_RTH.
  - 0DTE 50-Point Strike Gamma-Pin Filter:
      * abs(Close_1500 - 50 * round(Close_1500 / 50.0)) <= 8.0 AND 14:45-15:00 range < 15.0 NQ points -> Pinned, skip.
  - Risk & Exit Geometry:
      * Initial Hard Stop: SL_dist = clip(0.14 * (DayHigh_1500 - DayLow_1500), 16.0, 32.0) NQ points.
      * Zero Breakeven Scratches (be_trigger_r = None, scale_out_tp1 = False).
      * Structural Trail: Confirmed 5-minute bar close across VWAP_RTH against position.
      * Hard Session Flatten at 15:44:00 NY.
"""

from __future__ import annotations
import logging
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, TradeSignal

logger = logging.getLogger("H4_LETF_Rebalance")


class H4LETF1500Rebalance(BaseStrategy):
    def __init__(
        self,
        rod_thresh: float = 0.0075,
        atr_move_thresh: float = 0.60,
        range_pos_long: float = 0.78,
        range_pos_short: float = 0.22,
        pin_strike_dist_max: float = 8.0,
        pin_range_max: float = 15.0,
        min_sl_dist: float = 16.0,
        max_sl_dist: float = 32.0,
        sl_range_factor: float = 0.14
    ):
        super().__init__(name="H4_LETF_1500_Rebalance", strategy_id="H4")
        self.rod_thresh = rod_thresh
        self.atr_move_thresh = atr_move_thresh
        self.range_pos_long = range_pos_long
        self.range_pos_short = range_pos_short
        self.pin_strike_dist_max = pin_strike_dist_max
        self.pin_range_max = pin_range_max
        self.min_sl_dist = min_sl_dist
        self.max_sl_dist = max_sl_dist
        self.sl_range_factor = sl_range_factor

    def evaluate_session(
        self,
        session_date,
        df_session_1m: pd.DataFrame,
        daily_atr: float,
        prev_close: Optional[float] = None,
        preprocessor=None,
        **kwargs
    ) -> List[TradeSignal]:
        if prev_close is None or np.isnan(prev_close) or prev_close <= 0:
            return []
        if daily_atr is None or np.isnan(daily_atr) or daily_atr <= 0:
            return []

        # RTH bars from 09:30 to 15:44
        rth_bars = df_session_1m.between_time("09:30", "15:44")
        if len(rth_bars) < 200:
            return []

        # Bar at 10:00:00 (r_1 calculation)
        time_1000_mask = rth_bars.index.time == pd.Timestamp("10:00:00").time()
        if not time_1000_mask.any():
            return []
        close_1000 = float(rth_bars.loc[time_1000_mask].iloc[-1]["Close"])
        r_1 = close_1000 - prev_close

        # Bar at 13:30:00 (midday check)
        time_1330_mask = rth_bars.index.time == pd.Timestamp("13:30:00").time()
        close_1330 = float(rth_bars.loc[time_1330_mask].iloc[-1]["Close"]) if time_1330_mask.any() else None

        # Bar at 15:00:00 (rebalance entry trigger)
        time_1500_mask = rth_bars.index.time == pd.Timestamp("15:00:00").time()
        if not time_1500_mask.any():
            return []
        bar_1500 = rth_bars.loc[time_1500_mask].iloc[-1]
        t_1500 = rth_bars.loc[time_1500_mask].index[-1]
        close_1500 = float(bar_1500["Close"])

        # Running Day Range up to 15:00:00
        bars_to_1500 = rth_bars.loc[rth_bars.index <= t_1500]
        high_1500 = float(bars_to_1500["High"].max())
        low_1500 = float(bars_to_1500["Low"].min())
        range_1500 = high_1500 - low_1500
        if range_1500 <= 0:
            return []

        # Displacements
        r_rod = (close_1500 - prev_close) / prev_close
        open_0930 = float(rth_bars.iloc[0]["Open"])
        norm_move_atr = abs(close_1500 - open_0930) / daily_atr
        range_pos = (close_1500 - low_1500) / range_1500

        # Anchored RTH VWAP at 15:00
        tp = (rth_bars["High"] + rth_bars["Low"] + rth_bars["Close"]) / 3.0
        cum_vol = rth_bars["Volume"].cumsum()
        vwap_series = (tp * rth_bars["Volume"]).cumsum() / cum_vol.replace(0, np.nan)
        vwap_1500 = float(vwap_series.loc[t_1500])

        # 0DTE Gamma Pin Filter
        s_50 = 50.0 * round(close_1500 / 50.0)
        if abs(close_1500 - s_50) <= self.pin_strike_dist_max:
            pin_bars = df_session_1m.between_time("14:45", "15:00")
            if not pin_bars.empty:
                rng = float(pin_bars["High"].max() - pin_bars["Low"].min())
                if rng < self.pin_range_max:
                    return []

        # High-Conviction Displacement Filter (0.0085 rod or 0.70 ATR)
        if not (abs(r_rod) >= self.rod_thresh or norm_move_atr >= self.atr_move_thresh):
            return []

        # Triple Trend Alignment: Midday Continuation check
        midday_ok_long = (close_1330 is not None and close_1500 >= close_1330)
        midday_ok_short = (close_1330 is not None and close_1500 <= close_1330)

        future_1m = df_session_1m.loc[df_session_1m.index > t_1500]
        if future_1m.empty:
            return []

        sl_dist = float(np.clip(self.sl_range_factor * range_1500, self.min_sl_dist, self.max_sl_dist))
        open_exec = float(future_1m.iloc[0]["Open"])
        t_exec = future_1m.index[0]

        signals: List[TradeSignal] = []

        # Long Flow
        if r_rod > 0 and r_1 > 0 and midday_ok_long and close_1500 > vwap_1500 and range_pos >= self.range_pos_long:
            entry_price = open_exec + 0.50
            sl_price = entry_price - sl_dist
            signals.append(TradeSignal(
                strategy_id="H4",
                strategy_name="H4_LETF_1500_Rebalance",
                timestamp=t_1500,
                execution_time=t_exec,
                direction=1,
                entry_price=entry_price,
                stop_loss_price=sl_price,
                tp1_price=None,
                sl_dist=sl_dist,
                scale_out_tp1=False,
                be_trigger_r=None,
                trail_rule="VWAP",
                hard_time_exit="15:44:00",
                metadata={
                    "r_rod": r_rod,
                    "norm_move": norm_move_atr,
                    "range_pos": range_pos,
                    "sl_dist": sl_dist
                }
            ))

        # Short Flow
        elif r_rod < 0 and r_1 < 0 and midday_ok_short and close_1500 < vwap_1500 and range_pos <= self.range_pos_short:
            entry_price = open_exec - 0.50
            sl_price = entry_price + sl_dist
            signals.append(TradeSignal(
                strategy_id="H4",
                strategy_name="H4_LETF_1500_Rebalance",
                timestamp=t_1500,
                execution_time=t_exec,
                direction=-1,
                entry_price=entry_price,
                stop_loss_price=sl_price,
                tp1_price=None,
                sl_dist=sl_dist,
                scale_out_tp1=False,
                be_trigger_r=None,
                trail_rule="VWAP",
                hard_time_exit="15:44:00",
                metadata={
                    "r_rod": r_rod,
                    "norm_move": norm_move_atr,
                    "range_pos": range_pos,
                    "sl_dist": sl_dist
                }
            ))

        return signals
