"""
Module 4: 15:00-15:44 Leveraged ETF Rebalancing & 0DTE Gamma Pin Filter (h4_letf_1500_rebalance.py)
Author: Principal Quantitative Architect

Strategy Rules:
  - Signal Evaluation at 15:00:00 NY:
      * r_ROD = (Close_1500 - PrevDay_Close_1600) / PrevDay_Close_1600
      * r_1   = (Close_1000 - PrevDay_Close_1600) / PrevDay_Close_1600
      * RangePos = (Close_1500 - DayLow_1500) / (DayHigh_1500 - DayLow_1500)
      * Long Trigger:  r_ROD >= +0.010 (+1.0%), r_1 > 0, and RangePos >= 0.80.
      * Short Trigger: r_ROD <= -0.010 (-1.0%), r_1 < 0, and RangePos <= 0.20.
  - 0DTE Gamma Pin Filter:
      * S_50 = 50 * round(Close_1500 / 50.0).
      * If abs(Close_1500 - S_50) <= 8.0 AND range_1445_1500 < 15.0 pts:
        Pinned by positive 0DTE dealer gamma -> SKIP TRADE.
  - Risk & Execution:
      * SL_dist = min(35.0, 0.15 * (DayHigh_1500 - DayLow_1500)).
      * TP1 (50%): Next 50-point strike magnet (S_next_50 -/+ 2.0 pts).
      * TP2 (50%): Hard flatten at 15:44:00 NY.
      * Fill at 15:01:00 Open + 0.50 pt slippage.
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
        min_return_rod: float = 0.010,       # 1.0% return
        range_pos_long: float = 0.80,
        range_pos_short: float = 0.20,
        pin_strike_dist_max: float = 8.0,
        pin_range_max: float = 15.0,
        max_sl_dist: float = 35.0,
        sl_range_factor: float = 0.15
    ):
        super().__init__(name="H4_LETF_1500_Rebalance", strategy_id="H4")
        self.min_return_rod = min_return_rod
        self.range_pos_long = range_pos_long
        self.range_pos_short = range_pos_short
        self.pin_strike_dist_max = pin_strike_dist_max
        self.pin_range_max = pin_range_max
        self.max_sl_dist = max_sl_dist
        self.sl_range_factor = sl_range_factor

    def evaluate_session(
        self,
        session_date,
        df_session_1m: pd.DataFrame,
        daily_atr: float,
        prev_close: Optional[float] = None,
        **kwargs
    ) -> List[TradeSignal]:
        if prev_close is None or np.isnan(prev_close) or prev_close <= 0:
            return []

        # Bars up to 15:00:00 NY
        bars_to_1500 = df_session_1m.between_time("09:30", "15:00")
        if len(bars_to_1500) < 200:
            return []

        # Find bar at 15:00:00
        time_1500_mask = bars_to_1500.index.time == pd.Timestamp("15:00:00").time()
        if not time_1500_mask.any():
            return []

        close_1500 = float(bars_to_1500.loc[time_1500_mask].iloc[-1]["Close"])
        bar_1500_time = bars_to_1500.loc[time_1500_mask].index[-1]

        # Find bar at 10:00:00
        time_1000_mask = bars_to_1500.index.time == pd.Timestamp("10:00:00").time()
        if not time_1000_mask.any():
            return []
        close_1000 = float(bars_to_1500.loc[time_1000_mask].iloc[-1]["Close"])

        day_high_1500 = float(bars_to_1500["High"].max())
        day_low_1500 = float(bars_to_1500["Low"].min())
        day_range = day_high_1500 - day_low_1500
        if day_range <= 0:
            return []

        # 1. Compute Returns & Metrics
        r_rod = (close_1500 - prev_close) / prev_close
        r_1 = (close_1000 - prev_close) / prev_close
        range_pos = (close_1500 - day_low_1500) / day_range

        # 2. 0DTE Gamma Pin Filter
        s_50 = 50.0 * round(close_1500 / 50.0)
        dist_to_strike = abs(close_1500 - s_50)

        # Range between 14:45 and 15:00
        bars_pin_check = df_session_1m.between_time("14:45", "15:00")
        if not bars_pin_check.empty:
            range_15m = float(bars_pin_check["High"].max() - bars_pin_check["Low"].min())
        else:
            range_15m = 999.0

        if dist_to_strike <= self.pin_strike_dist_max and range_15m < self.pin_range_max:
            # Pinned by dealer gamma - abort trade
            return []

        # 3. Execution Bar: 15:01:00 NY
        future_bars = df_session_1m.loc[df_session_1m.index > bar_1500_time]
        if future_bars.empty:
            return []

        exec_bar = future_bars.iloc[0]
        exec_time = future_bars.index[0]

        sl_dist = min(self.max_sl_dist, self.sl_range_factor * day_range)
        if sl_dist < 10.0:
            sl_dist = 10.0

        signals: List[TradeSignal] = []

        # Long Trigger: r_ROD >= +1.0%, r_1 > 0, RangePos >= 0.80
        if r_rod >= self.min_return_rod and r_1 > 0 and range_pos >= self.range_pos_long:
            open_val = float(exec_bar["Open"])
            entry_price = open_val + 0.50
            sl_price = entry_price - sl_dist

            # Next 50-point strike magnet
            s_next_50 = 50.0 * np.ceil(entry_price / 50.0)
            if s_next_50 - entry_price < 10.0:
                s_next_50 += 50.0
            tp1_price = s_next_50 - 2.0

            sig = TradeSignal(
                strategy_id="H4",
                strategy_name="H4_LETF_1500_Rebalance",
                timestamp=bar_1500_time,
                execution_time=exec_time,
                direction=1,
                entry_price=entry_price,
                stop_loss_price=sl_price,
                tp1_price=tp1_price,
                tp2_price=None,  # TP2 is hard exit at 15:44:00
                sl_dist=sl_dist,
                tp1_dist=tp1_price - entry_price,
                tp2_dist=0.0,
                breakeven_trigger_price=entry_price + 2.0,
                hard_time_exit="15:44:00",
                metadata={"r_rod": r_rod, "range_pos": range_pos, "s_next_50": s_next_50}
            )
            signals.append(sig)

        # Short Trigger: r_ROD <= -1.0%, r_1 < 0, RangePos <= 0.20
        elif r_rod <= -self.min_return_rod and r_1 < 0 and range_pos <= self.range_pos_short:
            open_val = float(exec_bar["Open"])
            entry_price = open_val - 0.50
            sl_price = entry_price + sl_dist

            s_next_50 = 50.0 * np.floor(entry_price / 50.0)
            if entry_price - s_next_50 < 10.0:
                s_next_50 -= 50.0
            tp1_price = s_next_50 + 2.0

            sig = TradeSignal(
                strategy_id="H4",
                strategy_name="H4_LETF_1500_Rebalance",
                timestamp=bar_1500_time,
                execution_time=exec_time,
                direction=-1,
                entry_price=entry_price,
                stop_loss_price=sl_price,
                tp1_price=tp1_price,
                tp2_price=None,
                sl_dist=sl_dist,
                tp1_dist=entry_price - tp1_price,
                tp2_dist=0.0,
                breakeven_trigger_price=entry_price - 2.0,
                hard_time_exit="15:44:00",
                metadata={"r_rod": r_rod, "range_pos": range_pos, "s_next_50": s_next_50}
            )
            signals.append(sig)

        return signals
