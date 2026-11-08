"""
Module 2: NY Open London Low/High Sweep Reversal (h2_ny_london_sweep.py)
Author: Principal Quantitative Architect

Strategy Rules:
  - DST-Safe London Range: 08:00:00 Europe/London to 09:29:59 America/New_York.
  - Require 0.15 * ATR_14 <= LON_W <= 0.85 * ATR_14.
  - Sweep Detection Window: 09:30:00 - 10:30:00 NY.
      * Bullish Sweep: Low_1m < LON_L - 2.0 pts with depth <= 0.30 * ATR_14.
      * Bearish Sweep: High_1m > LON_H + 2.0 pts with depth <= 0.30 * ATR_14.
  - 1-Minute MSS / CISD Reversal Confirmation:
      * Within 15 minutes of sweep, candle closes back inside London Range (Close > LON_L / Close < LON_H)
        AND closes beyond prior 4-bar counter-swing pivot (SwingHigh / SwingLow).
  - Stop & Targets:
      * SL = Sweep Extreme +/- 5.0 pts (12.0 <= SL_dist <= 60.0 pts).
      * TP1 (50%): Entry +/- 1.0 * SL_dist (Move SL to BE + 2.0 pts).
      * TP2 (50%): London Midpoint (LON_L + 0.50 * LON_W) or RTH VWAP.
      * Hard Time Exit: 11:30:00 NY.
"""

from __future__ import annotations
import logging
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import pytz

from strategies.base_strategy import BaseStrategy, TradeSignal

logger = logging.getLogger("H2_London_Sweep")


class H2NYLondonSweep(BaseStrategy):
    def __init__(
        self,
        min_lon_atr_ratio: float = 0.15,
        max_lon_atr_ratio: float = 0.85,
        min_sweep_pts: float = 2.0,
        max_sweep_atr_ratio: float = 0.30,
        confirmation_window_minutes: int = 15,
        min_sl_dist: float = 12.0,
        max_sl_dist: float = 60.0
    ):
        super().__init__(name="H2_NY_London_Sweep", strategy_id="H2")
        self.min_lon_atr_ratio = min_lon_atr_ratio
        self.max_lon_atr_ratio = max_lon_atr_ratio
        self.min_sweep_pts = min_sweep_pts
        self.max_sweep_atr_ratio = max_sweep_atr_ratio
        self.confirmation_window_minutes = confirmation_window_minutes
        self.min_sl_dist = min_sl_dist
        self.max_sl_dist = max_sl_dist
        self.tz_ny = pytz.timezone("America/New_York")
        self.tz_lon = pytz.timezone("Europe/London")

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

        # 1. Extract London Session (08:00 Europe/London to 09:29:59 America/New_York)
        # Note: df_session_1m passed in contains full 24h session or includes pre-market
        # If df_session_1m is indexed in America/New_York:
        ny_rth_start = self.tz_ny.localize(pd.Timestamp(f"{session_date} 09:30:00"))
        lon_session_start = self.tz_lon.localize(pd.Timestamp(f"{session_date} 08:00:00")).astimezone(self.tz_ny)

        lon_bars = df_session_1m.loc[(df_session_1m.index >= lon_session_start) & (df_session_1m.index < ny_rth_start)]
        if len(lon_bars) < 30:
            return []

        lon_h = float(lon_bars["High"].max())
        lon_l = float(lon_bars["Low"].min())
        lon_w = lon_h - lon_l
        lon_mid = lon_l + (0.50 * lon_w)

        # London Range Filter
        if not (self.min_lon_atr_ratio * daily_atr <= lon_w <= self.max_lon_atr_ratio * daily_atr):
            return []

        # 2. Sweep Detection in NY Open Window: 09:30:00 - 10:30:00 NY
        ny_open_bars = df_session_1m.between_time("09:30", "10:30")
        if len(ny_open_bars) < 10:
            return []

        max_sweep_depth = self.max_sweep_atr_ratio * daily_atr

        signals: List[TradeSignal] = []
        sweep_active = False
        sweep_type = None  # "BULLISH" or "BEARISH"
        sweep_time = None
        sweep_extreme = None

        rth_all = df_session_1m.between_time("09:30", "11:30")

        for i in range(len(ny_open_bars)):
            bar_time = ny_open_bars.index[i]
            bar = ny_open_bars.iloc[i]
            h = float(bar["High"])
            l = float(bar["Low"])

            # Check Bullish Sweep of London Low
            if not sweep_active:
                if l < (lon_l - self.min_sweep_pts):
                    depth = lon_l - l
                    if depth <= max_sweep_depth:
                        sweep_active = True
                        sweep_type = "BULLISH"
                        sweep_time = bar_time
                        sweep_extreme = l
                elif h > (lon_h + self.min_sweep_pts):
                    depth = h - lon_h
                    if depth <= max_sweep_depth:
                        sweep_active = True
                        sweep_type = "BEARISH"
                        sweep_time = bar_time
                        sweep_extreme = h

            if sweep_active:
                # Update sweep extreme if deeper penetration occurs
                if sweep_type == "BULLISH":
                    if l < sweep_extreme:
                        sweep_extreme = l
                        if (lon_l - sweep_extreme) > max_sweep_depth:
                            # Too deep, true structural breakdown - abort
                            sweep_active = False
                            continue
                elif sweep_type == "BEARISH":
                    if h > sweep_extreme:
                        sweep_extreme = h
                        if (sweep_extreme - lon_h) > max_sweep_depth:
                            # Too deep, true breakout - abort
                            sweep_active = False
                            continue

                # Confirmation Check (within 15 minutes of sweep_time)
                elapsed_minutes = (bar_time - sweep_time).total_seconds() / 60.0
                if elapsed_minutes > self.confirmation_window_minutes:
                    # Timeout without MSS confirmation
                    sweep_active = False
                    continue

                # Check MSS/CISD reversal confirmation:
                # Need candle closing back inside range AND beyond prior 4 bars counter-swing
                bar_close = float(bar["Close"])
                if sweep_type == "BULLISH":
                    if bar_close > lon_l:
                        # Find prior 4 bars before this bar
                        idx_in_all = rth_all.index.get_loc(bar_time)
                        if idx_in_all >= 4:
                            prior_4_highs = rth_all.iloc[idx_in_all-4:idx_in_all]["High"].max()
                            if bar_close > prior_4_highs:
                                # Confirmed Long Reversal!
                                # Execution at next bar open
                                future_bars = rth_all.iloc[idx_in_all+1:]
                                if not future_bars.empty:
                                    exec_bar = future_bars.iloc[0]
                                    exec_time = future_bars.index[0]
                                    entry_price = float(exec_bar["Open"]) + 0.50
                                    sl_price = sweep_extreme - 5.0
                                    sl_dist = entry_price - sl_price

                                    if self.min_sl_dist <= sl_dist <= self.max_sl_dist:
                                        tp1_price = entry_price + (1.0 * sl_dist)
                                        tp2_price = max(entry_price + (1.5 * sl_dist), lon_mid)

                                        sig = TradeSignal(
                                            strategy_id="H2",
                                            strategy_name="H2_NY_London_Sweep",
                                            timestamp=bar_time,
                                            execution_time=exec_time,
                                            direction=1,
                                            entry_price=entry_price,
                                            stop_loss_price=sl_price,
                                            tp1_price=tp1_price,
                                            tp2_price=tp2_price,
                                            sl_dist=sl_dist,
                                            tp1_dist=sl_dist,
                                            tp2_dist=tp2_price - entry_price,
                                            breakeven_trigger_price=entry_price + 2.0,
                                            hard_time_exit="11:30:00",
                                            metadata={"sweep_type": sweep_type, "sweep_depth": lon_l - sweep_extreme, "lon_w": lon_w}
                                        )
                                        signals.append(sig)
                                        return signals

                elif sweep_type == "BEARISH":
                    if bar_close < lon_h:
                        idx_in_all = rth_all.index.get_loc(bar_time)
                        if idx_in_all >= 4:
                            prior_4_lows = rth_all.iloc[idx_in_all-4:idx_in_all]["Low"].min()
                            if bar_close < prior_4_lows:
                                # Confirmed Short Reversal!
                                future_bars = rth_all.iloc[idx_in_all+1:]
                                if not future_bars.empty:
                                    exec_bar = future_bars.iloc[0]
                                    exec_time = future_bars.index[0]
                                    entry_price = float(exec_bar["Open"]) - 0.50
                                    sl_price = sweep_extreme + 5.0
                                    sl_dist = sl_price - entry_price

                                    if self.min_sl_dist <= sl_dist <= self.max_sl_dist:
                                        tp1_price = entry_price - (1.0 * sl_dist)
                                        tp2_price = min(entry_price - (1.5 * sl_dist), lon_mid)

                                        sig = TradeSignal(
                                            strategy_id="H2",
                                            strategy_name="H2_NY_London_Sweep",
                                            timestamp=bar_time,
                                            execution_time=exec_time,
                                            direction=-1,
                                            entry_price=entry_price,
                                            stop_loss_price=sl_price,
                                            tp1_price=tp1_price,
                                            tp2_price=tp2_price,
                                            sl_dist=sl_dist,
                                            tp1_dist=sl_dist,
                                            tp2_dist=entry_price - tp2_price,
                                            breakeven_trigger_price=entry_price - 2.0,
                                            hard_time_exit="11:30:00",
                                            metadata={"sweep_type": sweep_type, "sweep_depth": sweep_extreme - lon_h, "lon_w": lon_w}
                                        )
                                        signals.append(sig)
                                        return signals

        return signals
