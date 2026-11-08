"""
Module 3: Intraday LOB 4th-Touch Absorption Exhaustion Breakout Cascade (Vectorized / NumPy Optimized)
Author: Principal Quantitative Architect

Strategy Rules:
  - Causal Pivot & Wall Clustering (10:15:00 - 14:15:00 NY):
      * Pivot confirmed at k-4 when High[k-4] == max(High[k-8 : k+1]) (0 lookahead).
      * Group 3 distinct touches (p1, p2, p3) within a 4.0 NQ point band.
      * Separation between touches >= 8 minutes.
      * Pullback depth between touches >= 12.0 NQ points.
      * No close above L_wall + 1.5 pts before touch 4.
  - Absorption Compression Filter:
      * D_12: pullback depth between touch 1 and 2.
      * D_34: pullback depth after touch 3.
      * Require D_34 <= 0.65 * D_12 (sellers exhausted, higher lows coiling).
  - 4th-Touch Breakout Trigger:
      * Price crosses L_wall + 1.50 pts with 3-bar volume >= 1.25 * SMA(Volume_1m, 20).
      * Fill at next 1m Open + 0.50 pt slippage.
  - Risk & Exits:
      * SL_dist = max(15.0, min(35.0, D_34 + 3.0)).
      * Scratch Rule: After 5 minutes, if trade has not reached +0.5 * SL_dist, exit immediately at market.
      * TP1 (50%): +1.2 * SL_dist (Move SL to BE).
      * TP2 (50%): +2.5 * SL_dist or 1m 9-EMA close.
"""

from __future__ import annotations
import logging
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, TradeSignal

logger = logging.getLogger("H3_LOB_4th_Touch")


class H3LOB4thTouch(BaseStrategy):
    def __init__(
        self,
        wall_band_points: float = 4.0,
        min_separation_minutes: int = 8,
        min_pullback_points: float = 12.0,
        compression_ratio_max: float = 0.65,
        volume_factor: float = 1.25,
        volume_sma_bars: int = 20,
        scratch_bars: int = 5,
        scratch_threshold_r: float = 0.5
    ):
        super().__init__(name="H3_LOB_4th_Touch", strategy_id="H3")
        self.wall_band_points = wall_band_points
        self.min_separation_minutes = min_separation_minutes
        self.min_pullback_points = min_pullback_points
        self.compression_ratio_max = compression_ratio_max
        self.volume_factor = volume_factor
        self.volume_sma_bars = volume_sma_bars
        self.scratch_bars = scratch_bars
        self.scratch_threshold_r = scratch_threshold_r

    def evaluate_session(
        self,
        session_date,
        df_session_1m: pd.DataFrame,
        daily_atr: float,
        prev_close: Optional[float] = None,
        **kwargs
    ) -> List[TradeSignal]:
        rth_bars = df_session_1m.between_time("09:30", "15:00")
        if len(rth_bars) < 60:
            return []

        # Vectorized volume moving average
        vols = rth_bars["Volume"].to_numpy(dtype=float)
        vol_sma = pd.Series(vols).rolling(window=self.volume_sma_bars).mean().to_numpy(dtype=float)
        vol_3bar = pd.Series(vols).rolling(window=3).mean().to_numpy(dtype=float)

        highs = rth_bars["High"].to_numpy(dtype=float)
        lows = rth_bars["Low"].to_numpy(dtype=float)
        closes = rth_bars["Close"].to_numpy(dtype=float)
        opens = rth_bars["Open"].to_numpy(dtype=float)
        times = rth_bars.index
        n_bars = len(highs)

        # Precompute minutes from session start (09:30 is 0 min)
        t_minutes = np.array([t.hour * 60 + t.minute for t in times])
        # Window: 10:15 (615 min) to 14:15 (855 min)
        min_1015 = 10 * 60 + 15
        max_1415 = 14 * 60 + 15

        # 1. Strictly causal pivot highs and pivot lows
        pivot_highs = [] # (bar_idx, time, price, minute)
        pivot_lows = []

        for k in range(8, n_bars):
            p_idx = k - 4
            m = t_minutes[p_idx]
            if m < min_1015 or m > max_1415:
                continue

            if highs[p_idx] == np.max(highs[k-8 : k+1]):
                pivot_highs.append((p_idx, times[p_idx], highs[p_idx], m))

            if lows[p_idx] == np.min(lows[k-8 : k+1]):
                pivot_lows.append((p_idx, times[p_idx], lows[p_idx], m))

        signals: List[TradeSignal] = []

        # 2. Resistance Walls (Bullish 4th-Touch Breakouts)
        if len(pivot_highs) >= 3:
            for i in range(len(pivot_highs) - 2):
                p1_idx, t1, p1_h, m1 = pivot_highs[i]
                p2_idx, t2, p2_h, m2 = pivot_highs[i+1]
                p3_idx, t3, p3_h, m3 = pivot_highs[i+2]

                wall_high = max(p1_h, p2_h, p3_h)
                wall_low = min(p1_h, p2_h, p3_h)
                if (wall_high - wall_low) > self.wall_band_points:
                    continue

                if (m2 - m1) < self.min_separation_minutes or (m3 - m2) < self.min_separation_minutes:
                    continue

                trough_12 = np.min(lows[p1_idx : p2_idx + 1])
                d_12 = p1_h - trough_12
                if d_12 < self.min_pullback_points:
                    continue

                trough_23 = np.min(lows[p2_idx : p3_idx + 1])
                if (p2_h - trough_23) < 8.0:
                    continue

                l_wall = wall_high
                trough_34 = p3_h

                for b in range(p3_idx + 1, n_bars):
                    if t_minutes[b] > max_1415:
                        break

                    if lows[b] < trough_34:
                        trough_34 = lows[b]

                    if closes[b] > (l_wall + 1.50):
                        d_34 = p3_h - trough_34
                        if d_34 <= self.compression_ratio_max * d_12 and d_34 >= 5.0:
                            if not np.isnan(vol_sma[b]) and vol_3bar[b] >= self.volume_factor * vol_sma[b]:
                                if b + 1 < n_bars:
                                    exec_time = times[b+1]
                                    entry_price = opens[b+1] + 0.50

                                    sl_dist = max(15.0, min(35.0, d_34 + 3.0))
                                    sl_price = entry_price - sl_dist
                                    tp1_price = entry_price + (1.2 * sl_dist)
                                    tp2_price = entry_price + (2.5 * sl_dist)

                                    sig = TradeSignal(
                                        strategy_id="H3",
                                        strategy_name="H3_LOB_4th_Touch",
                                        timestamp=times[b],
                                        execution_time=exec_time,
                                        direction=1,
                                        entry_price=entry_price,
                                        stop_loss_price=sl_price,
                                        tp1_price=tp1_price,
                                        tp2_price=tp2_price,
                                        sl_dist=sl_dist,
                                        tp1_dist=1.2 * sl_dist,
                                        tp2_dist=2.5 * sl_dist,
                                        breakeven_trigger_price=entry_price + 2.0,
                                        trail_rule="EMA1_9",
                                        scratch_bars=self.scratch_bars,
                                        scratch_threshold_r=self.scratch_threshold_r,
                                        hard_time_exit="15:44:00",
                                        metadata={"l_wall": l_wall, "d_12": d_12, "d_34": d_34, "compression": d_34 / d_12}
                                    )
                                    signals.append(sig)
                                    return signals
                        break
                if signals:
                    return signals

        # 3. Support Walls (Bearish 4th-Touch Breakdowns)
        if len(pivot_lows) >= 3:
            for i in range(len(pivot_lows) - 2):
                p1_idx, t1, p1_l, m1 = pivot_lows[i]
                p2_idx, t2, p2_l, m2 = pivot_lows[i+1]
                p3_idx, t3, p3_l, m3 = pivot_lows[i+2]

                wall_high = max(p1_l, p2_l, p3_l)
                wall_low = min(p1_l, p2_l, p3_l)
                if (wall_high - wall_low) > self.wall_band_points:
                    continue

                if (m2 - m1) < self.min_separation_minutes or (m3 - m2) < self.min_separation_minutes:
                    continue

                peak_12 = np.max(highs[p1_idx : p2_idx + 1])
                d_12 = peak_12 - p1_l
                if d_12 < self.min_pullback_points:
                    continue

                peak_23 = np.max(highs[p2_idx : p3_idx + 1])
                if (peak_23 - p2_l) < 8.0:
                    continue

                l_wall = wall_low
                peak_34 = p3_l

                for b in range(p3_idx + 1, n_bars):
                    if t_minutes[b] > max_1415:
                        break

                    if highs[b] > peak_34:
                        peak_34 = highs[b]

                    if closes[b] < (l_wall - 1.50):
                        d_34 = peak_34 - p3_l
                        if d_34 <= self.compression_ratio_max * d_12 and d_34 >= 5.0:
                            if not np.isnan(vol_sma[b]) and vol_3bar[b] >= self.volume_factor * vol_sma[b]:
                                if b + 1 < n_bars:
                                    exec_time = times[b+1]
                                    entry_price = opens[b+1] - 0.50

                                    sl_dist = max(15.0, min(35.0, d_34 + 3.0))
                                    sl_price = entry_price + sl_dist
                                    tp1_price = entry_price - (1.2 * sl_dist)
                                    tp2_price = entry_price - (2.5 * sl_dist)

                                    sig = TradeSignal(
                                        strategy_id="H3",
                                        strategy_name="H3_LOB_4th_Touch",
                                        timestamp=times[b],
                                        execution_time=exec_time,
                                        direction=-1,
                                        entry_price=entry_price,
                                        stop_loss_price=sl_price,
                                        tp1_price=tp1_price,
                                        tp2_price=tp2_price,
                                        sl_dist=sl_dist,
                                        tp1_dist=1.2 * sl_dist,
                                        tp2_dist=2.5 * sl_dist,
                                        breakeven_trigger_price=entry_price - 2.0,
                                        trail_rule="EMA1_9",
                                        scratch_bars=self.scratch_bars,
                                        scratch_threshold_r=self.scratch_threshold_r,
                                        hard_time_exit="15:44:00",
                                        metadata={"l_wall": l_wall, "d_12": d_12, "d_34": d_34, "compression": d_34 / d_12}
                                    )
                                    signals.append(sig)
                                    return signals
                        break
                if signals:
                    return signals

        return signals
