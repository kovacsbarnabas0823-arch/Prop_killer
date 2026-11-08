"""
Module 2: NY Open London Sweep Reversal (Phase 2 High-Conviction 5m Displacement)
Author: Principal Quantitative Architect

Strategy Rules:
  - DST-Safe London Range: 08:00 Europe/London to 09:29:59 America/New_York.
  - London Range Quality: 0.25 * ATR_14 <= LON_W <= 0.80 * ATR_14.
  - Dalton Overnight Inventory Exhaustion:
      * Bullish sweep of LON_L requires LON_L < PrevDay_Close_1600 (net short inventory).
      * Bearish sweep of LON_H requires LON_H > PrevDay_Close_1600 (net long inventory).
  - 5-Minute Displacement Reversal Trigger (09:30:00 to 10:15:00 NY Killzone):
      * Bullish: 5m candle sweeps below LON_L by 4.0 to 0.25 * ATR_14.
        Either that candle or immediately following candle closes back inside (Close_5m > LON_L),
        closes in top 40% of its range ((Close - Low)/(High - Low) >= 0.60), and Volume >= 1.25 * SMA(Volume_5m, 12).
      * Bearish: 5m candle sweeps above LON_H by 4.0 to 0.25 * ATR_14.
        Closes back inside (Close_5m < LON_H), closes in bottom 40% of its range, and Volume >= 1.25 * SMA(Volume_5m, 12).
  - Asymmetric Exit Geometry:
      * SL = Sweep Extreme +/- 4.0 pts (20.0 <= SL_dist <= 55.0 pts).
      * scale_out_tp1 = False (Hold 100% position).
      * Delayed Breakeven (+1.4R): Move stop to Entry +/- 2.0 pts only after +1.4R.
      * Full Target: max(2.4 * SL_dist, 0.85 * LON_W) (targeting opposite London boundary / 2.4R+).
      * Hard Time Exit: 12:00:00 NY.
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
        min_lon_atr_ratio: float = 0.25,      # Tightened from 0.15 to 0.25
        max_lon_atr_ratio: float = 0.80,
        min_sweep_pts: float = 4.0,           # Raised from 2.0 to 4.0
        max_sweep_atr_ratio: float = 0.25,    # 0.25 * ATR_14 max penetration
        volume_factor: float = 1.25,
        volume_sma_bars: int = 12,
        min_sl_dist: float = 20.0,
        max_sl_dist: float = 55.0,
        be_trigger_r: float = 1.4,
        target_r_multiple: float = 2.4
    ):
        super().__init__(name="H2_NY_London_Sweep_v2", strategy_id="H2")
        self.min_lon_atr_ratio = min_lon_atr_ratio
        self.max_lon_atr_ratio = max_lon_atr_ratio
        self.min_sweep_pts = min_sweep_pts
        self.max_sweep_atr_ratio = max_sweep_atr_ratio
        self.volume_factor = volume_factor
        self.volume_sma_bars = volume_sma_bars
        self.min_sl_dist = min_sl_dist
        self.max_sl_dist = max_sl_dist
        self.be_trigger_r = be_trigger_r
        self.target_r_multiple = target_r_multiple
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
        if prev_close is None or np.isnan(prev_close) or prev_close <= 0:
            return []

        # 1. London Session Extraction (08:00 Europe/London to 09:29:59 America/New_York)
        ny_rth_start = self.tz_ny.localize(pd.Timestamp(f"{session_date} 09:30:00"))
        lon_session_start = self.tz_lon.localize(pd.Timestamp(f"{session_date} 08:00:00")).astimezone(self.tz_ny)

        lon_bars = df_session_1m.loc[(df_session_1m.index >= lon_session_start) & (df_session_1m.index < ny_rth_start)]
        if len(lon_bars) < 30:
            return []

        lon_h = float(lon_bars["High"].max())
        lon_l = float(lon_bars["Low"].min())
        lon_w = lon_h - lon_l

        # London Range Filter: 0.25 * ATR_14 <= LON_W <= 0.80 * ATR_14
        if not (self.min_lon_atr_ratio * daily_atr <= lon_w <= self.max_lon_atr_ratio * daily_atr):
            return []

        # Dalton Overnight Inventory Filter:
        # Bullish sweep of LON_L requires LON_L < PrevDay_Close_1600 (net short inventory)
        # Bearish sweep of LON_H requires LON_H > PrevDay_Close_1600 (net long inventory)
        bullish_inventory_ok = (lon_l < prev_close)
        bearish_inventory_ok = (lon_h > prev_close)

        if not (bullish_inventory_ok or bearish_inventory_ok):
            return []

        # 2. Resample 5m bars for 09:30 - 10:15 Killzone (use 08:00+ for pre-market volume baseline)
        vol_bars = df_session_1m.between_time("08:00", "12:00")
        if vol_bars.empty:
            return []

        resampled_5m = vol_bars.resample("5min", closed="left", label="left").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }).dropna()

        vol_sma = resampled_5m["Volume"].rolling(window=self.volume_sma_bars, min_periods=3).mean()

        killzone_5m = resampled_5m.between_time("09:30", "10:15")
        if len(killzone_5m) < 2:
            return []

        max_penetration = self.max_sweep_atr_ratio * daily_atr

        # Iterate over 5m bars in killzone
        for i in range(len(killzone_5m)):
            bar_start = killzone_5m.index[i]
            bar_5m = killzone_5m.iloc[i]
            bar_end_time = bar_start + pd.Timedelta(minutes=5)

            h = float(bar_5m["High"])
            l = float(bar_5m["Low"])
            c = float(bar_5m["Close"])
            rng = h - l
            if rng <= 0:
                continue

            v_sma = vol_sma.loc[bar_start] if bar_start in vol_sma.index else np.nan
            vol_ok = (not np.isnan(v_sma) and bar_5m["Volume"] >= self.volume_factor * v_sma)

            # Check Bullish Sweep of London Low
            if bullish_inventory_ok and l < (lon_l - self.min_sweep_pts):
                depth = lon_l - l
                if depth <= max_penetration:
                    # Check reversal close in this candle or next candle
                    candidate_bars = [bar_5m]
                    if i + 1 < len(resampled_5m):
                        candidate_bars.append(resampled_5m.iloc[i+1])

                    for idx_c, c_bar in enumerate(candidate_bars):
                        c_close = float(c_bar["Close"])
                        c_high = float(c_bar["High"])
                        c_low = float(c_bar["Low"])
                        c_rng = c_high - c_low
                        c_time_end = c_bar.name + pd.Timedelta(minutes=5)

                        # Candle closes back inside range (Close > LON_L)
                        # Closes in top 40% of its range ((Close - Low)/Range >= 0.60)
                        if c_rng > 0 and c_close > lon_l and ((c_close - c_low) / c_rng >= 0.60):
                            c_v_sma = vol_sma.loc[c_bar.name] if c_bar.name in vol_sma.index else np.nan
                            if not np.isnan(c_v_sma) and c_bar["Volume"] >= self.volume_factor * c_v_sma:
                                # Confirmed Bullish Reversal!
                                future_1m = df_session_1m.loc[df_session_1m.index >= c_time_end]
                                if not future_1m.empty:
                                    open_1m = float(future_1m.iloc[0]["Open"])
                                    entry_price = open_1m + 0.50
                                    sweep_min = min(l, float(c_bar["Low"]))
                                    sl_price = sweep_min - 4.0
                                    sl_dist = entry_price - sl_price

                                    if self.min_sl_dist <= sl_dist <= self.max_sl_dist:
                                        tp_target = entry_price + max(self.target_r_multiple * sl_dist, 0.85 * lon_w)
                                        return [TradeSignal(
                                            strategy_id="H2",
                                            strategy_name="H2_NY_London_Sweep_v2",
                                            timestamp=c_time_end,
                                            execution_time=c_time_end,
                                            direction=1,
                                            entry_price=entry_price,
                                            stop_loss_price=sl_price,
                                            tp1_price=tp_target,
                                            sl_dist=sl_dist,
                                            scale_out_tp1=False,
                                            be_trigger_r=self.be_trigger_r,
                                            hard_time_exit="12:00:00",
                                            metadata={"lon_w": lon_w, "depth": depth, "sl_dist": sl_dist}
                                        )]

            # Check Bearish Sweep of London High
            if bearish_inventory_ok and h > (lon_h + self.min_sweep_pts):
                depth = h - lon_h
                if depth <= max_penetration:
                    candidate_bars = [bar_5m]
                    if i + 1 < len(resampled_5m):
                        candidate_bars.append(resampled_5m.iloc[i+1])

                    for idx_c, c_bar in enumerate(candidate_bars):
                        c_close = float(c_bar["Close"])
                        c_high = float(c_bar["High"])
                        c_low = float(c_bar["Low"])
                        c_rng = c_high - c_low
                        c_time_end = c_bar.name + pd.Timedelta(minutes=5)

                        # Closes back inside range (Close < LON_H)
                        # Closes in bottom 40% of range ((High - Close)/Range >= 0.60)
                        if c_rng > 0 and c_close < lon_h and ((c_high - c_close) / c_rng >= 0.60):
                            c_v_sma = vol_sma.loc[c_bar.name] if c_bar.name in vol_sma.index else np.nan
                            if not np.isnan(c_v_sma) and c_bar["Volume"] >= self.volume_factor * c_v_sma:
                                future_1m = df_session_1m.loc[df_session_1m.index >= c_time_end]
                                if not future_1m.empty:
                                    open_1m = float(future_1m.iloc[0]["Open"])
                                    entry_price = open_1m - 0.50
                                    sweep_max = max(h, float(c_bar["High"]))
                                    sl_price = sweep_max + 4.0
                                    sl_dist = sl_price - entry_price

                                    if self.min_sl_dist <= sl_dist <= self.max_sl_dist:
                                        tp_target = entry_price - max(self.target_r_multiple * sl_dist, 0.85 * lon_w)
                                        return [TradeSignal(
                                            strategy_id="H2",
                                            strategy_name="H2_NY_London_Sweep_v2",
                                            timestamp=c_time_end,
                                            execution_time=c_time_end,
                                            direction=-1,
                                            entry_price=entry_price,
                                            stop_loss_price=sl_price,
                                            tp1_price=tp_target,
                                            sl_dist=sl_dist,
                                            scale_out_tp1=False,
                                            be_trigger_r=self.be_trigger_r,
                                            hard_time_exit="12:00:00",
                                            metadata={"lon_w": lon_w, "depth": depth, "sl_dist": sl_dist}
                                        )]

        return []
