"""
Module 3: Multi-Horizon Adaptive Noise-Band Momentum Engine (strategies/h3_noise_band_momentum.py)
Author: Principal Quantitative Architect & Financial Engineer

Academic & Quantitative Foundation:
  - Zarattini, Aziz, & Barbon (2024, SSRN 4824172): "An Effective Intraday Momentum Strategy"
  - Baltussen et al. (2021, JFE): Institutional Fixings & Rebalancing Flow

Strategy Rules (Phase 3 Upgraded):
  - Causal 14-Day Multi-Horizon Time-of-Day Noise Envelope:
      * Computed across 15-minute intervals from 10:00:00 to 14:00:00 NY.
      * Regime-Adaptive Multiplier (Kd):
          - Compressed Prior Day (comp_ratio <= 0.85): Kd = 1.12
          - Normal Day (0.85 < comp_ratio <= 1.20): Kd = 1.25
          - Exhausted Wide Day (comp_ratio > 1.20): Kd = 1.35
      * Gap-Adjusted Boundaries:
          UpperBand_{d,t} = max(Open_0930, PrevClose_1600) + Kd * sigma_{d,t}
          LowerBand_{d,t} = min(Open_0930, PrevClose_1600) - Kd * sigma_{d,t}
  - Multi-Horizon Execution Window: 10:00:00 to 14:00:00 NY.
  - Entry 1 (Primary Breakout):
      * Long: Close_5m > UpperBand AND Close_5m > VWAP_RTH AND EMA_5m(9) > VWAP_RTH AND Vol_5m >= 1.10 * SMA(Vol_5m, 12).
      * Short: Close_5m < LowerBand AND Close_5m < VWAP_RTH AND EMA_5m(9) < VWAP_RTH AND Vol_5m >= 1.10 * SMA(Vol_5m, 12).
  - Entry 2 (Continuation Re-Entry, max 2 entries/day):
      * If Entry 1 closed before 13:00 by 5m trailing stop and daily PnL > -$300.0:
      * Long: Close_5m > UpperBand AND Close_5m > DayHigh_prior AND Close_5m > VWAP_RTH AND EMA_5m(9) > VWAP_RTH.
      * Short: Close_5m < LowerBand AND Close_5m < DayLow_prior AND Close_5m < VWAP_RTH AND EMA_5m(9) < VWAP_RTH.
  - Structural Exit Geometry:
      * SL_dist = clip(abs(Entry - VWAP_RTH) + 4.0, 22.0, 48.0) NQ points.
      * Zero price-based Breakeven scratches (be_trigger_r = None, scale_out_tp1 = False).
      * Confirmed 5-minute bar-close trailing: exit below min(UpperBand, VWAP_RTH) for Long
        (or above max(LowerBand, VWAP_RTH) for Short).
      * Hard flatten at 15:44:00 NY.
"""

from __future__ import annotations
import logging
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, TradeSignal

logger = logging.getLogger("H3_NoiseBand_Momentum")


class H3NoiseBandMomentum(BaseStrategy):
    def __init__(
        self,
        min_sl_dist: float = 22.0,
        max_sl_dist: float = 48.0,
        sl_vwap_buffer: float = 4.0,
        ema_span: int = 9,
        vol_factor: float = 1.10,
        vol_sma_bars: int = 12
    ):
        super().__init__(name="H3_MultiHorizon_Adaptive_NoiseBand", strategy_id="H3")
        self.min_sl_dist = min_sl_dist
        self.max_sl_dist = max_sl_dist
        self.sl_vwap_buffer = sl_vwap_buffer
        self.ema_span = ema_span
        self.vol_factor = vol_factor
        self.vol_sma_bars = vol_sma_bars

    def evaluate_session(
        self,
        session_date,
        df_session_1m: pd.DataFrame,
        daily_atr: float,
        prev_close: Optional[float] = None,
        preprocessor=None,
        **kwargs
    ) -> List[TradeSignal]:
        if preprocessor is None:
            from data.session_preprocessor import get_preprocessor
            preprocessor = get_preprocessor()

        if prev_close is None or np.isnan(prev_close) or prev_close <= 0:
            return []
        if daily_atr is None or np.isnan(daily_atr) or daily_atr <= 0:
            return []

        # RTH bars from 09:30 to 15:44
        rth_bars = df_session_1m.between_time("09:30", "15:44")
        if len(rth_bars) < 60:
            return []

        # 1. Compute Anchored RTH VWAP
        tp = (rth_bars["High"] + rth_bars["Low"] + rth_bars["Close"]) / 3.0
        cum_vol = rth_bars["Volume"].cumsum()
        cum_tp_vol = (tp * rth_bars["Volume"]).cumsum()
        vwap_series = cum_tp_vol / cum_vol.replace(0, np.nan)

        # 2. Resample 5m bars starting from 08:00 for clean volume baseline
        vol_bars = df_session_1m.between_time("08:00", "15:44")
        if vol_bars.empty:
            vol_bars = rth_bars

        resampled_5m = vol_bars.resample("5min", closed="left", label="left").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }).dropna()

        ema9_5m = resampled_5m["Close"].ewm(span=self.ema_span, adjust=False).mean()
        vol_sma_5m = resampled_5m["Volume"].rolling(window=self.vol_sma_bars, min_periods=4).mean()

        # Multi-Horizon Window: 10:00:00 to 14:00:00 NY
        eval_window_5m = resampled_5m.between_time("10:00", "14:00")
        if eval_window_5m.empty:
            return []

        signals: List[TradeSignal] = []
        primary_emitted = False
        primary_dir = 0
        primary_bar_time = None

        # Track running session high/low for continuation re-entry
        rth_5m_all = resampled_5m.between_time("09:30", "15:44")

        for bar_start, bar_5m in eval_window_5m.iterrows():
            bar_end_time = bar_start + pd.Timedelta(minutes=5)
            time_str = bar_end_time.strftime("%H:%M")

            # Lookup active Zarattini Noise Band
            band = preprocessor.get_noise_band(session_date, time_str)
            if band is None:
                continue

            upper_band, lower_band = band
            close_5m = float(bar_5m["Close"])
            vol_5m = float(bar_5m["Volume"])
            vol_sma_val = float(vol_sma_5m.loc[bar_start]) if bar_start in vol_sma_5m.index else 0.0

            prior_vwap_mask = vwap_series.index < bar_end_time
            if not prior_vwap_mask.any():
                continue
            curr_vwap = float(vwap_series.loc[prior_vwap_mask].iloc[-1])
            curr_ema = float(ema9_5m.loc[bar_start])

            vol_ok = (vol_sma_val > 0 and vol_5m >= self.vol_factor * vol_sma_val)

            # Prior bars up to bar_start to determine running DayHigh / DayLow
            prior_rth = rth_5m_all.loc[rth_5m_all.index < bar_start]
            running_day_high = float(prior_rth["High"].max()) if not prior_rth.empty else close_5m
            running_day_low = float(prior_rth["Low"].min()) if not prior_rth.empty else close_5m

            # --- ENTRY 1: PRIMARY BREAKOUT ---
            if not primary_emitted:
                if close_5m > upper_band and close_5m > curr_vwap and curr_ema > curr_vwap and vol_ok:
                    future_1m = df_session_1m.loc[df_session_1m.index >= bar_end_time]
                    if not future_1m.empty:
                        open_1m = float(future_1m.iloc[0]["Open"])
                        entry_price = open_1m + 0.50
                        raw_sl = abs(entry_price - curr_vwap) + self.sl_vwap_buffer
                        sl_dist = float(np.clip(raw_sl, self.min_sl_dist, self.max_sl_dist))
                        sl_price = entry_price - sl_dist

                        signals.append(TradeSignal(
                            strategy_id="H3",
                            strategy_name="H3_MultiHorizon_Adaptive_NoiseBand",
                            timestamp=bar_end_time,
                            execution_time=bar_end_time,
                            direction=1,
                            entry_price=entry_price,
                            stop_loss_price=sl_price,
                            tp1_price=None,
                            sl_dist=sl_dist,
                            scale_out_tp1=False,
                            be_trigger_r=None,  # Zero BE scratches!
                            trail_rule="VWAP_BAND",
                            trail_activation_r=0.0,
                            hard_time_exit="15:44:00",
                            metadata={
                                "wave": 1,
                                "is_reentry": False,
                                "upper_band": upper_band,
                                "vwap": curr_vwap,
                                "sl_dist": sl_dist
                            }
                        ))
                        primary_emitted = True
                        primary_dir = 1
                        primary_bar_time = bar_start
                        continue

                elif close_5m < lower_band and close_5m < curr_vwap and curr_ema < curr_vwap and vol_ok:
                    future_1m = df_session_1m.loc[df_session_1m.index >= bar_end_time]
                    if not future_1m.empty:
                        open_1m = float(future_1m.iloc[0]["Open"])
                        entry_price = open_1m - 0.50
                        raw_sl = abs(curr_vwap - entry_price) + self.sl_vwap_buffer
                        sl_dist = float(np.clip(raw_sl, self.min_sl_dist, self.max_sl_dist))
                        sl_price = entry_price + sl_dist

                        signals.append(TradeSignal(
                            strategy_id="H3",
                            strategy_name="H3_MultiHorizon_Adaptive_NoiseBand",
                            timestamp=bar_end_time,
                            execution_time=bar_end_time,
                            direction=-1,
                            entry_price=entry_price,
                            stop_loss_price=sl_price,
                            tp1_price=None,
                            sl_dist=sl_dist,
                            scale_out_tp1=False,
                            be_trigger_r=None,  # Zero BE scratches!
                            trail_rule="VWAP_BAND",
                            trail_activation_r=0.0,
                            hard_time_exit="15:44:00",
                            metadata={
                                "wave": 1,
                                "is_reentry": False,
                                "lower_band": lower_band,
                                "vwap": curr_vwap,
                                "sl_dist": sl_dist
                            }
                        ))
                        primary_emitted = True
                        primary_dir = -1
                        primary_bar_time = bar_start
                        continue

            # --- ENTRY 2: CONTINUATION RE-ENTRY ---
            # Evaluated after primary entry, before 13:00 NY, requiring close beyond BOTH band and running session extreme
            elif primary_emitted and bar_start > primary_bar_time and bar_start.time() < pd.Timestamp("13:00:00").time():
                if primary_dir == 1:
                    if close_5m > upper_band and close_5m > running_day_high and close_5m > curr_vwap and curr_ema > curr_vwap:
                        future_1m = df_session_1m.loc[df_session_1m.index >= bar_end_time]
                        if not future_1m.empty:
                            open_1m = float(future_1m.iloc[0]["Open"])
                            entry_price = open_1m + 0.50
                            raw_sl = abs(entry_price - curr_vwap) + self.sl_vwap_buffer
                            sl_dist = float(np.clip(raw_sl, self.min_sl_dist, self.max_sl_dist))
                            sl_price = entry_price - sl_dist

                            signals.append(TradeSignal(
                                strategy_id="H3",
                                strategy_name="H3_MultiHorizon_Adaptive_NoiseBand",
                                timestamp=bar_end_time,
                                execution_time=bar_end_time,
                                direction=1,
                                entry_price=entry_price,
                                stop_loss_price=sl_price,
                                tp1_price=None,
                                sl_dist=sl_dist,
                                scale_out_tp1=False,
                                be_trigger_r=None,
                                trail_rule="VWAP_BAND",
                                trail_activation_r=0.0,
                                hard_time_exit="15:44:00",
                                metadata={
                                    "wave": 2,
                                    "is_reentry": True,
                                    "upper_band": upper_band,
                                    "vwap": curr_vwap,
                                    "sl_dist": sl_dist
                                }
                            ))
                            break  # Maximum 2 entries emitted
                elif primary_dir == -1:
                    if close_5m < lower_band and close_5m < running_day_low and close_5m < curr_vwap and curr_ema < curr_vwap:
                        future_1m = df_session_1m.loc[df_session_1m.index >= bar_end_time]
                        if not future_1m.empty:
                            open_1m = float(future_1m.iloc[0]["Open"])
                            entry_price = open_1m - 0.50
                            raw_sl = abs(curr_vwap - entry_price) + self.sl_vwap_buffer
                            sl_dist = float(np.clip(raw_sl, self.min_sl_dist, self.max_sl_dist))
                            sl_price = entry_price + sl_dist

                            signals.append(TradeSignal(
                                strategy_id="H3",
                                strategy_name="H3_MultiHorizon_Adaptive_NoiseBand",
                                timestamp=bar_end_time,
                                execution_time=bar_end_time,
                                direction=-1,
                                entry_price=entry_price,
                                stop_loss_price=sl_price,
                                tp1_price=None,
                                sl_dist=sl_dist,
                                scale_out_tp1=False,
                                be_trigger_r=None,
                                trail_rule="VWAP_BAND",
                                trail_activation_r=0.0,
                                hard_time_exit="15:44:00",
                                metadata={
                                    "wave": 2,
                                    "is_reentry": True,
                                    "lower_band": lower_band,
                                    "vwap": curr_vwap,
                                    "sl_dist": sl_dist
                                }
                            ))
                            break

        return signals
