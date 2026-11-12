"""
Module 3: Zarattini-Baltussen Noise-Band Momentum with Seeck (2026) Nasdaq Asymmetry
Author: Principal Quantitative Architect & Financial Engineer

Academic & Quantitative Foundations:
  - Zarattini, Aziz, & Barbon (2024, SSRN 4824172): "An Effective Intraday Momentum Strategy"
  - Seeck (2026, SSRN 7364204): "Nasdaq Asymmetry & Directional Breakout Conditioning"
  - Baltussen, Swinkels, & van Vliet (2021, JFE): "Predictable Intraday Rebalancing Flows"

Phase 4 Institutional Architecture:
  - Strict Execution Window: 10:30:00 to 13:30:00 NY (allowing 60m Initial Balance 09:30-10:30 and 10:00 AM macro news to settle).
  - Single Trade Per Day: Occam's razor (eliminating Phase 3 false-breakout noise and Entry 2 whipsaws).
  - Robust Noise Envelope: Kd = 1.25 multiplier (eliminating the tight Kd=1.12 false-breakout trigger).
  - Seeck (2026) Long/Short Asymmetry Filter:
      * Long: Close > UpperBand AND Close > VWAP_RTH AND EMA_5m(9) > VWAP_RTH AND Vol_5m >= 1.10 * SMA(Vol_5m, 12).
      * Short: Close < LowerBand AND Close < VWAP_RTH AND EMA_5m(9) < VWAP_RTH AND Vol_5m >= 1.10 * SMA(Vol_5m, 12)
               AND Bearish Structure: (Open_0930 <= PrevClose_1600 OR Close_5m < PrevDay_Low OR ATR_14 >= SMA(ATR_14, 20)).
  - Risk & Exit Geometry:
      * Initial Structural Hard Stop: SL_dist = clip(abs(Entry - VWAP_RTH) + 4.0, 22.0, 48.0) NQ points.
      * Zero Price-Based Breakeven Scratch: (be_trigger_r = None, scale_out_tp1 = False).
      * Confirmed 5-Minute Bar-Close Structural Trailing Stop:
          - Long: Exit on confirmed 5m bar close below min(UpperBand, VWAP_RTH).
          - Short: Exit on confirmed 5m bar close above max(LowerBand, VWAP_RTH).
      * Hard Session Flatten at 15:44:00 NY.
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
        vol_sma_bars: int = 12,
        use_seeck_asymmetry: bool = True,
        start_time: str = "10:15",
        end_time: str = "13:30",
        kd: float = 1.10,
        vol_filter: bool = False,
        seeck_mode: str = "surgical"
    ):
        super().__init__(name="H3_Zarattini_NoiseBand_Momentum", strategy_id="H3")
        self.min_sl_dist = min_sl_dist
        self.max_sl_dist = max_sl_dist
        self.sl_vwap_buffer = sl_vwap_buffer
        self.ema_span = ema_span
        self.vol_factor = vol_factor
        self.vol_sma_bars = vol_sma_bars
        self.use_seeck_asymmetry = use_seeck_asymmetry
        self.start_time = start_time
        self.end_time = end_time
        self.kd = kd
        self.vol_filter = vol_filter
        self.seeck_mode = seeck_mode

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

        # Phase 6 Calibrated Execution Window: self.start_time to self.end_time (default 10:15 - 13:30 NY)
        eval_window_5m = resampled_5m.between_time(self.start_time, self.end_time)
        if eval_window_5m.empty:
            return []

        # Reference daily features for Seeck Asymmetry & Bull-Trap Guard
        rth_sub = df_session_1m.between_time("09:30", "09:30")
        d_open = float(rth_sub.iloc[0]["Open"]) if not rth_sub.empty else prev_close
        prev_high = preprocessor.prev_day_high_map.get(session_date) if preprocessor else None
        prev_low = preprocessor.prev_day_low_map.get(session_date) if preprocessor else None
        sma_atr20 = preprocessor.daily_atr_sma20_map.get(session_date, 0.0) if preprocessor and hasattr(preprocessor, "daily_atr_sma20_map") else daily_atr

        signals: List[TradeSignal] = []

        for bar_start, bar_5m in eval_window_5m.iterrows():
            bar_end_time = bar_start + pd.Timedelta(minutes=5)
            time_str = bar_end_time.strftime("%H:%M")

            # Lookup active Zarattini Noise Band with calibrated Kd
            band = preprocessor.get_noise_band(session_date, time_str, kd=self.kd)
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

            vol_ok = True
            if self.vol_filter:
                vol_ok = (vol_sma_val > 0 and vol_5m >= self.vol_factor * vol_sma_val)

            # --- LONG BREAKOUT ---
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
                        strategy_name="H3_Zarattini_NoiseBand_Momentum",
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
                        trail_band_mode="WIDE",
                        trail_activation_r=0.0,
                        hard_time_exit="15:44:00",
                        metadata={
                            "wave": 1,
                            "upper_band": upper_band,
                            "vwap": curr_vwap,
                            "sl_dist": sl_dist
                        }
                    ))
                    break  # Exactly 1 trade per day!

            # --- SHORT BREAKOUT ---
            elif close_5m < lower_band and close_5m < curr_vwap and curr_ema < curr_vwap and vol_ok:
                short_valid = True
                if self.seeck_mode == "surgical":
                    # Surgical Bull-Trap Guard: ONLY block shorts when gapping up above yesterday's high in low-vol regime
                    is_bull_trap = (prev_high is not None and d_open > prev_high) and (daily_atr < sma_atr20)
                    if is_bull_trap:
                        short_valid = False
                elif self.seeck_mode == "strict" and self.use_seeck_asymmetry:
                    # Bearish Intraday/Daily Structure: Non-gap-up day OR close < prev_low OR expanding ATR
                    is_bearish = (d_open <= prev_close) or (prev_low is not None and close_5m < prev_low) or (daily_atr >= sma_atr20)
                    if not is_bearish:
                        short_valid = False

                if short_valid:
                    future_1m = df_session_1m.loc[df_session_1m.index >= bar_end_time]
                    if not future_1m.empty:
                        open_1m = float(future_1m.iloc[0]["Open"])
                        entry_price = open_1m - 0.50
                        raw_sl = abs(curr_vwap - entry_price) + self.sl_vwap_buffer
                        sl_dist = float(np.clip(raw_sl, self.min_sl_dist, self.max_sl_dist))
                        sl_price = entry_price + sl_dist

                        signals.append(TradeSignal(
                            strategy_id="H3",
                            strategy_name="H3_Zarattini_NoiseBand_Momentum",
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
                            trail_band_mode="WIDE",
                            trail_activation_r=0.0,
                            hard_time_exit="15:44:00",
                            metadata={
                                "wave": 1,
                                "lower_band": lower_band,
                                "vwap": curr_vwap,
                                "sl_dist": sl_dist
                            }
                        ))
                        break  # Exactly 1 trade per day!

        return signals
