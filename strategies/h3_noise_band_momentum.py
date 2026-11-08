"""
Module 3: Zarattini-Baltussen Noise-Boundary Intraday Momentum (strategies/h3_noise_band_momentum.py)
Author: Principal Quantitative Architect

Academic Foundation:
  - Zarattini, Aziz, & Barbon (2024, SSRN 4824172): "An Effective Intraday Momentum Strategy"
  - Baltussen et al. (2021, JFE): Institutional Fixings & Rebalancing Flow

Strategy Rules:
  - Causal 14-Day Time-of-Day Noise Envelope (UpperBand_{d,t} and LowerBand_{d,t}).
  - Prime Institutional Window (10:30:00 - 13:30:00 NY, encompassing the 11:00-11:30 London Close fixing).
  - Entry Criteria:
      * Long: Close_5m > UpperBand_{d,t} AND Close_5m > VWAP_RTH AND EMA_5m(9) > VWAP_RTH.
      * Short: Close_5m < LowerBand_{d,t} AND Close_5m < VWAP_RTH AND EMA_5m(9) < VWAP_RTH.
  - Risk & Dynamic Trailing Exit:
      * SL_dist = min(45.0, max(20.0, abs(Entry - VWAP_RTH) + 5.0)).
      * scale_out_tp1 = False (Hold 100% position to capture right-tail trend runs).
      * Dynamic Trailing Exit: Confirmed 5m candle close below min(UpperBand, VWAP_RTH) for Long
        (or above max(LowerBand, VWAP_RTH) for Short), or hard flatten at 15:44:00 NY.
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
        min_sl_dist: float = 20.0,
        max_sl_dist: float = 45.0,
        sl_vwap_buffer: float = 5.0,
        ema_span: int = 9
    ):
        super().__init__(name="H3_NoiseBand_Intraday_Momentum", strategy_id="H3")
        self.min_sl_dist = min_sl_dist
        self.max_sl_dist = max_sl_dist
        self.sl_vwap_buffer = sl_vwap_buffer
        self.ema_span = ema_span

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

        rth_bars = df_session_1m.between_time("09:30", "15:44")
        if len(rth_bars) < 60:
            return []

        # 1. Compute Anchored RTH VWAP
        tp = (rth_bars["High"] + rth_bars["Low"] + rth_bars["Close"]) / 3.0
        cum_vol = rth_bars["Volume"].cumsum()
        cum_tp_vol = (tp * rth_bars["Volume"]).cumsum()
        vwap_series = cum_tp_vol / cum_vol.replace(0, np.nan)

        # 2. Resample 5m bars
        resampled_5m = rth_bars.resample("5min", closed="left", label="left").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }).dropna()

        ema9_5m = resampled_5m["Close"].ewm(span=self.ema_span, adjust=False).mean()

        # Window: 10:30:00 to 13:30:00 NY
        eval_window_5m = resampled_5m.between_time("10:30", "13:30")
        if eval_window_5m.empty:
            return []

        for bar_start, bar_5m in eval_window_5m.iterrows():
            bar_end_time = bar_start + pd.Timedelta(minutes=5)
            time_str = bar_end_time.strftime("%H:%M")

            # Lookup active Zarattini Noise Band
            band = preprocessor.get_noise_band(session_date, time_str)
            if band is None:
                continue

            upper_band, lower_band = band
            close_5m = float(bar_5m["Close"])

            prior_vwap_mask = vwap_series.index < bar_end_time
            if not prior_vwap_mask.any():
                continue
            curr_vwap = float(vwap_series.loc[prior_vwap_mask].iloc[-1])
            curr_ema = float(ema9_5m.loc[bar_start])

            # Check Long Trigger: Close_5m > UpperBand AND Close_5m > VWAP AND EMA_5m > VWAP
            if close_5m > upper_band and close_5m > curr_vwap and curr_ema > curr_vwap:
                future_1m = df_session_1m.loc[df_session_1m.index >= bar_end_time]
                if not future_1m.empty:
                    open_1m = float(future_1m.iloc[0]["Open"])
                    entry_price = open_1m + 0.50
                    raw_sl_dist = abs(entry_price - curr_vwap) + self.sl_vwap_buffer
                    sl_dist = min(self.max_sl_dist, max(self.min_sl_dist, raw_sl_dist))
                    sl_price = entry_price - sl_dist

                    return [TradeSignal(
                        strategy_id="H3",
                        strategy_name="H3_NoiseBand_Intraday_Momentum",
                        timestamp=bar_end_time,
                        execution_time=bar_end_time,
                        direction=1,
                        entry_price=entry_price,
                        stop_loss_price=sl_price,
                        tp1_price=None, # Pure asymmetric trailing / EOD exit
                        sl_dist=sl_dist,
                        scale_out_tp1=False,
                        be_trigger_r=1.5,
                        trail_rule="VWAP_BAND",
                        trail_activation_r=0.0,
                        hard_time_exit="15:44:00",
                        metadata={"upper_band": upper_band, "vwap": curr_vwap, "sl_dist": sl_dist}
                    )]

            # Check Short Trigger: Close_5m < LowerBand AND Close_5m < VWAP AND EMA_5m < VWAP
            elif close_5m < lower_band and close_5m < curr_vwap and curr_ema < curr_vwap:
                future_1m = df_session_1m.loc[df_session_1m.index >= bar_end_time]
                if not future_1m.empty:
                    open_1m = float(future_1m.iloc[0]["Open"])
                    entry_price = open_1m - 0.50
                    raw_sl_dist = abs(curr_vwap - entry_price) + self.sl_vwap_buffer
                    sl_dist = min(self.max_sl_dist, max(self.min_sl_dist, raw_sl_dist))
                    sl_price = entry_price + sl_dist

                    return [TradeSignal(
                        strategy_id="H3",
                        strategy_name="H3_NoiseBand_Intraday_Momentum",
                        timestamp=bar_end_time,
                        execution_time=bar_end_time,
                        direction=-1,
                        entry_price=entry_price,
                        stop_loss_price=sl_price,
                        tp1_price=None,
                        sl_dist=sl_dist,
                        scale_out_tp1=False,
                        be_trigger_r=1.5,
                        trail_rule="VWAP_BAND",
                        trail_activation_r=0.0,
                        hard_time_exit="15:44:00",
                        metadata={"lower_band": lower_band, "vwap": curr_vwap, "sl_dist": sl_dist}
                    )]

        return []
