"""
Module 4: Two-Wave Late-Day Leveraged ETF Flow & 0DTE Gamma Pin Module (h4_letf_1500_rebalance.py)
Author: Principal Quantitative Architect & Systems Engineer

Empirical Foundations:
  - Gao, Han, Li, & Zhou (2018, JFE): "Market Intraday Momentum"
  - Baltussen, Swinkels, & van Vliet (2021, JFE): "Predictable Rebalancing Flows"
  - Zarattini (2024, SSRN 4824172): "An Effective Intraday Momentum Strategy"

Features:
  - Two-Wave Execution:
      * Wave 1 (14:30 NY): Second-to-last half hour pre-rebalancing & short-gamma flow.
      * Wave 2 (15:00 NY): Leveraged ETF L(L-1) forced rebalancing + joint momentum.
  - Volatility-Normalized Displacement (norm_move = disp_pts / ATR_14).
  - 0DTE 50-Point Strike Gamma Pin Filter.
  - Asymmetric 100% position holding to 15:44:00 NY (No premature TP1 cap!).
  - Delayed Breakeven (+1.2R) and confirmed 5m VWAP trailing exit.
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
        enable_wave1: bool = False,
        wave1_norm_thresh: float = 0.65,
        wave2_norm_thresh: float = 0.60,
        wave2_rod_thresh: float = 0.010,
        range_pos_long: float = 0.80,
        range_pos_short: float = 0.20,
        pin_strike_dist_max: float = 8.0,
        pin_range_max: float = 15.0,
        min_sl_dist: float = 10.0,
        max_sl_dist: float = 35.0,
        sl_range_factor: float = 0.15,
        be_trigger_r: float = 1.2
    ):
        super().__init__(name="H4_TwoWave_LETF_Rebalance", strategy_id="H4")
        self.enable_wave1 = enable_wave1
        self.wave1_norm_thresh = wave1_norm_thresh
        self.wave2_norm_thresh = wave2_norm_thresh
        self.wave2_rod_thresh = wave2_rod_thresh
        self.range_pos_long = range_pos_long
        self.range_pos_short = range_pos_short
        self.pin_strike_dist_max = pin_strike_dist_max
        self.pin_range_max = pin_range_max
        self.min_sl_dist = min_sl_dist
        self.max_sl_dist = max_sl_dist
        self.sl_range_factor = sl_range_factor
        self.be_trigger_r = be_trigger_r

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
        if daily_atr is None or np.isnan(daily_atr) or daily_atr <= 0:
            return []

        # RTH bars from 09:30 to 15:44
        rth_bars = df_session_1m.between_time("09:30", "15:44")
        if len(rth_bars) < 200:
            return []

        # Open at 09:30
        open_0930 = float(rth_bars.iloc[0]["Open"])

        # Bar at 10:00:00 (r_1 calculation)
        time_1000_mask = rth_bars.index.time == pd.Timestamp("10:00:00").time()
        if not time_1000_mask.any():
            return []
        close_1000 = float(rth_bars.loc[time_1000_mask].iloc[-1]["Close"])
        r_1 = close_1000 - prev_close

        # Compute Anchored RTH VWAP
        tp = (rth_bars["High"] + rth_bars["Low"] + rth_bars["Close"]) / 3.0
        cum_vol = rth_bars["Volume"].cumsum()
        cum_tp_vol = (tp * rth_bars["Volume"]).cumsum()
        vwap_series = cum_tp_vol / cum_vol.replace(0, np.nan)

        # Helper to check 0DTE gamma pinning
        def is_gamma_pinned(close_t: float, check_start_time: str, check_end_time: str) -> bool:
            s_50 = 50.0 * round(close_t / 50.0)
            if abs(close_t - s_50) <= self.pin_strike_dist_max:
                pin_bars = df_session_1m.between_time(check_start_time, check_end_time)
                if not pin_bars.empty:
                    rng = float(pin_bars["High"].max() - pin_bars["Low"].min())
                    if rng < self.pin_range_max:
                        return True
            return False

        # Extract 14:30 close if available
        time_1430_mask = rth_bars.index.time == pd.Timestamp("14:30:00").time()
        close_1430 = float(rth_bars.loc[time_1430_mask].iloc[-1]["Close"]) if time_1430_mask.any() else None

        # --- WAVE 1 EVALUATION (14:30:00 NY - Optional Early Wave) ---
        if self.enable_wave1 and time_1430_mask.any():
            bar_1430 = rth_bars.loc[time_1430_mask].iloc[-1]
            t_1430 = rth_bars.loc[time_1430_mask].index[-1]
            close_1430_val = float(bar_1430["Close"])

            bars_to_1430 = rth_bars.loc[rth_bars.index <= t_1430]
            high_1430 = float(bars_to_1430["High"].max())
            low_1430 = float(bars_to_1430["Low"].min())
            range_1430 = high_1430 - low_1430

            if range_1430 > 0:
                disp_pts = close_1430_val - prev_close
                disp_open = close_1430_val - open_0930
                norm_move = disp_pts / daily_atr
                norm_open = disp_open / daily_atr
                range_pos_1430 = (close_1430_val - low_1430) / range_1430
                vwap_1430 = float(vwap_series.loc[t_1430])

                # Check 0DTE Gamma Pinning (14:15 - 14:30)
                pinned_1430 = is_gamma_pinned(close_1430_val, "14:15", "14:30")

                if not pinned_1430:
                    sl_dist_w1 = max(self.min_sl_dist, min(self.max_sl_dist, self.sl_range_factor * range_1430))
                    future_bars_w1 = df_session_1m.loc[df_session_1m.index > t_1430]

                    # Wave 1 Long
                    if (norm_move >= self.wave1_norm_thresh or norm_open >= self.wave1_norm_thresh) and r_1 > 0 and close_1430_val > vwap_1430 and range_pos_1430 >= self.range_pos_long:
                        if not future_bars_w1.empty:
                            exec_bar = future_bars_w1.iloc[0]
                            exec_time = future_bars_w1.index[0]
                            entry_price = float(exec_bar["Open"]) + 0.50
                            return [TradeSignal(
                                strategy_id="H4",
                                strategy_name="H4_TwoWave_LETF_Rebalance",
                                timestamp=t_1430,
                                execution_time=exec_time,
                                direction=1,
                                entry_price=entry_price,
                                stop_loss_price=entry_price - sl_dist_w1,
                                tp1_price=None,  # Hold 100% to 15:44 EOD
                                sl_dist=sl_dist_w1,
                                scale_out_tp1=False,
                                be_trigger_r=self.be_trigger_r,
                                trail_rule="VWAP",
                                hard_time_exit="15:44:00",
                                metadata={"wave": 1, "norm_move": norm_move, "range_pos": range_pos_1430}
                            )]

                    # Wave 1 Short
                    elif (norm_move <= -self.wave1_norm_thresh or norm_open <= -self.wave1_norm_thresh) and r_1 < 0 and close_1430_val < vwap_1430 and range_pos_1430 <= self.range_pos_short:
                        if not future_bars_w1.empty:
                            exec_bar = future_bars_w1.iloc[0]
                            exec_time = future_bars_w1.index[0]
                            entry_price = float(exec_bar["Open"]) - 0.50
                            return [TradeSignal(
                                strategy_id="H4",
                                strategy_name="H4_TwoWave_LETF_Rebalance",
                                timestamp=t_1430,
                                execution_time=exec_time,
                                direction=-1,
                                entry_price=entry_price,
                                stop_loss_price=entry_price + sl_dist_w1,
                                tp1_price=None,
                                sl_dist=sl_dist_w1,
                                scale_out_tp1=False,
                                be_trigger_r=self.be_trigger_r,
                                trail_rule="VWAP",
                                hard_time_exit="15:44:00",
                                metadata={"wave": 1, "norm_move": norm_move, "range_pos": range_pos_1430}
                            )]

        # --- WAVE 2 EVALUATION (15:00:00 NY - Primary Rebalancing Execution) ---
        time_1500_mask = rth_bars.index.time == pd.Timestamp("15:00:00").time()
        if not time_1500_mask.any():
            return []

        bar_1500 = rth_bars.loc[time_1500_mask].iloc[-1]
        t_1500 = rth_bars.loc[time_1500_mask].index[-1]
        close_1500 = float(bar_1500["Close"])

        bars_to_1500 = rth_bars.loc[rth_bars.index <= t_1500]
        high_1500 = float(bars_to_1500["High"].max())
        low_1500 = float(bars_to_1500["Low"].min())
        range_1500 = high_1500 - low_1500
        if range_1500 <= 0:
            return []

        disp_pts_1500 = close_1500 - prev_close
        norm_move_1500 = disp_pts_1500 / daily_atr
        r_rod = disp_pts_1500 / prev_close
        range_pos_1500 = (close_1500 - low_1500) / range_1500
        vwap_1500 = float(vwap_series.loc[t_1500])

        # Check 0DTE Gamma Pinning (14:45 - 15:00)
        pinned_1500 = is_gamma_pinned(close_1500, "14:45", "15:00")
        if pinned_1500:
            return []

        future_bars_w2 = df_session_1m.loc[df_session_1m.index > t_1500]
        if future_bars_w2.empty:
            return []

        sl_dist_w2 = max(self.min_sl_dist, min(self.max_sl_dist, self.sl_range_factor * range_1500))

        # Wave 2 Long: r_rod >= +1.0% (or norm_move >= 0.60), r_1 > 0, RangePos >= 0.80
        if (r_rod >= self.wave2_rod_thresh or norm_move_1500 >= self.wave2_norm_thresh) and r_1 > 0 and range_pos_1500 >= self.range_pos_long:
            exec_bar = future_bars_w2.iloc[0]
            exec_time = future_bars_w2.index[0]
            entry_price = float(exec_bar["Open"]) + 0.50
            return [TradeSignal(
                strategy_id="H4",
                strategy_name="H4_TwoWave_LETF_Rebalance",
                timestamp=t_1500,
                execution_time=exec_time,
                direction=1,
                entry_price=entry_price,
                stop_loss_price=entry_price - sl_dist_w2,
                tp1_price=None,  # Hold 100% full position to 15:44 EOD
                sl_dist=sl_dist_w2,
                scale_out_tp1=False,
                be_trigger_r=self.be_trigger_r,
                trail_rule="VWAP",
                hard_time_exit="15:44:00",
                metadata={"wave": 2, "r_rod": r_rod, "range_pos": range_pos_1500}
            )]

        # Wave 2 Short: r_rod <= -1.0% (or norm_move <= -0.60), r_1 < 0, RangePos <= 0.20
        elif (r_rod <= -self.wave2_rod_thresh or norm_move_1500 <= -self.wave2_norm_thresh) and r_1 < 0 and range_pos_1500 <= self.range_pos_short:
            exec_bar = future_bars_w2.iloc[0]
            exec_time = future_bars_w2.index[0]
            entry_price = float(exec_bar["Open"]) - 0.50
            return [TradeSignal(
                strategy_id="H4",
                strategy_name="H4_TwoWave_LETF_Rebalance",
                timestamp=t_1500,
                execution_time=exec_time,
                direction=-1,
                entry_price=entry_price,
                stop_loss_price=entry_price + sl_dist_w2,
                tp1_price=None,  # Hold 100% full position to 15:44 EOD
                sl_dist=sl_dist_w2,
                scale_out_tp1=False,
                be_trigger_r=self.be_trigger_r,
                trail_rule="VWAP",
                hard_time_exit="15:44:00",
                metadata={"wave": 2, "r_rod": r_rod, "range_pos": range_pos_1500}
            )]

        return []
