"""
Module 4: Two-Wave Late-Day Leveraged ETF Flow & 0DTE Gamma Pin Module (strategies/h4_letf_1500_rebalance.py)
Author: Principal Quantitative Architect & Financial Engineer

Empirical Foundations:
  - Gao, Han, Li, & Zhou (2018, JFE): "Market Intraday Momentum"
  - Baltussen, Swinkels, & van Vliet (2021, JFE): "Predictable Rebalancing Flows"
  - Zarattini (2024, SSRN 4824172): "An Effective Intraday Momentum Strategy"

Features (Phase 3 Upgraded):
  - Two-Wave Execution:
      * Wave 1 (14:30 NY): Second-to-last half hour pre-rebalancing flow.
        Trigger: abs(Close_1430 - PrevClose) / ATR_14 >= 0.58 (or abs(r_ROD) >= 0.0080),
                 aligned with r_1 (09:30-10:00 return) and VWAP_RTH, RangePos >= 0.78 (Long) or <= 0.22 (Short).
      * Wave 2 (15:00 NY, if not already in Wave 1):
        Same condition, plus joint momentum confirmation: sign(Close_1500 - Close_1430) == sign(r_1).
  - 0DTE 50-Point Strike Gamma Pin Filter:
      * abs(Close_t - S_50) <= 8.0 AND 15m range < 15.0 pts -> Pinned, skip trade.
  - Risk & Exit Geometry:
      * Initial Hard Stop: SL_dist = clip(0.15 * (DayHigh_t - DayLow_t), 18.0, 36.0) NQ points.
      * Zero Breakeven scratch! (be_trigger_r = None, scale_out_tp1 = False).
      * Structural Trail: Confirmed 5-minute bar close across VWAP_RTH against position.
      * Hard flatten at 15:44:00 NY.
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
        norm_move_thresh: float = 0.58,
        rod_thresh: float = 0.0080,
        range_pos_long: float = 0.78,
        range_pos_short: float = 0.22,
        pin_strike_dist_max: float = 8.0,
        pin_range_max: float = 15.0,
        min_sl_dist: float = 18.0,
        max_sl_dist: float = 36.0,
        sl_range_factor: float = 0.15
    ):
        super().__init__(name="H4_TwoWave_LETF_Rebalance_v3", strategy_id="H4")
        self.enable_wave1 = enable_wave1
        self.norm_move_thresh = norm_move_thresh
        self.rod_thresh = rod_thresh
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

        signals: List[TradeSignal] = []

        # Extract 14:30 close if available
        time_1430_mask = rth_bars.index.time == pd.Timestamp("14:30:00").time()
        close_1430 = float(rth_bars.loc[time_1430_mask].iloc[-1]["Close"]) if time_1430_mask.any() else None

        # --- WAVE 1 EVALUATION (14:30:00 NY) ---
        wave1_fired = False
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
                norm_move = disp_pts / daily_atr
                r_rod_1430 = disp_pts / prev_close
                range_pos_1430 = (close_1430_val - low_1430) / range_1430
                vwap_1430 = float(vwap_series.loc[t_1430])

                pinned_1430 = is_gamma_pinned(close_1430_val, "14:15", "14:30")

                if not pinned_1430:
                    sl_dist_w1 = float(np.clip(self.sl_range_factor * range_1430, self.min_sl_dist, self.max_sl_dist))
                    future_bars_w1 = df_session_1m.loc[df_session_1m.index > t_1430]

                    # Long Wave 1
                    if (norm_move >= self.norm_move_thresh or r_rod_1430 >= self.rod_thresh) and r_1 > 0 and close_1430_val > vwap_1430 and range_pos_1430 >= self.range_pos_long:
                        if not future_bars_w1.empty:
                            exec_bar = future_bars_w1.iloc[0]
                            exec_time = future_bars_w1.index[0]
                            entry_price = float(exec_bar["Open"]) + 0.50
                            signals.append(TradeSignal(
                                strategy_id="H4",
                                strategy_name="H4_TwoWave_LETF_Rebalance_v3",
                                timestamp=t_1430,
                                execution_time=exec_time,
                                direction=1,
                                entry_price=entry_price,
                                stop_loss_price=entry_price - sl_dist_w1,
                                tp1_price=None,  # 100% full position to 15:44
                                sl_dist=sl_dist_w1,
                                scale_out_tp1=False,
                                be_trigger_r=None,  # No BE scratch!
                                trail_rule="VWAP",
                                trail_activation_r=0.0,
                                hard_time_exit="15:44:00",
                                metadata={"wave": 1, "r_rod": r_rod_1430, "range_pos": range_pos_1430}
                            ))
                            wave1_fired = True

                    # Short Wave 1
                    elif (norm_move <= -self.norm_move_thresh or r_rod_1430 <= -self.rod_thresh) and r_1 < 0 and close_1430_val < vwap_1430 and range_pos_1430 <= self.range_pos_short:
                        if not future_bars_w1.empty:
                            exec_bar = future_bars_w1.iloc[0]
                            exec_time = future_bars_w1.index[0]
                            entry_price = float(exec_bar["Open"]) - 0.50
                            signals.append(TradeSignal(
                                strategy_id="H4",
                                strategy_name="H4_TwoWave_LETF_Rebalance_v3",
                                timestamp=t_1430,
                                execution_time=exec_time,
                                direction=-1,
                                entry_price=entry_price,
                                stop_loss_price=entry_price + sl_dist_w1,
                                tp1_price=None,
                                sl_dist=sl_dist_w1,
                                scale_out_tp1=False,
                                be_trigger_r=None,
                                trail_rule="VWAP",
                                trail_activation_r=0.0,
                                hard_time_exit="15:44:00",
                                metadata={"wave": 1, "r_rod": r_rod_1430, "range_pos": range_pos_1430}
                            ))
                            wave1_fired = True

        # --- WAVE 2 EVALUATION (15:00:00 NY) ---
        # Only evaluate Wave 2 if Wave 1 did not trigger
        if not wave1_fired:
            time_1500_mask = rth_bars.index.time == pd.Timestamp("15:00:00").time()
            if time_1500_mask.any():
                bar_1500 = rth_bars.loc[time_1500_mask].iloc[-1]
                t_1500 = rth_bars.loc[time_1500_mask].index[-1]
                close_1500 = float(bar_1500["Close"])

                bars_to_1500 = rth_bars.loc[rth_bars.index <= t_1500]
                high_1500 = float(bars_to_1500["High"].max())
                low_1500 = float(bars_to_1500["Low"].min())
                range_1500 = high_1500 - low_1500

                if range_1500 > 0:
                    disp_pts_1500 = close_1500 - prev_close
                    norm_move_1500 = disp_pts_1500 / daily_atr
                    r_rod_1500 = disp_pts_1500 / prev_close
                    range_pos_1500 = (close_1500 - low_1500) / range_1500
                    vwap_1500 = float(vwap_series.loc[t_1500])

                    close_1430_ref = close_1430 if close_1430 is not None else close_1500
                    m_diff = close_1500 - close_1430_ref
                    joint_momentum_confirmed = (m_diff * r_1 >= 0)

                    pinned_1500 = is_gamma_pinned(close_1500, "14:45", "15:00")

                    if not pinned_1500 and joint_momentum_confirmed:
                        sl_dist_w2 = float(np.clip(self.sl_range_factor * range_1500, self.min_sl_dist, self.max_sl_dist))
                        future_bars_w2 = df_session_1m.loc[df_session_1m.index > t_1500]

                        # Wave 2 Long
                        if (norm_move_1500 >= self.norm_move_thresh or r_rod_1500 >= self.rod_thresh) and r_1 > 0 and range_pos_1500 >= self.range_pos_long and close_1500 > vwap_1500:
                            if not future_bars_w2.empty:
                                exec_bar = future_bars_w2.iloc[0]
                                exec_time = future_bars_w2.index[0]
                                entry_price = float(exec_bar["Open"]) + 0.50
                                signals.append(TradeSignal(
                                    strategy_id="H4",
                                    strategy_name="H4_TwoWave_LETF_Rebalance_v3",
                                    timestamp=t_1500,
                                    execution_time=exec_time,
                                    direction=1,
                                    entry_price=entry_price,
                                    stop_loss_price=entry_price - sl_dist_w2,
                                    tp1_price=None,
                                    sl_dist=sl_dist_w2,
                                    scale_out_tp1=False,
                                    be_trigger_r=None,
                                    trail_rule="VWAP",
                                    trail_activation_r=0.0,
                                    hard_time_exit="15:44:00",
                                    metadata={"wave": 2, "r_rod": r_rod_1500, "range_pos": range_pos_1500}
                                ))

                        # Wave 2 Short
                        elif (norm_move_1500 <= -self.norm_move_thresh or r_rod_1500 <= -self.rod_thresh) and r_1 < 0 and range_pos_1500 <= self.range_pos_short and close_1500 < vwap_1500:
                            if not future_bars_w2.empty:
                                exec_bar = future_bars_w2.iloc[0]
                                exec_time = future_bars_w2.index[0]
                                entry_price = float(exec_bar["Open"]) - 0.50
                                signals.append(TradeSignal(
                                    strategy_id="H4",
                                    strategy_name="H4_TwoWave_LETF_Rebalance_v3",
                                    timestamp=t_1500,
                                    execution_time=exec_time,
                                    direction=-1,
                                    entry_price=entry_price,
                                    stop_loss_price=entry_price + sl_dist_w2,
                                    tp1_price=None,
                                    sl_dist=sl_dist_w2,
                                    scale_out_tp1=False,
                                    be_trigger_r=None,
                                    trail_rule="VWAP",
                                    trail_activation_r=0.0,
                                    hard_time_exit="15:44:00",
                                    metadata={"wave": 2, "r_rod": r_rod_1500, "range_pos": range_pos_1500}
                                ))

        return signals
