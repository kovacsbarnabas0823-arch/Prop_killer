"""
Institutional 5-Minute ORB Strategy Engine with 12 EMA Filter
Target: CME Micro E-mini Nasdaq-100 (MNQ) Futures
Prop Firm Focus: Tradeify $50K EOD Trailing Drawdown ($2,000 Barrier, $3,000 Target)
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger("Institutional5mORB")


@dataclass
class TradeSignal:
    date: object
    action: str  # "LONG", "SHORT", "FLAT"
    entry_time: object
    entry_price: float
    initial_sl: float
    risk_pts: float
    target_price: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class Institutional5mORBStrategy:
    """
    High-Velocity Institutional 5-Minute ORB Strategy:
      - 5-Minute Opening Candle (09:30–09:35 EST) establishes high/low and directional bias
      - 12 EMA Filter on 5m timeframe (Long if close > 12 EMA, Short if close < 12 EMA)
      - Breakout Entry: triggers when price breaches the 09:30–09:35 candle extreme (09:35–11:00 EST)
      - Trailing Stop Engine: Breakeven lock + dynamic R-trailing
      - Force close cutoff: 15:45 EST
    """

    def __init__(self, config: dict):
        cfg = config.get("strategy", config) if "strategy" in config else config
        self.config = cfg
        self.name = cfg.get("name", "NQ_Institutional_5m_ORB_Velocity")
        self._load_parameters()
        self.reset_session()

    def _load_parameters(self):
        or_cfg = self.config.get("opening_range", {})
        self.rth_start_time = datetime.strptime(or_cfg.get("rth_start", "09:30"), "%H:%M").time()
        self.or_end_time = datetime.strptime(or_cfg.get("or_end", "09:35"), "%H:%M").time()

        ind_cfg = self.config.get("indicators", {})
        self.ema_period = int(ind_cfg.get("ema_period", 12))

        f_cfg = self.config.get("filters", {})
        self.ema_rule = f_cfg.get("ema_alignment", {}).get("rule", "CLOSE_VS_EMA")
        self.min_candle_pts = float(f_cfg.get("noise_floor", {}).get("min_candle_points", 6.0))

        ex_cfg = self.config.get("execution", {})
        self.entry_mode = ex_cfg.get("entry_mode", "BREAKOUT_WITH_EMA")
        self.entry_start = datetime.strptime(ex_cfg.get("entry_window_start", "09:35"), "%H:%M").time()
        self.entry_end = datetime.strptime(ex_cfg.get("entry_window_end", "11:00"), "%H:%M").time()
        self.force_close_time = datetime.strptime(ex_cfg.get("force_close_time", "15:45"), "%H:%M").time()
        self.max_trades_per_day = int(ex_cfg.get("max_trades_per_day", 1))

        tr_cfg = self.config.get("trailing_stop", {})
        self.trailing_mode = tr_cfg.get("mode", "BE_AT_1R_THEN_TRAIL_0.75R")
        self.be_trigger_r = float(tr_cfg.get("be_trigger_r", 1.0))
        self.be_offset_ticks = float(tr_cfg.get("be_offset_ticks", 2.0))
        self.trail_distance_r = float(tr_cfg.get("trail_distance_r", 0.75))

        r_cfg = self.config.get("risk_management", {})
        self.point_value = float(r_cfg.get("point_value", 2.0))
        self.commission_per_contract = float(r_cfg.get("commission_per_contract", 1.24))
        self.slippage_points = float(r_cfg.get("slippage_points", 0.50))
        self.min_stop_points = float(r_cfg.get("min_stop_points", 6.0))
        self.max_stop_points = float(r_cfg.get("max_stop_points", 35.0))

    def reset_session(self):
        """Resets session state machine at the start of a trading day."""
        self.session_date = None
        self.candle_5m_established = False
        self.c_open = None
        self.c_high = None
        self.c_low = None
        self.c_close = None
        self.c_ema = None
        self.bias_direction = 0  # +1 Long, -1 Short, 0 None
        self.trades_taken_today = 0
        self.position = None  # Active trade state dict

    def establish_opening_candle(
        self,
        date: object,
        c_open: float,
        c_high: float,
        c_low: float,
        c_close: float,
        c_ema: float
    ) -> bool:
        """
        Processes the completed 09:30–09:35 5-minute candle and 12 EMA.
        Returns True if candle qualifies, False if noise floor or invalid.
        """
        self.reset_session()
        self.session_date = date
        self.c_open = c_open
        self.c_high = c_high
        self.c_low = c_low
        self.c_close = c_close
        self.c_ema = c_ema

        # Noise floor check
        candle_range = c_high - c_low
        if candle_range < self.min_candle_pts:
            return False

        # Directional alignment
        if c_close > c_ema:
            self.bias_direction = 1
        elif c_close < c_ema:
            self.bias_direction = -1
        else:
            self.bias_direction = 0

        self.candle_5m_established = (self.bias_direction != 0)
        return self.candle_5m_established

    def evaluate_1m_bar(
        self,
        t_bar: datetime,
        b_open: float,
        b_high: float,
        b_low: float,
        b_close: float
    ) -> Tuple[Optional[TradeSignal], Optional[str]]:
        """
        Evaluates incoming 1-minute bar for:
        1. Entry trigger (if no active position and within window)
        2. Trailing stop or profit exit (if active position)
        Returns:
            (signal, exit_reason)
        """
        bar_time = t_bar.time()

        # Hard EOD close
        if self.position is not None and bar_time >= self.force_close_time:
            exit_reason = "FORCE_CLOSE_EOD"
            return None, exit_reason

        # Manage active position
        if self.position is not None:
            return self._manage_position(t_bar, b_high, b_low)

        # Check for new entry
        if not self.candle_5m_established or self.trades_taken_today >= self.max_trades_per_day:
            return None, None

        if not (self.entry_start <= bar_time <= self.entry_end):
            return None, None

        # Check breakout triggers
        if self.bias_direction == 1:
            if b_high > self.c_high:
                entry_price = self.c_high + self.slippage_points
                risk_pts = min(max(entry_price - self.c_low, self.min_stop_points), self.max_stop_points)
                initial_sl = entry_price - risk_pts

                signal = TradeSignal(
                    date=self.session_date,
                    action="LONG",
                    entry_time=t_bar,
                    entry_price=entry_price,
                    initial_sl=initial_sl,
                    risk_pts=risk_pts
                )
                self.trades_taken_today += 1
                self.position = {
                    "direction": 1,
                    "entry_price": entry_price,
                    "initial_sl": initial_sl,
                    "current_sl": initial_sl,
                    "risk_pts": risk_pts,
                    "peak_favorable": entry_price,
                    "bars_held": 0
                }
                return signal, None

        elif self.bias_direction == -1:
            if b_low < self.c_low:
                entry_price = self.c_low - self.slippage_points
                risk_pts = min(max(self.c_high - entry_price, self.min_stop_points), self.max_stop_points)
                initial_sl = entry_price + risk_pts

                signal = TradeSignal(
                    date=self.session_date,
                    action="SHORT",
                    entry_time=t_bar,
                    entry_price=entry_price,
                    initial_sl=initial_sl,
                    risk_pts=risk_pts
                )
                self.trades_taken_today += 1
                self.position = {
                    "direction": -1,
                    "entry_price": entry_price,
                    "initial_sl": initial_sl,
                    "current_sl": initial_sl,
                    "risk_pts": risk_pts,
                    "peak_favorable": entry_price,
                    "bars_held": 0
                }
                return signal, None

        return None, None

    def _manage_position(
        self,
        t_bar: datetime,
        b_high: float,
        b_low: float
    ) -> Tuple[Optional[TradeSignal], Optional[str]]:
        pos = self.position
        pos["bars_held"] += 1
        direction = pos["direction"]
        current_sl = pos["current_sl"]
        entry_price = pos["entry_price"]
        risk_pts = pos["risk_pts"]

        # Pessimistic Stop Loss Check
        if direction == 1:
            if b_low <= current_sl:
                exit_reason = "TRAILING_STOP" if current_sl > entry_price else "INITIAL_STOP"
                self.position = None
                return None, exit_reason

            # Update Peak & Trailing Stop
            if b_high > pos["peak_favorable"]:
                pos["peak_favorable"] = b_high

            peak = pos["peak_favorable"]
            # Breakeven + trail logic
            if "BE_AT_1R" in self.trailing_mode:
                if peak >= entry_price + (self.be_trigger_r * risk_pts):
                    be_level = entry_price + (self.be_offset_ticks * 0.25)
                    trail_level = peak - (self.trail_distance_r * risk_pts)
                    cand_sl = max(be_level, trail_level)
                    if cand_sl > current_sl:
                        pos["current_sl"] = cand_sl
            elif "TRAIL_" in self.trailing_mode:
                trail_level = peak - (self.trail_distance_r * risk_pts)
                if trail_level > current_sl:
                    pos["current_sl"] = trail_level

        else: # Short
            if b_high >= current_sl:
                exit_reason = "TRAILING_STOP" if current_sl < entry_price else "INITIAL_STOP"
                self.position = None
                return None, exit_reason

            # Update Peak & Trailing Stop
            if b_low < pos["peak_favorable"]:
                pos["peak_favorable"] = b_low

            peak = pos["peak_favorable"]
            # Breakeven + trail logic
            if "BE_AT_1R" in self.trailing_mode:
                if peak <= entry_price - (self.be_trigger_r * risk_pts):
                    be_level = entry_price - (self.be_offset_ticks * 0.25)
                    trail_level = peak + (self.trail_distance_r * risk_pts)
                    cand_sl = min(be_level, trail_level)
                    if cand_sl < current_sl:
                        pos["current_sl"] = cand_sl
            elif "TRAIL_" in self.trailing_mode:
                trail_level = peak + (self.trail_distance_r * risk_pts)
                if trail_level < current_sl:
                    pos["current_sl"] = trail_level

        return None, None
