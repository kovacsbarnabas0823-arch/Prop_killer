"""
NQ/MNQ Quantitative Alpha Strategy: Filtered Quantitative ORB (Narrow-Range Trend Day Model)
Author: Senior Quantitative Developer & Algorithmic Trader
Target: CME E-mini Nasdaq-100 (NQ/MNQ) Futures (Tradeify / Lucid Prop Firm EOD Drawdown)

Strategy Architecture:
  1. Opening Range (OR):
     - Formed strictly between 09:30:00 and 09:59:59 EST (first 30 1-minute bars).
     - Locked at 10:00:00 EST: OR_High, OR_Low, and OR_Mid ((OR_High + OR_Low) / 2).
     - Strict flat / no-trade zone from 09:30 to 10:00 EST.
  2. ATR Exhaustion Filter:
     - Calculates rolling 14-day Daily ATR over preceding sessions.
     - ABORTS the session if (OR_High - OR_Low) > (0.50 * Daily_ATR_14).
  3. VWAP Alignment & Close-Confirmed Trigger:
     - Evaluated between 10:00:00 and 11:30:00 EST on confirmation candle closes (e.g. 5-minute bar close).
     - LONG Trigger: 5-min Close > OR_High AND 5-min Close > Session_VWAP (anchored at 09:30 EST).
     - SHORT Trigger: 5-min Close < OR_Low AND 5-min Close < Session_VWAP (anchored at 09:30 EST).
  4. Asymmetric Risk/Reward (Strict 1:2 R:R):
     - Stop Loss = OR_Mid
     - Long: Risk = Entry - OR_Mid; Take Profit = Entry + (2.0 * Risk)
     - Short: Risk = OR_Mid - Entry; Take Profit = Entry - (2.0 * Risk)
  5. One-and-Done & Time Invalidation:
     - Maximum 1 trade per day.
     - No new entries permitted after 11:30:00 EST.
     - Any open position is flattened at 15:45:00 EST.
"""

from __future__ import annotations
import logging
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import pytz

logger = logging.getLogger("FilteredQuantORB")


class NQORBStrategy:
    """
    Filtered Quantitative ORB with 14-day ATR Exhaustion Filter & VWAP Confirmation.
    """

    def __init__(self, config: dict):
        self.config = config.get("strategy", config) if "strategy" in config else config
        self.name = self.config.get("name", "NQ_Filtered_Quant_ORB")
        self.instrument = self.config.get("instrument", "MNQ")

        self._load_parameters()
        self.reset_session()

    def _load_parameters(self):
        # 1. Opening Range Parameters
        or_cfg = self.config.get("opening_range", {})
        self.rth_start_time = datetime.strptime(or_cfg.get("rth_start", "09:30"), "%H:%M").time()
        self.or_end_time = datetime.strptime(or_cfg.get("or_end", "10:00"), "%H:%M").time()

        # 2. Filters & Trigger Parameters
        filters_cfg = self.config.get("filters", {})
        atr_cfg = filters_cfg.get("atr_exhaustion", {})
        self.atr_exhaustion_enabled = atr_cfg.get("enabled", True)
        self.atr_lookback = int(atr_cfg.get("lookback_days", 14))
        self.max_or_atr_ratio = float(atr_cfg.get("max_or_atr_ratio", 0.50))

        noise_cfg = filters_cfg.get("noise_floor", {})
        self.noise_floor_enabled = noise_cfg.get("enabled", True)
        self.min_or_points = float(noise_cfg.get("min_or_points", 15.0))
        self.min_or_atr_ratio = float(noise_cfg.get("min_or_atr_ratio", 0.08))

        self.vwap_enabled = filters_cfg.get("vwap_confirmation", {}).get("enabled", True)
        self.confirmation_timeframe = int(filters_cfg.get("confirmation_timeframe_minutes", 3))

        htf_cfg = filters_cfg.get("htf_alignment", {})
        self.htf_enabled = htf_cfg.get("enabled", True)
        self.htf_ema_period = int(htf_cfg.get("ema_period", 20))

        # 3. Execution & Invalidation Parameters
        exec_cfg = self.config.get("execution", {})
        entry_end_str = exec_cfg.get("entry_window_end", "11:30")
        if isinstance(entry_end_str, str):
            self.entry_window_end = datetime.strptime(entry_end_str, "%H:%M").time()
        elif isinstance(entry_end_str, time):
            self.entry_window_end = entry_end_str
        else:
            self.entry_window_end = time(11, 30)

        force_close_str = exec_cfg.get("force_close_time", "15:45")
        if isinstance(force_close_str, str):
            self.force_close_time = datetime.strptime(force_close_str, "%H:%M").time()
        elif isinstance(force_close_str, time):
            self.force_close_time = force_close_str
        else:
            self.force_close_time = time(15, 45)

        self.max_trades_per_day = int(exec_cfg.get("max_trades_per_day", 1))

        # 4. Risk Management (Asymmetric 1:2 R:R)
        risk_cfg = self.config.get("risk_management", {})
        self.point_value = float(risk_cfg.get("point_value", 2.0))  # MNQ: $2/pt
        self.target_rr = float(risk_cfg.get("target_rr", 2.0))      # Strict 1:2 R:R
        self.min_stop_points = float(risk_cfg.get("min_stop_points", 5.0))
        self.max_stop_points = float(risk_cfg.get("max_stop_points", 60.0))

        time_cfg = self.config.get("time_management", {})
        self.tz = pytz.timezone(time_cfg.get("timezone", "America/New_York"))

    def reset_session(self):
        """Resets all intraday tracking variables for a new trading day."""
        self.or_high: Optional[float] = None
        self.or_low: Optional[float] = None
        self.or_mid: Optional[float] = None
        self.or_range: Optional[float] = None
        self.or_defined: bool = False
        self.session_aborted: bool = False
        self.abort_reason: str = ""
        self.trade_executed_today: bool = False
        self.long_breakout_detected: bool = False
        self.short_breakout_detected: bool = False

    def calculate_opening_range(self, bars_0930_to_1000: pd.DataFrame, daily_atr_14: float) -> bool:
        """
        Defines Opening Range from 09:30:00 to 09:59:59 EST.
        Applies:
          1. ATR Exhaustion Filter: aborts if OR_Range > 0.50 * Daily_ATR_14.
          2. Noise Floor Cutoff: aborts if OR_Range < 15.0 pts or < 0.08 * Daily_ATR_14.
        """
        if bars_0930_to_1000.empty or len(bars_0930_to_1000) < 15:
            self.or_defined = False
            return False

        self.or_high = float(bars_0930_to_1000["High"].max())
        self.or_low = float(bars_0930_to_1000["Low"].min())
        self.or_mid = round((self.or_high + self.or_low) / 2.0, 2)
        self.or_range = round(self.or_high - self.or_low, 2)
        self.or_defined = True

        # 1. ATR Exhaustion Filter Check (Upper Cap)
        if self.atr_exhaustion_enabled and daily_atr_14 > 0:
            max_allowed_or = self.max_or_atr_ratio * daily_atr_14
            if self.or_range > max_allowed_or:
                self.session_aborted = True
                self.abort_reason = (
                    f"ATR Exhaustion: OR_Range ({self.or_range:.2f}) > "
                    f"{self.max_or_atr_ratio:.0%} * 14d_ATR ({max_allowed_or:.2f})"
                )
                return False

        # 2. Noise Floor Cutoff Check (Lower Bound)
        if self.noise_floor_enabled:
            atr_floor = (self.min_or_atr_ratio * daily_atr_14) if daily_atr_14 > 0 else 0.0
            if self.or_range < self.min_or_points or (atr_floor > 0 and self.or_range < atr_floor):
                self.session_aborted = True
                self.abort_reason = (
                    f"Noise Floor Cutoff: OR_Range ({self.or_range:.2f}) < "
                    f"threshold (min {self.min_or_points} pts or {self.min_or_atr_ratio:.0%} * ATR)"
                )
                return False

        return True

    def evaluate_bar(
        self,
        current_time: datetime,
        current_bar: pd.Series,
        context_data: dict
    ) -> dict:
        """
        Evaluates a 1-minute bar against the Filtered Quant ORB rules.
        Supports:
          1. Primary Breakout Entry (3-min close confirmation)
          2. Secondary Retest Limit Entry (touch of OR boundary prior to 11:15 EST)
        """
        signal = {"action": "NONE", "reasons": []}

        t = (
            current_time.astimezone(self.tz).time()
            if current_time.tzinfo is not None
            else current_time.time()
        )

        # 1. Enforce strict flat / no-trade zone from 09:30 to 10:00 EST
        if t < self.or_end_time:
            signal["reasons"].append("Flat zone: Opening Range formation (09:30 - 10:00 EST)")
            return signal

        # 2. Check if session was aborted by ATR exhaustion filter
        if self.session_aborted:
            signal["reasons"].append(self.abort_reason)
            return signal

        # 3. One-and-Done: Maximum 1 trade per day
        if self.trade_executed_today:
            signal["reasons"].append("Max 1 trade per day already executed")
            return signal

        # 4. Entry Window Cutoff (no new entries after 11:30:00 EST)
        if t > self.entry_window_end:
            signal["reasons"].append(f"Past entry window cutoff ({self.entry_window_end})")
            return signal

        # 5. Check OR definition
        if not self.or_defined or self.or_high is None or self.or_low is None or self.or_mid is None:
            signal["reasons"].append("Opening Range not defined")
            return signal

        close_p = float(current_bar["Close"])
        high_p = float(current_bar["High"])
        low_p = float(current_bar["Low"])
        session_vwap = float(context_data.get("session_vwap", self.or_mid))

        # Track breakout states once price closes outside the OR boundary
        if close_p > self.or_high:
            self.long_breakout_detected = True
        if close_p < self.or_low:
            self.short_breakout_detected = True

        # Check Higher-Timeframe (HTF) Daily 20 EMA Alignment
        open_price = float(context_data.get("exchange_open_price", 0.0))
        daily_20_ema = float(context_data.get("daily_20_ema", 0.0))

        long_htf_aligned = True
        short_htf_aligned = True
        if self.htf_enabled and daily_20_ema > 0 and open_price > 0:
            if open_price <= daily_20_ema:
                long_htf_aligned = False
            if open_price >= daily_20_ema:
                short_htf_aligned = False

        # ----------------------------------------------------------------------
        # TRIGGER 1: Primary Breakout Entry (3-min confirmation close)
        # ----------------------------------------------------------------------
        is_confirmation_bar = ((t.minute + 1) % self.confirmation_timeframe == 0)
        if is_confirmation_bar:
            # LONG Breakout: Close > OR_High AND Close > VWAP AND 09:30 Open > Daily 20 EMA
            if long_htf_aligned and close_p > self.or_high and (not self.vwap_enabled or close_p > session_vwap):
                entry_p = close_p
                stop_loss = self.or_mid
                risk = entry_p - stop_loss

                if self.min_stop_points <= risk <= self.max_stop_points:
                    tp1 = round(entry_p + (1.2 * risk), 2)
                    tp2 = round(entry_p + (self.target_rr * risk), 2)
                    self.trade_executed_today = True
                    return {
                        "action": "LONG",
                        "entry_price": round(entry_p, 2),
                        "sizing": {
                            "stop_loss": round(stop_loss, 2),
                            "tp1": tp1,
                            "tp2": tp2,
                            "take_profit": tp2,
                            "stop_points": round(risk, 2),
                            "target_points": round(self.target_rr * risk, 2),
                            "target_rr": self.target_rr
                        },
                        "trigger_type": "BREAKOUT_LONG",
                        "trade_type": "BREAKOUT",
                        "or_high": self.or_high,
                        "or_mid": self.or_mid,
                        "session_vwap": round(session_vwap, 2),
                        "daily_20_ema": round(daily_20_ema, 2)
                    }

            # SHORT Breakout: Close < OR_Low AND Close < VWAP AND 09:30 Open < Daily 20 EMA
            if short_htf_aligned and close_p < self.or_low and (not self.vwap_enabled or close_p < session_vwap):
                entry_p = close_p
                stop_loss = self.or_mid
                risk = stop_loss - entry_p

                if self.min_stop_points <= risk <= self.max_stop_points:
                    tp1 = round(entry_p - (1.2 * risk), 2)
                    tp2 = round(entry_p - (self.target_rr * risk), 2)
                    self.trade_executed_today = True
                    return {
                        "action": "SHORT",
                        "entry_price": round(entry_p, 2),
                        "sizing": {
                            "stop_loss": round(stop_loss, 2),
                            "tp1": tp1,
                            "tp2": tp2,
                            "take_profit": tp2,
                            "stop_points": round(risk, 2),
                            "target_points": round(self.target_rr * risk, 2),
                            "target_rr": self.target_rr
                        },
                        "trigger_type": "BREAKOUT_SHORT",
                        "trade_type": "BREAKOUT",
                        "or_low": self.or_low,
                        "or_mid": self.or_mid,
                        "session_vwap": round(session_vwap, 2),
                        "daily_20_ema": round(daily_20_ema, 2)
                    }

        # ----------------------------------------------------------------------
        # TRIGGER 2: Secondary Retest Limit Entry (Prior to 11:15 EST)
        # ----------------------------------------------------------------------
        retest_cutoff_time = time(11, 15)
        if t <= retest_cutoff_time:
            # LONG Retest: Broke above OR_High, hasn't reached TP1, now pulls back to touch OR_High
            theo_risk_long = self.or_high - self.or_mid
            theo_tp1_long = self.or_high + (1.2 * theo_risk_long)

            if long_htf_aligned and self.long_breakout_detected and high_p < theo_tp1_long and low_p <= self.or_high:
                entry_p = self.or_high
                stop_loss = self.or_mid
                risk = theo_risk_long

                if self.min_stop_points <= risk <= self.max_stop_points:
                    tp1 = round(entry_p + (1.2 * risk), 2)
                    tp2 = round(entry_p + (self.target_rr * risk), 2)
                    self.trade_executed_today = True
                    return {
                        "action": "LONG",
                        "entry_price": round(entry_p, 2),
                        "sizing": {
                            "stop_loss": round(stop_loss, 2),
                            "tp1": tp1,
                            "tp2": tp2,
                            "take_profit": tp2,
                            "stop_points": round(risk, 2),
                            "target_points": round(self.target_rr * risk, 2),
                            "target_rr": self.target_rr
                        },
                        "trigger_type": "RETEST_LONG",
                        "trade_type": "RETEST",
                        "or_high": self.or_high,
                        "or_mid": self.or_mid,
                        "session_vwap": round(session_vwap, 2),
                        "daily_20_ema": round(daily_20_ema, 2)
                    }

            # SHORT Retest: Broke below OR_Low, hasn't reached TP1, now pulls back to touch OR_Low
            theo_risk_short = self.or_mid - self.or_low
            theo_tp1_short = self.or_low - (1.2 * theo_risk_short)

            if short_htf_aligned and self.short_breakout_detected and low_p > theo_tp1_short and high_p >= self.or_low:
                entry_p = self.or_low
                stop_loss = self.or_mid
                risk = theo_risk_short

                if self.min_stop_points <= risk <= self.max_stop_points:
                    tp1 = round(entry_p - (1.2 * risk), 2)
                    tp2 = round(entry_p - (self.target_rr * risk), 2)
                    self.trade_executed_today = True
                    return {
                        "action": "SHORT",
                        "entry_price": round(entry_p, 2),
                        "sizing": {
                            "stop_loss": round(stop_loss, 2),
                            "tp1": tp1,
                            "tp2": tp2,
                            "take_profit": tp2,
                            "stop_points": round(risk, 2),
                            "target_points": round(self.target_rr * risk, 2),
                            "target_rr": self.target_rr
                        },
                        "trigger_type": "RETEST_SHORT",
                        "trade_type": "RETEST",
                        "or_low": self.or_low,
                        "or_mid": self.or_mid,
                        "session_vwap": round(session_vwap, 2),
                        "daily_20_ema": round(daily_20_ema, 2)
                    }

        return signal

    def evaluate_entry(
        self,
        current_time: datetime,
        current_bar: pd.Series,
        context_data: dict
    ) -> dict:
        """Alias for evaluate_bar to ensure backward compatibility."""
        return self.evaluate_bar(current_time, current_bar, context_data)
