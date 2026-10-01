"""
Institutional 15-Minute ORB Alpha Engine — High Capital Velocity Overhaul
Target: CME Micro E-mini Nasdaq-100 (MNQ) Futures
Prop Firm: Tradeify $50K EOD Trailing Drawdown ($2,000 Barrier, $3,000 Target)

Architecture Overhaul:
  1. Dynamic Sizing & Structural Stop:
     - Natural structural stop: OR_Mid ((OR_High + OR_Low) / 2).
     - Hard catastrophic cap at 40.0 points (|Entry - OR_Mid| capped at 40.0).
     - Evaluation sizing allocated at 8.5% of Remaining DD (min $140, max $200).
  2. Restored Payoff Asymmetry:
     - Leg 1 (50%): TP1 = Entry ± 1.25R.
     - Ratchet on TP1: Immediate stop ratchet to Entry ± 2 ticks (0.50 pts).
     - Leg 2 (50% runner): TP2 = Entry ± 2.5R.
     - Profit Lock: On +2.0R expansion, ratchet Leg 2 SL to +1.0R.
  3. Filter Rebalance:
     - ATR Exhaustion ceiling raised to 0.55 × ATR_14 (unblocks trend sessions).
     - Noise floor set to 8.0 points.
     - Surgical Tier-1 news blackout window: ±2 minutes (CPI, PPI, NFP, FOMC).
     - 100% focus on high-momentum Primary Breakouts (09:45–11:00 EST).
     - Secondary Retest logic permanently deleted.
"""

from __future__ import annotations
import logging
from datetime import datetime, time
from typing import Optional
import pandas as pd
import pytz

from strategies.orb_15m.src.news_filter import InstitutionalNewsFilter

logger = logging.getLogger("Institutional15mORB_Velocity")


class Institutional15mORBStrategy:
    """
    High-Velocity Institutional 15-Minute ORB Strategy:
      - Primary Breakout entries (09:45–11:00 EST)
      - HTF Daily 20 EMA alignment (strictly lagged 1 day)
      - ATR Exhaustion Ceiling (0.55) & Noise Floor (8.0 pts)
      - Session VWAP confirmation
      - Surgical Tier-1 News Blackout (±2 minutes)
      - Structural OR_Mid Stop Loss (catastrophic cap at 40.0 pts)
      - High-Velocity Two-Tier Targets (1.25R / 2.50R)
    """

    def __init__(self, config: dict):
        self.config = config.get("strategy", config) if "strategy" in config else config
        self.name = self.config.get("name", "NQ_Institutional_15m_ORB_Velocity")
        self._load_parameters()
        self.news_filter = InstitutionalNewsFilter(
            blackout_window_minutes=self.news_blackout_minutes,
            timezone=self.tz_str
        )
        self.reset_session()

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------
    def _load_parameters(self):
        or_cfg = self.config.get("opening_range", {})
        self.rth_start_time = datetime.strptime(or_cfg.get("rth_start", "09:30"), "%H:%M").time()
        self.or_end_time = datetime.strptime(or_cfg.get("or_end", "09:45"), "%H:%M").time()

        # Filters
        f = self.config.get("filters", {})
        atr = f.get("atr_exhaustion", {})
        self.atr_exhaustion_enabled = atr.get("enabled", True)
        self.atr_lookback = int(atr.get("lookback_days", 14))
        self.max_or_atr_ratio = float(atr.get("max_or_atr_ratio", 0.55))

        nf = f.get("noise_floor", {})
        self.noise_floor_enabled = nf.get("enabled", True)
        self.min_or_points = float(nf.get("min_or_points", 8.0))

        self.vwap_enabled = f.get("vwap_confirmation", {}).get("enabled", True)

        htf = f.get("htf_alignment", {})
        self.htf_enabled = htf.get("enabled", True)
        self.htf_ema_period = int(htf.get("ema_period", 20))

        nws = f.get("news_filter", {})
        self.news_filter_enabled = nws.get("enabled", True)
        self.news_blackout_minutes = int(nws.get("blackout_window_minutes", 2))

        # Execution
        ex = self.config.get("execution", {})
        self.entry_window_start = datetime.strptime(ex.get("entry_window_start", "09:45"), "%H:%M").time()
        self.primary_cutoff_time = datetime.strptime(ex.get("primary_cutoff", "11:00"), "%H:%M").time()
        self.force_close_time = datetime.strptime(ex.get("force_close_time", "15:45"), "%H:%M").time()
        self.max_trades_per_day = int(ex.get("max_trades_per_day", 1))

        tt = ex.get("two_tier_scale_out", {})
        self.tp1_r = float(tt.get("tp1_r", 1.25))
        self.tp2_r = float(tt.get("tp2_r", 2.50))

        # Risk
        r = self.config.get("risk_management", {})
        self.point_value = float(r.get("point_value", 2.0))
        self.min_stop_points = float(r.get("min_stop_points", 4.0))
        self.max_stop_points = float(r.get("max_stop_points", 40.0))

        # Timezone
        t_cfg = self.config.get("time_management", {})
        self.tz_str = t_cfg.get("timezone", "America/New_York")
        self.tz = pytz.timezone(self.tz_str)

    # ------------------------------------------------------------------
    # Session state
    # ------------------------------------------------------------------
    def reset_session(self):
        """Reset all intraday state for a new trading day."""
        self.or_high: Optional[float] = None
        self.or_low: Optional[float] = None
        self.or_mid: Optional[float] = None
        self.or_range: Optional[float] = None
        self.or_defined: bool = False
        self.session_aborted: bool = False
        self.abort_reason: str = ""
        self.trade_executed_today: bool = False
        self.breakout_detected: bool = False
        self.breakout_direction: int = 0

    # ------------------------------------------------------------------
    # Opening Range
    # ------------------------------------------------------------------
    def calculate_opening_range(self, bars_0930_to_0945: pd.DataFrame, daily_atr_14: float) -> bool:
        """
        Define 15-minute Opening Range (09:30:00–09:44:59 EST).
        Apply Task 3 ATR exhaustion (0.55) and noise floor (8.0 pts) filters.
        """
        if bars_0930_to_0945.empty or len(bars_0930_to_0945) < 10:
            self.or_defined = False
            return False

        self.or_high = float(bars_0930_to_0945["High"].max())
        self.or_low = float(bars_0930_to_0945["Low"].min())
        self.or_mid = round((self.or_high + self.or_low) / 2.0, 2)
        self.or_range = round(self.or_high - self.or_low, 2)
        self.or_defined = True

        # ATR Exhaustion: OR_Range > 0.55 × ATR_14 → ABORT
        if self.atr_exhaustion_enabled and daily_atr_14 > 0:
            ceiling = self.max_or_atr_ratio * daily_atr_14
            if self.or_range > ceiling:
                self.session_aborted = True
                self.abort_reason = (
                    f"ATR Exhaustion: OR ({self.or_range:.1f}) > "
                    f"{self.max_or_atr_ratio:.0%} × ATR14 ({ceiling:.1f})"
                )
                return False

        # Noise Floor: OR_Range < 8.0 pts → ABORT
        if self.noise_floor_enabled and self.or_range < self.min_or_points:
            self.session_aborted = True
            self.abort_reason = (
                f"Noise Floor: OR ({self.or_range:.1f}) < "
                f"min ({self.min_or_points:.1f} pts)"
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Signal builder (Task 1: Natural Structural Stop OR_Mid, max 40 pts)
    # ------------------------------------------------------------------
    def _build_signal(self, action: str, entry_price: float, trigger_type: str) -> dict:
        """
        Build a trade signal with TP1 (+1.25R), TP2 (+2.50R), and structural SL (OR_Mid).
        Cap stop distance only at an absolute catastrophic threshold of 40.0 points.
        """
        d = 1 if action == "LONG" else -1
        raw_risk = abs(entry_price - self.or_mid)
        risk = min(raw_risk, self.max_stop_points)

        if risk < self.min_stop_points:
            return {"action": "NONE", "reasons": [f"Risk {risk:.1f} < min {self.min_stop_points}"]}

        stop_loss = round(entry_price - d * risk, 2)
        tp1 = round(entry_price + d * self.tp1_r * risk, 2)
        tp2 = round(entry_price + d * self.tp2_r * risk, 2)

        self.trade_executed_today = True
        return {
            "action": action,
            "entry_price": round(entry_price, 2),
            "sizing": {
                "stop_loss": stop_loss,
                "tp1": tp1,
                "tp2": tp2,
                "stop_points": round(risk, 2),
            },
            "trigger_type": trigger_type,
            "or_high": self.or_high,
            "or_low": self.or_low,
            "or_mid": self.or_mid,
        }

    # ------------------------------------------------------------------
    # Bar-by-bar evaluation (100% Primary Breakout Focus)
    # ------------------------------------------------------------------
    def evaluate_bar(
        self,
        current_time: datetime,
        current_bar: pd.Series,
        context_data: dict
    ) -> dict:
        """
        Evaluate a 1-minute bar against high-velocity primary breakout rules.
        Triggers immediately upon 1-minute close outside OR boundary with
        HTF Daily 20 EMA and Session VWAP confirmation.
        """
        signal = {"action": "NONE", "reasons": []}

        t = (
            current_time.astimezone(self.tz).time()
            if current_time.tzinfo is not None
            else current_time.time()
        )

        # ---- Pre-checks ------------------------------------------------
        if t < self.or_end_time:
            return signal
        if self.session_aborted:
            return signal
        if self.trade_executed_today:
            return signal
        if not self.or_defined or self.or_high is None:
            return signal
        if t > self.primary_cutoff_time:
            return signal

        # Surgical News Blackout (±2 min around Tier-1 releases)
        if self.news_filter_enabled:
            is_blackout, reason = self.news_filter.is_in_blackout(current_time)
            if is_blackout:
                signal["reasons"].append(reason or "News Blackout")
                return signal

        # ---- Extract data -----------------------------------------------
        close_p = float(current_bar["Close"])
        vwap = float(context_data.get("session_vwap", self.or_mid))
        ema20 = float(context_data.get("daily_20_ema", 0.0))
        open_p = float(context_data.get("exchange_open_price", 0.0))

        # HTF Alignment (strictly lagged Daily 20 EMA)
        long_htf = True
        short_htf = True
        if self.htf_enabled and ema20 > 0 and open_p > 0:
            if open_p <= ema20:
                long_htf = False
            if open_p >= ema20:
                short_htf = False

        # ================================================================
        # HIGH-MOMENTUM PRIMARY BREAKOUT (09:45 – 11:00 EST)
        # ================================================================
        if not self.breakout_detected:
            # LONG breakout: 1-min Close > OR_High, Open > Daily 20 EMA, Close > VWAP
            if close_p > self.or_high and long_htf:
                vwap_ok = (not self.vwap_enabled) or (close_p > vwap)
                if vwap_ok:
                    self.breakout_detected = True
                    self.breakout_direction = 1
                    return self._build_signal("LONG", close_p, "PRIMARY_BREAKOUT")

            # SHORT breakout: 1-min Close < OR_Low, Open < Daily 20 EMA, Close < VWAP
            elif close_p < self.or_low and short_htf:
                vwap_ok = (not self.vwap_enabled) or (close_p < vwap)
                if vwap_ok:
                    self.breakout_detected = True
                    self.breakout_direction = -1
                    return self._build_signal("SHORT", close_p, "PRIMARY_BREAKOUT")

        return signal
