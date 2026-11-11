"""
Tradeify $50K Growth Convex Risk & Bellman Sizing Engine
Author: Principal Quantitative Architect & Lead Systems Architect

Target: Tradeify $50K Growth Account (CME NQ/MNQ) with End-of-Day (EOD) Trailing Drawdown.
Enforces:
  - Exact EOD Trailing Drawdown mechanics ($2,000 trailing from closed EOD HWM).
  - Permanent Floor Lock at $50,100.00 once EOD HWM touches $52,100.00 (converting to static DD).
  - Internal Hard Daily Stop-Loss (-$600.00) protecting the EOD trailing buffer.
  - Tradeify Daily Loss Limit Soft Breach (-$1,250.00).
  - Strict contract exclusivity: Micro E-mini Nasdaq (MNQ, $2.00/pt) only.
  - Two-Phase State-Dependent Bellman Dynamic Sizing (Evaluation Sprint vs Funded Extraction).
  - Funded Consistency Governor (+825.00 daily profit ceiling protecting the 35% rule).
  - Hard EOD Flatten at 15:44:00 EST.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger("TradeifyRules")


class AccountPhase(str, Enum):
    EVALUATION = "EVALUATION"
    FUNDED = "FUNDED"


@dataclass
class TradeifyAccountConfig:
    INITIAL_BALANCE: float = 50000.0
    EVAL_PROFIT_TARGET: float = 53000.0          # +$3,000 target ($87 fee, $0 activation)
    EOD_TRAILING_DD: float = 2000.0             # $2,000 EOD trailing drawdown
    FLOOR_LOCK_TRIGGER: float = 52100.0          # Floor locks at $50,100 once EOD HWM >= $52,100
    LOCKED_FLOOR: float = 50100.0                # Permanent static floor post-lock
    DAILY_LOSS_LIMIT_HARD: float = -1250.0       # Tradeify daily circuit breaker
    INTERNAL_DAILY_STOP: float = -600.0          # Internal protective daily stop
    HARD_EOD_FLATTEN_TIME: str = "15:44:00"      # All positions flattened 60s before 15:45:00
    MAX_FUNDED_DAILY_PROFIT: float = 920.0       # Funded 35% consistency governor ceiling ($2,650 * 0.35 = $927.50)
    FUNDED_PAYOUT_TARGET: float = 52650.0        # +$2,650 for first payout
    MIN_FUNDED_WINNING_DAYS: int = 5             # 5 trading days with profit >= $150
    MIN_WINNING_DAY_PROFIT: float = 150.0
    MNQ_POINT_VALUE: float = 2.0                 # $2.00 per point
    COMMISSION_PER_MNQ_RT: float = 2.48          # $1.24 per side per contract
    SLIPPAGE_POINTS_PER_FILL: float = 0.50       # 0.50 NQ/MNQ points per market execution


def get_strategy_unit_risk_usd(strategy_id: str, atr_14: float) -> float:
    """
    Computes bounded, strategy-specific risk denominator in USD per MNQ contract.
    H3: Stop distance [22, 48] pts -> ref_pts clipped to [30, 40] pts ($64.48 - $84.48/MNQ).
    H4: Stop distance [16, 32] pts -> ref_pts clipped to [16, 22] pts ($36.48 - $48.48/MNQ).
    """
    if strategy_id == "H4":
        ref_pts_h4 = float(np.clip(0.09 * atr_14, 16.0, 22.0))
        return (ref_pts_h4 * 2.00) + 4.48
    else:  # H3
        ref_pts_h3 = float(np.clip(0.18 * atr_14, 30.0, 40.0))
        return (ref_pts_h3 * 2.00) + 4.48


class TradeifyAccountState:
    """
    Mathematical state machine modeling the Tradeify $50K Growth account.
    """

    def __init__(
        self,
        phase: AccountPhase = AccountPhase.EVALUATION,
        config: Optional[TradeifyAccountConfig] = None
    ):
        self.config = config or TradeifyAccountConfig()
        self.phase = phase

        # Account balances
        self.balance: float = self.config.INITIAL_BALANCE
        self.eod_hwm: float = self.config.INITIAL_BALANCE
        self.current_floor: float = self.config.INITIAL_BALANCE - self.config.EOD_TRAILING_DD
        self.is_floor_locked: bool = False

        # Intraday tracking
        self.day_start_balance: float = self.config.INITIAL_BALANCE
        self.intraday_pnl: float = 0.0
        self.is_daily_stopped: bool = False
        self.is_governor_halted: bool = False

        # Life cycle flags
        self.is_breached: bool = False
        self.is_passed: bool = False
        self.is_funded_payout_eligible: bool = False
        self.breach_reason: Optional[str] = None

        # Funded statistics
        self.funded_winning_days: int = 0
        self.total_trading_days: int = 0

    @property
    def current_equity(self) -> float:
        return self.balance + self.intraday_pnl

    @property
    def usable_buffer(self) -> float:
        return self.current_equity - self.current_floor

    @property
    def net_profit(self) -> float:
        return self.current_equity - self.config.INITIAL_BALANCE

    def start_session(self, current_date=None):
        """Prepares state for the start of a new trading session."""
        self.day_start_balance = self.balance
        self.intraday_pnl = 0.0
        self.is_daily_stopped = False
        self.is_governor_halted = False
        self.total_trading_days += 1

    def get_optimal_mnq_size(
        self,
        strategy_id: str = "H3",
        sl_dist_pts: float = 30.0,
        atr_14: Optional[float] = None,
        morning_pnl: float = 0.0
    ) -> int:
        """
        Phase 5 Bounded Risk-Parity Sizer with House-Money Floor Guard.
        - Strategy-specific bounded risk denominator prevents ATR starvation in 2024-2026.
        - Two-regime tiers: Evaluation Sprint vs Funded Capital Preservation.
        - House-Money Floor Guard for H4: If morning PnL >= +$165, size H4 from surplus (morning_pnl - $160),
          guaranteeing net day PnL >= +$160 even on full stop-out (preserving qualifying days).
        """
        if self.is_breached or self.is_daily_stopped or self.is_governor_halted:
            return 0

        buffer = self.usable_buffer
        profit = self.net_profit
        current_equity = self.current_equity
        atr = atr_14 if (atr_14 is not None and atr_14 > 0) else 200.0

        if self.phase == AccountPhase.EVALUATION:
            # Regime 1: Evaluation Sprint Tiers
            if current_equity >= 51900.0:
                r_target, min_c, max_c = 330.0, 4, 8
            elif profit >= 800.0 and buffer >= 1350.0:
                r_target, min_c, max_c = 250.0, 3, 6
            elif buffer >= 950.0:
                r_target, min_c, max_c = 170.0, 2, 4
            else:
                r_target, min_c, max_c = 120.0, 1, 2
        else:
            # Regime 2: Funded Capital Preservation Tiers
            if buffer < 1050.0:
                r_target, min_c, max_c = 90.0, 1, 2
            elif not self.is_floor_locked and self.eod_hwm < self.config.FLOOR_LOCK_TRIGGER:
                r_target, min_c, max_c = 165.0, 2, 4
            else:
                r_target, min_c, max_c = 245.0, 3, 6

            # House-Money Floor Guard for H4 in Funded Pre-Lock Mode
            if strategy_id == "H4" and not self.is_floor_locked and self.eod_hwm < self.config.FLOOR_LOCK_TRIGGER:
                if morning_pnl >= 165.0:
                    house_money_budget = morning_pnl - 160.0
                    r_effective = min(r_target, house_money_budget)
                    per_c_risk = (sl_dist_pts * self.config.MNQ_POINT_VALUE) + 4.48
                    h4_contracts = int(np.floor(r_effective / per_c_risk))
                    return max(0, min(max_c, h4_contracts))

        unit_risk_usd = get_strategy_unit_risk_usd(strategy_id, atr)
        raw_contracts = int(np.round(r_target / unit_risk_usd))

        per_contract_sl_risk = (sl_dist_pts * self.config.MNQ_POINT_VALUE) + 4.48
        max_safe_contracts = int(np.max([1, np.floor((1.35 * r_target) / per_contract_sl_risk)]))

        contracts = int(np.clip(min(raw_contracts, max_safe_contracts), min_c, max_c))
        return contracts

    def update_intraday_pnl(self, delta_pnl: float) -> Tuple[bool, Optional[str]]:
        """
        Updates intraday PnL and validates all Tradeify risk constraints.
        Returns (can_continue_trading, halt_or_breach_reason).
        """
        self.intraday_pnl += delta_pnl
        current_equity = self.current_equity

        # 1. Catastrophic Account Breach Check against EOD Trailing Floor
        if current_equity <= self.current_floor:
            self.is_breached = True
            self.breach_reason = (
                f"LIQUIDATION BREACH: Equity (${current_equity:.2f}) fell below active floor (${self.current_floor:.2f})."
            )
            return False, self.breach_reason

        # 2. Daily Loss Limit Hard Check (-$1,250.00)
        if self.intraday_pnl <= self.config.DAILY_LOSS_LIMIT_HARD:
            self.is_daily_stopped = True
            self.breach_reason = (
                f"DAILY LOSS LIMIT BREACH: Intraday PnL (${self.intraday_pnl:.2f}) breached -${abs(self.config.DAILY_LOSS_LIMIT_HARD):.2f} DLL."
            )
            return False, self.breach_reason

        # 3. Internal Protective Daily Stop-Loss (-$600.00)
        if self.intraday_pnl <= self.config.INTERNAL_DAILY_STOP:
            self.is_daily_stopped = True
            return False, f"INTERNAL DAILY STOP ENGAGED: Intraday PnL (${self.intraday_pnl:.2f}) reached internal stop limit."

        # 4. Funded Consistency Governor Ceiling (+$920.00 protecting 35% rule on $2,650 payout target)
        if self.phase == AccountPhase.FUNDED:
            if self.intraday_pnl >= self.config.MAX_FUNDED_DAILY_PROFIT:
                self.is_governor_halted = True
                return False, f"FUNDED CONSISTENCY GOVERNOR: Daily profit (${self.intraday_pnl:.2f}) reached +${self.config.MAX_FUNDED_DAILY_PROFIT:.2f} ceiling. Halting to protect 35% rule."

        # 5. Evaluation Target Check ($53,000.00)
        if self.phase == AccountPhase.EVALUATION and current_equity >= self.config.EVAL_PROFIT_TARGET:
            self.is_passed = True

        return True, None

    def close_trade(self, net_realized_pnl: float):
        """Records completed trade PnL to cash balance and intraday PnL."""
        self.balance += net_realized_pnl
        # Also check evaluation pass on closed balance
        if self.phase == AccountPhase.EVALUATION and self.balance >= self.config.EVAL_PROFIT_TARGET:
            self.is_passed = True

    def end_session(self, current_date=None) -> Dict[str, Any]:
        """
        Executes end-of-day reconciliation at 15:45:00 EST:
          - Realizes intraday balance.
          - Updates EOD_HWM if closed balance reached a new high.
          - Recalculates EOD Trailing Floor.
          - Engages Permanent Floor Lock if EOD_HWM >= $52,100.
          - Updates Funded Consistency and Payout criteria.
        """
        day_pnl = self.balance - self.day_start_balance

        # Track funded winning days
        if self.phase == AccountPhase.FUNDED:
            if day_pnl >= self.config.MIN_WINNING_DAY_PROFIT:
                self.funded_winning_days += 1

            if (
                self.balance >= self.config.FUNDED_PAYOUT_TARGET
                and self.funded_winning_days >= self.config.MIN_FUNDED_WINNING_DAYS
            ):
                self.is_funded_payout_eligible = True

        # Check Evaluation Pass
        if self.phase == AccountPhase.EVALUATION and self.balance >= self.config.EVAL_PROFIT_TARGET:
            self.is_passed = True

        # Check if closed balance breached floor
        if self.balance <= self.current_floor:
            self.is_breached = True
            self.breach_reason = f"EOD BREACH: Closed balance (${self.balance:.2f}) at or below floor (${self.current_floor:.2f})."

        # Update EOD High Watermark
        if self.balance > self.eod_hwm:
            self.eod_hwm = self.balance

        # Trailing Floor Logic & Permanent Floor Lock
        if not self.is_floor_locked:
            if self.eod_hwm >= self.config.FLOOR_LOCK_TRIGGER:
                self.is_floor_locked = True
                self.current_floor = self.config.LOCKED_FLOOR
                logger.debug(
                    f"FLOOR LOCK ACTIVATED: EOD HWM (${self.eod_hwm:.2f}) >= ${self.config.FLOOR_LOCK_TRIGGER:.2f}. "
                    f"Floor permanently locked at ${self.config.LOCKED_FLOOR:.2f}."
                )
            else:
                new_floor = self.eod_hwm - self.config.EOD_TRAILING_DD
                if new_floor > self.current_floor:
                    self.current_floor = new_floor

        session_summary = {
            "date": current_date,
            "closing_balance": self.balance,
            "day_pnl": day_pnl,
            "eod_hwm": self.eod_hwm,
            "current_floor": self.current_floor,
            "is_floor_locked": self.is_floor_locked,
            "is_breached": self.is_breached,
            "is_passed": self.is_passed,
            "is_funded_payout_eligible": self.is_funded_payout_eligible,
            "funded_winning_days": self.funded_winning_days
        }

        # Reset intraday tracking for next day
        self.intraday_pnl = 0.0
        self.is_daily_stopped = False
        self.is_governor_halted = False

        return session_summary
