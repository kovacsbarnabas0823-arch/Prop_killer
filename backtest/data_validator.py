"""
Backtesting & Data Validation Engine: Pessimistic Execution Rule & Dynamic Break-Even
Enforces zero look-ahead bias for 1-minute and tick data resolution.

Rules:
  1. Pessimistic Execution Rule:
     If a single 1-minute bar touches both the Stop-Loss and Take-Profit / +1.0R levels,
     the simulator strictly logs it as a STOP-LOSS (full loss).
  2. Dynamic Break-Even (BE) Trade Management:
     If an active position reaches an unrealized profit equal to its initial risk (+1.0R),
     the Stop Loss is automatically moved to Entry Price + 2 ticks (for Longs) or
     Entry Price - 2 ticks (for Shorts) to cover commission drag.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class TradeOutcome:
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    ACTIVE = "ACTIVE"
    PESSIMISTIC_STOP = "PESSIMISTIC_STOP_COLLISION"
    BREAK_EVEN = "BREAK_EVEN"
    PROFIT_LOCK_1R = "PROFIT_LOCK_1R"
    SCALE_OUT_BREAK_EVEN = "SCALE_OUT_BREAK_EVEN"
    SCALE_OUT_PROFIT_LOCK_1R = "SCALE_OUT_PROFIT_LOCK_1R"
    SCALE_OUT_TAKE_PROFIT = "SCALE_OUT_TAKE_PROFIT"
    TIME_EXIT_1545 = "TIME_EXIT_1545"


@dataclass
class BacktestTrade:
    trade_id: str
    symbol: str
    direction: int  # 1 for Long, -1 for Short
    entry_price: float
    stop_loss: float
    take_profit: float  # Default / TP2 level (+2.5R)
    contracts: int      # Total contracts (even number, min 2)
    entry_time: pd.Timestamp
    initial_risk: float = 0.0
    initial_stop_loss: float = 0.0
    tp1: float = 0.0    # Leg 1 target (+1.2R)
    tp2: float = 0.0    # Leg 2 target (+2.5R)
    leg1_contracts: int = 0
    leg2_contracts: int = 0
    leg1_filled: bool = False
    leg1_exit_price: Optional[float] = None
    leg1_outcome: Optional[str] = None
    leg2_filled: bool = False
    leg2_exit_price: Optional[float] = None
    leg2_outcome: Optional[str] = None
    be_triggered: bool = False
    leg2_ratcheted_to_1r: bool = False
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    outcome: Optional[str] = None
    realized_pnl: float = 0.0
    trade_type: str = "BREAKOUT"  # "BREAKOUT" or "RETEST"

    def __post_init__(self):
        if self.initial_risk == 0.0:
            self.initial_risk = abs(self.entry_price - self.stop_loss)
        if self.initial_stop_loss == 0.0:
            self.initial_stop_loss = self.stop_loss

        # Calculate TP1 (+1.2R) and TP2 (+2.5R)
        if self.tp1 == 0.0:
            if self.direction == 1:
                self.tp1 = round(self.entry_price + (1.2 * self.initial_risk), 2)
            else:
                self.tp1 = round(self.entry_price - (1.2 * self.initial_risk), 2)

        if self.tp2 == 0.0:
            if self.take_profit > 0:
                self.tp2 = self.take_profit
            else:
                if self.direction == 1:
                    self.tp2 = round(self.entry_price + (2.5 * self.initial_risk), 2)
                else:
                    self.tp2 = round(self.entry_price - (2.5 * self.initial_risk), 2)

        # Enforce clean 50/50 two-tier contract split (minimum 2 contracts)
        if self.contracts < 2:
            self.contracts = 2
        elif self.contracts % 2 != 0:
            self.contracts -= 1

        if self.leg1_contracts == 0:
            self.leg1_contracts = self.contracts // 2
        if self.leg2_contracts == 0:
            self.leg2_contracts = self.contracts - self.leg1_contracts


def evaluate_bar_with_pessimistic_rule(
    bar: pd.Series,
    trade: BacktestTrade,
    point_value: float = 2.0,  # $2/pt for MNQ, $1.25/pt for MES
    commission_per_contract: float = 1.24,
    slippage_points: float = 0.5,
    be_offset_ticks: int = 2,
    tick_size: float = 0.25
) -> Optional[BacktestTrade]:
    """
    Evaluates a single 1-minute OHLC bar against an open trade.
    Enforces the Two-Tier Scale-Out Engine, Dynamic Profit-Lock (+2.0R -> +1.0R),
    and Pessimistic Execution Rule.

    Leg 1: 50% position exits at +1.2R (TP1).
    Ratchet 1: Once TP1 fills, move Stop Loss for Leg 2 to Entry +/- 2 ticks.
    Ratchet 2 (Dynamic Profit-Lock): If price touches +2.0R of unrealized gain,
              immediately ratchet Leg 2 Stop Loss to +1.0R.
    Leg 2: 50% position exits at +2.5R (TP2), +1.0R (Profit-Lock), or Break-Even.
    """
    low = bar["Low"]
    high = bar["High"]
    bar_time = bar.name if isinstance(bar.name, pd.Timestamp) else pd.to_datetime(bar.get("Timestamp", 0))
    be_offset = be_offset_ticks * tick_size  # 2 ticks = 0.50 pts

    if trade.direction == 1:  # Long
        r1_level = round(trade.entry_price + (1.0 * trade.initial_risk), 2)
        r2_level = round(trade.entry_price + (2.0 * trade.initial_risk), 2)
        be_sl = round(trade.entry_price + be_offset, 2)

        if not trade.leg1_filled:
            sl_touched = (low <= trade.stop_loss)
            tp1_touched = (high >= trade.tp1)

            # Pessimistic Rule: Stop Loss and TP1 breached in same candle
            if sl_touched and tp1_touched:
                trade.outcome = TradeOutcome.PESSIMISTIC_STOP
                trade.leg1_outcome = TradeOutcome.PESSIMISTIC_STOP
                trade.leg2_outcome = TradeOutcome.PESSIMISTIC_STOP
                trade.exit_price = trade.stop_loss - slippage_points
                trade.leg1_exit_price = trade.exit_price
                trade.leg2_exit_price = trade.exit_price
                trade.leg1_filled = True
                trade.leg2_filled = True
                gross_pnl = (trade.exit_price - trade.entry_price) * point_value * trade.contracts
                trade.realized_pnl = gross_pnl - (commission_per_contract * trade.contracts)
                trade.exit_time = bar_time
                return trade

            elif sl_touched:
                # Both legs stopped out at original Stop Loss
                trade.outcome = TradeOutcome.STOP_LOSS
                trade.leg1_outcome = TradeOutcome.STOP_LOSS
                trade.leg2_outcome = TradeOutcome.STOP_LOSS
                trade.exit_price = trade.stop_loss - slippage_points
                trade.leg1_exit_price = trade.exit_price
                trade.leg2_exit_price = trade.exit_price
                trade.leg1_filled = True
                trade.leg2_filled = True
                gross_pnl = (trade.exit_price - trade.entry_price) * point_value * trade.contracts
                trade.realized_pnl = gross_pnl - (commission_per_contract * trade.contracts)
                trade.exit_time = bar_time
                return trade

            elif tp1_touched:
                # Leg 1 fills at TP1 (+1.2R)!
                trade.leg1_filled = True
                trade.leg1_exit_price = trade.tp1
                trade.leg1_outcome = "TAKE_PROFIT_1"
                trade.be_triggered = True

                # Evaluate Leg 2 on the same candle
                if low <= be_sl:
                    # Pessimistic: pull-back to BE occurred
                    trade.leg2_filled = True
                    trade.leg2_exit_price = be_sl
                    trade.leg2_outcome = TradeOutcome.BREAK_EVEN
                    trade.outcome = TradeOutcome.SCALE_OUT_BREAK_EVEN
                    trade.exit_price = round((trade.leg1_exit_price + trade.leg2_exit_price) / 2.0, 2)
                    pnl1 = ((trade.leg1_exit_price - trade.entry_price) * point_value - commission_per_contract) * trade.leg1_contracts
                    pnl2 = ((trade.leg2_exit_price - trade.entry_price) * point_value - commission_per_contract) * trade.leg2_contracts
                    trade.realized_pnl = round(pnl1 + pnl2, 2)
                    trade.exit_time = bar_time
                    return trade
                else:
                    # BE stop not touched. Check for +2.0R expansion
                    if high >= r2_level:
                        trade.stop_loss = r1_level
                        trade.leg2_ratcheted_to_1r = True
                        lock_stop_touched = (low <= trade.stop_loss)
                        tp2_touched = (high >= trade.tp2)

                        if lock_stop_touched and tp2_touched:
                            trade.leg2_filled = True
                            trade.leg2_exit_price = trade.stop_loss
                            trade.leg2_outcome = TradeOutcome.PROFIT_LOCK_1R
                            trade.outcome = TradeOutcome.SCALE_OUT_PROFIT_LOCK_1R
                            trade.exit_price = round((trade.leg1_exit_price + trade.leg2_exit_price) / 2.0, 2)
                            pnl1 = ((trade.leg1_exit_price - trade.entry_price) * point_value - commission_per_contract) * trade.leg1_contracts
                            pnl2 = ((trade.leg2_exit_price - trade.entry_price) * point_value - commission_per_contract) * trade.leg2_contracts
                            trade.realized_pnl = round(pnl1 + pnl2, 2)
                            trade.exit_time = bar_time
                            return trade
                        elif lock_stop_touched:
                            trade.leg2_filled = True
                            trade.leg2_exit_price = trade.stop_loss
                            trade.leg2_outcome = TradeOutcome.PROFIT_LOCK_1R
                            trade.outcome = TradeOutcome.SCALE_OUT_PROFIT_LOCK_1R
                            trade.exit_price = round((trade.leg1_exit_price + trade.leg2_exit_price) / 2.0, 2)
                            pnl1 = ((trade.leg1_exit_price - trade.entry_price) * point_value - commission_per_contract) * trade.leg1_contracts
                            pnl2 = ((trade.leg2_exit_price - trade.entry_price) * point_value - commission_per_contract) * trade.leg2_contracts
                            trade.realized_pnl = round(pnl1 + pnl2, 2)
                            trade.exit_time = bar_time
                            return trade
                        elif tp2_touched:
                            trade.leg2_filled = True
                            trade.leg2_exit_price = trade.tp2
                            trade.leg2_outcome = TradeOutcome.TAKE_PROFIT
                            trade.outcome = TradeOutcome.SCALE_OUT_TAKE_PROFIT
                            trade.exit_price = round((trade.leg1_exit_price + trade.leg2_exit_price) / 2.0, 2)
                            pnl1 = ((trade.leg1_exit_price - trade.entry_price) * point_value - commission_per_contract) * trade.leg1_contracts
                            pnl2 = ((trade.leg2_exit_price - trade.entry_price) * point_value - commission_per_contract) * trade.leg2_contracts
                            trade.realized_pnl = round(pnl1 + pnl2, 2)
                            trade.exit_time = bar_time
                            return trade
                    else:
                        trade.stop_loss = be_sl

        else:
            # Leg 1 is already closed; Leg 2 is active
            if trade.leg2_ratcheted_to_1r:
                # Stop Loss is at +1.0R
                stop_touched = (low <= trade.stop_loss)
                tp2_touched = (high >= trade.tp2)

                if stop_touched and tp2_touched:
                    # Pessimistic: Stop hit first
                    trade.leg2_filled = True
                    trade.leg2_exit_price = trade.stop_loss
                    trade.leg2_outcome = TradeOutcome.PROFIT_LOCK_1R
                    trade.outcome = TradeOutcome.SCALE_OUT_PROFIT_LOCK_1R
                    trade.exit_price = round((trade.leg1_exit_price + trade.leg2_exit_price) / 2.0, 2)
                    pnl1 = ((trade.leg1_exit_price - trade.entry_price) * point_value - commission_per_contract) * trade.leg1_contracts
                    pnl2 = ((trade.leg2_exit_price - trade.entry_price) * point_value - commission_per_contract) * trade.leg2_contracts
                    trade.realized_pnl = round(pnl1 + pnl2, 2)
                    trade.exit_time = bar_time
                    return trade
                elif stop_touched:
                    trade.leg2_filled = True
                    trade.leg2_exit_price = trade.stop_loss
                    trade.leg2_outcome = TradeOutcome.PROFIT_LOCK_1R
                    trade.outcome = TradeOutcome.SCALE_OUT_PROFIT_LOCK_1R
                    trade.exit_price = round((trade.leg1_exit_price + trade.leg2_exit_price) / 2.0, 2)
                    pnl1 = ((trade.leg1_exit_price - trade.entry_price) * point_value - commission_per_contract) * trade.leg1_contracts
                    pnl2 = ((trade.leg2_exit_price - trade.entry_price) * point_value - commission_per_contract) * trade.leg2_contracts
                    trade.realized_pnl = round(pnl1 + pnl2, 2)
                    trade.exit_time = bar_time
                    return trade
                elif tp2_touched:
                    trade.leg2_filled = True
                    trade.leg2_exit_price = trade.tp2
                    trade.leg2_outcome = TradeOutcome.TAKE_PROFIT
                    trade.outcome = TradeOutcome.SCALE_OUT_TAKE_PROFIT
                    trade.exit_price = round((trade.leg1_exit_price + trade.leg2_exit_price) / 2.0, 2)
                    pnl1 = ((trade.leg1_exit_price - trade.entry_price) * point_value - commission_per_contract) * trade.leg1_contracts
                    pnl2 = ((trade.leg2_exit_price - trade.entry_price) * point_value - commission_per_contract) * trade.leg2_contracts
                    trade.realized_pnl = round(pnl1 + pnl2, 2)
                    trade.exit_time = bar_time
                    return trade

            else:
                # Leg 2 Stop Loss is at Entry + 2 ticks
                be_touched = (low <= trade.stop_loss)
                if be_touched:
                    trade.leg2_filled = True
                    trade.leg2_exit_price = trade.stop_loss
                    trade.leg2_outcome = TradeOutcome.BREAK_EVEN
                    trade.outcome = TradeOutcome.SCALE_OUT_BREAK_EVEN
                    trade.exit_price = round((trade.leg1_exit_price + trade.leg2_exit_price) / 2.0, 2)
                    pnl1 = ((trade.leg1_exit_price - trade.entry_price) * point_value - commission_per_contract) * trade.leg1_contracts
                    pnl2 = ((trade.leg2_exit_price - trade.entry_price) * point_value - commission_per_contract) * trade.leg2_contracts
                    trade.realized_pnl = round(pnl1 + pnl2, 2)
                    trade.exit_time = bar_time
                    return trade
                else:
                    # Check if high reaches +2.0R to ratchet Leg 2 to +1.0R
                    if high >= r2_level:
                        trade.stop_loss = r1_level
                        trade.leg2_ratcheted_to_1r = True
                        lock_stop_touched = (low <= trade.stop_loss)
                        tp2_touched = (high >= trade.tp2)

                        if lock_stop_touched and tp2_touched:
                            trade.leg2_filled = True
                            trade.leg2_exit_price = trade.stop_loss
                            trade.leg2_outcome = TradeOutcome.PROFIT_LOCK_1R
                            trade.outcome = TradeOutcome.SCALE_OUT_PROFIT_LOCK_1R
                            trade.exit_price = round((trade.leg1_exit_price + trade.leg2_exit_price) / 2.0, 2)
                            pnl1 = ((trade.leg1_exit_price - trade.entry_price) * point_value - commission_per_contract) * trade.leg1_contracts
                            pnl2 = ((trade.leg2_exit_price - trade.entry_price) * point_value - commission_per_contract) * trade.leg2_contracts
                            trade.realized_pnl = round(pnl1 + pnl2, 2)
                            trade.exit_time = bar_time
                            return trade
                        elif lock_stop_touched:
                            trade.leg2_filled = True
                            trade.leg2_exit_price = trade.stop_loss
                            trade.leg2_outcome = TradeOutcome.PROFIT_LOCK_1R
                            trade.outcome = TradeOutcome.SCALE_OUT_PROFIT_LOCK_1R
                            trade.exit_price = round((trade.leg1_exit_price + trade.leg2_exit_price) / 2.0, 2)
                            pnl1 = ((trade.leg1_exit_price - trade.entry_price) * point_value - commission_per_contract) * trade.leg1_contracts
                            pnl2 = ((trade.leg2_exit_price - trade.entry_price) * point_value - commission_per_contract) * trade.leg2_contracts
                            trade.realized_pnl = round(pnl1 + pnl2, 2)
                            trade.exit_time = bar_time
                            return trade
                        elif tp2_touched:
                            trade.leg2_filled = True
                            trade.leg2_exit_price = trade.tp2
                            trade.leg2_outcome = TradeOutcome.TAKE_PROFIT
                            trade.outcome = TradeOutcome.SCALE_OUT_TAKE_PROFIT
                            trade.exit_price = round((trade.leg1_exit_price + trade.leg2_exit_price) / 2.0, 2)
                            pnl1 = ((trade.leg1_exit_price - trade.entry_price) * point_value - commission_per_contract) * trade.leg1_contracts
                            pnl2 = ((trade.leg2_exit_price - trade.entry_price) * point_value - commission_per_contract) * trade.leg2_contracts
                            trade.realized_pnl = round(pnl1 + pnl2, 2)
                            trade.exit_time = bar_time
                            return trade
                    elif high >= trade.tp2:
                        trade.leg2_filled = True
                        trade.leg2_exit_price = trade.tp2
                        trade.leg2_outcome = TradeOutcome.TAKE_PROFIT
                        trade.outcome = TradeOutcome.SCALE_OUT_TAKE_PROFIT
                        trade.exit_price = round((trade.leg1_exit_price + trade.leg2_exit_price) / 2.0, 2)
                        pnl1 = ((trade.leg1_exit_price - trade.entry_price) * point_value - commission_per_contract) * trade.leg1_contracts
                        pnl2 = ((trade.leg2_exit_price - trade.entry_price) * point_value - commission_per_contract) * trade.leg2_contracts
                        trade.realized_pnl = round(pnl1 + pnl2, 2)
                        trade.exit_time = bar_time
                        return trade

    elif trade.direction == -1:  # Short
        r1_level = round(trade.entry_price - (1.0 * trade.initial_risk), 2)
        r2_level = round(trade.entry_price - (2.0 * trade.initial_risk), 2)
        be_sl = round(trade.entry_price - be_offset, 2)

        if not trade.leg1_filled:
            sl_touched = (high >= trade.stop_loss)
            tp1_touched = (low <= trade.tp1)

            # Pessimistic Rule: Stop Loss and TP1 breached in same candle
            if sl_touched and tp1_touched:
                trade.outcome = TradeOutcome.PESSIMISTIC_STOP
                trade.leg1_outcome = TradeOutcome.PESSIMISTIC_STOP
                trade.leg2_outcome = TradeOutcome.PESSIMISTIC_STOP
                trade.exit_price = trade.stop_loss + slippage_points
                trade.leg1_exit_price = trade.exit_price
                trade.leg2_exit_price = trade.exit_price
                trade.leg1_filled = True
                trade.leg2_filled = True
                gross_pnl = (trade.entry_price - trade.exit_price) * point_value * trade.contracts
                trade.realized_pnl = gross_pnl - (commission_per_contract * trade.contracts)
                trade.exit_time = bar_time
                return trade

            elif sl_touched:
                trade.outcome = TradeOutcome.STOP_LOSS
                trade.leg1_outcome = TradeOutcome.STOP_LOSS
                trade.leg2_outcome = TradeOutcome.STOP_LOSS
                trade.exit_price = trade.stop_loss + slippage_points
                trade.leg1_exit_price = trade.exit_price
                trade.leg2_exit_price = trade.exit_price
                trade.leg1_filled = True
                trade.leg2_filled = True
                gross_pnl = (trade.entry_price - trade.exit_price) * point_value * trade.contracts
                trade.realized_pnl = gross_pnl - (commission_per_contract * trade.contracts)
                trade.exit_time = bar_time
                return trade

            elif tp1_touched:
                # Leg 1 fills at TP1 (-1.2R)
                trade.leg1_filled = True
                trade.leg1_exit_price = trade.tp1
                trade.leg1_outcome = "TAKE_PROFIT_1"
                trade.be_triggered = True

                if high >= be_sl:
                    trade.leg2_filled = True
                    trade.leg2_exit_price = be_sl
                    trade.leg2_outcome = TradeOutcome.BREAK_EVEN
                    trade.outcome = TradeOutcome.SCALE_OUT_BREAK_EVEN
                    trade.exit_price = round((trade.leg1_exit_price + trade.leg2_exit_price) / 2.0, 2)
                    pnl1 = ((trade.entry_price - trade.leg1_exit_price) * point_value - commission_per_contract) * trade.leg1_contracts
                    pnl2 = ((trade.entry_price - trade.leg2_exit_price) * point_value - commission_per_contract) * trade.leg2_contracts
                    trade.realized_pnl = round(pnl1 + pnl2, 2)
                    trade.exit_time = bar_time
                    return trade
                else:
                    if low <= r2_level:
                        trade.stop_loss = r1_level
                        trade.leg2_ratcheted_to_1r = True
                        lock_stop_touched = (high >= trade.stop_loss)
                        tp2_touched = (low <= trade.tp2)

                        if lock_stop_touched and tp2_touched:
                            trade.leg2_filled = True
                            trade.leg2_exit_price = trade.stop_loss
                            trade.leg2_outcome = TradeOutcome.PROFIT_LOCK_1R
                            trade.outcome = TradeOutcome.SCALE_OUT_PROFIT_LOCK_1R
                            trade.exit_price = round((trade.leg1_exit_price + trade.leg2_exit_price) / 2.0, 2)
                            pnl1 = ((trade.entry_price - trade.leg1_exit_price) * point_value - commission_per_contract) * trade.leg1_contracts
                            pnl2 = ((trade.entry_price - trade.leg2_exit_price) * point_value - commission_per_contract) * trade.leg2_contracts
                            trade.realized_pnl = round(pnl1 + pnl2, 2)
                            trade.exit_time = bar_time
                            return trade
                        elif lock_stop_touched:
                            trade.leg2_filled = True
                            trade.leg2_exit_price = trade.stop_loss
                            trade.leg2_outcome = TradeOutcome.PROFIT_LOCK_1R
                            trade.outcome = TradeOutcome.SCALE_OUT_PROFIT_LOCK_1R
                            trade.exit_price = round((trade.leg1_exit_price + trade.leg2_exit_price) / 2.0, 2)
                            pnl1 = ((trade.entry_price - trade.leg1_exit_price) * point_value - commission_per_contract) * trade.leg1_contracts
                            pnl2 = ((trade.entry_price - trade.leg2_exit_price) * point_value - commission_per_contract) * trade.leg2_contracts
                            trade.realized_pnl = round(pnl1 + pnl2, 2)
                            trade.exit_time = bar_time
                            return trade
                        elif tp2_touched:
                            trade.leg2_filled = True
                            trade.leg2_exit_price = trade.tp2
                            trade.leg2_outcome = TradeOutcome.TAKE_PROFIT
                            trade.outcome = TradeOutcome.SCALE_OUT_TAKE_PROFIT
                            trade.exit_price = round((trade.leg1_exit_price + trade.leg2_exit_price) / 2.0, 2)
                            pnl1 = ((trade.entry_price - trade.leg1_exit_price) * point_value - commission_per_contract) * trade.leg1_contracts
                            pnl2 = ((trade.entry_price - trade.leg2_exit_price) * point_value - commission_per_contract) * trade.leg2_contracts
                            trade.realized_pnl = round(pnl1 + pnl2, 2)
                            trade.exit_time = bar_time
                            return trade
                    else:
                        trade.stop_loss = be_sl

        else:
            if trade.leg2_ratcheted_to_1r:
                stop_touched = (high >= trade.stop_loss)
                tp2_touched = (low <= trade.tp2)

                if stop_touched and tp2_touched:
                    trade.leg2_filled = True
                    trade.leg2_exit_price = trade.stop_loss
                    trade.leg2_outcome = TradeOutcome.PROFIT_LOCK_1R
                    trade.outcome = TradeOutcome.SCALE_OUT_PROFIT_LOCK_1R
                    trade.exit_price = round((trade.leg1_exit_price + trade.leg2_exit_price) / 2.0, 2)
                    pnl1 = ((trade.entry_price - trade.leg1_exit_price) * point_value - commission_per_contract) * trade.leg1_contracts
                    pnl2 = ((trade.entry_price - trade.leg2_exit_price) * point_value - commission_per_contract) * trade.leg2_contracts
                    trade.realized_pnl = round(pnl1 + pnl2, 2)
                    trade.exit_time = bar_time
                    return trade
                elif stop_touched:
                    trade.leg2_filled = True
                    trade.leg2_exit_price = trade.stop_loss
                    trade.leg2_outcome = TradeOutcome.PROFIT_LOCK_1R
                    trade.outcome = TradeOutcome.SCALE_OUT_PROFIT_LOCK_1R
                    trade.exit_price = round((trade.leg1_exit_price + trade.leg2_exit_price) / 2.0, 2)
                    pnl1 = ((trade.entry_price - trade.leg1_exit_price) * point_value - commission_per_contract) * trade.leg1_contracts
                    pnl2 = ((trade.entry_price - trade.leg2_exit_price) * point_value - commission_per_contract) * trade.leg2_contracts
                    trade.realized_pnl = round(pnl1 + pnl2, 2)
                    trade.exit_time = bar_time
                    return trade
                elif tp2_touched:
                    trade.leg2_filled = True
                    trade.leg2_exit_price = trade.tp2
                    trade.leg2_outcome = TradeOutcome.TAKE_PROFIT
                    trade.outcome = TradeOutcome.SCALE_OUT_TAKE_PROFIT
                    trade.exit_price = round((trade.leg1_exit_price + trade.leg2_exit_price) / 2.0, 2)
                    pnl1 = ((trade.entry_price - trade.leg1_exit_price) * point_value - commission_per_contract) * trade.leg1_contracts
                    pnl2 = ((trade.entry_price - trade.leg2_exit_price) * point_value - commission_per_contract) * trade.leg2_contracts
                    trade.realized_pnl = round(pnl1 + pnl2, 2)
                    trade.exit_time = bar_time
                    return trade

            else:
                be_touched = (high >= trade.stop_loss)
                if be_touched:
                    trade.leg2_filled = True
                    trade.leg2_exit_price = trade.stop_loss
                    trade.leg2_outcome = TradeOutcome.BREAK_EVEN
                    trade.outcome = TradeOutcome.SCALE_OUT_BREAK_EVEN
                    trade.exit_price = round((trade.leg1_exit_price + trade.leg2_exit_price) / 2.0, 2)
                    pnl1 = ((trade.entry_price - trade.leg1_exit_price) * point_value - commission_per_contract) * trade.leg1_contracts
                    pnl2 = ((trade.entry_price - trade.leg2_exit_price) * point_value - commission_per_contract) * trade.leg2_contracts
                    trade.realized_pnl = round(pnl1 + pnl2, 2)
                    trade.exit_time = bar_time
                    return trade
                else:
                    if low <= r2_level:
                        trade.stop_loss = r1_level
                        trade.leg2_ratcheted_to_1r = True
                        lock_stop_touched = (high >= trade.stop_loss)
                        tp2_touched = (low <= trade.tp2)

                        if lock_stop_touched and tp2_touched:
                            trade.leg2_filled = True
                            trade.leg2_exit_price = trade.stop_loss
                            trade.leg2_outcome = TradeOutcome.PROFIT_LOCK_1R
                            trade.outcome = TradeOutcome.SCALE_OUT_PROFIT_LOCK_1R
                            trade.exit_price = round((trade.leg1_exit_price + trade.leg2_exit_price) / 2.0, 2)
                            pnl1 = ((trade.entry_price - trade.leg1_exit_price) * point_value - commission_per_contract) * trade.leg1_contracts
                            pnl2 = ((trade.entry_price - trade.leg2_exit_price) * point_value - commission_per_contract) * trade.leg2_contracts
                            trade.realized_pnl = round(pnl1 + pnl2, 2)
                            trade.exit_time = bar_time
                            return trade
                        elif lock_stop_touched:
                            trade.leg2_filled = True
                            trade.leg2_exit_price = trade.stop_loss
                            trade.leg2_outcome = TradeOutcome.PROFIT_LOCK_1R
                            trade.outcome = TradeOutcome.SCALE_OUT_PROFIT_LOCK_1R
                            trade.exit_price = round((trade.leg1_exit_price + trade.leg2_exit_price) / 2.0, 2)
                            pnl1 = ((trade.entry_price - trade.leg1_exit_price) * point_value - commission_per_contract) * trade.leg1_contracts
                            pnl2 = ((trade.entry_price - trade.leg2_exit_price) * point_value - commission_per_contract) * trade.leg2_contracts
                            trade.realized_pnl = round(pnl1 + pnl2, 2)
                            trade.exit_time = bar_time
                            return trade
                        elif tp2_touched:
                            trade.leg2_filled = True
                            trade.leg2_exit_price = trade.tp2
                            trade.leg2_outcome = TradeOutcome.TAKE_PROFIT
                            trade.outcome = TradeOutcome.SCALE_OUT_TAKE_PROFIT
                            trade.exit_price = round((trade.leg1_exit_price + trade.leg2_exit_price) / 2.0, 2)
                            pnl1 = ((trade.entry_price - trade.leg1_exit_price) * point_value - commission_per_contract) * trade.leg1_contracts
                            pnl2 = ((trade.entry_price - trade.leg2_exit_price) * point_value - commission_per_contract) * trade.leg2_contracts
                            trade.realized_pnl = round(pnl1 + pnl2, 2)
                            trade.exit_time = bar_time
                            return trade
                    elif low <= trade.tp2:
                        trade.leg2_filled = True
                        trade.leg2_exit_price = trade.tp2
                        trade.leg2_outcome = TradeOutcome.TAKE_PROFIT
                        trade.outcome = TradeOutcome.SCALE_OUT_TAKE_PROFIT
                        trade.exit_price = round((trade.leg1_exit_price + trade.leg2_exit_price) / 2.0, 2)
                        pnl1 = ((trade.entry_price - trade.leg1_exit_price) * point_value - commission_per_contract) * trade.leg1_contracts
                        pnl2 = ((trade.entry_price - trade.leg2_exit_price) * point_value - commission_per_contract) * trade.leg2_contracts
                        trade.realized_pnl = round(pnl1 + pnl2, 2)
                        trade.exit_time = bar_time
                        return trade

    return None
