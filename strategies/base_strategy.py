"""
Base Strategy Interface & Standardized Data Structures
Author: Principal Quantitative Architect

Standardizes signal output, risk geometry, order execution contracts,
and trade reporting across all quantitative alpha modules.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Dict, Any, List, Optional
import pandas as pd


@dataclass
class TradeSignal:
    strategy_id: str                              # e.g. "H1", "H2", "H3", "H4"
    strategy_name: str                            # Human-readable name
    timestamp: pd.Timestamp                       # Signal timestamp (bar close / evaluation)
    execution_time: pd.Timestamp                  # Exact fill timestamp (next bar open)
    direction: int                                # +1 for Long, -1 for Short
    entry_price: float                            # Simulated fill price with slippage
    stop_loss_price: float                        # Initial structural stop
    tp1_price: Optional[float] = None             # Target price (if scale_out_tp1=False, this is 100% target)
    tp2_price: Optional[float] = None             # Runner target (if scale_out_tp1=True)
    sl_dist: float = 0.0                          # Distance to stop in points
    tp1_dist: float = 0.0                         # Distance to TP1 in points
    tp2_dist: float = 0.0                         # Distance to TP2 in points
    scale_out_tp1: bool = False                   # False = Hold 100% position to asymmetric target/trail/EOD
    be_trigger_r: float = 1.5                     # Move stop to BE only after reaching +be_trigger_r * sl_dist
    breakeven_trigger_price: Optional[float] = None
    trail_rule: Optional[str] = None              # "VWAP", "VWAP_BAND", "EMA15_12"
    trail_activation_r: float = 0.0               # R-multiple needed to engage trail (0.0 = immediate)
    scratch_bars: Optional[int] = None            # Fast scratch timeout
    scratch_threshold_r: float = 0.5
    hard_time_exit: Optional[str] = None          # e.g. "12:00:00", "15:44:00"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletedTrade:
    trade_id: int
    strategy_id: str
    strategy_name: str
    direction: int
    contracts: int
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    gross_points: float
    gross_pnl: float
    commission: float
    net_pnl: float
    exit_reason: str                              # "TP1", "TP_FULL", "SL", "BE", "SCRATCH", "TRAIL_EXIT", "TIME_EXIT", "EOD_FLATTEN"
    hold_duration_seconds: float
    hold_duration_minutes: float
    r_multiple: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """
    Abstract interface for all quantitative alpha strategies.
    """

    def __init__(self, name: str, strategy_id: str):
        self.name = name
        self.strategy_id = strategy_id

    @abstractmethod
    def evaluate_session(
        self,
        session_date,
        df_session_1m: pd.DataFrame,
        daily_atr: float,
        prev_close: Optional[float] = None,
        **kwargs
    ) -> List[TradeSignal]:
        """
        Evaluates a single session's 1-minute bars and emits zero or more TradeSignals.
        Must be strictly causal and zero lookahead.
        """
        pass
