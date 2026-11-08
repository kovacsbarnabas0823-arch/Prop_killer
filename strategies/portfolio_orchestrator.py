"""
Multi-Alpha Portfolio Orchestrator (strategies/portfolio_orchestrator.py)
Author: Principal Quantitative Architect

Coordinates:
  - H1: Upgraded 30m Filtered ORB (10:00 - 13:30 NY)
  - H2: NY Open London Sweep Reversal (09:30 - 10:30 NY)
  - H3: LOB 4th-Touch Breakout Cascade (10:15 - 14:15 NY)
  - H4: Leveraged ETF 15:00 Rebalance + 0DTE Gamma Filter (15:00 - 15:44 NY)

Enforces:
  - Strict chronological order flow and single-position concurrency guard.
  - Dynamic Bellman Position Sizing via TradeifyAccountState.
  - Internal Protective Daily Stop (-$600.00).
  - Funded Consistency Governor Ceiling (+$825.00).
  - H4 Profit Protection (caps risk to 25% of morning profits if up >= +$400).
  - Hard EOD session reconciliation at 15:45:00 EST.
"""

from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from engine.tradeify_rules import TradeifyAccountState, AccountPhase
from engine.pessimistic_sim import PessimisticExecutionSimulator
from strategies.base_strategy import TradeSignal, CompletedTrade
from strategies.h1_orb_30m_regime import H1ORB30mRegime
from strategies.h2_ny_london_sweep import H2NYLondonSweep
from strategies.h3_lob_4th_touch import H3LOB4thTouch
from strategies.h4_letf_1500_rebalance import H4LETF1500Rebalance

logger = logging.getLogger("PortfolioOrchestrator")


class PortfolioOrchestrator:
    def __init__(
        self,
        account_state: Optional[TradeifyAccountState] = None,
        simulator: Optional[PessimisticExecutionSimulator] = None
    ):
        self.account = account_state or TradeifyAccountState()
        self.sim = simulator or PessimisticExecutionSimulator()

        # Instantiate all 4 alpha modules
        self.h1 = H1ORB30mRegime()
        self.h2 = H2NYLondonSweep()
        self.h3 = H3LOB4thTouch()
        self.h4 = H4LETF1500Rebalance()

        self.all_completed_trades: List[CompletedTrade] = []
        self.daily_pnl_series: Dict[Any, float] = {}
        self.strategy_pnl_series: Dict[str, List[CompletedTrade]] = {
            "H1": [], "H2": [], "H3": [], "H4": []
        }

    def simulate_session(
        self,
        session_date,
        df_session_1m: pd.DataFrame,
        daily_atr: float,
        prev_close: Optional[float] = None
    ) -> List[CompletedTrade]:
        """
        Executes a single session across all 4 alpha modules with real-time risk checks.
        """
        self.account.start_session(session_date)
        session_trades: List[CompletedTrade] = []

        if df_session_1m.empty or daily_atr is None or np.isnan(daily_atr):
            self.account.end_session(session_date)
            return []

        # 1. Collect candidate signals from all strategies
        # (Strategies evaluate strictly causal features)
        signals: List[TradeSignal] = []
        signals.extend(self.h2.evaluate_session(session_date, df_session_1m, daily_atr, prev_close))
        signals.extend(self.h1.evaluate_session(session_date, df_session_1m, daily_atr, prev_close))
        signals.extend(self.h3.evaluate_session(session_date, df_session_1m, daily_atr, prev_close))
        signals.extend(self.h4.evaluate_session(session_date, df_session_1m, daily_atr, prev_close))

        if not signals:
            self.account.end_session(session_date)
            return []

        # Sort signals strictly by execution timestamp
        signals.sort(key=lambda s: s.execution_time)

        # 2. Sequential execution loop with state-dependent risk controls
        last_exit_time = pd.Timestamp.min.tz_localize(df_session_1m.index.tz)

        for sig in signals:
            # Concurrency guard: Do not open overlapping trades
            if sig.execution_time < last_exit_time:
                continue

            # Check if account is daily stopped or breached
            if self.account.is_breached or self.account.is_daily_stopped or self.account.is_governor_halted:
                break

            # Bellman dynamic contract sizing
            optimal_qty = self.account.get_optimal_mnq_size()
            if optimal_qty <= 0:
                break

            # H4 Intraday Profit Protection Check:
            # If account already up >= +$400 on the day, cap dollar risk to 25% of profit
            if sig.strategy_id == "H4" and self.account.intraday_pnl >= 400.0:
                max_risk_allowed = 0.25 * self.account.intraday_pnl
                risk_per_contract = sig.sl_dist * 2.0  # $2/pt for MNQ
                capped_qty = int(max_risk_allowed / risk_per_contract)
                if capped_qty < 2:
                    # Risk exceeds 25% profit cap - skip trade
                    continue
                optimal_qty = min(optimal_qty, capped_qty)

            # Build indicator slice for dynamic trail (e.g. 15m 12-EMA, 1m 9-EMA)
            rth_bars = df_session_1m.between_time("09:30", "15:44")
            df_ind = pd.DataFrame(index=rth_bars.index)
            if sig.trail_rule == "EMA15_12":
                res_15m = rth_bars.resample("15min", closed="left", label="left")["Close"].last().dropna()
                ema12 = res_15m.ewm(span=12, adjust=False).mean()
                df_ind["EMA15_12"] = ema12.reindex(rth_bars.index, method="ffill")
            elif sig.trail_rule == "EMA1_9":
                df_ind["EMA1_9"] = rth_bars["Close"].ewm(span=9, adjust=False).mean()

            # Execute trade through pessimistic simulator
            trades = self.sim.simulate_trade(
                signal=sig,
                session_bars=df_session_1m,
                contract_qty=optimal_qty,
                df_indicators=df_ind
            )

            if not trades:
                continue

            # Process completed trade legs
            trade_net_pnl = sum(t.net_pnl for t in trades)
            last_exit_time = max(t.exit_time for t in trades)

            # Update account state
            can_continue, halt_reason = self.account.update_intraday_pnl(trade_net_pnl)
            self.account.close_trade(trade_net_pnl)

            session_trades.extend(trades)
            self.all_completed_trades.extend(trades)
            self.strategy_pnl_series[sig.strategy_id].extend(trades)

            if not can_continue:
                # Account halted or breached
                break

        # Reconcile EOD state
        self.account.end_session(session_date)
        day_pnl = sum(t.net_pnl for t in session_trades)
        self.daily_pnl_series[session_date] = day_pnl

        return session_trades
