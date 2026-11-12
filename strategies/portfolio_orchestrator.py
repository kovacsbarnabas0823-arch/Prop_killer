"""
Multi-Alpha Portfolio Orchestrator (Phase 3 Institutional)
Author: Principal Quantitative Architect & Financial Engineer

Coordinates:
  - H3: Multi-Horizon Adaptive Noise-Band Momentum Engine (10:00 - 14:00 NY)
  - H4: Late-Day Leveraged ETF Flow & 0DTE Gamma Pin (14:30 / 15:00 NY)

Enforces:
  - Strict chronological order flow and single-position concurrency guard.
  - Decoupled Two-Regime Volatility-Normalized Stochastic Position Sizing.
  - Internal Protective Daily Stop (-$600.00).
  - Skewness-Preserving Funded Consistency Governor (+$920.00 ceiling, $52,650 payout target).
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
from strategies.h3_noise_band_momentum import H3NoiseBandMomentum
from strategies.h4_letf_1500_rebalance import H4LETF1500Rebalance

logger = logging.getLogger("PortfolioOrchestrator")


class PortfolioOrchestrator:
    def __init__(
        self,
        account_state: Optional[TradeifyAccountState] = None,
        simulator: Optional[PessimisticExecutionSimulator] = None,
        enabled_modules: Optional[List[str]] = None
    ):
        self.account = account_state or TradeifyAccountState()
        self.sim = simulator or PessimisticExecutionSimulator()
        self.enabled_modules = enabled_modules or ["H3", "H4"]

        # Instantiate alpha modules
        self.h3 = H3NoiseBandMomentum()
        self.h4 = H4LETF1500Rebalance()

        self.all_completed_trades: List[CompletedTrade] = []
        self.daily_pnl_series: Dict[Any, float] = {}
        self.strategy_pnl_series: Dict[str, List[CompletedTrade]] = {
            "H3": [], "H4": []
        }

    def simulate_session(
        self,
        session_date,
        df_session_1m: pd.DataFrame,
        daily_atr: float,
        prev_close: Optional[float] = None,
        prep=None,
        signals: Optional[List[TradeSignal]] = None
    ) -> List[CompletedTrade]:
        self.account.start_session(session_date)
        session_trades: List[CompletedTrade] = []

        if df_session_1m.empty or daily_atr is None or np.isnan(daily_atr):
            self.account.end_session(session_date)
            return []

        # 1. Collect signals from enabled strategies if not provided
        if signals is None:
            signals = []
            if "H3" in self.enabled_modules:
                signals.extend(self.h3.evaluate_session(
                    session_date, df_session_1m, daily_atr, prev_close, preprocessor=prep
                ))

            if "H4" in self.enabled_modules:
                signals.extend(self.h4.evaluate_session(session_date, df_session_1m, daily_atr, prev_close))
        else:
            signals = list(signals)

        if not signals:
            self.account.end_session(session_date)
            return []

        # Sort signals strictly chronologically
        signals.sort(key=lambda s: s.execution_time)

        # Precompute VWAP and noise bands for dynamic trailing
        rth_bars = df_session_1m.between_time("09:30", "15:44")
        df_ind = pd.DataFrame(index=rth_bars.index)
        if not rth_bars.empty:
            tp = (rth_bars["High"] + rth_bars["Low"] + rth_bars["Close"]) / 3.0
            cum_vol = rth_bars["Volume"].cumsum()
            df_ind["VWAP"] = (tp * rth_bars["Volume"]).cumsum() / cum_vol.replace(0, np.nan)

            if prep and session_date in prep.noise_band_map:
                u_bands = []
                l_bands = []
                for t in rth_bars.index:
                    t_str = t.strftime("%H:%M")
                    b = prep.get_noise_band(session_date, t_str)
                    if b:
                        u_bands.append(b[0])
                        l_bands.append(b[1])
                    else:
                        u_bands.append(np.nan)
                        l_bands.append(np.nan)
                df_ind["UpperBand"] = pd.Series(u_bands, index=rth_bars.index).ffill()
                df_ind["LowerBand"] = pd.Series(l_bands, index=rth_bars.index).ffill()

        # 2. Sequential execution loop with concurrent late-day pyramiding
        last_exit_time = pd.Timestamp.min.tz_localize(df_session_1m.index.tz)
        last_exit_reason: Optional[str] = None
        morning_trade_pnl = 0.0

        for sig in signals:
            is_concurrent_pyramid = False
            effective_morning_pnl = morning_trade_pnl

            # Check if previous trade is still open
            if sig.execution_time < last_exit_time:
                # Concurrent Same-Direction Pyramiding Unlock for H4
                if sig.strategy_id == "H4":
                    open_trades = [t for t in session_trades if t.exit_time > sig.execution_time]
                    if open_trades and open_trades[0].direction == sig.direction:
                        h3_tr = open_trades[0]
                        # Compute H3's unrealized PnL at H4 execution time
                        bars_to_exec = df_session_1m.loc[:sig.execution_time]
                        if not bars_to_exec.empty:
                            close_exec = float(bars_to_exec.iloc[-1]["Close"])
                            h3_unreal = (close_exec - h3_tr.entry_price) * h3_tr.direction * 2.00 * h3_tr.contracts - (2.48 * h3_tr.contracts)
                            if self.account.phase == AccountPhase.EVALUATION:
                                if h3_unreal >= 100.0:
                                    is_concurrent_pyramid = True
                            else: # FUNDED
                                effective_morning_pnl = morning_trade_pnl + max(0.0, 0.70 * h3_unreal)
                                if effective_morning_pnl >= 165.0:
                                    is_concurrent_pyramid = True

                if not is_concurrent_pyramid:
                    continue

            # Continuation Re-entry Filter: Requires daily PnL > -$300 and prior exit was not hard SL
            if sig.metadata.get("is_reentry", False):
                if self.account.intraday_pnl <= -300.0 or last_exit_reason == "SL":
                    continue

            if self.account.is_breached or self.account.is_daily_stopped or self.account.is_governor_halted:
                break

            optimal_qty = self.account.get_optimal_mnq_size(
                strategy_id=sig.strategy_id,
                sl_dist_pts=sig.sl_dist,
                atr_14=daily_atr,
                morning_pnl=effective_morning_pnl
            )
            if optimal_qty <= 0:
                continue

            # Execute trade through pessimistic simulator
            trades = self.sim.simulate_trade(
                signal=sig,
                session_bars=df_session_1m,
                contract_qty=optimal_qty,
                df_indicators=df_ind
            )

            if not trades:
                continue

            trade_net_pnl = sum(t.net_pnl for t in trades)
            last_exit_time = max(last_exit_time, max(t.exit_time for t in trades))
            last_exit_reason = trades[-1].exit_reason

            if sig.strategy_id == "H3":
                morning_trade_pnl += trade_net_pnl

            can_continue, halt_reason = self.account.update_intraday_pnl(trade_net_pnl)
            self.account.close_trade(trade_net_pnl)

            session_trades.extend(trades)
            self.all_completed_trades.extend(trades)
            self.strategy_pnl_series[sig.strategy_id].extend(trades)

            if not can_continue:
                break

        self.account.end_session(session_date)
        day_pnl = sum(t.net_pnl for t in session_trades)
        self.daily_pnl_series[session_date] = day_pnl

        return session_trades
