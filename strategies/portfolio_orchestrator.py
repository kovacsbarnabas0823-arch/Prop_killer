"""
Multi-Alpha Portfolio Orchestrator (Phase 2 Upgraded)
Author: Principal Quantitative Architect

Coordinates:
  - H1: Upgraded 30m Filtered ORB (10:00 - 11:15 NY, Asymmetric 1:2.5R)
  - H2: NY Open London Sweep Reversal (09:30 - 10:15 NY, 5m Reversal Wick)
  - H3: Zarattini-Baltussen Noise-Boundary Momentum (10:30 - 13:30 NY)
  - H4: Two-Wave Leveraged ETF Flow & 0DTE Gamma Pin (14:30 / 15:00 NY)

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
        self.enabled_modules = enabled_modules or ["H1", "H2", "H3", "H4"]

        # Instantiate alpha modules
        self.h1 = H1ORB30mRegime()
        self.h2 = H2NYLondonSweep()
        self.h3 = H3NoiseBandMomentum()
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
        prev_close: Optional[float] = None,
        prep=None
    ) -> List[CompletedTrade]:
        self.account.start_session(session_date)
        session_trades: List[CompletedTrade] = []

        if df_session_1m.empty or daily_atr is None or np.isnan(daily_atr):
            self.account.end_session(session_date)
            return []

        # Extract prior day features from preprocessor
        prev_h = prep.prev_day_high_map.get(session_date) if prep else None
        prev_l = prep.prev_day_low_map.get(session_date) if prep else None
        prev_rng = prep.prev_day_range_map.get(session_date) if prep else None
        prev2_h = prep.prev2_day_high_map.get(session_date) if prep else None
        prev2_l = prep.prev2_day_low_map.get(session_date) if prep else None

        # 1. Collect signals from enabled strategies
        signals: List[TradeSignal] = []
        if "H2" in self.enabled_modules:
            signals.extend(self.h2.evaluate_session(session_date, df_session_1m, daily_atr, prev_close))

        if "H1" in self.enabled_modules:
            signals.extend(self.h1.evaluate_session(
                session_date, df_session_1m, daily_atr, prev_close,
                prev_high=prev_h, prev_low=prev_l, prev_range=prev_rng,
                prev2_high=prev2_h, prev2_low=prev2_l
            ))

        if "H3" in self.enabled_modules:
            signals.extend(self.h3.evaluate_session(
                session_date, df_session_1m, daily_atr, prev_close, preprocessor=prep
            ))

        if "H4" in self.enabled_modules:
            signals.extend(self.h4.evaluate_session(session_date, df_session_1m, daily_atr, prev_close))

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

        # 2. Sequential execution loop with risk controls
        last_exit_time = pd.Timestamp.min.tz_localize(df_session_1m.index.tz)

        for sig in signals:
            if sig.execution_time < last_exit_time:
                continue

            if self.account.is_breached or self.account.is_daily_stopped or self.account.is_governor_halted:
                break

            optimal_qty = self.account.get_optimal_mnq_size()
            if optimal_qty <= 0:
                break

            # H4 Intraday Profit Protection
            if sig.strategy_id == "H4" and self.account.intraday_pnl >= 400.0:
                max_risk_allowed = 0.25 * self.account.intraday_pnl
                risk_per_contract = sig.sl_dist * 2.0
                capped_qty = int(max_risk_allowed / risk_per_contract)
                if capped_qty < 2:
                    continue
                optimal_qty = min(optimal_qty, capped_qty)

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
            last_exit_time = max(t.exit_time for t in trades)

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
