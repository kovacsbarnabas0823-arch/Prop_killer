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
from datetime import datetime, date
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

            # Execute trade through pessimistic simulator with Phase 7 absorbing barrier & ratchet monitoring
            trades = self.sim.simulate_trade(
                signal=sig,
                session_bars=df_session_1m,
                contract_qty=optimal_qty,
                df_indicators=df_ind,
                account_phase=self.account.phase,
                current_equity_before_trade=self.account.current_equity,
                intraday_realized_pnl=self.account.intraday_pnl,
                qualifying_days=self.account.funded_winning_days,
                max_historical_day_pnl=self.account.max_historical_day_pnl
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

            # Phase 7: Absorbing upper barrier auto-flattening halts further session activity
            if trades[-1].exit_reason == "TP_EVAL_PASS":
                self.account.is_passed = True
                can_continue = False
            elif trades[-1].exit_reason == "TP_FUNDED_PAYOUT":
                self.account.is_funded_payout_eligible = True
                can_continue = False

            session_trades.extend(trades)
            self.all_completed_trades.extend(trades)
            self.strategy_pnl_series[sig.strategy_id].extend(trades)

            if not can_continue:
                break

        self.account.end_session(session_date)
        day_pnl = sum(t.net_pnl for t in session_trades)
        self.daily_pnl_series[session_date] = day_pnl

        return session_trades


# =====================================================================
# Phase 9 State-Pure Risk-Tier Lookup & Realistic Reset Simulators
# =====================================================================

TIER_SPECS = {
    90:  {"r_target": 90.0,  "min_c": 1, "max_c": 2, "is_funded": True,  "pre_lock": True},
    120: {"r_target": 120.0, "min_c": 1, "max_c": 2, "is_funded": False, "pre_lock": False},
    165: {"r_target": 165.0, "min_c": 2, "max_c": 3, "is_funded": True,  "pre_lock": True},
    170: {"r_target": 170.0, "min_c": 2, "max_c": 3, "is_funded": False, "pre_lock": False},
    245: {"r_target": 245.0, "min_c": 3, "max_c": 6, "is_funded": True,  "pre_lock": False},
    250: {"r_target": 250.0, "min_c": 3, "max_c": 4, "is_funded": False, "pre_lock": False},
    330: {"r_target": 270.0, "min_c": 3, "max_c": 5, "is_funded": False, "pre_lock": False},
}


def get_tier_contracts(sig: TradeSignal, atr: float, r_target: float, min_c: int, max_c: int) -> int:
    eff_atr = atr if (atr and not np.isnan(atr) and atr > 0) else 200.0
    ref_pts = max(16.0, min(22.0, 0.09 * eff_atr)) if sig.strategy_id == "H4" else max(30.0, min(40.0, 0.18 * eff_atr))
    unit_risk_usd = (ref_pts * 2.00) + 4.48
    raw_contracts = int(np.round(r_target / unit_risk_usd))
    per_c_sl_risk = (sig.sl_dist * 2.00) + 4.48
    max_safe_contracts = int(np.max([1, np.floor((1.35 * r_target) / per_c_sl_risk)]))
    return int(np.clip(min(raw_contracts, max_safe_contracts), min_c, max_c))


def precompute_tier_daily_records(
    all_dates: List[date],
    prep,
    daily_signals: Dict[date, List[TradeSignal]],
    sim: Optional[PessimisticExecutionSimulator] = None
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Precomputes deterministic daily PnL, MFE, and MAE records across all 7 target risk tiers:
    R_tier in {90, 120, 165, 170, 245, 250, 330}.
    Returns a dictionary of fast NumPy arrays indexed by tier.
    """
    if sim is None:
        sim = PessimisticExecutionSimulator(slippage_points=0.50)

    raw_data = {k: {"pnl": [], "mfe": [], "mae": []} for k in TIER_SPECS}

    for d in all_dates:
        sigs = daily_signals.get(d, [])
        if not sigs:
            for k in TIER_SPECS:
                raw_data[k]["pnl"].append(0.0)
                raw_data[k]["mfe"].append(0.0)
                raw_data[k]["mae"].append(0.0)
            continue

        df_day = prep.get_session_bars(d)
        atr = prep.daily_atr_map.get(d)
        rth_bars = df_day.between_time("09:30", "15:44")
        df_ind = pd.DataFrame(index=rth_bars.index)
        if not rth_bars.empty:
            tp = (rth_bars["High"] + rth_bars["Low"] + rth_bars["Close"]) / 3.0
            cum_vol = rth_bars["Volume"].cumsum()
            df_ind["VWAP"] = (tp * rth_bars["Volume"]).cumsum() / cum_vol.replace(0, np.nan)
            if d in prep.noise_band_map:
                u_bands, l_bands = [], []
                for t in rth_bars.index:
                    t_str = t.strftime("%H:%M")
                    b = prep.get_noise_band(d, t_str, kd=1.10)
                    if b:
                        u_bands.append(b[0])
                        l_bands.append(b[1])
                    else:
                        u_bands.append(np.nan)
                        l_bands.append(np.nan)
                df_ind["UpperBand"] = pd.Series(u_bands, index=rth_bars.index).ffill()
                df_ind["LowerBand"] = pd.Series(l_bands, index=rth_bars.index).ffill()

        for k, spec in TIER_SPECS.items():
            r_target = spec["r_target"]
            min_c = spec["min_c"]
            max_c = spec["max_c"]
            is_funded = spec["is_funded"]
            pre_lock = spec["pre_lock"]

            session_trades = []
            morning_pnl = 0.0
            last_exit_time = pd.Timestamp.min.tz_localize(df_day.index.tz)

            for sig in sigs:
                is_pyramid = False
                if sig.execution_time < last_exit_time:
                    if sig.strategy_id == "H4":
                        open_tr = [t for t in session_trades if t.exit_time > sig.execution_time]
                        if open_tr and open_tr[0].direction == sig.direction:
                            h3_tr = open_tr[0]
                            bars_exec = df_day.loc[:sig.execution_time]
                            if not bars_exec.empty:
                                close_exec = float(bars_exec.iloc[-1]["Close"])
                                h3_unreal = (close_exec - h3_tr.entry_price) * h3_tr.direction * 2.00 * h3_tr.contracts - (2.48 * h3_tr.contracts)
                                if not is_funded and h3_unreal >= 100.0:
                                    is_pyramid = True
                                elif is_funded and (morning_pnl + max(0.0, 0.70 * h3_unreal)) >= 165.0:
                                    is_pyramid = True
                    if not is_pyramid:
                        continue

                # Sizing
                if pre_lock and sig.strategy_id == "H4":
                    if morning_pnl >= 165.0:
                        house_money = morning_pnl - 160.0
                        per_c_risk = (sig.sl_dist * 2.00) + 4.48
                        qty = min(max_c, max(0, int(house_money // per_c_risk)))
                    else:
                        qty = 0
                else:
                    qty = get_tier_contracts(sig, atr, r_target, min_c, max_c)

                if qty <= 0:
                    continue

                trades = sim.simulate_trade(sig, df_day, contract_qty=qty, df_indicators=df_ind)
                if not trades:
                    continue

                t_net = sum(t.net_pnl for t in trades)
                last_exit_time = max(last_exit_time, max(t.exit_time for t in trades))
                if sig.strategy_id == "H3":
                    morning_pnl += t_net
                session_trades.extend(trades)

            if not session_trades:
                raw_data[k]["pnl"].append(0.0)
                raw_data[k]["mfe"].append(0.0)
                raw_data[k]["mae"].append(0.0)
            elif len(session_trades) == 1:
                tr0 = session_trades[0]
                raw_data[k]["pnl"].append(tr0.net_pnl)
                raw_data[k]["mfe"].append(max(0.0, tr0.mfe_usd, tr0.net_pnl))
                raw_data[k]["mae"].append(min(0.0, tr0.mae_usd, tr0.net_pnl))
            else:
                p0 = session_trades[0].net_pnl
                tot_pnl = sum(t.net_pnl for t in session_trades)
                peak_mfe = max(0.0, session_trades[0].mfe_usd, p0, p0 + session_trades[1].mfe_usd, tot_pnl)
                worst_mae = min(0.0, session_trades[0].mae_usd, p0, p0 + session_trades[1].mae_usd, tot_pnl)
                raw_data[k]["pnl"].append(tot_pnl)
                raw_data[k]["mfe"].append(peak_mfe)
                raw_data[k]["mae"].append(worst_mae)

    tier_arrays = {}
    for k in TIER_SPECS:
        tier_arrays[k] = {
            "pnl": np.array(raw_data[k]["pnl"], dtype=float),
            "mfe": np.array(raw_data[k]["mfe"], dtype=float),
            "mae": np.array(raw_data[k]["mae"], dtype=float)
        }

    return tier_arrays


def simulate_realistic_funded_resets(
    all_dates: List[date],
    tier_data: Dict[int, Dict[str, np.ndarray]]
) -> Dict[str, Any]:
    """
    Simulates the 8.9-year Funded account lifecycle with realistic payout cycling & account resets:
    - Starts with $50,000 capital and trailing floor at $48,000 (Pre-Lock zone, R_165).
    - When EOD HWM reaches $52,100, floor permanently locks at $50,100 (Post-Lock zone, R_245).
    - When Equity reaches >= $52,660 and qualifying_days >= 5:
        * Payout of $1,500 ($1,350 net) is withdrawn.
        * Equity drops by $1,500 ($51,160), floor remains at $50,100.
        * qualifying_days resets to 0.
    - If Equity <= active_floor:
        * Account breaches, resets to fresh $50,000 account (Pre-Lock zone).
    """
    accounts_deployed = 1
    total_payouts = 0
    payout_dates = []
    breach_count = 0

    equity = 50000.0
    eod_hwm = 50000.0
    floor = 48000.0
    is_floor_locked = False
    qual_days = 0
    target_equity = 52660.0

    daily_net_cashflow = []
    cumulative_cashflow = []
    running_cash = 0.0

    for d_idx, d in enumerate(all_dates):
        buf = equity - floor
        if buf < 1050.0:
            tier = 90
        elif not is_floor_locked and eod_hwm < 52100.0:
            tier = 165
        else:
            tier = 245

        mfe = tier_data[tier]["mfe"][d_idx]
        mae = tier_data[tier]["mae"][d_idx]
        raw_pnl = tier_data[tier]["pnl"][d_idx]
        pnl = min(920.0, max(-600.0, raw_pnl))

        day_cash = 0.0

        # Payout-on-Touch check ($52,660 with >= 4 qualifying days)
        if qual_days >= 4 and (equity + mfe >= target_equity) and (mfe >= 160.0):
            total_payouts += 1
            day_cash += 1350.0
            payout_dates.append(d)
            equity = target_equity - 1500.0
            target_equity = equity + 1500.0
            is_floor_locked = True
            floor = 50100.0
            qual_days = 0
        elif equity + mae <= floor:
            breach_count += 1
            accounts_deployed += 1
            equity = 50000.0
            eod_hwm = 50000.0
            floor = 48000.0
            is_floor_locked = False
            qual_days = 0
            target_equity = 52660.0
        else:
            equity += pnl
            if pnl >= 150.0:
                qual_days += 1
            if equity > eod_hwm:
                eod_hwm = equity
            if eod_hwm >= 52100.0:
                is_floor_locked = True
                floor = 50100.0
            elif not is_floor_locked:
                floor = max(floor, eod_hwm - 2000.0)

            if equity <= floor:
                breach_count += 1
                accounts_deployed += 1
                equity = 50000.0
                eod_hwm = 50000.0
                floor = 48000.0
                is_floor_locked = False
                qual_days = 0
                target_equity = 52660.0
            elif equity >= target_equity and qual_days >= 5:
                total_payouts += 1
                day_cash += 1350.0
                payout_dates.append(d)
                equity -= 1500.0
                target_equity = equity + 1500.0
                is_floor_locked = True
                floor = 50100.0
                qual_days = 0

        running_cash += day_cash
        daily_net_cashflow.append(day_cash)
        cumulative_cashflow.append(running_cash)

    total_net_cash = total_payouts * 1350.0
    return {
        "accounts_deployed": accounts_deployed,
        "total_payouts": total_payouts,
        "total_net_cash": total_net_cash,
        "breach_count": breach_count,
        "payout_dates": payout_dates,
        "cumulative_cashflow": cumulative_cashflow
    }

