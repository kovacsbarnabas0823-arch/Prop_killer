"""
Tradeify $50K Growth Monte Carlo & Horizon Challenge Simulator (Phase 3 Upgraded)
Author: Principal Quantitative Architect & Financial Engineer

Simulates Tradeify $50K Growth EOD Trailing Drawdown accounts:
  - 10,000 Stationary Block-Bootstrap iterations (preserving volatility clustering).
  - Multi-Horizon Accounting: 90-day, 180-day, and 250-day (1 trading year) evaluations.
  - No artificial 30-day calendar timeout (Tradeify has no calendar expiry, only trailing DD breach).
  - Two-Phase modeling: EVALUATION ($50K -> $53K) followed by FUNDED ($50K -> $52,650 payout).
  - Decoupled Skewness-Preserving Consistency Governor ($920 daily profit cap, protecting 35% rule).
  - Reports:
      * True Pass Rate (%) at 90d, 180d, 250d horizons.
      * Hard Drawdown Breach Rate (%) vs Timeout Rate (%).
      * Conditional Funded-to-Payout Conversion Rate (%) = P(Payout | Pass).
      * Unconditional 1st Payout Rate (%).
      * Expected Net Value per $87 Attempt (EV = P_payout * $1,500 * 0.90 - $87).
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from engine.tradeify_rules import TradeifyAccountState, AccountPhase, TradeifyAccountConfig

logger = logging.getLogger("MonteCarloSim")


@dataclass
class SimulationResults:
    model_name: str
    pass_rate_pct: float                  # At 250 days
    pass_rate_90d: float                  # At 90 days
    pass_rate_180d: float                 # At 180 days
    breach_rate_pct: float                # Account failed via trailing DD breach
    timeout_rate_pct: float               # Ended without pass or breach at 250 days
    median_days_to_pass: float
    mean_days_to_pass: float
    avg_fee_cost_per_pass: float          # $87 / (pass_rate / 100)
    conditional_payout_rate_pct: float    # P(Payout | Pass)
    unconditional_payout_rate_pct: float  # P(Payout) across all attempts
    mean_days_to_payout: float
    expected_net_value: float             # EV = P_uncond * $1,500 * 0.90 - $87
    total_evaluations: int


class TradeifyChallengeSimulator:
    def __init__(
        self,
        config: Optional[TradeifyAccountConfig] = None,
        max_evaluation_days: int = 250,
        max_funded_days: int = 250
    ):
        self.config = config or TradeifyAccountConfig()
        self.max_eval_days = max_evaluation_days
        self.max_funded_days = max_funded_days

    def simulate_path_from_daily_pnls(
        self,
        daily_pnls_eval: np.ndarray,
        daily_pnls_funded: np.ndarray
    ) -> Dict[str, Any]:
        """
        Simulates a single challenge path:
        Evaluation Phase -> if passed -> Funded Phase -> Payout.
        """
        # --- PHASE 1: EVALUATION ---
        eval_state = TradeifyAccountState(phase=AccountPhase.EVALUATION, config=self.config)
        eval_days = 0
        passed_eval = False
        breached_eval = False

        for pnl in daily_pnls_eval:
            eval_days += 1
            eval_state.start_session()
            eval_state.update_intraday_pnl(pnl)
            eval_state.close_trade(pnl)
            eval_state.end_session()

            if eval_state.is_passed:
                passed_eval = True
                break
            if eval_state.is_breached:
                breached_eval = True
                break
            if eval_days >= self.max_eval_days:
                break

        if not passed_eval:
            return {
                "passed_eval": False,
                "eval_days": eval_days,
                "breached_eval": breached_eval,
                "got_payout": False,
                "funded_days": 0
            }

        # --- PHASE 2: FUNDED ---
        funded_state = TradeifyAccountState(phase=AccountPhase.FUNDED, config=self.config)
        funded_days = 0
        got_payout = False
        breached_funded = False

        for pnl in daily_pnls_funded:
            funded_days += 1
            funded_state.start_session()

            # Apply Consistency Governor ceiling on funded days ($920 cap)
            clamped_pnl = min(pnl, self.config.MAX_FUNDED_DAILY_PROFIT)

            funded_state.update_intraday_pnl(clamped_pnl)
            funded_state.close_trade(clamped_pnl)
            funded_state.end_session()

            if funded_state.is_funded_payout_eligible:
                got_payout = True
                break
            if funded_state.is_breached:
                breached_funded = True
                break
            if funded_days >= self.max_funded_days:
                break

        return {
            "passed_eval": True,
            "eval_days": eval_days,
            "breached_eval": False,
            "got_payout": got_payout,
            "funded_days": funded_days
        }

    def run_monte_carlo(
        self,
        model_name: str,
        daily_pnl_array: np.ndarray,
        num_simulations: int = 10000,
        block_size: int = 5,
        daily_pnl_array_funded: Optional[np.ndarray] = None
    ) -> SimulationResults:
        """
        Executes Stationary Block-Bootstrap Monte Carlo simulations across 10,000 iterations.
        If daily_pnl_array_funded is provided, uses daily_pnl_array for Evaluation phase
        and daily_pnl_array_funded for Funded phase, preserving regime-specific position sizing.
        """
        n_days = len(daily_pnl_array)
        if n_days == 0:
            raise ValueError("Daily PnL array is empty.")
        if daily_pnl_array_funded is not None and len(daily_pnl_array_funded) != n_days:
            raise ValueError("daily_pnl_array_funded must match length of daily_pnl_array.")

        passed_eval_count = 0
        passed_90d_count = 0
        passed_180d_count = 0
        breached_eval_count = 0
        timeout_eval_count = 0

        eval_days_list = []
        payout_count = 0
        payout_days_list = []

        np.random.seed(42)

        # Generate block indices
        num_blocks = (self.max_eval_days + self.max_funded_days) // block_size + 2

        for _ in range(num_simulations):
            # Block-bootstrap daily PnL stream
            sampled_eval = []
            sampled_funded = []
            for _ in range(num_blocks):
                start_idx = np.random.randint(0, max(1, n_days - block_size))
                sampled_eval.extend(daily_pnl_array[start_idx : start_idx + block_size])
                if daily_pnl_array_funded is not None:
                    sampled_funded.extend(daily_pnl_array_funded[start_idx : start_idx + block_size])
                else:
                    sampled_funded.extend(daily_pnl_array[start_idx : start_idx + block_size])

            eval_pnls = np.array(sampled_eval[:self.max_eval_days])
            funded_pnls = np.array(sampled_funded[self.max_eval_days : self.max_eval_days + self.max_funded_days])

            res = self.simulate_path_from_daily_pnls(eval_pnls, funded_pnls)

            if res["passed_eval"]:
                passed_eval_count += 1
                e_days = res["eval_days"]
                eval_days_list.append(e_days)
                if e_days <= 90:
                    passed_90d_count += 1
                if e_days <= 180:
                    passed_180d_count += 1
                if res["got_payout"]:
                    payout_count += 1
                    payout_days_list.append(res["funded_days"])
            else:
                if res["breached_eval"]:
                    breached_eval_count += 1
                else:
                    timeout_eval_count += 1

        pass_rate = (passed_eval_count / num_simulations) * 100.0
        pass_rate_90d = (passed_90d_count / num_simulations) * 100.0
        pass_rate_180d = (passed_180d_count / num_simulations) * 100.0
        breach_rate = (breached_eval_count / num_simulations) * 100.0
        timeout_rate = (timeout_eval_count / num_simulations) * 100.0

        conditional_payout_rate = (payout_count / max(1, passed_eval_count)) * 100.0
        unconditional_payout_rate = (payout_count / num_simulations) * 100.0

        median_eval_days = float(np.median(eval_days_list)) if eval_days_list else 0.0
        mean_eval_days = float(np.mean(eval_days_list)) if eval_days_list else 0.0
        mean_payout_days = float(np.mean(payout_days_list)) if payout_days_list else 0.0

        avg_fee_cost = (87.0 / (pass_rate / 100.0)) if pass_rate > 0 else 99999.0

        # Net Expected Value per $87 attempt:
        # First payout is typically up to $1,500 with 90% payout split ($1,350 net payout)
        ev_net = (unconditional_payout_rate / 100.0) * (1500.0 * 0.90) - 87.0

        return SimulationResults(
            model_name=model_name,
            pass_rate_pct=pass_rate,
            pass_rate_90d=pass_rate_90d,
            pass_rate_180d=pass_rate_180d,
            breach_rate_pct=breach_rate,
            timeout_rate_pct=timeout_rate,
            median_days_to_pass=median_eval_days,
            mean_days_to_pass=mean_eval_days,
            avg_fee_cost_per_pass=avg_fee_cost,
            conditional_payout_rate_pct=conditional_payout_rate,
            unconditional_payout_rate_pct=unconditional_payout_rate,
            mean_days_to_payout=mean_payout_days,
            expected_net_value=ev_net,
            total_evaluations=num_simulations
        )

    def simulate_multi_account_portfolio(
        self,
        daily_pnl_eval: np.ndarray,
        daily_pnl_funded: np.ndarray,
        num_simulations: int = 10000,
        block_size: int = 5,
        stagger_days: int = 10,
        timeline_days: int = 300
    ) -> Dict[str, Any]:
        """
        Simulates 1-Account, 3-Account, and 5-Account Staggered Portfolios over 10,000 iterations.
        Accounts are staggered by 10 trading days to avoid concurrent drawdown clusters.
        Evaluates at 180-day and 250-day horizons.
        """
        n_days = len(daily_pnl_eval)
        num_blocks = timeline_days // block_size + 2
        np.random.seed(42)

        configs = [1, 3, 5]
        payouts_180 = {c: [] for c in configs}
        payouts_250 = {c: [] for c in configs}

        for _ in range(num_simulations):
            # Sample continuous stream
            stream_eval = []
            stream_funded = []
            for _ in range(num_blocks):
                idx = np.random.randint(0, max(1, n_days - block_size))
                stream_eval.extend(daily_pnl_eval[idx : idx + block_size])
                stream_funded.extend(daily_pnl_funded[idx : idx + block_size])

            s_eval = np.array(stream_eval[:timeline_days])
            s_funded = np.array(stream_funded[:timeline_days])

            # Run 5 accounts
            acc_payout_days = [None] * 5

            for acc_idx in range(5):
                start_d = acc_idx * stagger_days
                eval_bal = 50000.0
                eval_hwm = 50000.0
                eval_floor = 48000.0
                passed_day = None
                breached = False

                for t in range(start_d, timeline_days):
                    if (t - start_d) >= self.max_eval_days:
                        break
                    pnl = s_eval[t]
                    if pnl <= -600.0:
                        pnl = -600.0
                    eval_bal += pnl
                    if eval_bal > eval_hwm:
                        eval_hwm = eval_bal
                    eval_floor = max(eval_floor, eval_hwm - 2000.0)
                    if eval_bal <= eval_floor:
                        breached = True
                        break
                    if eval_bal >= self.config.EVAL_PROFIT_TARGET:
                        passed_day = t + 1
                        break

                if not breached and passed_day is not None:
                    fund_bal = 50000.0
                    fund_hwm = 50000.0
                    fund_floor = 48000.0
                    is_locked = False
                    qual_days = 0

                    for t in range(passed_day, timeline_days):
                        if (t - passed_day) >= self.max_funded_days:
                            break
                        pnl = min(self.config.MAX_FUNDED_DAILY_PROFIT, s_funded[t])
                        if pnl <= -600.0:
                            pnl = -600.0
                        fund_bal += pnl
                        if pnl >= self.config.MIN_WINNING_DAY_PROFIT:
                            qual_days += 1
                        if fund_bal > fund_hwm:
                            fund_hwm = fund_bal
                        if not is_locked:
                            if fund_hwm >= self.config.FLOOR_LOCK_TRIGGER:
                                is_locked = True
                                fund_floor = self.config.LOCKED_FLOOR
                            else:
                                fund_floor = max(fund_floor, fund_hwm - self.config.EOD_TRAILING_DD)
                        if fund_bal <= fund_floor:
                            break
                        if fund_bal >= self.config.FUNDED_PAYOUT_TARGET and qual_days >= self.config.MIN_FUNDED_WINNING_DAYS:
                            acc_payout_days[acc_idx] = t + 1
                            break

            for c in configs:
                p_180 = sum(1 for i in range(c) if acc_payout_days[i] is not None and acc_payout_days[i] <= 180)
                p_250 = sum(1 for i in range(c) if acc_payout_days[i] is not None and acc_payout_days[i] <= 250)
                payouts_180[c].append(p_180)
                payouts_250[c].append(p_250)

        results = {}
        b = 1350.0 / 87.0  # 15.517 payout to ticket fee ratio
        for c in configs:
            arr_180 = np.array(payouts_180[c])
            arr_250 = np.array(payouts_250[c])

            prob_ge1_180 = float(np.mean(arr_180 >= 1) * 100.0)
            prob_ge1_250 = float(np.mean(arr_250 >= 1) * 100.0)
            prob_ge2_180 = float(np.mean(arr_180 >= 2) * 100.0)
            prob_ge2_250 = float(np.mean(arr_250 >= 2) * 100.0)
            prob_ge3_180 = float(np.mean(arr_180 >= 3) * 100.0)
            prob_ge3_250 = float(np.mean(arr_250 >= 3) * 100.0)

            mean_payouts_250 = float(np.mean(arr_250))
            ticket_cost = c * 87.0
            expected_gross = mean_payouts_250 * 1350.0
            expected_net = expected_gross - ticket_cost
            ret_on_fees = (expected_net / ticket_cost) * 100.0

            p_win = prob_ge1_250 / 100.0
            kelly = p_win - ((1.0 - p_win) / b)

            results[f"{c}_Accounts"] = {
                "num_accounts": c,
                "ticket_cost": ticket_cost,
                "prob_ge1_180d": prob_ge1_180,
                "prob_ge1_250d": prob_ge1_250,
                "prob_ge2_180d": prob_ge2_180,
                "prob_ge2_250d": prob_ge2_250,
                "prob_ge3_180d": prob_ge3_180,
                "prob_ge3_250d": prob_ge3_250,
                "mean_payouts_250d": mean_payouts_250,
                "expected_gross_cash": expected_gross,
                "expected_net_cash": expected_net,
                "return_on_fees_pct": ret_on_fees,
                "kelly_fraction": kelly
            }

        return results
