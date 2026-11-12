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
        block_size: int = 5
    ) -> Dict[str, Any]:
        """
        Phase 8 Multi-Account Institutional Benchmark & Funded Lifetime Value (LTV) Engine:
        1. Architecture A: 5-Ticket Sequential / Non-Overlapping Campaign (Independent 250d-Eval + 250d-Funded lifecycles).
        2. Architecture B: 5-Account Milestone-Gated Parallel Scaling (Tier 2 $50,800 milestone-activated desynchronization).
        3. Architecture C: Funded Account Lifetime Value (Multi-Payout Compounding across Payouts #1, #2, and #3 on permanent $50,100 locked floor).
        """
        n_days = len(daily_pnl_eval)
        np.random.seed(42)

        def run_single_ticket(eval_pnl_path, fund_pnl_path, max_eval=250, max_fund=250):
            # Phase 1: Eval
            e_bal = 50000.0
            e_hwm = 50000.0
            e_floor = 48000.0
            passed = False
            t_pass = None

            for t, pnl in enumerate(eval_pnl_path[:max_eval]):
                pnl = max(-600.0, pnl)
                e_bal += pnl
                if e_bal > e_hwm:
                    e_hwm = e_bal
                e_floor = max(e_floor, e_hwm - 2000.0)
                if e_bal <= e_floor:
                    return {"payouts": 0, "eval_passed": False, "t_pass": None, "eval_bal": e_bal}
                if e_bal >= 53000.0:
                    passed = True
                    t_pass = t + 1
                    break

            if not passed:
                return {"payouts": 0, "eval_passed": False, "t_pass": None, "eval_bal": e_bal}

            # Phase 2: Funded Multi-Payout Compounding (Payouts #1, #2, #3)
            f_bal = 50000.0
            f_hwm = 50000.0
            f_floor = 48000.0
            is_locked = False
            payouts = 0
            qual_days = 0
            target = 52650.0

            for t, raw_pnl in enumerate(fund_pnl_path):
                pnl = max(-600.0, min(920.0, raw_pnl))
                f_bal += pnl
                if pnl >= 150.0:
                    qual_days += 1
                if f_bal > f_hwm:
                    f_hwm = f_bal
                if not is_locked:
                    if f_hwm >= 52100.0:
                        is_locked = True
                        f_floor = 50100.0
                    else:
                        f_floor = max(f_floor, f_hwm - 2000.0)

                if f_bal <= f_floor:
                    break

                if f_bal >= target and qual_days >= 5:
                    payouts += 1
                    f_bal -= 1500.0
                    target = f_bal + 1500.0
                    qual_days = 0
                    is_locked = True
                    f_floor = 50100.0
                    if payouts >= 3 or t >= (max_fund * 3):
                        break

            return {"payouts": payouts, "eval_passed": True, "t_pass": t_pass, "eval_bal": e_bal}

        num_blocks = (250 + 750) // block_size + 2

        # Architecture A collections
        arch_a_payouts = {1: [], 3: [], 5: []}
        # Architecture B collections
        arch_b_payouts = {1: [], 3: [], 5: []}
        # Architecture C counters
        c_p1, c_p2, c_p3 = 0, 0, 0

        for sim_i in range(num_simulations):
            # Architecture A: 5 independent non-overlapping tickets
            p_a = []
            for ticket_i in range(5):
                s_e, s_f = [], []
                for _ in range(num_blocks):
                    idx = np.random.randint(0, max(1, n_days - block_size))
                    s_e.extend(daily_pnl_eval[idx : idx + block_size])
                    s_f.extend(daily_pnl_funded[idx : idx + block_size])
                res = run_single_ticket(s_e, s_f)
                p_a.append(min(1, res["payouts"]))
                if ticket_i == 0:
                    if res["payouts"] >= 1: c_p1 += 1
                    if res["payouts"] >= 2: c_p2 += 1
                    if res["payouts"] >= 3: c_p3 += 1

            arch_a_payouts[1].append(p_a[0])
            arch_a_payouts[3].append(sum(p_a[:3]))
            arch_a_payouts[5].append(sum(p_a))

            # Architecture B: Milestone-Gated Parallel Scaling on a shared timeline
            s_e_b, s_f_b = [], []
            for _ in range(300):
                idx = np.random.randint(0, max(1, n_days - block_size))
                s_e_b.extend(daily_pnl_eval[idx : idx + block_size])
                s_f_b.extend(daily_pnl_funded[idx : idx + block_size])

            b_res = [0] * 5
            curr_t = 0
            for acc_i in range(5):
                st = curr_t
                bal, hwm, flr = 50000.0, 50000.0, 48000.0
                p_day, m_day = None, None
                for t in range(st, st + 250):
                    if t >= len(s_e_b): break
                    pnl = max(-600.0, s_e_b[t])
                    bal += pnl
                    if bal > hwm: hwm = bal
                    flr = max(flr, hwm - 2000.0)
                    if bal <= flr:
                        if m_day is None: curr_t = t + 1
                        break
                    if m_day is None and bal >= 50800.0:
                        m_day = t + 1
                        curr_t = m_day
                    if bal >= 53000.0:
                        p_day = t + 1
                        if m_day is None: curr_t = p_day
                        break

                if p_day is not None:
                    fbal, fhwm, fflr = 50000.0, 50000.0, 48000.0
                    flck = False
                    qdays = 0
                    for t in range(p_day, p_day + 250):
                        if t >= len(s_f_b): break
                        pnl = max(-600.0, min(920.0, s_f_b[t]))
                        fbal += pnl
                        if pnl >= 150.0: qdays += 1
                        if fbal > fhwm: fhwm = fbal
                        if not flck:
                            if fhwm >= 52100.0:
                                flck = True
                                fflr = 50100.0
                            else:
                                fflr = max(fflr, fhwm - 2000.0)
                        if fbal <= fflr: break
                        if fbal >= 52650.0 and qdays >= 5:
                            b_res[acc_i] = 1
                            break

            arch_b_payouts[1].append(b_res[0])
            arch_b_payouts[3].append(sum(b_res[:3]))
            arch_b_payouts[5].append(sum(b_res))

        b = 1350.0 / 87.0
        results = {"architecture_a": {}, "architecture_b": {}, "architecture_c": {}}

        for k in [1, 3, 5]:
            arr_a = np.array(arch_a_payouts[k])
            p1 = float(np.mean(arr_a >= 1) * 100.0)
            p2 = float(np.mean(arr_a >= 2) * 100.0)
            p3 = float(np.mean(arr_a >= 3) * 100.0)
            mp = float(np.mean(arr_a))
            cost = k * 87.0
            net = mp * 1350.0 - cost
            roi = (net / cost) * 100.0
            p_win = p1 / 100.0
            kelly = p_win - ((1.0 - p_win) / b)
            results["architecture_a"][k] = {
                "num_accounts": k, "ticket_cost": cost, "p_ge1_payout": p1, "p_ge2_payout": p2, "p_ge3_payout": p3,
                "mean_payouts": mp, "expected_net_cash": net, "return_on_fees_pct": roi, "kelly_fraction": kelly
            }

            arr_b = np.array(arch_b_payouts[k])
            bp1 = float(np.mean(arr_b >= 1) * 100.0)
            bp2 = float(np.mean(arr_b >= 2) * 100.0)
            bp3 = float(np.mean(arr_b >= 3) * 100.0)
            bmp = float(np.mean(arr_b))
            bnet = bmp * 1350.0 - cost
            broi = (bnet / cost) * 100.0
            bp_win = bp1 / 100.0
            bkelly = bp_win - ((1.0 - bp_win) / b)
            results["architecture_b"][k] = {
                "num_accounts": k, "ticket_cost": cost, "p_ge1_payout": bp1, "p_ge2_payout": bp2, "p_ge3_payout": bp3,
                "mean_payouts": bmp, "expected_net_cash": bnet, "return_on_fees_pct": broi, "kelly_fraction": bkelly
            }

            # Backwards compatibility
            results[f"{k}_Accounts"] = results["architecture_a"][k]

        p1_pct = (c_p1 / num_simulations) * 100.0
        p2_pct = (c_p2 / num_simulations) * 100.0
        p3_pct = (c_p3 / num_simulations) * 100.0
        p2_g_p1 = (c_p2 / max(1, c_p1)) * 100.0
        p3_g_p2 = (c_p3 / max(1, c_p2)) * 100.0
        exp_payouts = (c_p1 + c_p2 + c_p3) / num_simulations
        ltv_net = exp_payouts * 1350.0 - 87.0

        results["architecture_c"] = {
            "p1_pct": p1_pct,
            "p2_pct": p2_pct,
            "p3_pct": p3_pct,
            "p2_given_p1": p2_g_p1,
            "p3_given_p2": p3_g_p2,
            "expected_payouts_per_ticket": exp_payouts,
            "lifetime_net_ev": ltv_net
        }

        return results
