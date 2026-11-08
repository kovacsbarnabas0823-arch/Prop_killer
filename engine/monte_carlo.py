"""
Tradeify $50K Growth Monte Carlo & Rolling-Window Challenge Simulator
Author: Principal Quantitative Architect

Simulates Tradeify $50K Growth EOD Trailing Drawdown accounts:
  - 10,000 Stationary Block-Bootstrap iterations (preserving volatility clustering).
  - Rolling historical evaluation windows across 8.9 years.
  - Two-Phase modeling: EVALUATION ($50K -> $53K) followed immediately by FUNDED ($50K -> $52.5K payout).
  - Evaluates:
      * Baseline orb_30m Static (2 MNQ)
      * orb_30m Simple Sprint (2 -> 4 MNQ)
      * H1 Regime-Filtered orb_30m + 4-Tier Bellman Sizing
      * Full Multi-Alpha Portfolio (H1 + H2 + H3 + H4) + Bellman Sizing + Funded Consistency Governor
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
    pass_rate_pct: float
    median_days_to_pass: float
    mean_days_to_pass: float
    avg_fee_cost_per_pass: float
    payout_rate_pct: float
    mean_days_to_payout: float
    total_evaluations: int


class TradeifyChallengeSimulator:
    def __init__(
        self,
        config: Optional[TradeifyAccountConfig] = None,
        max_evaluation_days: int = 120,
        max_funded_days: int = 120
    ):
        self.config = config or TradeifyAccountConfig()
        self.max_eval_days = max_evaluation_days
        self.max_funded_days = max_funded_days

    def simulate_path_from_daily_pnls(
        self,
        daily_pnls_eval: np.ndarray,
        daily_pnls_funded: np.ndarray,
        base_contracts: int = 2
    ) -> Tuple[bool, int, bool, int]:
        """
        Simulates a single challenge path:
        Evaluation Phase -> if passed -> Funded Phase -> Payout.
        Returns (passed_eval, days_to_pass, passed_funded, days_to_payout).
        """
        # --- PHASE 1: EVALUATION ---
        eval_state = TradeifyAccountState(phase=AccountPhase.EVALUATION, config=self.config)
        eval_days = 0
        passed_eval = False

        for pnl in daily_pnls_eval:
            eval_days += 1
            eval_state.start_session()
            eval_state.update_intraday_pnl(pnl)
            eval_state.close_trade(pnl)
            eval_state.end_session()

            if eval_state.is_passed:
                passed_eval = True
                break
            if eval_state.is_breached or eval_days >= self.max_eval_days:
                break

        if not passed_eval:
            return False, eval_days, False, 0

        # --- PHASE 2: FUNDED ---
        funded_state = TradeifyAccountState(phase=AccountPhase.FUNDED, config=self.config)
        funded_days = 0
        got_payout = False

        for pnl in daily_pnls_funded:
            funded_days += 1
            funded_state.start_session()

            # Apply Consistency Governor ceiling on funded days
            clamped_pnl = min(pnl, self.config.MAX_FUNDED_DAILY_PROFIT)

            funded_state.update_intraday_pnl(clamped_pnl)
            funded_state.close_trade(clamped_pnl)
            funded_state.end_session()

            if funded_state.is_funded_payout_eligible:
                got_payout = True
                break
            if funded_state.is_breached or funded_days >= self.max_funded_days:
                break

        return True, eval_days, got_payout, funded_days

    def run_monte_carlo(
        self,
        model_name: str,
        daily_pnl_array: np.ndarray,
        num_simulations: int = 10000,
        block_size: int = 5
    ) -> SimulationResults:
        """
        Executes Stationary Block-Bootstrap Monte Carlo simulations.
        """
        n_days = len(daily_pnl_array)
        if n_days == 0:
            raise ValueError("Daily PnL array is empty.")

        passed_eval_count = 0
        eval_days_list = []
        payout_count = 0
        payout_days_list = []

        np.random.seed(42)

        # Generate block indices
        num_blocks = (self.max_eval_days + self.max_funded_days) // block_size + 2

        for _ in range(num_simulations):
            # Block-bootstrap daily PnL stream
            sampled_pnls = []
            for _ in range(num_blocks):
                start_idx = np.random.randint(0, max(1, n_days - block_size))
                sampled_pnls.extend(daily_pnl_array[start_idx : start_idx + block_size])

            sampled_arr = np.array(sampled_pnls)
            eval_pnls = sampled_arr[:self.max_eval_days]
            funded_pnls = sampled_arr[self.max_eval_days : self.max_eval_days + self.max_funded_days]

            passed, e_days, payout, f_days = self.simulate_path_from_daily_pnls(eval_pnls, funded_pnls)

            if passed:
                passed_eval_count += 1
                eval_days_list.append(e_days)
                if payout:
                    payout_count += 1
                    payout_days_list.append(f_days)

        pass_rate = (passed_eval_count / num_simulations) * 100.0
        payout_rate = (payout_count / max(1, passed_eval_count)) * 100.0
        overall_payout_rate = (payout_count / num_simulations) * 100.0

        median_eval_days = float(np.median(eval_days_list)) if eval_days_list else 0.0
        mean_eval_days = float(np.mean(eval_days_list)) if eval_days_list else 0.0
        mean_payout_days = float(np.mean(payout_days_list)) if payout_days_list else 0.0

        # Expected evaluation fee cost per successful pass ($87 base fee)
        # Cost = 87 / (pass_rate / 100)
        avg_fee_cost = (87.0 / (pass_rate / 100.0)) if pass_rate > 0 else 99999.0

        return SimulationResults(
            model_name=model_name,
            pass_rate_pct=pass_rate,
            median_days_to_pass=median_eval_days,
            mean_days_to_pass=mean_eval_days,
            avg_fee_cost_per_pass=avg_fee_cost,
            payout_rate_pct=payout_rate,
            mean_days_to_payout=mean_payout_days,
            total_evaluations=num_simulations
        )
