"""
Tradeify $50K Evaluation Challenge Monte Carlo & Capital Velocity Engine
Target: Tradeify EOD Trailing Drawdown ($2,000 Max DD, +$3,000 Profit Target, $87 Fee)
Focus: Optimizing Time-to-Funded, Expected Challenge Cost, and Monthly Payout Velocity
"""

from __future__ import annotations
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TradeifyChallengeSimulator:
    """
    Monte Carlo bootstrap engine simulating Tradeify 50K evaluation challenge economics.
    EOD Trailing Drawdown ratchet mechanics:
      - Starting balance: $50,000
      - Profit target: $53,000 (+$3,000)
      - Trailing drawdown floor starts at $48,000 (-$2,000)
      - Floor trails the EOD high-water mark until reaching $50,100 (breakeven + $100 safety), where it freezes
      - Daily loss limit: $1,250
      - Max trading days per eval: 250 days cutoff
    """

    def __init__(
        self,
        matrix_path: Path = PROJECT_ROOT / "strategies" / "orb_5m" / "backtest" / "session_pnl_matrix.npy",
        leaderboard_path: Path = PROJECT_ROOT / "strategies" / "orb_5m" / "backtest" / "parameter_sweep_leaderboard.csv",
        challenge_fee: float = 87.0,
        profit_target: float = 3000.0,
        max_drawdown: float = 2000.0,
        daily_loss_limit: float = 1250.0,
        n_simulations: int = 5000,
        random_seed: int = 42
    ):
        self.matrix = np.load(matrix_path)
        self.leaderboard = pd.read_csv(leaderboard_path)
        self.fee = challenge_fee
        self.profit_target = profit_target
        self.max_drawdown = max_drawdown
        self.daily_loss_limit = daily_loss_limit
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(random_seed)

    def simulate_model(self, model_id: int) -> Dict[str, Any]:
        """Runs 5,000 bootstrap paths for a single model configuration."""
        daily_pnls = self.matrix[:, model_id]
        n_days_pool = len(daily_pnls)

        passed_paths = 0
        busted_paths = 0
        timed_out_paths = 0
        days_to_pass_list = []
        days_to_bust_list = []

        max_days = 250

        # Run vectorized or looped paths
        for _ in range(self.n_simulations):
            # Sample random days with replacement
            sampled_pnls = self.rng.choice(daily_pnls, size=max_days, replace=True)

            balance = 50000.0
            peak_balance = 50000.0
            trail_floor = 50000.0 - self.max_drawdown  # 48,000

            path_status = None
            for day_idx, pnl in enumerate(sampled_pnls):
                # Check daily loss limit
                if pnl <= -self.daily_loss_limit:
                    path_status = "BUST"
                    days_to_bust_list.append(day_idx + 1)
                    break

                balance += pnl

                # Update EOD peak and trail floor
                if balance > peak_balance:
                    peak_balance = balance
                    new_floor = peak_balance - self.max_drawdown
                    # Ratchet freeze at $50,100
                    if new_floor > 50100.0:
                        new_floor = 50100.0
                    if new_floor > trail_floor:
                        trail_floor = new_floor

                # Check trailing drawdown breach
                if balance <= trail_floor:
                    path_status = "BUST"
                    days_to_bust_list.append(day_idx + 1)
                    break

                # Check profit target
                if (balance - 50000.0) >= self.profit_target:
                    path_status = "PASS"
                    days_to_pass_list.append(day_idx + 1)
                    break

            if path_status == "PASS":
                passed_paths += 1
            elif path_status == "BUST":
                busted_paths += 1
            else:
                timed_out_paths += 1

        pass_rate = passed_paths / self.n_simulations
        bust_rate = busted_paths / self.n_simulations
        avg_days_pass = float(np.mean(days_to_pass_list)) if days_to_pass_list else 999.0
        median_days_pass = float(np.median(days_to_pass_list)) if days_to_pass_list else 999.0
        p90_days_pass = float(np.percentile(days_to_pass_list, 90)) if days_to_pass_list else 999.0

        # Economic Metrics
        # Expected attempts to get 1 funded account = 1 / pass_rate
        expected_attempts = (1.0 / pass_rate) if pass_rate > 0 else 999.0
        expected_fee_cost = expected_attempts * self.fee

        # Time to funded in trading days and calendar months (21 trading days/month)
        trading_days_to_funded = avg_days_pass
        months_to_funded = trading_days_to_funded / 21.0

        # Payout Velocity = Net Value per month
        # Target ($3,000) minus fees paid ($87 * attempts), divided by months
        net_eval_ev = self.profit_target - expected_fee_cost
        monthly_velocity = (net_eval_ev / months_to_funded) if months_to_funded > 0 else 0.0
        daily_velocity = (net_eval_ev / trading_days_to_funded) if trading_days_to_funded > 0 else 0.0

        return {
            "model_id": model_id,
            "pass_rate": pass_rate,
            "bust_rate": bust_rate,
            "avg_days_pass": avg_days_pass,
            "median_days_pass": median_days_pass,
            "p90_days_pass": p90_days_pass,
            "expected_attempts": expected_attempts,
            "expected_fee_cost": expected_fee_cost,
            "trading_days_to_funded": trading_days_to_funded,
            "months_to_funded": months_to_funded,
            "net_eval_ev": net_eval_ev,
            "monthly_velocity_usd": monthly_velocity,
            "daily_velocity_usd": daily_velocity
        }

    def evaluate_top_models(self, top_n: int = 10) -> pd.DataFrame:
        print(f"\n=========================================================================")
        print(f"TRADEIFY $50K CHALLENGE EVALUATION ({self.n_simulations:,} MONTE CARLO BOOTSTRAP PATHS)")
        print(f"Optimizing: Time-to-Funded, Total Cost EV, and Monthly Payout Velocity")
        print(f"=========================================================================\n", flush=True)

        rows = []
        top_candidates = self.leaderboard.head(top_n)

        for _, row in top_candidates.iterrows():
            m_id = int(row["model_id"])
            metrics = self.simulate_model(m_id)
            metrics.update({
                "entry_mode": row["entry_mode"],
                "trail_mode": row["trail_mode"],
                "max_sl_pts": row["max_sl_pts"],
                "sharpe_ratio": row["sharpe_ratio"],
                "profit_factor": row["profit_factor"],
                "win_rate": row["win_rate"],
                "max_dd_usd": row["max_dd_usd"]
            })
            rows.append(metrics)

        df = pd.DataFrame(rows).sort_values("monthly_velocity_usd", ascending=False)
        out_csv = PROJECT_ROOT / "strategies" / "orb_5m" / "backtest" / "tradeify_velocity_leaderboard.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved Tradeify Velocity Leaderboard to {out_csv}")

        # Formatted output table
        summary_cols = [
            "model_id", "trail_mode", "max_sl_pts", "pass_rate", "avg_days_pass",
            "expected_attempts", "expected_fee_cost", "months_to_funded", "monthly_velocity_usd"
        ]
        print("\n--- TRADEIFY VELOCITY & CAPITAL EFFICIENCY RANKING ---")
        print(df[summary_cols].to_string(index=False))
        return df


if __name__ == "__main__":
    sim = TradeifyChallengeSimulator()
    sim.evaluate_top_models(top_n=10)
