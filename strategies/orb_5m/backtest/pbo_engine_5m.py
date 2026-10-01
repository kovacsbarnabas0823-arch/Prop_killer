"""
Probability of Backtest Overfitting (PBO) Engine for 5-Minute ORB Strategy
Methodology: Combinatorial Symmetric Cross-Validation (CSCV)
Author: Quantitative Research & Risk Architecture

Theoretical Basis:
  - David H. Bailey, Jonathan M. Borwein, Marcos Lopez de Prado, Qiji Jim Zhu (2016)
  - Evaluates N models across T sessions partitioned into S = 16 blocks (12,870 combinatorial splits).
  - Quantifies:
      1. Empirical PBO (Probability that In-Sample optimal model underperforms median OOS)
      2. Probability of Out-of-Sample Loss P(OOS PnL <= 0)
      3. Deflated Sharpe Ratio (DSR) adjusted for skewness, kurtosis, and trial selection bias
      4. Logit distribution of relative OOS performance ranks
"""

from __future__ import annotations
import sys
import os
import itertools
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class PBOEngine5m:
    """Executes Combinatorial Symmetric Cross-Validation (CSCV) to audit overfitting."""

    def __init__(
        self,
        matrix_path: Path = PROJECT_ROOT / "strategies" / "orb_5m" / "backtest" / "session_pnl_matrix.npy",
        leaderboard_path: Path = PROJECT_ROOT / "strategies" / "orb_5m" / "backtest" / "parameter_sweep_leaderboard.csv",
        n_blocks: int = 16
    ):
        self.matrix_path = matrix_path
        self.leaderboard_path = leaderboard_path
        self.n_blocks = n_blocks

        if not self.matrix_path.exists():
            raise FileNotFoundError(f"Session matrix not found at {self.matrix_path}")
        if not self.leaderboard_path.exists():
            raise FileNotFoundError(f"Leaderboard not found at {self.leaderboard_path}")

        self.matrix = np.load(self.matrix_path)  # (T, N)
        self.leaderboard = pd.read_csv(self.leaderboard_path)
        self.T, self.N = self.matrix.shape
        print(f"Loaded matrix: {self.T} sessions x {self.N} models.")

    def run_cscv(self) -> Dict[str, Any]:
        """
        Runs CSCV with S = 16 blocks (binom(16, 8) = 12,870 splits).
        """
        S = self.n_blocks
        half_S = S // 2
        block_size = self.T // S

        # Split session indices into S contiguous blocks
        blocks = [np.arange(i * block_size, (i + 1) * block_size if i < S - 1 else self.T) for i in range(S)]

        # Precompute block sums and statistics for high performance
        # block_means: (S, N)
        block_sums = np.zeros((S, self.N), dtype=np.float64)
        block_counts = np.zeros(S, dtype=np.int32)
        for i in range(S):
            block_sums[i, :] = np.sum(self.matrix[blocks[i], :], axis=0)
            block_counts[i] = len(blocks[i])

        all_combinations = list(itertools.combinations(range(S), half_S))
        M = len(all_combinations)
        print(f"CSCV Partitioning: S={S} blocks, generating {M:,} combinatorial splits...", flush=True)

        t0 = time.time()
        logits = []
        ranks = []
        is_best_models = []
        oos_best_sharpes = []
        oos_losses = 0

        # Vectorized batch processing of splits
        for c in all_combinations:
            is_blocks = list(c)
            oos_blocks = [b for b in range(S) if b not in is_blocks]

            # Fast evaluation of daily Sharpe on IS
            is_indices = np.concatenate([blocks[b] for b in is_blocks])
            oos_indices = np.concatenate([blocks[b] for b in oos_blocks])

            matrix_is = self.matrix[is_indices, :]
            matrix_oos = self.matrix[oos_indices, :]

            # IS Sharpe
            mean_is = np.mean(matrix_is, axis=0)
            std_is = np.std(matrix_is, axis=0)
            std_is[std_is == 0] = 1e-9
            sharpe_is = (mean_is / std_is) * np.sqrt(252)

            # Pick best IS model
            best_model_idx = int(np.argmax(sharpe_is))
            is_best_models.append(best_model_idx)

            # OOS Sharpe for all models
            mean_oos = np.mean(matrix_oos, axis=0)
            std_oos = np.std(matrix_oos, axis=0)
            std_oos[std_oos == 0] = 1e-9
            sharpe_oos = (mean_oos / std_oos) * np.sqrt(252)

            best_model_oos_sharpe = sharpe_oos[best_model_idx]
            oos_best_sharpes.append(best_model_oos_sharpe)

            if np.sum(matrix_oos[:, best_model_idx]) <= 0:
                oos_losses += 1

            # Determine relative rank of best_model_idx in OOS
            # Rank from 1 (lowest) to N (highest)
            order = np.argsort(sharpe_oos)
            rank = np.where(order == best_model_idx)[0][0] + 1
            relative_rank = rank / (self.N + 1.0)
            ranks.append(relative_rank)

            # Logit transform with epsilon clamp
            r_clamped = np.clip(relative_rank, 1e-5, 1.0 - 1e-5)
            logit = np.log(r_clamped / (1.0 - r_clamped))
            logits.append(logit)

        elapsed = time.time() - t0
        print(f"Completed {M:,} splits in {elapsed:.2f}s ({M / elapsed:,.0f} splits/sec).", flush=True)

        ranks = np.array(ranks)
        logits = np.array(logits)
        oos_best_sharpes = np.array(oos_best_sharpes)

        # 1. PBO: Proportion of splits where relative rank < 0.50
        pbo = np.mean(ranks < 0.50)

        # 2. Probability of Loss
        prob_loss = oos_losses / M

        # 3. Model #1 (overall best model) OOS statistics
        top_model_id = int(self.leaderboard.iloc[0]["model_id"])
        top_model_row = self.leaderboard.iloc[0]
        top_model_full_pnl = self.matrix[:, top_model_id]

        # Calculate DSR (Deflated Sharpe Ratio)
        full_sharpe = top_model_row["sharpe_ratio"]
        daily_skew = float(stats.skew(top_model_full_pnl))
        daily_kurt = float(stats.kurtosis(top_model_full_pnl, fisher=False))

        # Expected maximum Sharpe under null (Euler-Mascheroni constant gamma ~ 0.5772)
        euler_gamma = 0.5772156649
        z_n = (1.0 - euler_gamma) * stats.norm.ppf(1.0 - 1.0 / self.N) + euler_gamma * stats.norm.ppf(1.0 - 1.0 / (self.N * np.e))
        sr_benchmark = z_n * (1.0 / np.sqrt(252)) * np.sqrt(252) # annualized

        # DSR statistic
        t_years = self.T / 252.0
        denom = np.sqrt(1.0 - daily_skew * (full_sharpe / np.sqrt(252)) + ((daily_kurt - 1.0) / 4.0) * (full_sharpe / np.sqrt(252))**2)
        if denom > 0:
            dsr_stat = ((full_sharpe - sr_benchmark) * np.sqrt(self.T - 1)) / (denom * np.sqrt(252))
            dsr_prob = stats.norm.cdf(dsr_stat)
        else:
            dsr_prob = 1.0

        results = {
            "pbo": float(pbo),
            "prob_loss": float(prob_loss),
            "median_oos_rank": float(np.median(ranks)),
            "mean_oos_rank": float(np.mean(ranks)),
            "pct_positive_oos_sharpe": float(np.mean(oos_best_sharpes > 0)),
            "mean_oos_sharpe": float(np.mean(oos_best_sharpes)),
            "top_model_id": top_model_id,
            "top_model_entry": str(top_model_row["entry_mode"]),
            "top_model_trail": str(top_model_row["trail_mode"]),
            "top_model_sl": float(top_model_row["max_sl_pts"]),
            "top_model_full_sharpe": float(full_sharpe),
            "top_model_full_pf": float(top_model_row["profit_factor"]),
            "top_model_full_win_rate": float(top_model_row["win_rate"]),
            "top_model_max_dd": float(top_model_row["max_dd_usd"]),
            "daily_skewness": daily_skew,
            "daily_kurtosis": daily_kurt,
            "deflated_sharpe_ratio": float(dsr_prob),
            "expected_max_null_sharpe": float(sr_benchmark)
        }

        self._save_report(results, ranks, logits)
        return results

    def _save_report(self, res: dict, ranks: np.ndarray, logits: np.ndarray):
        report_path = PROJECT_ROOT / "strategies" / "orb_5m" / "backtest" / "pbo_audit_report.txt"
        report_text = f"""========================================================================================
INSTITUTIONAL QUANTITATIVE AUDIT: PROBABILITY OF BACKTEST OVERFITTING (PBO)
Methodology: Combinatorial Symmetric Cross-Validation (CSCV | S=16 blocks, 12,870 splits)
Target Strategy: CME NQ/MNQ 5-Minute Opening Candle with 12 EMA Filter
========================================================================================

1. OVERFITTING & OUT-OF-SAMPLE AUDIT METRICS:
----------------------------------------------------------------------------------------
Empirical Probability of Backtest Overfitting (PBO): {res['pbo']*100:.2f}%
   [Threshold Guide: PBO < 10% = Robust / Institutional | PBO > 50% = High Overfitting]
Probability of Out-of-Sample Net Loss P(OOS <= 0) : {res['prob_loss']*100:.2f}%
Proportion of Positive Out-of-Sample Sharpe Splits : {res['pct_positive_oos_sharpe']*100:.2f}%
Mean Out-of-Sample Relative Percentile Rank       : {res['mean_oos_rank']*100:.2f}%
Median Out-of-Sample Relative Percentile Rank     : {res['median_oos_rank']*100:.2f}%
Mean Out-of-Sample Annualized Sharpe Ratio        : {res['mean_oos_sharpe']:.2f}

2. DEFLATED SHARPE RATIO (DSR) & SELECTION BIAS CORRECTION:
----------------------------------------------------------------------------------------
Full Period Annualized Sharpe Ratio               : {res['top_model_full_sharpe']:.2f}
Expected Maximum Null Sharpe across Trials (SR*) : {res['expected_max_null_sharpe']:.2f}
Daily Return Skewness                             : {res['daily_skewness']:.3f}
Daily Return Kurtosis                             : {res['daily_kurtosis']:.3f}
Deflated Sharpe Ratio (DSR)                       : {res['deflated_sharpe_ratio']*100:.2f}%
   [Interpretation: > 95% confirms statistically significant edge after accounting for {self.N} trials]

3. WINNING PARAMETER SPECIFICATION:
----------------------------------------------------------------------------------------
Model ID                                          : #{res['top_model_id']}
Entry Mode                                        : {res['top_model_entry']}
Trailing Stop Mode                                : {res['top_model_trail']}
Max Stop Loss Cap                                 : {res['top_model_sl']} points
Win Rate                                          : {res['top_model_full_win_rate']*100:.2f}%
Profit Factor                                     : {res['top_model_full_pf']:.2f}
Max Historical Drawdown                           : ${res['top_model_max_dd']:,.2f}
========================================================================================
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\nSaved Institutional PBO Audit Report to {report_path}")
        print(report_text)


if __name__ == "__main__":
    pbo_engine = PBOEngine5m()
    pbo_engine.run_cscv()
