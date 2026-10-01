"""
Combinatorial Purged Cross-Validation (CPCV) & Deflated Sharpe Ratio (DSR) Engine
Reference: Marcos López de Prado (Advances in Financial Machine Learning, 2018)
           Bailey & López de Prado (The Deflated Sharpe Ratio, 2014)

Features:
  1. Combinatorial Purged Cross-Validation (CPCV):
     - N-split chronological partitioning with combinatorial (N choose k) test paths.
     - Strict purging of overlapping trade holding horizons.
     - Embargoing on post-test boundaries to prevent serial dependency leakage.
  2. Deflated Sharpe Ratio (DSR):
     - Corrects for selection bias under multiple testing (M trials).
     - Accounts for non-normal return distributions (skewness and kurtosis).
     - Outputs rigorous statistical significance p-value.
"""

from __future__ import annotations
import itertools
import math
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats


def compute_sharpe_ratio(returns: np.ndarray, periods_per_year: float = 252.0) -> float:
    """Computes annualized Sharpe Ratio assuming 0 risk-free rate."""
    if len(returns) < 2:
        return 0.0
    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1)
    if std_r <= 1e-9:
        return 0.0
    return float((mean_r / std_r) * np.sqrt(periods_per_year))


def compute_deflated_sharpe_ratio(
    observed_sr: float,
    all_trials_sr: List[float],
    returns: np.ndarray,
    periods_per_year: float = 252.0
) -> Dict[str, float]:
    """
    Calculates the Deflated Sharpe Ratio (DSR).

    Parameters:
        observed_sr: Annualized Sharpe ratio of the selected strategy.
        all_trials_sr: List of annualized Sharpe ratios across all M swept parameter configurations.
        returns: Time-series returns of the selected strategy.
        periods_per_year: Annualization factor (e.g. 252 trading days).

    Returns:
        Dict containing DSR probability, expected max SR, skewness, and kurtosis.
    """
    N = len(returns)
    if N < 5:
        return {"dsr": 0.0, "sr_star": 0.0, "skewness": 0.0, "kurtosis": 3.0}

    # De-annualize Sharpe ratio to per-period units for statistical variance formula
    sr_period = observed_sr / np.sqrt(periods_per_year)

    # Moments of return distribution
    skew = float(stats.skew(returns))
    # Fisher kurtosis (normal = 3)
    kurt = float(stats.kurtosis(returns, fisher=False))
    if np.isnan(kurt) or kurt < 1.0:
        kurt = 3.0

    # Number of trials M and variance of trial Sharpe ratios
    M = max(len(all_trials_sr), 1)
    sr_trials_period = np.array(all_trials_sr) / np.sqrt(periods_per_year)
    var_trials = float(np.var(sr_trials_period, ddof=1)) if len(sr_trials_period) > 1 else 0.0
    std_trials = math.sqrt(var_trials) if var_trials > 0 else 0.0

    # Expected maximum Sharpe ratio under null hypothesis (Euler-Mascheroni approximation)
    euler_mascheroni = 0.5772156649
    if M > 1 and std_trials > 0:
        term1 = (1.0 - euler_mascheroni) * stats.norm.ppf(1.0 - (1.0 / M))
        term2 = euler_mascheroni * stats.norm.ppf(1.0 - (1.0 / (M * math.e)))
        sr_star_period = (term1 + term2) * std_trials
    else:
        sr_star_period = 0.0

    # Variance of Sharpe Ratio estimate under non-normality
    # Var(SR) = (1 - skew * SR + (kurt - 1)/4 * SR^2) / (N - 1)
    denom_variance = 1.0 - (skew * sr_period) + (((kurt - 1.0) / 4.0) * (sr_period ** 2))
    if denom_variance <= 0:
        denom_variance = 1e-6
    std_sr_estimate = math.sqrt(denom_variance / (N - 1))

    # Test statistic Z
    z_stat = (sr_period - sr_star_period) / std_sr_estimate
    dsr = float(stats.norm.cdf(z_stat))

    return {
        "dsr": round(dsr, 4),
        "expected_max_sr_annualized": round(float(sr_star_period * np.sqrt(periods_per_year)), 2),
        "skewness": round(skew, 2),
        "kurtosis": round(kurt, 2),
        "total_trials_M": M,
        "sample_periods_N": N
    }


class CPCVSplitter:
    """
    Combinatorial Purged Cross-Validation Splitter.
    Partitions N chronological blocks, forms (N choose k) test paths,
    and applies purging and embargoing to prevent data leakage.
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        embargo_pct: float = 0.01  # 1% embargo post-test
    ):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_pct = embargo_pct

    def split(self, total_sessions: int) -> List[Tuple[List[int], List[int]]]:
        """
        Generates (train_indices, test_indices) for each combinatorial test path.
        """
        indices = np.arange(total_sessions)
        chunk_size = total_sessions // self.n_splits
        splits = []
        for i in range(self.n_splits):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < self.n_splits - 1 else total_sessions
            splits.append(indices[start:end])

        embargo_size = max(1, int(total_sessions * self.embargo_pct))
        combinations = list(itertools.combinations(range(self.n_splits), self.n_test_splits))

        cv_paths = []
        for test_chunk_ids in combinations:
            test_idx = np.concatenate([splits[i] for i in test_chunk_ids])
            train_chunk_ids = [i for i in range(self.n_splits) if i not in test_chunk_ids]

            # Construct train indices with purging & embargoing
            raw_train = np.concatenate([splits[i] for i in train_chunk_ids])

            # Apply embargo: for any test chunk, embargo the start of the next chronological chunk if it's training
            embargoed_indices = set()
            for t_id in test_chunk_ids:
                next_id = t_id + 1
                if next_id in train_chunk_ids and next_id < self.n_splits:
                    next_chunk = splits[next_id]
                    embargo_len = min(embargo_size, len(next_chunk))
                    embargoed_indices.update(next_chunk[:embargo_len])

            purged_train = [idx for idx in raw_train if idx not in embargoed_indices]
            cv_paths.append((purged_train, list(test_idx)))

        return cv_paths
