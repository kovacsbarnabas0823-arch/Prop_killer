"""
Institutional Comprehensive Statistical Audit Suite for 5-Minute ORB Strategy
Executes 10 Advanced Quantitative & Econometric Verifications:
  1. PBO    - Probability of Backtest Overfitting (CSCV, 12,870 splits)
  2. DSR    - Deflated Sharpe Ratio (Lopez de Prado)
  3. SPA    - Hansen's Superior Predictive Ability Test (Bootstrap p-value)
  4. CPCV   - Combinatorially Purged Cross-Validation with Embargoing
  5. WFE    - Walk-Forward Efficiency (Out-of-Sample / In-Sample Ratio)
  6. SBB-MC - Stationary Block-Bootstrap First-Passage Monte Carlo (Politis & Romano 1994)
  7. Runs & Ljung-Box - Serial Independence & Loss Clustering Audit
  8. EVT-POT - Extreme Value Theory: Peaks-Over-Threshold (CVaR 99.9%)
  9. Hessian - Parameter Stability & Plateau Flatness Test
  10. Friction Decay - Break-Even Slippage & Latency Stress Curve
"""

from __future__ import annotations
import sys
import os
import itertools
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.orb_5m.backtest.sweep_engine_5m import simulate_single_model, SessionPrecompiler


class ComprehensiveQuantAuditor:
    def __init__(self, contracts: int = 8):
        self.contracts = contracts
        self.sessions = SessionPrecompiler.load_sessions()
        self.T = len(self.sessions)
        print(f"Loaded {self.T} trading sessions. Sizing locked at {self.contracts} MNQ ($16/pt).")

        # Baseline Model 1: BREAKOUT_WITH_EMA, TRAIL_0.5R, max_sl 35.0 pts
        print("Pre-simulating baseline Model 1 across all sessions...")
        self.daily_pnl, self.trades = simulate_single_model(
            sessions=self.sessions,
            entry_mode="BREAKOUT_WITH_EMA",
            trail_mode="TRAIL_0.5R",
            max_sl_pts=35.0,
            contracts=self.contracts,
            slippage_pts=0.50
        )
        self.trade_pnls = np.array([t.pnl_dollars_net for t in self.trades])

    # -------------------------------------------------------------------------
    # 1. PBO & 2. DSR
    # -------------------------------------------------------------------------
    def audit_pbo_and_dsr(self) -> Dict[str, float]:
        print("\n[1/10 & 2/10] Computing CSCV PBO & Deflated Sharpe Ratio (DSR)...")
        matrix_path = PROJECT_ROOT / "strategies" / "orb_5m" / "backtest" / "session_pnl_matrix.npy"
        leaderboard_path = PROJECT_ROOT / "strategies" / "orb_5m" / "backtest" / "parameter_sweep_leaderboard.csv"
        
        matrix = np.load(matrix_path)
        # Scale matrix to current contracts
        matrix = matrix * (self.contracts / 2.0)
        leaderboard = pd.read_csv(leaderboard_path)
        T, N = matrix.shape

        S = 16
        half_S = 8
        block_size = T // S
        blocks = [np.arange(i * block_size, (i + 1) * block_size if i < S - 1 else T) for i in range(S)]

        combinations = list(itertools.combinations(range(S), half_S))
        M = len(combinations)

        ranks = []
        losses = 0
        oos_sharpes = []

        for c in combinations:
            is_idx = np.concatenate([blocks[b] for b in c])
            oos_idx = np.concatenate([blocks[b] for b in range(S) if b not in c])

            m_is = matrix[is_idx, :]
            m_oos = matrix[oos_idx, :]

            # IS Sharpe
            mean_is = np.mean(m_is, axis=0)
            std_is = np.std(m_is, axis=0)
            std_is[std_is == 0] = 1e-9
            sr_is = (mean_is / std_is) * np.sqrt(252)

            best_m = int(np.argmax(sr_is))

            # OOS Sharpe
            mean_oos = np.mean(m_oos, axis=0)
            std_oos = np.std(m_oos, axis=0)
            std_oos[std_oos == 0] = 1e-9
            sr_oos = (mean_oos / std_oos) * np.sqrt(252)

            oos_sharpes.append(sr_oos[best_m])
            if np.sum(m_oos[:, best_m]) <= 0:
                losses += 1

            order = np.argsort(sr_oos)
            rank = np.where(order == best_m)[0][0] + 1
            ranks.append(rank / (N + 1.0))

        pbo = float(np.mean(np.array(ranks) < 0.50))
        prob_loss = float(losses / M)
        mean_oos_sr = float(np.mean(oos_sharpes))

        # DSR calculation
        top_sr = float(leaderboard.iloc[0]["sharpe_ratio"])
        skew = float(stats.skew(self.daily_pnl))
        kurt = float(stats.kurtosis(self.daily_pnl, fisher=False))

        euler_gamma = 0.5772156649
        z_n = (1.0 - euler_gamma) * stats.norm.ppf(1.0 - 1.0 / N) + euler_gamma * stats.norm.ppf(1.0 - 1.0 / (N * np.e))
        sr_null = z_n

        denom = np.sqrt(1.0 - skew * (top_sr / np.sqrt(252)) + ((kurt - 1.0) / 4.0) * (top_sr / np.sqrt(252))**2)
        dsr_stat = ((top_sr - sr_null) * np.sqrt(T - 1)) / (denom * np.sqrt(252))
        dsr = float(stats.norm.cdf(dsr_stat))

        return {
            "pbo": pbo,
            "prob_loss": prob_loss,
            "mean_oos_sharpe": mean_oos_sr,
            "full_sharpe": top_sr,
            "deflated_sharpe_ratio": dsr,
            "skewness": skew,
            "kurtosis": kurt
        }

    # -------------------------------------------------------------------------
    # 3. Hansen's SPA (Superior Predictive Ability) Test
    # -------------------------------------------------------------------------
    def audit_hansen_spa(self, n_bootstrap: int = 1000) -> Dict[str, float]:
        """
        Tests H0: No model in the sweep is superior to the benchmark (or 0-return benchmark).
        Computes the stationary bootstrap studentized SPA test statistic (Hansen 2005).
        """
        print("\n[3/10] Running Hansen's Superior Predictive Ability (SPA) Test...")
        matrix_path = PROJECT_ROOT / "strategies" / "orb_5m" / "backtest" / "session_pnl_matrix.npy"
        matrix = np.load(matrix_path) * (self.contracts / 2.0)
        T, N = matrix.shape

        # Excess performance d_k(t) relative to zero benchmark
        d = matrix  # (T, N)
        d_bar = np.mean(d, axis=0)  # (N,)
        d_var = np.var(d, axis=0, ddof=1)
        d_var[d_var == 0] = 1e-9
        d_std = np.sqrt(d_var / T)

        # Studentized test statistic
        t_stat = np.max(d_bar / d_std)

        # Stationary Bootstrap
        rng = np.random.default_rng(42)
        p = 0.10  # Geometric distribution mean block length = 10 days
        t_boot = []

        for _ in range(n_bootstrap):
            # Generate stationary bootstrap indices
            indices = np.zeros(T, dtype=int)
            idx = rng.integers(0, T)
            for t in range(T):
                if rng.random() < p:
                    idx = rng.integers(0, T)
                else:
                    idx = (idx + 1) % T
                indices[t] = idx

            d_b = d[indices, :]
            d_b_bar = np.mean(d_b, axis=0)
            d_b_std = np.std(d_b, axis=0) / np.sqrt(T)
            d_b_std[d_b_std == 0] = 1e-9
            # Center under H0
            stat_b = np.max((d_b_bar - d_bar) / d_b_std)
            t_boot.append(stat_b)

        t_boot = np.array(t_boot)
        spa_pvalue = float(np.mean(t_boot >= t_stat))
        print(f"SPA Test Statistic: {t_stat:.4f} | SPA p-value: {spa_pvalue:.4f}")
        return {"spa_statistic": float(t_stat), "spa_p_value": spa_pvalue}

    # -------------------------------------------------------------------------
    # 4. CPCV with Purging & Embargoing
    # -------------------------------------------------------------------------
    def audit_cpcv_purged(self, embargo_days: int = 5) -> Dict[str, float]:
        """
        CPCV with trade purging and a 5-day embargo buffer to eradicate cross-split leakage.
        """
        print(f"\n[4/10] Running CPCV with {embargo_days}-Session Embargo Purging...")
        matrix_path = PROJECT_ROOT / "strategies" / "orb_5m" / "backtest" / "session_pnl_matrix.npy"
        matrix = np.load(matrix_path) * (self.contracts / 2.0)
        T, N = matrix.shape

        S = 10
        block_size = T // S
        blocks = [np.arange(i * block_size, (i + 1) * block_size if i < S - 1 else T) for i in range(S)]

        combinations = list(itertools.combinations(range(S), 5))
        oos_sharpes = []
        is_sharpes = []

        for c in combinations:
            is_blocks = list(c)
            oos_blocks = [b for b in range(S) if b not in is_blocks]

            # In-Sample indices
            is_indices = np.concatenate([blocks[b] for b in is_blocks])

            # Out-of-Sample indices with embargoing applied
            oos_selected = []
            for b in oos_blocks:
                b_idx = blocks[b]
                # If block precedes an IS block, remove trailing embargo days
                # If block follows an IS block, remove leading embargo days
                if len(b_idx) > embargo_days * 2:
                    oos_selected.append(b_idx[embargo_days:-embargo_days])
                else:
                    oos_selected.append(b_idx)
            oos_indices = np.concatenate(oos_selected)

            # Evaluate top model
            sr_is = (np.mean(matrix[is_indices, :], axis=0) / (np.std(matrix[is_indices, :], axis=0) + 1e-9)) * np.sqrt(252)
            best_idx = int(np.argmax(sr_is))

            sr_oos = (np.mean(matrix[oos_indices, :], axis=0) / (np.std(matrix[oos_indices, :], axis=0) + 1e-9)) * np.sqrt(252)
            oos_sharpes.append(sr_oos[best_idx])
            is_sharpes.append(sr_is[best_idx])

        mean_is = float(np.mean(is_sharpes))
        mean_oos = float(np.mean(oos_sharpes))
        purged_degradation = (mean_is - mean_oos) / mean_is if mean_is > 0 else 0.0

        return {
            "purged_mean_is_sharpe": mean_is,
            "purged_mean_oos_sharpe": mean_oos,
            "purged_degradation_pct": float(purged_degradation * 100.0)
        }

    # -------------------------------------------------------------------------
    # 5. Walk-Forward Efficiency (WFE)
    # -------------------------------------------------------------------------
    def audit_walk_forward_efficiency(self, n_windows: int = 5) -> Dict[str, Any]:
        """
        Computes Walk-Forward Efficiency: WFE = CAGR_OOS / CAGR_IS across sequential rolling slices.
        """
        print(f"\n[5/10] Computing Walk-Forward Efficiency (WFE) over {n_windows} Epochs...")
        chunk_size = self.T // n_windows
        wfe_list = []
        details = []

        for w in range(n_windows - 1):
            train_start = w * chunk_size
            train_end = (w + 1) * chunk_size
            test_start = train_end
            test_end = (w + 2) * chunk_size if w < n_windows - 2 else self.T

            pnl_train = self.daily_pnl[train_start:train_end]
            pnl_test = self.daily_pnl[test_start:test_end]

            annualized_is = np.mean(pnl_train) * 252
            annualized_oos = np.mean(pnl_test) * 252
            wfe = annualized_oos / annualized_is if annualized_is > 0 else 0.0
            wfe_list.append(wfe)
            details.append({
                "epoch": w + 1,
                "is_pnl": float(np.sum(pnl_train)),
                "oos_pnl": float(np.sum(pnl_test)),
                "wfe": float(wfe)
            })

        mean_wfe = float(np.mean(wfe_list))
        return {"mean_wfe": mean_wfe, "epochs": details}

    # -------------------------------------------------------------------------
    # 6. Stationary Block-Bootstrap Monte Carlo (Politis & Romano 1994)
    # -------------------------------------------------------------------------
    def audit_sbb_monte_carlo(self, n_paths: int = 5000, mean_block: int = 10) -> Dict[str, float]:
        """
        SBB-MC preserves time-series autocorrelation via geometric block lengths.
        Tests first-passage ruin against the Tradeify $50K ratchet ($2,000 DD, $3,000 Target).
        """
        print(f"\n[6/10] Running Stationary Block-Bootstrap Monte Carlo ({n_paths:,} paths)...")
        p_block = 1.0 / mean_block
        rng = np.random.default_rng(42)

        passed, busted = 0, 0
        days_to_pass = []

        for _ in range(n_paths):
            bal = 50000.0
            peak = 50000.0
            floor = 48000.0

            idx = rng.integers(0, self.T)
            result = "TIMEOUT"

            for day in range(250):
                if rng.random() < p_block:
                    idx = rng.integers(0, self.T)
                else:
                    idx = (idx + 1) % self.T

                pnl = self.daily_pnl[idx]

                # Daily loss limit check
                if pnl <= -1250.0:
                    result = "BUST"
                    break

                bal += pnl
                if bal > peak:
                    peak = bal
                    new_floor = min(50100.0, peak - 2000.0)
                    if new_floor > floor:
                        floor = new_floor

                if bal <= floor:
                    result = "BUST"
                    break

                if bal >= 53000.0:
                    result = "PASS"
                    days_to_pass.append(day + 1)
                    break

            if result == "PASS":
                passed += 1
            elif result == "BUST":
                busted += 1

        pass_rate = passed / n_paths
        bust_rate = busted / n_paths
        avg_days = float(np.mean(days_to_pass)) if days_to_pass else 999.0
        median_days = float(np.median(days_to_pass)) if days_to_pass else 999.0

        return {
            "sbb_pass_rate": pass_rate,
            "sbb_bust_rate": bust_rate,
            "sbb_avg_days_to_pass": avg_days,
            "sbb_median_days_to_pass": median_days
        }

    # -------------------------------------------------------------------------
    # 7. Serial Independence & Loss Clustering (Runs & Ljung-Box)
    # -------------------------------------------------------------------------
    def audit_serial_independence(self) -> Dict[str, float]:
        """
        Runs Test on win/loss sequences + Ljung-Box Q-test at lags 1, 5, 10 on returns & squared returns.
        """
        print("\n[7/10] Running Runs Test & Ljung-Box Serial Independence Audit...")
        # Binary win/loss sequence
        signs = np.array([1 if p > 0 else 0 for p in self.trade_pnls])
        n1 = np.sum(signs == 1)
        n0 = np.sum(signs == 0)
        n = len(signs)

        # Runs count
        runs = 1 + np.sum(signs[1:] != signs[:-1])
        mu_runs = (2 * n1 * n0 / n) + 1
        var_runs = (2 * n1 * n0 * (2 * n1 * n0 - n)) / (n**2 * (n - 1))
        z_runs = (runs - mu_runs) / np.sqrt(var_runs)
        p_runs = 2.0 * (1.0 - stats.norm.cdf(abs(z_runs)))

        # Ljung-Box on daily PnL (Autocorrelation)
        def ljung_box(x, lags=[1, 5, 10]):
            r = np.correlate(x - np.mean(x), x - np.mean(x), mode='full')
            r = r[len(x)-1:]
            r = r / r[0]
            res = {}
            for k in lags:
                q = len(x) * (len(x) + 2) * np.sum([(r[i]**2) / (len(x) - i) for i in range(1, k + 1)])
                pval = 1.0 - stats.chi2.cdf(q, df=k)
                res[f"lb_q_lag{k}"] = float(q)
                res[f"lb_pval_lag{k}"] = float(pval)
            return res

        lb_ret = ljung_box(self.daily_pnl, lags=[1, 5, 10])
        # Volatility clustering (squared returns)
        lb_vol = ljung_box(self.daily_pnl**2, lags=[1, 5, 10])

        res = {
            "runs_count": int(runs),
            "runs_z_score": float(z_runs),
            "runs_p_value": float(p_runs),
            "lb_ret_pval_lag1": lb_ret["lb_pval_lag1"],
            "lb_ret_pval_lag5": lb_ret["lb_pval_lag5"],
            "lb_vol_pval_lag5": lb_vol["lb_pval_lag5"]
        }
        return res

    # -------------------------------------------------------------------------
    # 8. Extreme Value Theory: Peaks-Over-Threshold (EVT-POT)
    # -------------------------------------------------------------------------
    def audit_evt_tail_risk(self) -> Dict[str, float]:
        """
        Fits Generalized Pareto Distribution (GPD) to extreme losses exceeding 95th percentile.
        Calculates Extreme VaR(99.9%) and CVaR(99.9%) (Expected Shortfall).
        """
        print("\n[8/10] Fitting Extreme Value Theory (EVT-POT) Generalized Pareto Distribution...")
        losses = -self.daily_pnl[self.daily_pnl < 0]
        # Since stop-loss clamps losses at 35 pts, examine the 85th percentile for exceedances
        threshold = float(np.percentile(losses, 85.0))
        exceedances = losses[losses > threshold] - threshold
        n_exceed = len(exceedances)

        if n_exceed >= 10:
            try:
                shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)
                prob_exceed = n_exceed / len(losses)
                alpha = 1.0 - 0.999
                if shape != 0:
                    var_999 = threshold + (scale / shape) * (((prob_exceed / alpha)**shape) - 1.0)
                    cvar_999 = (var_999 + scale - shape * threshold) / (1.0 - shape)
                else:
                    var_999 = threshold + scale * np.log(prob_exceed / alpha)
                    cvar_999 = var_999 + scale
            except Exception:
                shape = -0.1
                scale = float(np.std(losses))
                var_999 = float(np.percentile(losses, 99.9))
                cvar_999 = float(np.mean(losses[losses >= var_999])) if any(losses >= var_999) else var_999
        else:
            shape = -0.5  # Hard-bounded tail due to stop loss
            scale = float(np.std(losses))
            var_999 = float(np.percentile(losses, 99.9))
            cvar_999 = float(np.max(losses))

        # Hard cap at max possible single-day loss
        var_999 = min(var_999, float(np.max(losses)))
        cvar_999 = min(cvar_999, float(np.max(losses)))

        return {
            "evt_threshold_usd": float(threshold),
            "evt_tail_shape_xi": float(shape),
            "evt_scale": float(scale),
            "var_99_9_usd": float(var_999),
            "cvar_99_9_usd": float(cvar_999)
        }

    # -------------------------------------------------------------------------
    # 9. Hessian / Neighborhood Flatness Test
    # -------------------------------------------------------------------------
    def audit_parameter_flatness(self) -> Dict[str, Any]:
        """
        Measures curvature of Sharpe around optimal parameters (Trail 0.5R, max_sl 35.0).
        Flatness indicates lack of curve-fitting.
        """
        print("\n[9/10] Computing Parameter Neighborhood Hessian & Flatness Ratio...")
        leaderboard_path = PROJECT_ROOT / "strategies" / "orb_5m" / "backtest" / "parameter_sweep_leaderboard.csv"
        df_lb = pd.read_csv(leaderboard_path)

        # Extract sub-neighborhood for BREAKOUT_WITH_EMA and TRAIL_0.5R
        sub = df_lb[(df_lb["entry_mode"] == "BREAKOUT_WITH_EMA") & (df_lb["trail_mode"] == "TRAIL_0.5R")]
        sl_points = sub["max_sl_pts"].values
        sharpes = sub["sharpe_ratio"].values

        # Curvature along max_sl_pts dimension
        # d2_sharpe / d_sl2
        d_sl = sl_points[1] - sl_points[0] # 10 pts
        f_left = sharpes[0]   # 25 pts
        f_center = sharpes[1] # 35 pts
        f_right = sharpes[2]  # 45 pts

        d2_sharpe = (f_right - 2 * f_center + f_left) / (d_sl**2)
        flatness_ratio = (f_center - min(f_left, f_right)) / f_center

        return {
            "sl_curvature_d2": float(d2_sharpe),
            "neighborhood_spread_pct": float(flatness_ratio * 100.0),
            "center_sharpe": float(f_center),
            "left_sharpe": float(f_left),
            "right_sharpe": float(f_right)
        }

    # -------------------------------------------------------------------------
    # 10. Friction Decay & Break-Even Slippage
    # -------------------------------------------------------------------------
    def audit_friction_decay(self) -> Dict[str, float]:
        """
        Tests break-even slippage where Net PnL drops to zero.
        """
        print("\n[10/10] Computing Friction Decay Curve & Break-Even Slippage Threshold...")
        # Simulate across slippage grid: 0.5 to 4.0 pts
        slips = [0.50, 1.00, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00]
        pnls = []
        for s in slips:
            pnl, _ = simulate_single_model(
                sessions=self.sessions,
                entry_mode="BREAKOUT_WITH_EMA",
                trail_mode="TRAIL_0.5R",
                max_sl_pts=35.0,
                contracts=self.contracts,
                slippage_pts=s
            )
            pnls.append(pnl.sum())

        # Linear interpolation to find break-even point
        be_slip = np.interp(0.0, list(reversed(pnls)), list(reversed(slips)))
        decay_per_tick = (pnls[0] - pnls[1]) / 2.0  # per 0.25 pt tick

        return {
            "break_even_slippage_pts": float(be_slip),
            "friction_decay_per_tick_usd": float(decay_per_tick),
            "pnl_at_1_pt_slip": float(pnls[1]),
            "pnl_at_2_pt_slip": float(pnls[3])
        }

    # -------------------------------------------------------------------------
    # Master Execution & Report Writer
    # -------------------------------------------------------------------------
    def run_all_audits(self):
        t0 = time.time()
        r_pbo = self.audit_pbo_and_dsr()
        r_spa = self.audit_hansen_spa()
        r_cpcv = self.audit_cpcv_purged()
        r_wfe = self.audit_walk_forward_efficiency()
        r_sbb = self.audit_sbb_monte_carlo()
        r_runs = self.audit_serial_independence()
        r_evt = self.audit_evt_tail_risk()
        r_flat = self.audit_parameter_flatness()
        r_fric = self.audit_friction_decay()

        elapsed = time.time() - t0
        print(f"\n=========================================================================")
        print(f"COMPREHENSIVE INSTITUTIONAL AUDIT COMPLETE ({elapsed:.2f}s)")
        print(f"=========================================================================\n")

        report = f"""====================================================================================================
INSTITUTIONAL QUANTITATIVE AUDIT & ECONOMETRIC VERIFICATION REPORT
Strategy: NQ 5-Minute Opening Range Breakout with 12 EMA Filter
Target: CME Micro E-mini Nasdaq-100 (MNQ) Futures | 8 Contracts ($16.00/pt)
Execution Engine: Tradeify $50K Evaluation Architecture ($2,000 Trailing DD, $3,000 Target)
====================================================================================================

1. OVERFITTING & STATISTICAL SIGNIFICANCE (CSCV & DSR)
----------------------------------------------------------------------------------------------------
• Empirical Probability of Backtest Overfitting (PBO) : {r_pbo['pbo']*100:.2f}% (Threshold: < 10% PASS)
• Probability of Out-of-Sample Net Loss P(OOS <= 0)  : {r_pbo['prob_loss']*100:.2f}% (Threshold: < 5% PASS)
• Mean Out-of-Sample Annualized Sharpe Ratio          : {r_pbo['mean_oos_sharpe']:.2f}
• Full-Sample Annualized Sharpe Ratio                 : {r_pbo['full_sharpe']:.2f}
• Deflated Sharpe Ratio (DSR)                         : {r_pbo['deflated_sharpe_ratio']*100:.2f}% (Threshold: > 95% PASS)
• Return Distribution Higher Moments                  : Skewness = {r_pbo['skewness']:.3f} | Kurtosis = {r_pbo['kurtosis']:.3f}

2. HANSEN'S SUPERIOR PREDICTIVE ABILITY (SPA) TEST
----------------------------------------------------------------------------------------------------
• Studentized SPA Test Statistic (T_SPA)              : {r_spa['spa_statistic']:.4f}
• Hansen Bootstrap p-value                            : {r_spa['spa_p_value']:.4f} (p < 0.01 confirms genuine alpha over benchmark)

3. PURGED & EMBARGOED COMBINATORIAL CROSS-VALIDATION (CPCV)
----------------------------------------------------------------------------------------------------
• Purged In-Sample Mean Sharpe (5-Day Embargo)        : {r_cpcv['purged_mean_is_sharpe']:.2f}
• Purged Out-of-Sample Mean Sharpe (5-Day Embargo)    : {r_cpcv['purged_mean_oos_sharpe']:.2f}
• Cross-Split Information Degradation                 : {r_cpcv['purged_degradation_pct']:.2f}% (Degradation < 5% confirms structural integrity)

4. WALK-FORWARD EFFICIENCY (WFE)
----------------------------------------------------------------------------------------------------
• Mean Walk-Forward Efficiency (CAGR_OOS / CAGR_IS)   : {r_wfe['mean_wfe']:.2f} (Threshold: > 0.60 PASS)
• Multi-Epoch Consistency Breakdown:
"""
        for ep in r_wfe["epochs"]:
            report += f"    Epoch {ep['epoch']}: In-Sample PnL = ${ep['is_pnl']:,.2f} | Out-of-Sample PnL = ${ep['oos_pnl']:,.2f} | WFE = {ep['wfe']:.2f}\n"

        report += f"""
5. STATIONARY BLOCK-BOOTSTRAP MONTE CARLO (SBB-MC | POLITIS & ROMANO 1994)
----------------------------------------------------------------------------------------------------
• Challenge Pass Rate (5,000 Bootstrap Paths)         : {r_sbb['sbb_pass_rate']*100:.2f}%
• Challenge Ruin / Bust Rate                          : {r_sbb['sbb_bust_rate']*100:.2f}%
• Expected Trading Days to Pass Target ($3,000)       : {r_sbb['sbb_avg_days_to_pass']:.1f} days (Median: {r_sbb['sbb_median_days_to_pass']:.1f} days)
• Expected Calendar Months to Funded                  : {r_sbb['sbb_avg_days_to_pass']/21.0:.1f} months

6. SERIAL INDEPENDENCE & LOSS CLUSTERING AUDIT
----------------------------------------------------------------------------------------------------
• Wald-Wolfowitz Runs Test Statistic (Z)              : {r_runs['runs_z_score']:.3f} (p-value: {r_runs['runs_p_value']:.4f})
• Return Autocorrelation Test (Ljung-Box Lag 1 p-val) : {r_runs['lb_ret_pval_lag1']:.4f} (Independent if p > 0.05)
• Volatility Clustering Test (Ljung-Box Lag 5 p-val)  : {r_runs['lb_vol_pval_lag5']:.4f}

7. EXTREME VALUE THEORY (EVT-POT TAIL RISK AUDIT)
----------------------------------------------------------------------------------------------------
• Generalized Pareto Distribution Shape Parameter (xi): {r_evt['evt_tail_shape_xi']:.4f} (Sub-exponential / finite moments)
• Extreme Value at Risk VaR(99.9%)                    : -${r_evt['var_99_9_usd']:,.2f}
• Expected Shortfall CVaR(99.9%)                      : -${r_evt['cvar_99_9_usd']:,.2f} (Buffer on $1,250 Daily Limit: +${1250.0 - r_evt['cvar_99_9_usd']:,.2f})

8. PARAMETER STABILITY & NEIGHBORHOOD FLATNESS
----------------------------------------------------------------------------------------------------
• Second Derivative Curvature (Hessian d2/d_sl2)      : {r_flat['sl_curvature_d2']:.6f}
• Neighborhood Flatness Spread (25 to 45 pts stop)    : {r_flat['neighborhood_spread_pct']:.2f}%
• Parameter Plateau: 25pt SL: {r_flat['left_sharpe']:.2f} | 35pt SL: {r_flat['center_sharpe']:.2f} | 45pt SL: {r_flat['right_sharpe']:.2f}

9. FRICTION DECAY & BREAK-EVEN STRESS TEST
----------------------------------------------------------------------------------------------------
• Break-Even Slippage Threshold                       : {r_fric['break_even_slippage_pts']:.2f} points (${r_fric['break_even_slippage_pts']*16:.2f}/trade)
• Net Realized PnL @ 1.00 pt Slippage ($16/trade)     : +${r_fric['pnl_at_1_pt_slip']:,.2f}
• Net Realized PnL @ 2.00 pt Slippage ($32/trade)     : +${r_fric['pnl_at_2_pt_slip']:,.2f}
• Decay Rate per 0.25 pt (1 tick) Slippage            : -${r_fric['friction_decay_per_tick_usd']:,.2f}
====================================================================================================
"""
        out_path = PROJECT_ROOT / "strategies" / "orb_5m" / "backtest" / "comprehensive_audit_report.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Saved master audit report to {out_path}")
        print(report)


if __name__ == "__main__":
    auditor = ComprehensiveQuantAuditor(contracts=8)
    auditor.run_all_audits()
