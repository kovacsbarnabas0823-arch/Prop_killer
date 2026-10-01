"""
Institutional Parameter Sweep & Combinatorial Cross-Validation Engine
Strategy: Filtered Quantitative ORB (Narrow-Range Trend Day Model)
Author: Senior Quantitative Developer

Features:
  - Vectorized pre-computation of Daily 14-day ATR, Opening Range (09:30-10:00), and Session VWAP.
  - Parameter grid sweep:
      * max_or_atr_ratio: [0.35, 0.40, 0.45, 0.50, 0.55]
      * confirmation_timeframe: [3, 5, 10] minutes
      * target_rr: [1.5, 2.0, 2.5]
  - Combinatorial Purged Cross-Validation (CPCV) evaluation (N=6, k=2 -> 15 paths).
  - Deflated Sharpe Ratio (DSR) calculation correcting for selection bias under M trials.
  - Outputs optimization leaderboard to backtest/parameter_sweep_leaderboard.csv.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import logging
import itertools
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Support UTF-8 encoding on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from core.data_loader import load_continuous_nq_parquet, extract_daily_trading_sessions
from core.risk_state_machine import AccountType, RiskStateMachine
from backtest.data_validator import TradeOutcome
from backtest.cpcv_validator import CPCVSplitter, compute_sharpe_ratio, compute_deflated_sharpe_ratio

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("FilteredORBSweep")


class FastFilteredORBSweepEngine:
    def __init__(self, sessions: list[dict]):
        self.sessions = sessions
        self.n_sessions = len(sessions)
        self.risk_machine = RiskStateMachine(starting_balance=50000.0)
        self.point_value = 2.0  # MNQ: $2/pt
        self.commission_per_contract = 1.24
        self.slippage_points = 0.5
        self.cpcv = CPCVSplitter(n_splits=6, n_test_splits=2, embargo_pct=0.01)

        logger.info("Precomputing daily 14-day ATR, Opening Range (09:30-10:00), and Session VWAP...")
        self._precompute_session_data()

    def _precompute_session_data(self):
        """Precomputes rolling 14-day ATR, OR (09:30-10:00), and Session VWAP for each session."""
        # 1. Compute True Range for each daily session
        true_ranges = []
        for i, s in enumerate(self.sessions):
            h = s["daily_high"]
            l = s["daily_low"]
            if i > 0:
                prev_c = self.sessions[i - 1]["daily_close"]
                tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            else:
                tr = h - l
            true_ranges.append(tr)

        # 2. Compute 14-day rolling ATR strictly lagged by 1 day
        self.daily_atrs = []
        for i in range(self.n_sessions):
            if i < 14:
                atr = np.mean(true_ranges[:max(1, i)]) if i > 0 else 150.0
            else:
                atr = np.mean(true_ranges[i - 14:i])
            self.daily_atrs.append(float(atr))

        # 3. Precompute OR and numpy bar arrays
        self.session_or = []
        self.session_bars_data = []

        for s in self.sessions:
            rth_df = s["rth_bars"]
            if rth_df.empty:
                self.session_or.append((np.nan, np.nan, np.nan, np.nan))
                self.session_bars_data.append(np.empty((0, 8), dtype=np.float64))
                continue

            # OR from 09:30:00 to 09:59:59
            or_bars = rth_df.between_time("09:30", "09:59")
            if len(or_bars) < 15:
                self.session_or.append((np.nan, np.nan, np.nan, np.nan))
                self.session_bars_data.append(np.empty((0, 8), dtype=np.float64))
                continue

            or_h = float(or_bars["High"].max())
            or_l = float(or_bars["Low"].min())
            or_m = round((or_h + or_l) / 2.0, 2)
            or_rng = round(or_h - or_l, 2)
            self.session_or.append((or_h, or_l, or_m, or_rng))

            # Cumulative session VWAP from 09:30
            typical_price = (rth_df["High"] + rth_df["Low"] + rth_df["Close"]) / 3.0
            cum_vol = rth_df["Volume"].cumsum()
            cum_vol_price = (typical_price * rth_df["Volume"]).cumsum()
            day_vwap = cum_vol_price / np.maximum(cum_vol, 1)

            times = rth_df.index
            hours = times.hour.to_numpy(dtype=np.float64)
            minutes = times.minute.to_numpy(dtype=np.float64)
            minute_of_day = (hours * 60 + minutes).astype(np.float64)

            ohlc_vwap = np.column_stack([
                rth_df["Open"].to_numpy(dtype=np.float64),
                rth_df["High"].to_numpy(dtype=np.float64),
                rth_df["Low"].to_numpy(dtype=np.float64),
                rth_df["Close"].to_numpy(dtype=np.float64),
                day_vwap.to_numpy(dtype=np.float64),
                minute_of_day,
                hours,
                minutes
            ])
            self.session_bars_data.append(ohlc_vwap)

        logger.info(f"Precomputation complete for {self.n_sessions} trading sessions.")

    def run_single_parameter_set(
        self,
        max_or_atr_ratio: float = 0.50,
        confirmation_timeframe: int = 5,
        target_rr: float = 2.0
    ) -> pd.DataFrame:
        """Runs the Filtered Quant ORB across all sessions for a specific parameter set."""
        trades = []
        entry_cutoff_mod = 11 * 60 + 30  # 11:30 EST = 690 minutes
        force_close_mod = 15 * 60 + 45   # 15:45 EST = 945 minutes

        account_balance = 50000.0
        eod_barrier = 48000.0
        eod_hwm = 50000.0

        be_offset = 2 * 0.25  # 2 ticks = 0.50 pts

        for i in range(self.n_sessions):
            or_h, or_l, or_m, or_rng = self.session_or[i]
            if np.isnan(or_h):
                continue

            # Task 2: ATR Exhaustion Filter check
            daily_atr = self.daily_atrs[i]
            if or_rng > (max_or_atr_ratio * daily_atr):
                continue  # Abort session!

            s = self.sessions[i]
            bars = self.session_bars_data[i]
            n_bars = len(bars)
            if n_bars == 0:
                continue

            # HTF Institutional Alignment: Daily 20 EMA check
            open_p = float(s.get("exchange_open_price", 0.0))
            ema_20 = float(s.get("daily_20_ema", 0.0))
            long_htf_aligned = (open_p > ema_20) if (ema_20 > 0 and open_p > 0) else True
            short_htf_aligned = (open_p < ema_20) if (ema_20 > 0 and open_p > 0) else True

            active_trade_dir = 0
            entry_idx = -1
            entry_p = 0.0
            stop_p = 0.0
            tp_p = 0.0
            contracts = 1

            # Scan bars from 10:00 to 11:30 EST
            for b_idx in range(n_bars):
                cl = bars[b_idx, 3]
                vwap = bars[b_idx, 4]
                mod = int(bars[b_idx, 5])
                mn = int(bars[b_idx, 7])

                # Skip 09:30-09:59 flat zone
                if mod < 600:
                    continue

                # Stop searching after 11:30 EST
                if mod > entry_cutoff_mod:
                    break

                # Confirmation Candle Close check
                is_confirmation_close = ((mn + 1) % confirmation_timeframe == 0)
                if not is_confirmation_close:
                    continue

                # LONG Trigger: Close > OR_High AND Close > VWAP AND 09:30 Open > Daily 20 EMA
                if long_htf_aligned and cl > or_h and cl > vwap:
                    risk = cl - or_m
                    if 5.0 <= risk <= 60.0:
                        active_trade_dir = 1
                        entry_idx = b_idx
                        entry_p = cl
                        stop_p = round(or_m, 2)
                        tp_p = round(entry_p + (target_rr * risk), 2)

                        try:
                            sizing = self.risk_machine.calculate_sizing(
                                AccountType.EVALUATION, account_balance, eod_barrier, risk
                            )
                            contracts = max(1, sizing.contracts)
                        except Exception:
                            account_balance = 50000.0
                            eod_barrier = 48000.0
                            eod_hwm = 50000.0
                            contracts = 1
                        break

                # SHORT Trigger: Close < OR_Low AND Close < VWAP AND 09:30 Open < Daily 20 EMA
                if short_htf_aligned and cl < or_l and cl < vwap:
                    risk = or_m - cl
                    if 5.0 <= risk <= 60.0:
                        active_trade_dir = -1
                        entry_idx = b_idx
                        entry_p = cl
                        stop_p = round(or_m, 2)
                        tp_p = round(entry_p - (target_rr * risk), 2)

                        try:
                            sizing = self.risk_machine.calculate_sizing(
                                AccountType.EVALUATION, account_balance, eod_barrier, risk
                            )
                            contracts = max(1, sizing.contracts)
                        except Exception:
                            account_balance = 50000.0
                            eod_barrier = 48000.0
                            eod_hwm = 50000.0
                            contracts = 1
                        break

            # Forward Bar Resolution with Pessimistic Execution Rule & Dynamic Break-Even
            if active_trade_dir != 0:
                outcome = "TIME_EXIT_1545"
                exit_p = bars[-1, 3]
                be_triggered = False
                initial_risk = abs(entry_p - stop_p)

                for post_idx in range(entry_idx + 1, n_bars):
                    post_hi = bars[post_idx, 1]
                    post_lo = bars[post_idx, 2]
                    post_mod = int(bars[post_idx, 5])

                    if active_trade_dir == 1:  # Long
                        if not be_triggered:
                            sl_hit = (post_lo <= stop_p)
                            be_hit = (post_hi >= entry_p + initial_risk)
                            tp_hit = (post_hi >= tp_p)

                            if sl_hit and (be_hit or tp_hit):
                                outcome = TradeOutcome.PESSIMISTIC_STOP
                                exit_p = stop_p - self.slippage_points
                                break
                            elif sl_hit:
                                outcome = TradeOutcome.STOP_LOSS
                                exit_p = stop_p - self.slippage_points
                                break
                            elif tp_hit:
                                outcome = TradeOutcome.TAKE_PROFIT
                                exit_p = tp_p
                                break
                            elif be_hit:
                                be_triggered = True
                                stop_p = round(entry_p + be_offset, 2)
                                if post_lo <= stop_p:
                                    outcome = TradeOutcome.BREAK_EVEN
                                    exit_p = stop_p
                                    break
                        else:  # BE active
                            be_sl_hit = (post_lo <= stop_p)
                            tp_hit = (post_hi >= tp_p)
                            if be_sl_hit and tp_hit:
                                outcome = TradeOutcome.BREAK_EVEN
                                exit_p = stop_p
                                break
                            elif be_sl_hit:
                                outcome = TradeOutcome.BREAK_EVEN
                                exit_p = stop_p
                                break
                            elif tp_hit:
                                outcome = TradeOutcome.TAKE_PROFIT
                                exit_p = tp_p
                                break

                    else:  # Short
                        if not be_triggered:
                            sl_hit = (post_hi >= stop_p)
                            be_hit = (post_lo <= entry_p - initial_risk)
                            tp_hit = (post_lo <= tp_p)

                            if sl_hit and (be_hit or tp_hit):
                                outcome = TradeOutcome.PESSIMISTIC_STOP
                                exit_p = stop_p + self.slippage_points
                                break
                            elif sl_hit:
                                outcome = TradeOutcome.STOP_LOSS
                                exit_p = stop_p + self.slippage_points
                                break
                            elif tp_hit:
                                outcome = TradeOutcome.TAKE_PROFIT
                                exit_p = tp_p
                                break
                            elif be_hit:
                                be_triggered = True
                                stop_p = round(entry_p - be_offset, 2)
                                if post_hi >= stop_p:
                                    outcome = TradeOutcome.BREAK_EVEN
                                    exit_p = stop_p
                                    break
                        else:  # BE active
                            be_sl_hit = (post_hi >= stop_p)
                            tp_hit = (post_lo <= tp_p)
                            if be_sl_hit and tp_hit:
                                outcome = TradeOutcome.BREAK_EVEN
                                exit_p = stop_p
                                break
                            elif be_sl_hit:
                                outcome = TradeOutcome.BREAK_EVEN
                                exit_p = stop_p
                                break
                            elif tp_hit:
                                outcome = TradeOutcome.TAKE_PROFIT
                                exit_p = tp_p
                                break

                    # Force close at 15:45 EST
                    if post_mod >= force_close_mod:
                        outcome = "TIME_EXIT_1545"
                        exit_p = bars[post_idx, 3]
                        break

                if active_trade_dir == 1:
                    gross = (exit_p - entry_p) * self.point_value * contracts
                else:
                    gross = (entry_p - exit_p) * self.point_value * contracts
                realized_pnl = gross - (self.commission_per_contract * contracts)

                trades.append({
                    "session_idx": i,
                    "date": s["date"],
                    "direction": active_trade_dir,
                    "contracts": contracts,
                    "outcome": outcome,
                    "net_pnl": round(realized_pnl, 2)
                })

                # Update balance and trailing barrier
                account_balance += realized_pnl
                if account_balance <= eod_barrier or account_balance >= 53000.0:
                    account_balance = 50000.0
                    eod_barrier = 48000.0
                    eod_hwm = 50000.0
                elif account_balance > eod_hwm:
                    eod_hwm = account_balance
                    new_b = min(50000.0, eod_hwm - 2000.0)
                    if new_b > eod_barrier:
                        eod_barrier = new_b

        return pd.DataFrame(trades)

    def evaluate_cpcv(self, trades_df: pd.DataFrame) -> Tuple[float, float]:
        """Evaluates CPCV across 15 combinatorial paths, returns (mean_test_sr, min_test_sr)."""
        if len(trades_df) < 15:
            return 0.0, 0.0

        cv_paths = self.cpcv.split(self.n_sessions)
        test_sharpes = []
        trades_by_session = trades_df.set_index("session_idx")["net_pnl"].to_dict()

        for train_idx, test_idx in cv_paths:
            test_pnls = [trades_by_session.get(idx, 0.0) for idx in test_idx]
            test_arr = np.array(test_pnls)
            sr = compute_sharpe_ratio(test_arr)
            test_sharpes.append(sr)

        return float(np.mean(test_sharpes)), float(np.min(test_sharpes))


def run_parameter_sweep():
    logger.info("=" * 80)
    logger.info("STARTING FILTERED QUANTITATIVE ORB PARAMETER OPTIMIZATION & CPCV")
    logger.info("=" * 80)

    df = load_continuous_nq_parquet(filter_rth=False)
    sessions = extract_daily_trading_sessions(df, rth_start="09:30", rth_end="15:45")

    engine = FastFilteredORBSweepEngine(sessions)

    # Grid search space for Filtered Quant ORB
    max_or_atr_ratios = [0.35, 0.40, 0.45, 0.50, 0.55]
    confirmation_tfs = [3, 5, 10]
    target_rrs = [1.5, 2.0, 2.5]

    grid = list(itertools.product(max_or_atr_ratios, confirmation_tfs, target_rrs))
    logger.info(f"Total Parameter Combinations to Test: {len(grid)}")

    results = []
    all_trials_sr = []
    trade_dfs_cache = {}

    for idx, (atr_ratio, c_tf, r_r) in enumerate(grid):
        tdf = engine.run_single_parameter_set(atr_ratio, c_tf, r_r)
        trade_dfs_cache[idx] = tdf

        n_trades = len(tdf)
        if n_trades >= 10:
            daily_pnls = tdf.groupby("date")["net_pnl"].sum().values
            sr = compute_sharpe_ratio(daily_pnls)
        else:
            sr = 0.0
        all_trials_sr.append(sr)

        if (idx + 1) % 15 == 0 or (idx + 1) == len(grid):
            logger.info(f"Sweep progress: {idx + 1}/{len(grid)} combinations tested...")

    # Second pass: Compute CPCV Sharpe, DSR, Win Rate, Profit Factor, EV Velocity
    for idx, (atr_ratio, c_tf, r_r) in enumerate(grid):
        tdf = trade_dfs_cache[idx]
        n_trades = len(tdf)
        if n_trades < 10:
            continue

        wins = tdf[tdf["net_pnl"] > 0]
        losses = tdf[tdf["net_pnl"] <= 0]
        win_rate = (len(wins) / n_trades) * 100.0
        total_pnl = tdf["net_pnl"].sum()
        gross_win = wins["net_pnl"].sum()
        gross_loss = abs(losses["net_pnl"].sum())
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else 0.0

        daily_pnls = tdf.groupby("date")["net_pnl"].sum().values
        obs_sr = all_trials_sr[idx]

        # CPCV Evaluation
        cpcv_mean_sr, cpcv_min_sr = engine.evaluate_cpcv(tdf)

        # Deflated Sharpe Ratio (DSR)
        dsr_info = compute_deflated_sharpe_ratio(obs_sr, all_trials_sr, daily_pnls)

        expectancy = tdf["net_pnl"].mean()
        trades_per_day = n_trades / engine.n_sessions
        ev_velocity = expectancy * trades_per_day

        results.append({
            "max_or_atr_ratio": atr_ratio,
            "confirmation_tf": c_tf,
            "target_rr": r_r,
            "trades": n_trades,
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "net_pnl": round(total_pnl, 2),
            "expectancy": round(expectancy, 2),
            "sharpe_ratio": round(obs_sr, 2),
            "cpcv_mean_sr": round(cpcv_mean_sr, 2),
            "cpcv_min_sr": round(cpcv_min_sr, 2),
            "dsr_pval": dsr_info["dsr"],
            "ev_velocity_daily": round(ev_velocity, 2)
        })

    leaderboard = pd.DataFrame(results)
    leaderboard.sort_values(by=["cpcv_mean_sr", "dsr_pval", "ev_velocity_daily"], ascending=False, inplace=True)
    leaderboard.reset_index(drop=True, inplace=True)

    out_csv = PROJECT_ROOT / "backtest" / "parameter_sweep_leaderboard.csv"
    leaderboard.to_csv(out_csv, index=False)
    logger.info(f"Leaderboard saved to {out_csv}")

    print("\n" + "=" * 115)
    print("FILTERED QUANTITATIVE ORB OPTIMIZATION LEADERBOARD (CPCV & DSR)")
    print("=" * 115)
    print(leaderboard.head(10).to_string())
    print("=" * 115)

    # Select best configuration
    best = leaderboard.iloc[0]
    print(f"\nOPTIMAL PARAMETERS SELECTED:")
    print(f"  Max OR/ATR Ratio   : {best['max_or_atr_ratio']}")
    print(f"  Confirmation TF    : {int(best['confirmation_tf'])} minutes")
    print(f"  Target R:R         : {best['target_rr']}x")
    print(f"  Total Trades       : {int(best['trades'])} ({best['trades'] / (engine.n_sessions/252):.1f} trades/year)")
    print(f"  Win Rate           : {best['win_rate_pct']}% | Profit Factor: {best['profit_factor']}")
    print(f"  CPCV Mean Sharpe   : {best['cpcv_mean_sr']} | DSR p-value: {best['dsr_pval']}")
    print(f"  Daily EV Velocity  : ${best['ev_velocity_daily']} / day\n")

    # Update strategy_config.yaml with optimal parameters
    config_file = PROJECT_ROOT / "config" / "strategy_config.yaml"
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["strategy"]["filters"]["atr_exhaustion"]["max_or_atr_ratio"] = float(best["max_or_atr_ratio"])
    cfg["strategy"]["filters"]["confirmation_timeframe_minutes"] = int(best["confirmation_tf"])
    cfg["strategy"]["risk_management"]["target_rr"] = float(best["target_rr"])

    with open(config_file, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    logger.info(f"Updated {config_file} with optimal parameters.")

    return leaderboard


if __name__ == "__main__":
    run_parameter_sweep()
