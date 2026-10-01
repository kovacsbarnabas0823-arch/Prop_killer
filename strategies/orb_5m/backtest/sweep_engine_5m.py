"""
High-Performance Parameter Sweep Engine for 5-Minute ORB Strategy
Evaluates Trailing Stop Architectures, Entry Modes, and Risk Caps across 8.7 Years of Data
Produces Session PnL Matrix (T x N) for CSCV / PBO Analysis
"""

from __future__ import annotations
import sys
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data_loader import load_continuous_nq_parquet

POINT_VALUE_MNQ = 2.0
COMMISSION_PER_CONTRACT = 1.24
SLIPPAGE_POINTS = 0.50


@dataclass
class FastTrade:
    session_idx: int
    direction: int
    entry_price: float
    exit_price: float
    pnl_dollars_net: float
    r_multiple: float
    exit_reason: str


class SessionPrecompiler:
    """Precompiles continuous 1m and 5m parquet data into high-speed memory buffers."""

    @staticmethod
    def load_sessions():
        parquet_path = PROJECT_ROOT / "data" / "nq_1m_continuous.parquet"
        print(f"Loading parquet from {parquet_path}...", flush=True)
        df_1m = load_continuous_nq_parquet(filepath=parquet_path, filter_rth=False)

        print("Resampling 5m bars & calculating 12 EMA...", flush=True)
        df_5m = df_1m[['Open', 'High', 'Low', 'Close', 'Volume']].resample('5min', label='left', closed='left').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        df_5m['EMA_12'] = df_5m['Close'].ewm(span=12, adjust=False).mean()

        rth_1m = df_1m.between_time("09:35", "15:45")
        candle_0930 = df_5m.between_time("09:30", "09:30")

        groups_1m = dict(tuple(rth_1m.groupby(rth_1m.index.date)))
        candle_dict = {ts.date(): row for ts, row in candle_0930.iterrows()}

        all_dates = sorted(list(set(rth_1m.index.date).intersection(candle_dict.keys())))

        sessions = []
        for d in all_dates:
            c = candle_dict[d]
            df_day = groups_1m.get(d)
            if df_day is None or len(df_day) == 0:
                continue

            sessions.append({
                "date": d,
                "c_open": float(c['Open']),
                "c_high": float(c['High']),
                "c_low": float(c['Low']),
                "c_close": float(c['Close']),
                "c_ema": float(c['EMA_12']),
                "opens": df_day['Open'].to_numpy(dtype=np.float64),
                "highs": df_day['High'].to_numpy(dtype=np.float64),
                "lows": df_day['Low'].to_numpy(dtype=np.float64),
                "closes": df_day['Close'].to_numpy(dtype=np.float64),
                "times": df_day.index.values
            })

        print(f"Precompiled {len(sessions):,} daily trading sessions.", flush=True)
        return sessions


def simulate_single_model(
    sessions: list,
    entry_mode: str,
    trail_mode: str,
    max_sl_pts: float,
    min_sl_pts: float = 6.0,
    point_value: float = POINT_VALUE_MNQ,
    contracts: int = 2,
    comm_per_contract: float = COMMISSION_PER_CONTRACT,
    slippage_pts: float = SLIPPAGE_POINTS
) -> Tuple[np.ndarray, List[FastTrade]]:
    """
    Simulates a configuration across all sessions and returns:
    - daily_pnl: np.ndarray of shape (n_sessions,) with net USD PnL per session
    - trades: list of executed FastTrade objects
    """
    n_sessions = len(sessions)
    daily_pnl = np.zeros(n_sessions, dtype=np.float64)
    trades: List[FastTrade] = []

    is_breakout = (entry_mode == "BREAKOUT_WITH_EMA")

    for s_idx, sess in enumerate(sessions):
        c_open = sess["c_open"]
        c_high = sess["c_high"]
        c_low = sess["c_low"]
        c_close = sess["c_close"]
        c_ema = sess["c_ema"]

        # Direction determination
        direction = 0
        if entry_mode == "BREAKOUT_WITH_EMA":
            if c_close > c_ema:
                direction = 1
            elif c_close < c_ema:
                direction = -1
        elif entry_mode == "CANDLE_AND_EMA":
            if c_close > c_open and c_close > c_ema:
                direction = 1
            elif c_close < c_open and c_close < c_ema:
                direction = -1
        elif entry_mode == "EMA_ONLY":
            if c_close > c_ema:
                direction = 1
            elif c_close < c_ema:
                direction = -1

        if direction == 0:
            continue

        # Noise floor check
        if (c_high - c_low) < min_sl_pts:
            continue

        opens = sess["opens"]
        highs = sess["highs"]
        lows = sess["lows"]
        n_bars = len(opens)
        if n_bars == 0:
            continue

        # Entry logic
        entered = False
        start_idx = 0
        entry_price = 0.0
        risk_pts = 0.0
        initial_sl = 0.0

        if not is_breakout:
            # Enter market at 09:35 open
            base_entry = opens[0]
            if direction == 1:
                entry_price = base_entry + slippage_pts
                raw_risk = entry_price - c_low
            else:
                entry_price = base_entry - slippage_pts
                raw_risk = c_high - entry_price

            risk_pts = min(max(raw_risk, min_sl_pts), max_sl_pts)
            initial_sl = entry_price - risk_pts if direction == 1 else entry_price + risk_pts
            entered = True
            start_idx = 0
        else:
            # Breakout search up to 85 minutes (until 11:00)
            max_scan = min(85, n_bars)
            for i in range(max_scan):
                if direction == 1 and highs[i] > c_high:
                    entry_price = c_high + slippage_pts
                    risk_pts = min(max(entry_price - c_low, min_sl_pts), max_sl_pts)
                    initial_sl = entry_price - risk_pts
                    start_idx = i
                    entered = True
                    break
                elif direction == -1 and lows[i] < c_low:
                    entry_price = c_low - slippage_pts
                    risk_pts = min(max(c_high - entry_price, min_sl_pts), max_sl_pts)
                    initial_sl = entry_price + risk_pts
                    start_idx = i
                    entered = True
                    break

        if not entered:
            continue

        # Target definitions
        target_price = None
        is_two_tier = False
        tp1_price = None
        tp1_hit = False

        if trail_mode == "FIXED_TP_1.5R":
            target_price = entry_price + (1.5 * risk_pts) if direction == 1 else entry_price - (1.5 * risk_pts)
        elif trail_mode == "FIXED_TP_2.0R":
            target_price = entry_price + (2.0 * risk_pts) if direction == 1 else entry_price - (2.0 * risk_pts)
        elif trail_mode == "FIXED_TP_2.5R":
            target_price = entry_price + (2.5 * risk_pts) if direction == 1 else entry_price - (2.5 * risk_pts)
        elif "TWO_TIER" in trail_mode:
            is_two_tier = True
            tp_mult = 1.25 if "1.25R" in trail_mode else 1.50
            tp1_price = entry_price + (tp_mult * risk_pts) if direction == 1 else entry_price - (tp_mult * risk_pts)

        current_sl = initial_sl
        peak = entry_price
        exit_price = None
        exit_reason = None

        # Intra-day bar traversal
        for i in range(start_idx, n_bars):
            b_high = highs[i]
            b_low = lows[i]

            # 1. Pessimistic Stop check
            if direction == 1:
                if b_low <= current_sl:
                    exit_price = current_sl - slippage_pts
                    exit_reason = "TRAILING_STOP" if current_sl > entry_price else "STOP_LOSS"
                    break
            else:
                if b_high >= current_sl:
                    exit_price = current_sl + slippage_pts
                    exit_reason = "TRAILING_STOP" if current_sl < entry_price else "STOP_LOSS"
                    break

            # 2. Fixed target check
            if target_price is not None:
                if direction == 1 and b_high >= target_price:
                    exit_price = target_price - slippage_pts
                    exit_reason = "TARGET_PROFIT"
                    break
                elif direction == -1 and b_low <= target_price:
                    exit_price = target_price + slippage_pts
                    exit_reason = "TARGET_PROFIT"
                    break

            # 3. Two-Tier TP1 check
            if is_two_tier and not tp1_hit:
                if direction == 1 and b_high >= tp1_price:
                    tp1_hit = True
                    # Ratchet stop to breakeven + 2 ticks
                    be_level = entry_price + 0.50
                    if be_level > current_sl:
                        current_sl = be_level
                elif direction == -1 and b_low <= tp1_price:
                    tp1_hit = True
                    be_level = entry_price - 0.50
                    if be_level < current_sl:
                        current_sl = be_level

            # 4. Trailing stop updates
            if direction == 1:
                if b_high > peak:
                    peak = b_high

                if trail_mode == "TRAIL_0.5R":
                    cand = peak - (0.5 * risk_pts)
                    if cand > current_sl: current_sl = cand
                elif trail_mode == "TRAIL_0.75R":
                    cand = peak - (0.75 * risk_pts)
                    if cand > current_sl: current_sl = cand
                elif trail_mode == "TRAIL_1.0R":
                    cand = peak - (1.0 * risk_pts)
                    if cand > current_sl: current_sl = cand
                elif trail_mode == "TRAIL_1.25R":
                    cand = peak - (1.25 * risk_pts)
                    if cand > current_sl: current_sl = cand
                elif trail_mode == "TRAIL_1.5R":
                    cand = peak - (1.5 * risk_pts)
                    if cand > current_sl: current_sl = cand
                elif "BE_AT_1R" in trail_mode:
                    if peak >= entry_price + risk_pts:
                        be_level = entry_price + 0.50
                        mult = 0.5 if "0.5R" in trail_mode else (0.75 if "0.75R" in trail_mode else 1.0)
                        cand = max(be_level, peak - (mult * risk_pts))
                        if cand > current_sl: current_sl = cand
                elif "BE_AT_1.2R" in trail_mode:
                    if peak >= entry_price + (1.2 * risk_pts):
                        be_level = entry_price + 0.50
                        cand = max(be_level, peak - (1.0 * risk_pts))
                        if cand > current_sl: current_sl = cand
                elif is_two_tier and tp1_hit:
                    cand = peak - (1.0 * risk_pts)
                    if cand > current_sl: current_sl = cand

            else: # Short
                if b_low < peak:
                    peak = b_low

                if trail_mode == "TRAIL_0.5R":
                    cand = peak + (0.5 * risk_pts)
                    if cand < current_sl: current_sl = cand
                elif trail_mode == "TRAIL_0.75R":
                    cand = peak + (0.75 * risk_pts)
                    if cand < current_sl: current_sl = cand
                elif trail_mode == "TRAIL_1.0R":
                    cand = peak + (1.0 * risk_pts)
                    if cand < current_sl: current_sl = cand
                elif trail_mode == "TRAIL_1.25R":
                    cand = peak + (1.25 * risk_pts)
                    if cand < current_sl: current_sl = cand
                elif trail_mode == "TRAIL_1.5R":
                    cand = peak + (1.5 * risk_pts)
                    if cand < current_sl: current_sl = cand
                elif "BE_AT_1R" in trail_mode:
                    if peak <= entry_price - risk_pts:
                        be_level = entry_price - 0.50
                        mult = 0.5 if "0.5R" in trail_mode else (0.75 if "0.75R" in trail_mode else 1.0)
                        cand = min(be_level, peak + (mult * risk_pts))
                        if cand < current_sl: current_sl = cand
                elif "BE_AT_1.2R" in trail_mode:
                    if peak <= entry_price - (1.2 * risk_pts):
                        be_level = entry_price - 0.50
                        cand = min(be_level, peak + (1.0 * risk_pts))
                        if cand < current_sl: current_sl = cand
                elif is_two_tier and tp1_hit:
                    cand = peak + (1.0 * risk_pts)
                    if cand < current_sl: current_sl = cand

        # EOD settlement if still holding
        if exit_price is None:
            last_close = sess["closes"][-1]
            exit_price = last_close - slippage_pts if direction == 1 else last_close + slippage_pts
            exit_reason = "FORCE_CLOSE_EOD"

        # Calculate PnL
        roundtrip_comm = 2.0 * comm_per_contract * contracts
        if not is_two_tier or not tp1_hit:
            pts = (exit_price - entry_price) if direction == 1 else (entry_price - exit_price)
            net_dollars = (pts * point_value * contracts) - roundtrip_comm
        else:
            # 1 contract closed at TP1, 1 contract closed at exit_price
            pts1 = (tp1_price - entry_price) if direction == 1 else (entry_price - tp1_price)
            pts2 = (exit_price - entry_price) if direction == 1 else (entry_price - exit_price)
            net_dollars = ((pts1 + pts2) * point_value * (contracts / 2.0)) - roundtrip_comm

        r_mult = net_dollars / (risk_pts * point_value * contracts)

        daily_pnl[s_idx] = net_dollars
        trades.append(FastTrade(
            session_idx=s_idx,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_dollars_net=net_dollars,
            r_multiple=r_mult,
            exit_reason=exit_reason
        ))

    return daily_pnl, trades


def run_full_grid_sweep():
    """Sweeps parameter configurations across all sessions."""
    sessions = SessionPrecompiler.load_sessions()
    n_sessions = len(sessions)

    # Grid specifications
    entry_modes = ["BREAKOUT_WITH_EMA", "CANDLE_AND_EMA"]
    trail_modes = [
        "TRAIL_0.5R",
        "TRAIL_0.75R",
        "TRAIL_1.0R",
        "TRAIL_1.25R",
        "TRAIL_1.5R",
        "BE_AT_1R_THEN_TRAIL_0.5R",
        "BE_AT_1R_THEN_TRAIL_0.75R",
        "BE_AT_1R_THEN_TRAIL_1.0R",
        "BE_AT_1.2R_THEN_TRAIL_1.0R",
        "FIXED_TP_1.5R",
        "FIXED_TP_2.0R",
        "FIXED_TP_2.5R",
        "TWO_TIER_1.25R_RUNNER_1.0R",
        "TWO_TIER_1.5R_RUNNER_1.0R",
    ]
    max_sl_list = [25.0, 35.0, 45.0]

    models_config = []
    for em in entry_modes:
        for tm in trail_modes:
            for sl in max_sl_list:
                models_config.append({
                    "entry_mode": em,
                    "trail_mode": tm,
                    "max_sl_pts": sl
                })

    n_models = len(models_config)
    print(f"\n=======================================================")
    print(f"RUNNING PARAMETER SWEEP: {n_models} CONFIGURATIONS x {n_sessions} SESSIONS")
    print(f"=======================================================\n", flush=True)

    t0 = time.time()
    session_matrix = np.zeros((n_sessions, n_models), dtype=np.float64)
    results = []

    for m_idx, cfg in enumerate(models_config):
        daily_pnl, trades = simulate_single_model(
            sessions=sessions,
            entry_mode=cfg["entry_mode"],
            trail_mode=cfg["trail_mode"],
            max_sl_pts=cfg["max_sl_pts"]
        )
        session_matrix[:, m_idx] = daily_pnl

        # Metrics
        n_trades = len(trades)
        if n_trades == 0:
            continue

        pnls = [t.pnl_dollars_net for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
        tot_win = sum(wins)
        tot_loss = abs(sum(losses))
        pf = (tot_win / tot_loss) if tot_loss > 0 else 999.0
        tot_pnl = sum(pnls)
        expectancy = tot_pnl / n_trades if n_trades > 0 else 0.0

        # Equity and Max Drawdown
        equity = np.cumsum(daily_pnl)
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        max_dd = np.max(dd)

        # Annualized Sharpe (assuming 252 sessions/yr)
        mean_d = np.mean(daily_pnl)
        std_d = np.std(daily_pnl)
        sharpe = (mean_d / std_d * np.sqrt(252)) if std_d > 0 else 0.0

        results.append({
            "model_id": m_idx,
            "entry_mode": cfg["entry_mode"],
            "trail_mode": cfg["trail_mode"],
            "max_sl_pts": cfg["max_sl_pts"],
            "n_trades": n_trades,
            "win_rate": win_rate,
            "profit_factor": pf,
            "total_net_pnl": tot_pnl,
            "expectancy_usd": expectancy,
            "max_dd_usd": max_dd,
            "sharpe_ratio": sharpe
        })

    elapsed = time.time() - t0
    print(f"Sweep complete in {elapsed:.2f}s ({n_models * n_sessions / elapsed:,.0f} session-evals/sec).", flush=True)

    df_res = pd.DataFrame(results).sort_values("sharpe_ratio", ascending=False)
    leaderboard_csv = PROJECT_ROOT / "strategies" / "orb_5m" / "backtest" / "parameter_sweep_leaderboard.csv"
    df_res.to_csv(leaderboard_csv, index=False)
    print(f"Saved leaderboard with {len(df_res)} models to {leaderboard_csv}")

    # Save session matrix for PBO
    matrix_path = PROJECT_ROOT / "strategies" / "orb_5m" / "backtest" / "session_pnl_matrix.npy"
    np.save(matrix_path, session_matrix)
    print(f"Saved session PnL matrix {session_matrix.shape} to {matrix_path}")

    # Print Top 10
    print("\n--- TOP 10 PARAMETER CONFIGURATIONS BY SHARPE RATIO ---")
    print(df_res[[
        "model_id", "entry_mode", "trail_mode", "max_sl_pts", "win_rate", "profit_factor", "sharpe_ratio", "max_dd_usd", "total_net_pnl"
    ]].head(10).to_string(index=False))

    return df_res, session_matrix, sessions


if __name__ == "__main__":
    run_full_grid_sweep()
