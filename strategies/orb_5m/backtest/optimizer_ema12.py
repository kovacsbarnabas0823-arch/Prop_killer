"""
Institutional Quantitative Optimization Engine: 12 EMA Directional Bias at 09:35 EST
Target: CME Micro E-mini Nasdaq-100 (MNQ) Futures
Objective: Optimize Risk Management, SL & TP for Tradeify $50K Evaluation Economics
Preventing Overfitting: Strict Train (2018-2022) / Test (2023-2026) Split + Monte Carlo Bootstrapping
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.orb_5m.backtest.sweep_engine_5m import SessionPrecompiler

POINT_VALUE_MNQ = 2.0
COMMISSION_PER_CONTRACT = 1.24  # $2.48 roundturn
SLIPPAGE_POINTS = 0.50          # $1.00 per contract per exit/entry


@dataclass
class TradeResult:
    session_idx: int
    date: Any
    direction: int
    entry_price: float
    exit_price: float
    pnl_dollars_net: float
    r_multiple: float
    exit_reason: str
    duration_bars: int


def run_single_simulation(
    session: dict,
    entry_type: str,           # 'MARKET_0935', 'CANDLE_CONFIRM', 'BREAKOUT_5M'
    sl_type: str,              # 'STRUCT_5M', 'STRUCT_MID', 'FIXED_PTS', 'ATR_MULT'
    sl_param: float,           # pts for FIXED_PTS, or multiplier, or max_cap
    tp_type: str,              # 'FIXED_R', 'FIXED_PTS', 'TIME_EXIT', 'TWO_TIER_R', 'BE_THEN_R'
    tp_param: float,           # R-multiple or points or minutes
    tp_param2: float = 0.0,    # Runner R or secondary target for two-tier
    contracts: int = 2,
    point_value: float = POINT_VALUE_MNQ,
    comm_per_contract: float = COMMISSION_PER_CONTRACT,
    slippage_pts: float = SLIPPAGE_POINTS
) -> TradeResult | None:
    """
    Simulates a single session under strict pessimistic intra-bar execution.
    Returns TradeResult or None if no trade triggered.
    """
    c_open = session["c_open"]
    c_high = session["c_high"]
    c_low = session["c_low"]
    c_close = session["c_close"]
    c_ema = session["c_ema"]
    opens = session["opens"]
    highs = session["highs"]
    lows = session["lows"]
    closes = session["closes"]
    n_bars = len(opens)
    if n_bars == 0:
        return None

    # 1. 12 EMA Directional Bias at 09:35 EST
    if c_close > c_ema:
        direction = 1   # Long
    elif c_close < c_ema:
        direction = -1  # Short
    else:
        return None

    # Entry Filtering / Execution
    entry_price = 0.0
    start_idx = 0
    entered = False

    if entry_type == "MARKET_0935":
        # Immediate entry at 09:35:00 open
        if direction == 1:
            entry_price = opens[0] + slippage_pts
        else:
            entry_price = opens[0] - slippage_pts
        start_idx = 0
        entered = True

    elif entry_type == "CANDLE_CONFIRM":
        # Only enter if 09:30-09:35 5m candle color matches direction
        if direction == 1 and c_close <= c_open:
            return None
        if direction == -1 and c_close >= c_open:
            return None
        if direction == 1:
            entry_price = opens[0] + slippage_pts
        else:
            entry_price = opens[0] - slippage_pts
        start_idx = 0
        entered = True

    elif entry_type == "BREAKOUT_5M":
        # Wait for breakout of 5m high / low
        max_scan = min(60, n_bars)
        for i in range(max_scan):
            if direction == 1 and highs[i] > c_high:
                entry_price = max(c_high, opens[i]) + slippage_pts
                start_idx = i
                entered = True
                break
            elif direction == -1 and lows[i] < c_low:
                entry_price = min(c_low, opens[i]) - slippage_pts
                start_idx = i
                entered = True
                break
        if not entered:
            return None

    if not entered:
        return None

    # 2. Stop Loss Calculation
    c_range = max(c_high - c_low, 6.0)
    if sl_type == "STRUCT_5M":
        raw_risk = (entry_price - c_low) if direction == 1 else (c_high - entry_price)
        risk_pts = min(max(raw_risk, 8.0), sl_param)  # sl_param acts as max cap
    elif sl_type == "STRUCT_MID":
        mid = (c_high + c_low) / 2.0
        raw_risk = (entry_price - mid) if direction == 1 else (mid - entry_price)
        risk_pts = min(max(raw_risk, 6.0), sl_param)
    elif sl_type == "FIXED_PTS":
        risk_pts = sl_param
    elif sl_type == "ATR_MULT":
        risk_pts = min(max(c_range * sl_param, 8.0), 60.0)
    else:
        risk_pts = 25.0

    initial_sl = (entry_price - risk_pts) if direction == 1 else (entry_price + risk_pts)
    current_sl = initial_sl

    # 3. Take Profit Target Calculation
    target_price = None
    tp1_price = None
    is_two_tier = False
    is_be_mode = False
    be_trigger_price = None

    if tp_type == "FIXED_R":
        target_price = (entry_price + tp_param * risk_pts) if direction == 1 else (entry_price - tp_param * risk_pts)
    elif tp_type == "FIXED_PTS":
        target_price = (entry_price + tp_param) if direction == 1 else (entry_price - tp_param)
    elif tp_type == "BE_THEN_R":
        is_be_mode = True
        be_trigger_price = (entry_price + 1.0 * risk_pts) if direction == 1 else (entry_price - 1.0 * risk_pts)
        target_price = (entry_price + tp_param * risk_pts) if direction == 1 else (entry_price - tp_param * risk_pts)
    elif tp_type == "TWO_TIER_R":
        is_two_tier = True
        tp1_price = (entry_price + tp_param * risk_pts) if direction == 1 else (entry_price - tp_param * risk_pts)
        target_price = (entry_price + tp_param2 * risk_pts) if direction == 1 else (entry_price - tp_param2 * risk_pts)

    # 4. Intra-day Bar Traversal (Strict Pessimistic Execution)
    exit_price = None
    exit_reason = None
    tp1_hit = False
    be_active = False
    duration = 0

    roundtrip_comm = 2.0 * comm_per_contract * contracts

    for i in range(start_idx, n_bars):
        duration += 1
        b_open = opens[i]
        b_high = highs[i]
        b_low = lows[i]

        # Time Exit check (if specified)
        if tp_type == "TIME_EXIT" and duration >= int(tp_param):
            exit_price = closes[i] - slippage_pts if direction == 1 else closes[i] + slippage_pts
            exit_reason = "TIME_EXIT"
            break

        # Check Stop Loss hit
        sl_hit = (b_low <= current_sl) if direction == 1 else (b_high >= current_sl)

        # Check Target / TP1 hit
        target_hit = False
        if target_price is not None:
            target_hit = (b_high >= target_price) if direction == 1 else (b_low <= target_price)

        tp1_touched = False
        if is_two_tier and not tp1_hit:
            tp1_touched = (b_high >= tp1_price) if direction == 1 else (b_low <= tp1_price)

        # PESSIMISTIC RULE: If both SL and Target are breached on the same bar, SL is hit first!
        if sl_hit and (target_hit or tp1_touched):
            fill = min(current_sl, b_open) if direction == 1 else max(current_sl, b_open)
            exit_price = (fill - slippage_pts) if direction == 1 else (fill + slippage_pts)
            exit_reason = "PESSIMISTIC_STOP"
            break

        if sl_hit:
            fill = min(current_sl, b_open) if direction == 1 else max(current_sl, b_open)
            exit_price = (fill - slippage_pts) if direction == 1 else (fill + slippage_pts)
            exit_reason = "BE_STOP" if be_active else "STOP_LOSS"
            break

        # Check Target hit
        if target_hit:
            fill = max(target_price, b_open) if direction == 1 else min(target_price, b_open)
            exit_price = (fill - slippage_pts) if direction == 1 else (fill + slippage_pts)
            exit_reason = "TARGET_PROFIT"
            break

        # Check Two-Tier TP1
        if is_two_tier and not tp1_hit and tp1_touched:
            tp1_hit = True
            # Move stop to Breakeven (+2 ticks = 0.50 pts)
            be_sl = (entry_price + 0.50) if direction == 1 else (entry_price - 0.50)
            if direction == 1 and be_sl > current_sl:
                current_sl = be_sl
                be_active = True
            elif direction == -1 and be_sl < current_sl:
                current_sl = be_sl
                be_active = True

        # Check BE Trigger for BE_THEN_R
        if is_be_mode and not be_active:
            be_touched = (b_high >= be_trigger_price) if direction == 1 else (b_low <= be_trigger_price)
            if be_touched:
                be_sl = (entry_price + 0.50) if direction == 1 else (entry_price - 0.50)
                if (direction == 1 and be_sl > current_sl) or (direction == -1 and be_sl < current_sl):
                    current_sl = be_sl
                    be_active = True

    # EOD Settlement if still open
    if exit_price is None:
        last_close = closes[-1]
        exit_price = (last_close - slippage_pts) if direction == 1 else (last_close + slippage_pts)
        exit_reason = "FORCE_CLOSE_EOD"

    # Calculate Net PnL
    if not is_two_tier or not tp1_hit:
        pts = (exit_price - entry_price) if direction == 1 else (entry_price - exit_price)
        net_dollars = (pts * point_value * contracts) - roundtrip_comm
    else:
        # Half position closed at TP1, half closed at exit_price
        pts1 = (tp1_price - entry_price) if direction == 1 else (entry_price - tp1_price)
        pts2 = (exit_price - entry_price) if direction == 1 else (entry_price - exit_price)
        net_dollars = ((pts1 + pts2) * point_value * (contracts / 2.0)) - roundtrip_comm

    r_mult = net_dollars / (risk_pts * point_value * contracts)

    return TradeResult(
        session_idx=session.get("idx", 0),
        date=session["date"],
        direction=direction,
        entry_price=entry_price,
        exit_price=exit_price,
        pnl_dollars_net=net_dollars,
        r_multiple=r_mult,
        exit_reason=exit_reason,
        duration_bars=duration
    )


def simulate_model_on_sessions(
    sessions: list,
    cfg: dict,
    contracts: int = 2
) -> Tuple[np.ndarray, List[TradeResult]]:
    """Simulates a model configuration across a list of sessions."""
    n = len(sessions)
    daily_pnls = np.zeros(n, dtype=np.float64)
    trades = []
    for i, s in enumerate(sessions):
        tr = run_single_simulation(
            session=s,
            entry_type=cfg["entry_type"],
            sl_type=cfg["sl_type"],
            sl_param=cfg["sl_param"],
            tp_type=cfg["tp_type"],
            tp_param=cfg["tp_param"],
            tp_param2=cfg.get("tp_param2", 0.0),
            contracts=contracts
        )
        if tr is not None:
            daily_pnls[i] = tr.pnl_dollars_net
            trades.append(tr)
    return daily_pnls, trades


def simulate_tradeify_challenge_monte_carlo(
    daily_pnls: np.ndarray,
    n_sims: int = 1500,
    profit_target: float = 3000.0,
    trailing_dd_limit: float = 2000.0,
    daily_loss_limit: float = 1250.0,
    block_size: int = 5,
    max_days: int = 150,
    initial_fee: float = 87.0,
    reset_fee: float = 95.0
) -> Tuple[float, float, float]:
    """
    Simulates Tradeify Growth $50K Challenge using Stationary Block Bootstrap.
    Returns:
      pass_rate (float 0.0 - 1.0)
      avg_days_to_pass (float)
      expected_challenge_cost (float in USD)
    """
    n_samples = len(daily_pnls)
    if n_samples == 0 or np.sum(daily_pnls) <= 0:
        return 0.0, 999.0, 9999.0

    passes = 0
    days_to_pass_list = []

    for _ in range(n_sims):
        path_pnls = []
        while len(path_pnls) < max_days:
            start = np.random.randint(0, n_samples)
            length = np.random.geometric(1.0 / block_size)
            chunk = [daily_pnls[(start + j) % n_samples] for j in range(length)]
            path_pnls.extend(chunk)
        path_pnls = np.array(path_pnls[:max_days])

        balance = 50000.0
        peak = 50000.0
        floor = 50000.0 - trailing_dd_limit  # $48,000
        passed = False
        failed = False

        for day_idx, pnl in enumerate(path_pnls):
            if pnl <= -daily_loss_limit:
                failed = True
                break

            balance += pnl

            if balance <= floor:
                failed = True
                break

            if balance > peak:
                peak = balance
                new_floor = min(peak - trailing_dd_limit, 50100.0)
                if new_floor > floor:
                    floor = new_floor

            if balance >= 50000.0 + profit_target:
                passed = True
                days_to_pass_list.append(day_idx + 1)
                break

        if passed:
            passes += 1

    pass_rate = passes / n_sims
    avg_days = float(np.mean(days_to_pass_list)) if len(days_to_pass_list) > 0 else float(max_days)

    if pass_rate > 0.01:
        expected_cost = initial_fee + ((1.0 / pass_rate) - 1.0) * reset_fee
    else:
        expected_cost = 9999.0

    return pass_rate, avg_days, expected_cost


def run_comprehensive_optimization():
    print("=========================================================================")
    print("INSTITUTIONAL QUANTITATIVE OPTIMIZATION: 12 EMA AT 09:35 EST")
    print("=========================================================================")
    t0 = time.time()

    sessions = SessionPrecompiler.load_sessions()
    for idx, s in enumerate(sessions):
        s["idx"] = idx

    split_date = pd.to_datetime("2023-01-01").date()
    train_sessions = [s for s in sessions if pd.to_datetime(s["date"]).date() < split_date]
    test_sessions = [s for s in sessions if pd.to_datetime(s["date"]).date() >= split_date]

    print(f"Total sessions: {len(sessions):,} | Train (2018-2022): {len(train_sessions):,} | Test (2023-2026): {len(test_sessions):,}")

    entry_types = ["MARKET_0935", "CANDLE_CONFIRM", "BREAKOUT_5M"]

    sl_configs = [
        ("STRUCT_5M", 35.0),
        ("STRUCT_5M", 50.0),
        ("STRUCT_5M", 70.0),
        ("STRUCT_MID", 30.0),
        ("STRUCT_MID", 45.0),
        ("FIXED_PTS", 25.0),
        ("FIXED_PTS", 35.0),
        ("FIXED_PTS", 50.0),
        ("FIXED_PTS", 75.0),
        ("ATR_MULT", 1.0),
        ("ATR_MULT", 1.5),
    ]

    tp_configs = [
        ("FIXED_R", 1.0, 0.0),
        ("FIXED_R", 1.25, 0.0),
        ("FIXED_R", 1.5, 0.0),
        ("FIXED_R", 2.0, 0.0),
        ("FIXED_R", 2.5, 0.0),
        ("FIXED_PTS", 35.0, 0.0),
        ("FIXED_PTS", 50.0, 0.0),
        ("FIXED_PTS", 75.0, 0.0),
        ("FIXED_PTS", 100.0, 0.0),
        ("TIME_EXIT", 30.0, 0.0),
        ("TIME_EXIT", 60.0, 0.0),
        ("TIME_EXIT", 120.0, 0.0),
        ("TWO_TIER_R", 1.0, 2.0),
        ("TWO_TIER_R", 1.25, 2.5),
        ("BE_THEN_R", 1.5, 0.0),
        ("BE_THEN_R", 2.0, 0.0),
    ]

    grid = []
    for et in entry_types:
        for sl_t, sl_p in sl_configs:
            for tp_t, tp_p, tp_p2 in tp_configs:
                grid.append({
                    "entry_type": et,
                    "sl_type": sl_t,
                    "sl_param": sl_p,
                    "tp_type": tp_t,
                    "tp_param": tp_p,
                    "tp_param2": tp_p2
                })

    print(f"\nEvaluating {len(grid):,} parameter configurations on In-Sample (Train) dataset...", flush=True)

    train_results = []
    for m_idx, cfg in enumerate(grid):
        daily_pnls, trades = simulate_model_on_sessions(train_sessions, cfg, contracts=2)
        n_tr = len(trades)
        if n_tr < 50:
            continue
        tot_pnl = np.sum(daily_pnls)
        wins = [t.pnl_dollars_net for t in trades if t.pnl_dollars_net > 0]
        losses = [abs(t.pnl_dollars_net) for t in trades if t.pnl_dollars_net <= 0]
        wr = len(wins) / n_tr if n_tr > 0 else 0.0
        pf = (sum(wins) / sum(losses)) if sum(losses) > 0 else 0.0
        expectancy = tot_pnl / n_tr

        cum = np.cumsum(daily_pnls)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        max_dd = np.max(dd) if len(dd) > 0 else 0.0

        sharpe = (np.mean(daily_pnls) / np.std(daily_pnls) * np.sqrt(252)) if np.std(daily_pnls) > 0 else 0.0

        train_results.append({
            "cfg": cfg,
            "n_trades": n_tr,
            "train_pnl": tot_pnl,
            "train_wr": wr,
            "train_pf": pf,
            "train_exp": expectancy,
            "train_max_dd": max_dd,
            "train_sharpe": sharpe
        })

    df_train = pd.DataFrame(train_results)
    df_train = df_train.sort_values(by="train_sharpe", ascending=False)
    print(f"Top 10 In-Sample Models by Sharpe Ratio:\n")
    for i, r in df_train.head(10).iterrows():
        c = r["cfg"]
        print(f"[{c['entry_type']}] SL: {c['sl_type']}({c['sl_param']}) | TP: {c['tp_type']}({c['tp_param']}) -> PnL: ${r['train_pnl']:,.0f} | WR: {r['train_wr']:.1%} | PF: {r['train_pf']:.2f} | Sharpe: {r['train_sharpe']:.2f} | MaxDD: ${r['train_max_dd']:,.0f}")

    print("\n-------------------------------------------------------------------------")
    print("VALIDATING TOP CANDIDATES ON OUT-OF-SAMPLE (TEST) 2023-2026...")
    print("-------------------------------------------------------------------------\n")

    top_candidates = df_train[df_train["train_pnl"] > 3000].head(30)
    test_evals = []

    for _, row in top_candidates.iterrows():
        cfg = row["cfg"]
        test_pnls, test_trades = simulate_model_on_sessions(test_sessions, cfg, contracts=2)
        n_test = len(test_trades)
        test_pnl = np.sum(test_pnls)
        wins_test = [t.pnl_dollars_net for t in test_trades if t.pnl_dollars_net > 0]
        losses_test = [abs(t.pnl_dollars_net) for t in test_trades if t.pnl_dollars_net <= 0]
        wr_test = len(wins_test) / n_test if n_test > 0 else 0.0
        pf_test = (sum(wins_test) / sum(losses_test)) if sum(losses_test) > 0 else 0.0
        exp_test = test_pnl / n_test if n_test > 0 else 0.0

        cum_test = np.cumsum(test_pnls)
        peak_test = np.maximum.accumulate(cum_test)
        dd_test = peak_test - cum_test
        max_dd_test = np.max(dd_test) if len(dd_test) > 0 else 0.0
        sharpe_test = (np.mean(test_pnls) / np.std(test_pnls) * np.sqrt(252)) if np.std(test_pnls) > 0 else 0.0

        full_pnls, full_trades = simulate_model_on_sessions(sessions, cfg, contracts=2)
        full_pnl = np.sum(full_pnls)
        full_sharpe = (np.mean(full_pnls) / np.std(full_pnls) * np.sqrt(252)) if np.std(full_pnls) > 0 else 0.0

        sizing_results = {}
        for k in [2, 3, 4]:
            k_pnls = full_pnls * (k / 2.0)
            pass_rate, avg_days, exp_cost = simulate_tradeify_challenge_monte_carlo(
                k_pnls, n_sims=1000, profit_target=3000.0, trailing_dd_limit=2000.0, daily_loss_limit=1250.0
            )
            sizing_results[k] = {
                "pass_rate": pass_rate,
                "avg_days": avg_days,
                "exp_cost": exp_cost
            }

        test_evals.append({
            "cfg": cfg,
            "train_pnl": row["train_pnl"],
            "train_sharpe": row["train_sharpe"],
            "test_pnl": test_pnl,
            "test_wr": wr_test,
            "test_pf": pf_test,
            "test_exp": exp_test,
            "test_sharpe": sharpe_test,
            "full_pnl": full_pnl,
            "full_sharpe": full_sharpe,
            "sizing": sizing_results
        })

    df_test = pd.DataFrame(test_evals)
    df_test = df_test.sort_values(by="test_sharpe", ascending=False)

    print("TOP OOS MODELS (Generalizing to 2023-2026):\n")
    for i, r in df_test.head(10).iterrows():
        c = r["cfg"]
        s3 = r["sizing"][3]
        print(f"[{c['entry_type']}] SL: {c['sl_type']}({c['sl_param']}) | TP: {c['tp_type']}({c['tp_param']})")
        print(f"   Train PnL: ${r['train_pnl']:,.0f} (Sh: {r['train_sharpe']:.2f}) | Test PnL: ${r['test_pnl']:,.0f} (Sh: {r['test_sharpe']:.2f}, WR: {r['test_wr']:.1%}, PF: {r['test_pf']:.2f})")
        print(f"   Tradeify 3 MNQ -> Pass Rate: {s3['pass_rate']:.1%} | Avg Days: {s3['avg_days']:.1f} d | Exp Cost: ${s3['exp_cost']:.2f}\n")

    print(f"Optimization completed in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    run_comprehensive_optimization()

