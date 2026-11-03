"""
Microstructure Anomaly Strategy: NY Open London Liquidity Sweep & Reversal (LLSR)
Target Asset: CME Micro E-mini Nasdaq-100 (MNQ) Futures ($2 / pt)
Data: 2018-2026 (8.9 Years CME continuous 1-minute Databento parquet)

Capitalizes on the empirical finding that:
  - NY Open sweeps of London Low fail to break down 89.0% of the time.
  - NY Open sweeps of London High fail to break out 72.1% of the time.

Rules:
  - London Session: 03:00 to 09:29 EST (calculate LH, LL, LMid).
  - NY Open Window: 09:30 to 10:30 EST.
  - Bullish Sweep: Price sweeps LL - 2.0 pts, then 1m bar closes back ABOVE LL.
  - Bearish Sweep: Price sweeps LH + 2.0 pts, then 1m bar closes back BELOW LH.
  - Target 1 (50%): London Midpoint (move stop to BE).
  - Target 2 (50%): Opposite London Extreme (or 15:45 force close).
  - Strictly Zero Overnight Holding.
"""

from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data_loader import load_continuous_nq_parquet

POINT_VALUE_MNQ = 2.0
COMMISSION_PER_CONTRACT = 1.24  # $2.48 roundturn per contract
SLIPPAGE_POINTS = 0.50          # 0.50 pt ($1.00 MNQ) pessimistic entry/exit


def run_session_sweep_backtest():
    print("Loading continuous CME NQ parquet dataset...")
    df = pd.read_parquet("data/nq_1m_continuous.parquet")
    df = df.sort_index()

    df["date"] = df.index.date
    df["hour_min"] = df.index.strftime("%H:%M")
    unique_dates = sorted(list(set(df[df["hour_min"] == "09:30"]["date"].unique())))

    print(f"Total trading days: {len(unique_dates):,}\n")
    contracts = 2
    roundtrip_comm = 2.0 * COMMISSION_PER_CONTRACT * contracts

    daily_pnls = []
    trades = []

    for cur_date in unique_dates:
        t_lon_start = pd.Timestamp(f"{cur_date} 03:00:00", tz="US/Eastern")
        t_lon_end = pd.Timestamp(f"{cur_date} 09:30:00", tz="US/Eastern")
        lon_bars = df.loc[(df.index >= t_lon_start) & (df.index < t_lon_end)]

        t_ny_start = pd.Timestamp(f"{cur_date} 09:30:00", tz="US/Eastern")
        t_ny_close = pd.Timestamp(f"{cur_date} 15:45:00", tz="US/Eastern")
        rth_bars = df.loc[(df.index >= t_ny_start) & (df.index <= t_ny_close)]

        if len(lon_bars) < 60 or len(rth_bars) < 120:
            continue

        lh = float(lon_bars["High"].max())
        ll = float(lon_bars["Low"].min())
        lmid = round((lh + ll) / 2.0, 2)
        lrange = lh - ll

        if lrange < 25.0:  # London range too compressed to trade reversals
            continue

        opens = rth_bars["Open"].to_numpy(dtype=np.float64)
        highs = rth_bars["High"].to_numpy(dtype=np.float64)
        lows = rth_bars["Low"].to_numpy(dtype=np.float64)
        closes = rth_bars["Close"].to_numpy(dtype=np.float64)
        times = rth_bars.index
        n_bars = len(opens)

        entered = False
        direction = 0
        entry_price = 0.0
        entry_idx = -1
        sweep_extreme = 0.0
        trade_side = ""

        # Scan 09:30 to 10:15 (first 45 bars)
        swept_low = False
        swept_low_min = ll
        swept_high = False
        swept_high_max = lh

        for i in range(min(45, n_bars)):
            b_o = opens[i]
            b_h = highs[i]
            b_l = lows[i]
            b_c = closes[i]

            # Track sweep
            if b_l < ll - 2.0:
                swept_low = True
                if b_l < swept_low_min:
                    swept_low_min = b_l

            if b_h > lh + 2.0:
                swept_high = True
                if b_h > swept_high_max:
                    swept_high_max = b_h

            # Trigger long: previously swept low, now closes back above LL
            if swept_low and b_c > ll and b_c < lmid:
                entry_price = b_c + SLIPPAGE_POINTS
                direction = 1
                entry_idx = i
                sweep_extreme = swept_low_min
                trade_side = "LONG_LL_FADE"
                entered = True
                break

            # Trigger short: previously swept high, now closes back below LH
            if swept_high and b_c < lh and b_c > lmid:
                entry_price = b_c - SLIPPAGE_POINTS
                direction = -1
                entry_idx = i
                sweep_extreme = swept_high_max
                trade_side = "SHORT_LH_FADE"
                entered = True
                break

        if not entered:
            continue

        # Stop loss calculation
        if direction == 1:
            raw_risk = max(entry_price - (sweep_extreme - 3.0), 12.0)
            risk_pts = min(raw_risk, 35.0)
            initial_sl = entry_price - risk_pts
            tp1 = lmid
            tp2 = lh
        else:
            raw_risk = max((sweep_extreme + 3.0) - entry_price, 12.0)
            risk_pts = min(raw_risk, 35.0)
            initial_sl = entry_price + risk_pts
            tp1 = lmid
            tp2 = ll

        current_sl = initial_sl
        tp1_hit = False
        exit_leg1 = None
        exit_leg2 = None
        exit_reason = None

        for b_i in range(entry_idx + 1, n_bars):
            b_o = opens[b_i]
            b_h = highs[b_i]
            b_l = lows[b_i]

            sl_hit = (b_l <= current_sl) if direction == 1 else (b_h >= current_sl)

            if not tp1_hit:
                hit_tp1 = (b_h >= tp1) if direction == 1 else (b_l <= tp1)
                if sl_hit and hit_tp1:
                    fill = min(current_sl, b_o) if direction == 1 else max(current_sl, b_o)
                    exit_leg1 = fill - SLIPPAGE_POINTS if direction == 1 else fill + SLIPPAGE_POINTS
                    exit_leg2 = exit_leg1
                    exit_reason = "STOP_LOSS"
                    break
                elif sl_hit:
                    fill = min(current_sl, b_o) if direction == 1 else max(current_sl, b_o)
                    exit_leg1 = fill - SLIPPAGE_POINTS if direction == 1 else fill + SLIPPAGE_POINTS
                    exit_leg2 = exit_leg1
                    exit_reason = "STOP_LOSS"
                    break
                elif hit_tp1:
                    tp1_hit = True
                    exit_leg1 = tp1
                    current_sl = entry_price + 0.50 if direction == 1 else entry_price - 0.50
            else:
                hit_tp2 = (b_h >= tp2) if direction == 1 else (b_l <= tp2)
                if sl_hit and hit_tp2:
                    fill = min(current_sl, b_o) if direction == 1 else max(current_sl, b_o)
                    exit_leg2 = fill - SLIPPAGE_POINTS if direction == 1 else fill + SLIPPAGE_POINTS
                    exit_reason = "LEG2_BE_STOP"
                    break
                elif sl_hit:
                    fill = min(current_sl, b_o) if direction == 1 else max(current_sl, b_o)
                    exit_leg2 = fill - SLIPPAGE_POINTS if direction == 1 else fill + SLIPPAGE_POINTS
                    exit_reason = "LEG2_BE_STOP"
                    break
                elif hit_tp2:
                    exit_leg2 = tp2
                    exit_reason = "FULL_TP2_OPPOSITE"
                    break

        if exit_leg1 is None:
            last_c = closes[-1]
            exit_leg1 = last_c - SLIPPAGE_POINTS if direction == 1 else last_c + SLIPPAGE_POINTS
            exit_leg2 = exit_leg1
            exit_reason = "FORCE_CLOSE_1545"
        elif exit_leg2 is None:
            last_c = closes[-1]
            exit_leg2 = last_c - SLIPPAGE_POINTS if direction == 1 else last_c + SLIPPAGE_POINTS
            exit_reason = "FORCE_CLOSE_1545_RUNNER"

        pts1 = (exit_leg1 - entry_price) if direction == 1 else (entry_price - exit_leg1)
        pts2 = (exit_leg2 - entry_price) if direction == 1 else (entry_price - exit_leg2)
        gross = (pts1 * 1.0 * POINT_VALUE_MNQ) + (pts2 * 1.0 * POINT_VALUE_MNQ)
        net_d = gross - roundtrip_comm

        daily_pnls.append(net_d)
        trades.append({
            "date": cur_date,
            "side": trade_side,
            "direction": direction,
            "entry": entry_price,
            "pnl": net_d,
            "exit_reason": exit_reason
        })

    print("=" * 75)
    print("NEW YORK OPEN LONDON LIQUIDITY SWEEP & REVERSAL (LLSR) RESULTS:")
    print("=" * 75)
    pnls = np.array([t["pnl"] for t in trades])
    n_t = len(trades)
    wins = pnls[pnls > 0]
    losses = np.abs(pnls[pnls <= 0])
    wr = len(wins) / n_t if n_t > 0 else 0.0
    pf = np.sum(wins) / np.sum(losses) if np.sum(losses) > 0 else 0.0
    tot_pnl = np.sum(pnls)
    exp = tot_pnl / n_t if n_t > 0 else 0.0

    cum = np.cumsum(daily_pnls)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = np.max(dd) if len(dd) > 0 else 0.0

    print(f"Total Net PnL (8.9y)       : ${tot_pnl:,.2f} (2 MNQ, net of comms & slippage)")
    print(f"Total Trades Executed      : {n_t:,} ({n_t / 8.9:.1f} trades / year)")
    print(f"Win Rate                   : {wr:.2%} ({len(wins)} Wins / {len(losses)} Losses)")
    print(f"Profit Factor              : {pf:.2f}")
    print(f"Average Expectancy         : ${exp:.2f} / trade")
    print(f"Maximum Realized Drawdown  : -${max_dd:,.2f}")

    # Trade side breakdown
    df_tr = pd.DataFrame(trades)
    for s_name, grp in df_tr.groupby("side"):
        s_wins = grp[grp["pnl"] > 0]
        s_losses = grp[grp["pnl"] <= 0]
        s_wr = len(s_wins) / len(grp)
        s_pnl = grp["pnl"].sum()
        s_pf = s_wins["pnl"].sum() / abs(s_losses["pnl"].sum()) if len(s_losses) > 0 else 0.0
        print(f"  • {s_name:16s} (N={len(grp):3d}): PnL: ${s_pnl:,.2f} | WR: {s_wr:.1%} | PF: {s_pf:.2f}")

    # Yearly breakdown
    df_tr["year"] = pd.to_datetime(df_tr["date"]).dt.year
    yearly = df_tr.groupby("year")["pnl"].agg(["count", "sum"]).rename(columns={"count": "trades", "sum": "net_pnl"})
    yearly["win_rate"] = df_tr.groupby("year")["pnl"].apply(lambda s: (s > 0).mean() * 100.0)
    print("\nYearly Performance Breakdown:")
    print(yearly.to_string())

    # Tradeify Challenge Monte Carlo
    eval_passes = 0
    n_sims = 3000
    eval_days = []
    for _ in range(n_sims):
        balance = 50000.0
        peak = 50000.0
        floor = 48000.0
        passed = False
        days = 0
        sampled = np.random.choice(len(pnls), size=150, replace=True)
        for idx in sampled:
            days += 4.5  # trades roughly every 4.5 days
            pnl = pnls[idx]
            if pnl <= -1250.0: break
            balance += pnl
            if balance <= floor: break
            if balance > peak:
                peak = balance
                new_floor = min(peak - 2000.0, 50100.0)
                if new_floor > floor: floor = new_floor
            if balance >= 53000.0:
                passed = True
                eval_days.append(days)
                break
        if passed: eval_passes += 1

    pass_r = eval_passes / n_sims
    avg_d = np.mean(eval_days) if eval_days else 999.0
    print(f"\nTradeify $50K Challenge Pass Rate: {pass_r:.1%} | Avg Days: {avg_d:.1f} | Expected Cost: ${87.0/pass_r:.2f}")


if __name__ == "__main__":
    run_session_sweep_backtest()
