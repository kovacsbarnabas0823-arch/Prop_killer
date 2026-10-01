"""
Empirical Backtester & Tearsheet Generator for 5-Minute ORB Strategy
Target: CME Micro E-mini Nasdaq-100 (MNQ) Futures
Executes Production Configuration across 8.7 Years (2018–2026, 2,139 Sessions)
"""

from __future__ import annotations
import sys
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.orb_5m.backtest.sweep_engine_5m import simulate_single_model, SessionPrecompiler

POINT_VALUE_MNQ = 2.0
COMMISSION_PER_CONTRACT = 1.24
SLIPPAGE_POINTS = 0.50


def run_production_backtest():
    print("=========================================================================")
    print("RUNNING PRODUCTION EMPIRICAL BACKTEST: 5-MINUTE ORB + 12 EMA")
    print("=========================================================================")

    sessions = SessionPrecompiler.load_sessions()

    # Optimal Model: Breakout with 12 EMA, TRAIL_0.5R, max_sl 35.0 pts, 4 MNQ contracts
    # (High velocity setup balancing drawdown and rapid challenge passing)
    contracts = 4
    daily_pnl, trades = simulate_single_model(
        sessions=sessions,
        entry_mode="BREAKOUT_WITH_EMA",
        trail_mode="TRAIL_0.5R",
        max_sl_pts=35.0,
        contracts=contracts
    )

    n_trades = len(trades)
    pnls = [t.pnl_dollars_net for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
    tot_win = sum(wins)
    tot_loss = abs(sum(losses))
    pf = (tot_win / tot_loss) if tot_loss > 0 else 999.0
    tot_pnl = sum(pnls)
    expectancy = tot_pnl / n_trades if n_trades > 0 else 0.0

    # Equity curve and Drawdown
    equity = np.cumsum(daily_pnl)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = np.max(dd)

    mean_d = np.mean(daily_pnl)
    std_d = np.std(daily_pnl)
    sharpe = (mean_d / std_d * np.sqrt(252)) if std_d > 0 else 0.0

    # Construct DataFrame of trades
    trade_dicts = []
    for t in trades:
        sess = sessions[t.session_idx]
        trade_dicts.append({
            "session_idx": t.session_idx,
            "date": sess["date"],
            "direction": "LONG" if t.direction == 1 else "SHORT",
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "pnl_dollars_net": t.pnl_dollars_net,
            "r_multiple": t.r_multiple,
            "exit_reason": t.exit_reason
        })

    df_trades = pd.DataFrame(trade_dicts)
    trade_log_csv = PROJECT_ROOT / "strategies" / "orb_5m" / "backtest" / "historical_trade_log_5m.csv"
    df_trades.to_csv(trade_log_csv, index=False)
    print(f"Saved complete trade log ({len(df_trades)} trades) to {trade_log_csv}")

    # Yearly breakdown
    df_trades["year"] = pd.to_datetime(df_trades["date"]).dt.year
    yearly = df_trades.groupby("year")["pnl_dollars_net"].agg(["count", "sum"]).rename(columns={"count": "trades", "sum": "net_pnl"})
    yearly["win_rate"] = df_trades.groupby("year")["pnl_dollars_net"].apply(lambda s: (s > 0).mean() * 100.0)

    tearsheet = f"""========================================================================================
INSTITUTIONAL QUANTITATIVE TEARSHEET: 5-MINUTE ORB + 12 EMA STRATEGY
Period: 2018–2026 (8.7 Years | 2,139 Daily Sessions)
Asset: CME Micro E-mini Nasdaq-100 (MNQ) Futures
Sizing: {contracts} MNQ Contracts ($8.00 per point)
Execution Friction: ${COMMISSION_PER_CONTRACT * 2 * contracts:.2f} Roundturn Comms + {SLIPPAGE_POINTS:.2f} pts Slippage
========================================================================================

1. CORE RETURN & EFFICIENCY METRICS:
----------------------------------------------------------------------------------------
Total Realized Net PnL    : +${tot_pnl:,.2f}
Total Trades Executed     : {n_trades:,} trades ({n_trades / len(sessions):.1%} of sessions)
Win Rate                  : {win_rate * 100:.2f}% ({len(wins)} Wins / {len(losses)} Losses)
Profit Factor             : {pf:.2f}
Average Trade Expectancy  : +${expectancy:.2f} per trade
Annualized Sharpe Ratio   : {sharpe:.2f}
Maximum Realized Drawdown : -${max_dd:,.2f} (Safety Buffer on $2,000 limit: +${2000.0 - max_dd:,.2f})

2. YEAR-BY-YEAR PERFORMANCE AUDIT:
----------------------------------------------------------------------------------------
{yearly.to_string()}

3. EXIT REASON ATTRIBUTION:
----------------------------------------------------------------------------------------
{df_trades['exit_reason'].value_counts().to_string()}
========================================================================================
"""
    tearsheet_path = PROJECT_ROOT / "strategies" / "orb_5m" / "backtest" / "tearsheet_5m.txt"
    with open(tearsheet_path, "w", encoding="utf-8") as f:
        f.write(tearsheet)

    print(f"\nSaved Institutional Tearsheet to {tearsheet_path}")
    print(tearsheet)


if __name__ == "__main__":
    run_production_backtest()
