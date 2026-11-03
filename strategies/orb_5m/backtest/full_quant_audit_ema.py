"""
Comprehensive Institutional Quantitative Audit Engine for 12 EMA Strategy
Executes full econometric and prop evaluation audit comparing:
  Case 1: The Repository's "Champion" 5m ORB + 12 EMA (0.5R Trail) under Realistic Execution
  Case 2: Pure 12 EMA Directional Bias at 09:35 (Fixed Risk & Invalidation Exit)
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.orb_5m.backtest.sweep_engine_5m import SessionPrecompiler, simulate_single_model
from strategies.orb_5m.backtest.optimizer_ema12 import run_single_simulation, simulate_tradeify_challenge_monte_carlo


def run_full_institutional_quant_audit():
    print("=========================================================================")
    print("RUNNING INSTITUTIONAL COMPREHENSIVE QUANT AUDIT ON 12 EMA STRATEGY")
    print("Period: 2018-2026 (8.7 Years, 2,139 Daily CME Sessions)")
    print("=========================================================================\n")

    sessions = SessionPrecompiler.load_sessions()
    for idx, s in enumerate(sessions):
        s["idx"] = idx

    split_date = pd.to_datetime("2023-01-01").date()

    # -------------------------------------------------------------------------
    # CASE 1: Original Repo Champion (BREAKOUT_WITH_EMA, TRAIL_0.5R, 4 MNQ)
    # -------------------------------------------------------------------------
    print("--- Simulating Case 1: Repo 'Champion' (Breakout + 12 EMA + 0.5R Trail, 4 MNQ) ---")
    daily_pnl_case1, trades_case1 = simulate_single_model(
        sessions=sessions,
        entry_mode="BREAKOUT_WITH_EMA",
        trail_mode="TRAIL_0.5R",
        max_sl_pts=35.0,
        contracts=4,
        slippage_pts=0.50
    )

    # -------------------------------------------------------------------------
    # CASE 2: Best Realistic 12 EMA Model (BREAKOUT_5M, SL: 75 pts, TP: BE then 2.0R, 3 MNQ)
    # -------------------------------------------------------------------------
    print("--- Simulating Case 2: Best Robust 12 EMA Model (Breakout + SL 75 pts + BE/2R, 3 MNQ) ---")
    daily_pnl_case2 = np.zeros(len(sessions), dtype=np.float64)
    trades_case2 = []
    cfg_case2 = {
        "entry_type": "BREAKOUT_5M",
        "sl_type": "FIXED_PTS",
        "sl_param": 75.0,
        "tp_type": "BE_THEN_R",
        "tp_param": 2.0,
        "tp_param2": 0.0
    }
    for i, s in enumerate(sessions):
        tr = run_single_simulation(s, **cfg_case2, contracts=3)
        if tr is not None:
            daily_pnl_case2[i] = tr.pnl_dollars_net
            trades_case2.append(tr)

    # -------------------------------------------------------------------------
    # CASE 3: Pure 12 EMA Market Entry at 09:35 (MARKET_0935, SL: EMA_12, TP: 2.0R, 3 MNQ)
    # -------------------------------------------------------------------------
    print("--- Simulating Case 3: Pure 12 EMA Market Entry at 09:35 Open (3 MNQ) ---")
    daily_pnl_case3 = np.zeros(len(sessions), dtype=np.float64)
    trades_case3 = []
    cfg_case3 = {
        "entry_type": "MARKET_0935",
        "sl_type": "STRUCT_5M",
        "sl_param": 50.0,
        "tp_type": "FIXED_R",
        "tp_param": 2.0,
        "tp_param2": 0.0
    }
    for i, s in enumerate(sessions):
        tr = run_single_simulation(s, **cfg_case3, contracts=3)
        if tr is not None:
            daily_pnl_case3[i] = tr.pnl_dollars_net
            trades_case3.append(tr)

    cases = [
        ("Case 1: Repo 'Champion' (0.5R Trail, 4 MNQ)", daily_pnl_case1, trades_case1, 4),
        ("Case 2: Best Robust 12 EMA Breakout (SL 75, BE/2R, 3 MNQ)", daily_pnl_case2, trades_case2, 3),
        ("Case 3: Pure 12 EMA Market Entry 09:35 (SL 50, TP 2R, 3 MNQ)", daily_pnl_case3, trades_case3, 3)
    ]

    for name, pnl_series, trades, contracts in cases:
        print("\n" + "="*85)
        print(f"QUANTITATIVE AUDIT REPORT: {name.upper()}")
        print("="*85)

        n_trades = len(trades)
        pnls = [t.pnl_dollars_net for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p <= 0]
        tot_pnl = sum(pnls)
        wr = len(wins) / n_trades if n_trades > 0 else 0.0
        pf = sum(wins) / sum(losses) if sum(losses) > 0 else 0.0
        exp = tot_pnl / n_trades if n_trades > 0 else 0.0

        # Equity & Drawdown
        equity = np.cumsum(pnl_series)
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        max_dd = np.max(dd)

        # Sharpe, Sortino, Calmar
        std = np.std(pnl_series)
        sharpe = (np.mean(pnl_series) / std * np.sqrt(252)) if std > 0 else 0.0
        downside_std = np.std([p for p in pnl_series if p < 0])
        sortino = (np.mean(pnl_series) / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0
        ann_return = np.mean(pnl_series) * 252.0
        calmar = (ann_return / max_dd) if max_dd > 0 else 0.0

        # Distribution / Tail Risk (EVT & VaR)
        skew = stats.skew(pnl_series)
        kurt = stats.kurtosis(pnl_series)
        var_95 = np.percentile(pnl_series, 5)
        var_99 = np.percentile(pnl_series, 1)
        tail_losses = [p for p in pnl_series if p <= var_99]
        cvar_99 = np.mean(tail_losses) if len(tail_losses) > 0 else var_99

        # In-Sample vs Out-of-Sample (WFE)
        is_pnls = [pnl_series[i] for i, s in enumerate(sessions) if pd.to_datetime(s["date"]).date() < split_date]
        oos_pnls = [pnl_series[i] for i, s in enumerate(sessions) if pd.to_datetime(s["date"]).date() >= split_date]
        is_mean = np.mean(is_pnls)
        oos_mean = np.mean(oos_pnls)
        wfe = (oos_mean / is_mean) if is_mean > 0 else 0.0

        # Tradeify Monte Carlo Challenge Evaluation (1,500 Bootstrap Paths)
        pass_rate, avg_days, exp_cost = simulate_tradeify_challenge_monte_carlo(
            pnl_series, n_sims=1500, profit_target=3000.0, trailing_dd_limit=2000.0, daily_loss_limit=1250.0
        )

        print(f"\n1. CORE RETURN & EFFICIENCY METRICS:")
        print(f"-----------------------------------------------------------------------------")
        print(f"Total Realized Net PnL    : ${tot_pnl:,.2f}")
        print(f"Total Trades Executed     : {n_trades:,} trades ({n_trades/len(sessions):.1%} of sessions)")
        print(f"Win Rate                  : {wr:.2%} ({len(wins)} Wins / {len(losses)} Losses)")
        print(f"Profit Factor             : {pf:.2f}")
        print(f"Average Trade Expectancy  : ${exp:.2f} per trade")
        print(f"Annualized Sharpe Ratio   : {sharpe:.2f}")
        print(f"Annualized Sortino Ratio  : {sortino:.2f}")
        print(f"Calmar Ratio              : {calmar:.2f}")
        print(f"Maximum Realized Drawdown : -${max_dd:,.2f}")

        print(f"\n2. DISTRIBUTION & TAIL RISK (TRADEIFY $1,250 DAILY LIMIT AUDIT):")
        print(f"-----------------------------------------------------------------------------")
        print(f"PnL Skewness              : {skew:+.3f}")
        print(f"PnL Excess Kurtosis       : {kurt:+.3f}")
        print(f"Value at Risk (95% Daily) : ${var_95:,.2f}")
        print(f"Value at Risk (99% Daily) : ${var_99:,.2f}")
        print(f"CVaR / Expected Shortfall : ${cvar_99:,.2f} (Buffer on $1,250 limit: ${1250.0 + cvar_99:,.2f})")

        print(f"\n3. WALK-FORWARD EFFICIENCY (WFE) OVERFITTING TEST:")
        print(f"-----------------------------------------------------------------------------")
        print(f"In-Sample Net PnL (2018-2022)  : ${sum(is_pnls):,.2f} (Mean/Day: ${is_mean:.2f})")
        print(f"Out-of-Sample PnL (2023-2026)  : ${sum(oos_pnls):,.2f} (Mean/Day: ${oos_mean:.2f})")
        print(f"Walk-Forward Efficiency (WFE)  : {wfe:.2f} (Benchmark: >= 0.70)")

        print(f"\n4. TRADEIFY $50K EVALUATION ECONOMICS (1,500 MONTE CARLO PATHS):")
        print(f"-----------------------------------------------------------------------------")
        print(f"Challenge Pass Rate       : {pass_rate:.1%}")
        print(f"Average Days to Pass      : {avg_days:.1f} trading days")
        print(f"Expected Challenge Cost   : ${exp_cost:,.2f}")
        print(f"Pass Velocity ($EV/Day)   : ${tot_pnl/len(sessions):.2f} / day")

        # Yearly breakdown
        trade_records = []
        for t in trades:
            d = getattr(t, "date", None)
            if d is None:
                d = sessions[t.session_idx]["date"]
            trade_records.append({"year": pd.to_datetime(d).year, "pnl": t.pnl_dollars_net})
        df_t = pd.DataFrame(trade_records)
        if len(df_t) > 0:
            yearly = df_t.groupby("year")["pnl"].agg(["count", "sum"]).rename(columns={"count": "trades", "sum": "net_pnl"})
            yearly["win_rate"] = df_t.groupby("year")["pnl"].apply(lambda s: (s > 0).mean() * 100.0)
            print(f"\n5. YEAR-BY-YEAR REALIZED PERFORMANCE:")
            print(f"-----------------------------------------------------------------------------")
            print(yearly.to_string())


if __name__ == "__main__":
    run_full_institutional_quant_audit()
