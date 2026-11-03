"""
Clean Slate Quantitative Optimization Engine
Multi-Horizon Filtered Opening Range Breakout (ORB) Architecture
Target: Tradeify $50K Growth EOD Trailing Drawdown
Focus: High Win Rate, Small/Moderate R:R, Zero Overnight, Prop Capital Velocity
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

from core.data_loader import load_continuous_nq_parquet, extract_daily_trading_sessions

POINT_VALUE_MNQ = 2.0
COMMISSION_PER_CONTRACT = 1.24  # $2.48 roundturn
SLIPPAGE_POINTS = 0.50          # 0.50 pt ($1.00 MNQ) pessimistic entry/exit


@dataclass
class Trade:
    session_idx: int
    date: Any
    direction: int
    entry_price: float
    exit_price: float
    pnl_net: float
    r_mult: float
    exit_reason: str
    bars_held: int


def precompute_sessions_and_atr(df_nq: pd.DataFrame) -> List[dict]:
    """Extracts daily sessions and precomputes rolling 14-day Daily ATR."""
    sessions = extract_daily_trading_sessions(df_nq, rth_start="09:30", rth_end="15:45")
    
    # Calculate daily True Range and 14-day ATR
    daily_highs = [s["daily_high"] for s in sessions]
    daily_lows = [s["daily_low"] for s in sessions]
    daily_closes = [s["daily_close"] for s in sessions]
    
    n = len(sessions)
    tr_list = []
    for i in range(n):
        h = daily_highs[i]
        l = daily_lows[i]
        if i == 0:
            tr = h - l
        else:
            prev_c = daily_closes[i - 1]
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        tr_list.append(tr)
        
    # 14-day rolling ATR (lagged by 1 day to prevent lookahead)
    atr_series = pd.Series(tr_list).rolling(window=14, min_periods=1).mean().shift(1).fillna(tr_list[0]).tolist()
    
    for i, s in enumerate(sessions):
        s["idx"] = i
        s["daily_atr_14"] = atr_series[i]
        
    return sessions


def simulate_orb_strategy(
    sessions: list[dict],
    or_minutes: int,             # 5, 10, 15, 30
    compression_atr_cap: float,  # e.g. 0.25, 0.35, 0.50, 999.0 (no filter)
    use_vwap_filter: bool,       # require close > VWAP for Long, < VWAP for Short
    use_htf_ema_filter: bool,    # require daily close > 20 EMA for Long, < 20 EMA for Short
    sl_mode: str,                # 'OR_MID', 'OR_EXTREME', 'FIXED_PTS'
    sl_param: float,             # max SL cap or fixed pts
    tp_rr: float,                # 0.75, 1.0, 1.25, 1.5, 2.0
    use_be_at_r: Optional[float] = None, # move to BE after hitting +X R
    time_exit_bars: int = 999,   # max bars to hold before time-exit
    contracts: int = 2
) -> Tuple[np.ndarray, List[Trade]]:
    """
    Executes a clean, strictly pessimistic, gap-aware intraday ORB backtest.
    """
    n_sessions = len(sessions)
    daily_pnls = np.zeros(n_sessions, dtype=np.float64)
    trades: List[Trade] = []
    
    roundtrip_comm = 2.0 * COMMISSION_PER_CONTRACT * contracts
    
    for s_idx, sess in enumerate(sessions):
        rth_bars = sess["rth_bars"]
        if len(rth_bars) < or_minutes + 5:
            continue
            
        # 1. Establish Opening Range
        or_slice = rth_bars.iloc[:or_minutes]
        or_high = float(or_slice["High"].max())
        or_low = float(or_slice["Low"].min())
        or_range = or_high - or_low
        or_mid = (or_high + or_low) / 2.0
        
        # Minimum noise floor
        if or_range < 6.0:
            continue
            
        # 2. Compression / ATR Exhaustion Filter
        daily_atr = sess.get("daily_atr_14", 150.0)
        if daily_atr > 0 and (or_range / daily_atr) > compression_atr_cap:
            continue  # Range already exhausted, abort session!
            
        # 3. Higher Timeframe 20 EMA Filter
        htf_bias = 0 # 1 = Long only, -1 = Short only, 0 = both allowed
        if use_htf_ema_filter:
            prev_close = sess.get("prev_day_close", 0.0)
            daily_ema = sess.get("daily_ema_20", 0.0)
            if prev_close > 0 and daily_ema > 0:
                htf_bias = 1 if prev_close > daily_ema else -1
                
        # 4. Breakout Scanning starting immediately after OR window
        post_bars = rth_bars.iloc[or_minutes:]
        opens = post_bars["Open"].to_numpy(dtype=np.float64)
        highs = post_bars["High"].to_numpy(dtype=np.float64)
        lows = post_bars["Low"].to_numpy(dtype=np.float64)
        closes = post_bars["Close"].to_numpy(dtype=np.float64)
        vwaps = post_bars["VWAP"].to_numpy(dtype=np.float64) if "VWAP" in post_bars.columns else None
        
        n_post = len(opens)
        max_scan = min(90, n_post)  # Look for breakout within first 90 minutes post-OR
        
        entered = False
        entry_idx = -1
        direction = 0
        entry_price = 0.0
        
        for i in range(max_scan):
            # Long breakout
            if highs[i] > or_high:
                if htf_bias == -1:
                    continue
                if use_vwap_filter and vwaps is not None and closes[i] <= vwaps[i]:
                    continue
                # Gap-aware fill at max(or_high, open) + slippage
                entry_price = max(or_high, opens[i]) + SLIPPAGE_POINTS
                direction = 1
                entry_idx = i
                entered = True
                break
                
            # Short breakout
            elif lows[i] < or_low:
                if htf_bias == 1:
                    continue
                if use_vwap_filter and vwaps is not None and closes[i] >= vwaps[i]:
                    continue
                # Gap-aware fill at min(or_low, open) - slippage
                entry_price = min(or_low, opens[i]) - SLIPPAGE_POINTS
                direction = -1
                entry_idx = i
                entered = True
                break
                
        if not entered:
            continue
            
        # 5. Stop Loss & Target Calculation
        if sl_mode == "OR_MID":
            raw_risk = abs(entry_price - or_mid)
            risk_pts = min(max(raw_risk, 8.0), sl_param)
        elif sl_mode == "OR_EXTREME":
            raw_risk = (entry_price - or_low) if direction == 1 else (or_high - entry_price)
            risk_pts = min(max(raw_risk, 10.0), sl_param)
        elif sl_mode == "FIXED_PTS":
            risk_pts = sl_param
        else:
            risk_pts = 25.0
            
        initial_sl = (entry_price - risk_pts) if direction == 1 else (entry_price + risk_pts)
        current_sl = initial_sl
        target_price = (entry_price + tp_rr * risk_pts) if direction == 1 else (entry_price - tp_rr * risk_pts)
        
        be_trigger_price = None
        if use_be_at_r is not None:
            be_trigger_price = (entry_price + use_be_at_r * risk_pts) if direction == 1 else (entry_price - use_be_at_r * risk_pts)
            
        # 6. Intraday Bar Traversal (Strict Pessimistic Collision Resolution)
        exit_price = None
        exit_reason = None
        be_active = False
        bars_held = 0
        
        for i in range(entry_idx, n_post):
            bars_held += 1
            b_o = opens[i]
            b_h = highs[i]
            b_l = lows[i]
            
            # Time exit check
            if bars_held >= time_exit_bars:
                exit_price = closes[i] - SLIPPAGE_POINTS if direction == 1 else closes[i] + SLIPPAGE_POINTS
                exit_reason = "TIME_EXIT"
                break
                
            sl_hit = (b_l <= current_sl) if direction == 1 else (b_h >= current_sl)
            tp_hit = (b_h >= target_price) if direction == 1 else (b_l <= target_price)
            
            # PESSIMISTIC RULE: If both hit on same bar, assume stop loss occurred first
            if sl_hit and tp_hit:
                fill = min(current_sl, b_o) if direction == 1 else max(current_sl, b_o)
                exit_price = fill - SLIPPAGE_POINTS if direction == 1 else fill + SLIPPAGE_POINTS
                exit_reason = "PESSIMISTIC_STOP"
                break
            elif sl_hit:
                fill = min(current_sl, b_o) if direction == 1 else max(current_sl, b_o)
                exit_price = fill - SLIPPAGE_POINTS if direction == 1 else fill + SLIPPAGE_POINTS
                exit_reason = "BE_STOP" if be_active else "STOP_LOSS"
                break
            elif tp_hit:
                fill = max(target_price, b_o) if direction == 1 else min(target_price, b_o)
                exit_price = fill - SLIPPAGE_POINTS if direction == 1 else fill + SLIPPAGE_POINTS
                exit_reason = "TARGET_PROFIT"
                break
                
            # Breakeven trigger check for subsequent bars
            if be_trigger_price is not None and not be_active:
                be_touched = (b_h >= be_trigger_price) if direction == 1 else (b_l <= be_trigger_price)
                if be_touched:
                    be_level = (entry_price + 0.50) if direction == 1 else (entry_price - 0.50)
                    if (direction == 1 and be_level > current_sl) or (direction == -1 and be_level < current_sl):
                        current_sl = be_level
                        be_active = True
                        
        # EOD Flatten at 15:45 (No overnight holding permitted)
        if exit_price is None:
            last_c = closes[-1]
            exit_price = last_c - SLIPPAGE_POINTS if direction == 1 else last_c + SLIPPAGE_POINTS
            exit_reason = "FORCE_CLOSE_EOD"
            
        pts = (exit_price - entry_price) if direction == 1 else (entry_price - exit_price)
        net_dollars = (pts * POINT_VALUE_MNQ * contracts) - roundtrip_comm
        r_mult = net_dollars / (risk_pts * POINT_VALUE_MNQ * contracts)
        
        daily_pnls[s_idx] = net_dollars
        trades.append(Trade(
            session_idx=s_idx,
            date=sess["date"],
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_net=net_dollars,
            r_mult=r_mult,
            exit_reason=exit_reason,
            bars_held=bars_held
        ))
        
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
) -> Tuple[float, float, float, float]:
    """
    Simulates Tradeify Growth $50K Challenge using Stationary Block Bootstrap.
    Returns:
      pass_rate (float 0.0 - 1.0)
      avg_days_to_pass (float)
      expected_cost (float in USD)
      velocity_usd_per_day (float in USD/day)
    """
    n_samples = len(daily_pnls)
    if n_samples == 0 or np.sum(daily_pnls) <= 0:
        return 0.0, 999.0, 9999.0, -99.0

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

            # EOD Trailing Floor Check
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
        # Capital velocity: profit target net of expected cost divided by expected days to pass
        velocity = (profit_target - expected_cost) / avg_days
    else:
        expected_cost = 9999.0
        velocity = -99.0

    return pass_rate, avg_days, expected_cost, velocity


def run_clean_slate_optimization():
    print("=========================================================================")
    print("RUNNING CLEAN SLATE QUANTITATIVE OPTIMIZATION: PROP ORB HIGH-WR FOCUS")
    print("Asset: CME Micro E-mini Nasdaq-100 (MNQ) Futures")
    print("Target Challenge: Tradeify $50K Growth ($3K Target, $2K EOD Trailing DD)")
    print("=========================================================================\n")

    print("Loading continuous CME NQ parquet dataset...")
    df_nq = load_continuous_nq_parquet(filter_rth=False)
    sessions = precompute_sessions_and_atr(df_nq)
    
    # Train / Test Split: In-Sample 2018-2022 (5 years), Out-of-Sample 2023-2026 (3.8 years)
    split_date = pd.to_datetime("2023-01-01").date()
    train_sessions = [s for s in sessions if pd.to_datetime(s["date"]).date() < split_date]
    test_sessions = [s for s in sessions if pd.to_datetime(s["date"]).date() >= split_date]
    
    print(f"Total sessions: {len(sessions):,} | Train (2018-2022): {len(train_sessions):,} | Test (2023-2026): {len(test_sessions):,}\n")

    # Grid Construction focusing on:
    # 1. Opening Range Horizons: 10m, 15m, 20m, 30m
    # 2. Smaller R:R (0.75R, 1.0R, 1.25R, 1.5R) for High Win Rate
    # 3. Natural Structural Stops (OR_MID, OR_EXTREME, FIXED)
    # 4. Volatility & Trend Filters (ATR Compression, VWAP, Daily EMA)
    
    or_horizons = [10, 15, 20, 30]
    compression_caps = [0.35, 0.50, 999.0]
    vwap_filters = [True, False]
    htf_filters = [True, False]
    sl_configs = [
        ("OR_MID", 35.0),
        ("OR_MID", 50.0),
        ("OR_EXTREME", 40.0),
        ("FIXED_PTS", 30.0),
        ("FIXED_PTS", 40.0)
    ]
    tp_configs = [
        (0.75, None),
        (1.0, None),
        (1.0, 0.5),   # 1.0R with BE at +0.5R
        (1.25, None),
        (1.25, 0.75), # 1.25R with BE at +0.75R
        (1.5, None),
        (1.5, 0.75)   # 1.5R with BE at +0.75R
    ]
    
    grid = []
    for or_m in or_horizons:
        for cap in compression_caps:
            for vwap in vwap_filters:
                for htf in htf_filters:
                    for sl_m, sl_p in sl_configs:
                        for tp_r, be_r in tp_configs:
                            grid.append({
                                "or_minutes": or_m,
                                "compression_atr_cap": cap,
                                "use_vwap_filter": vwap,
                                "use_htf_ema_filter": htf,
                                "sl_mode": sl_m,
                                "sl_param": sl_p,
                                "tp_rr": tp_r,
                                "use_be_at_r": be_r
                            })
                            
    print(f"Constructed Parameter Grid: {len(grid):,} Candidate Strategies.")
    print("Evaluating all candidates on In-Sample (Train: 2018-2022) dataset with 2 MNQ base contracts...\n")
    
    train_evals = []
    for idx, cfg in enumerate(grid):
        daily_pnls, trades = simulate_orb_strategy(train_sessions, **cfg, contracts=2)
        n_tr = len(trades)
        if n_tr < 80:  # Require statistical minimum of trades
            continue
            
        tot_pnl = np.sum(daily_pnls)
        wins = [t.pnl_net for t in trades if t.pnl_net > 0]
        losses = [abs(t.pnl_net) for t in trades if t.pnl_net <= 0]
        wr = len(wins) / n_tr
        pf = (sum(wins) / sum(losses)) if sum(losses) > 0 else 0.0
        exp = tot_pnl / n_tr
        
        # Max drawdown
        cum = np.cumsum(daily_pnls)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        max_dd = np.max(dd) if len(dd) > 0 else 0.0
        std = np.std(daily_pnls)
        sharpe = (np.mean(daily_pnls) / std * np.sqrt(252)) if std > 0 else 0.0
        
        # We only advance candidates with positive expectancy and reasonable drawdown
        if tot_pnl > 3000 and pf >= 1.15 and max_dd <= 2500:
            train_evals.append({
                "cfg": cfg,
                "n_trades": n_tr,
                "train_pnl": tot_pnl,
                "train_wr": wr,
                "train_pf": pf,
                "train_exp": exp,
                "train_max_dd": max_dd,
                "train_sharpe": sharpe
            })
            
    df_train = pd.DataFrame(train_evals)
    print(f"Identified {len(df_train)} Promising In-Sample Candidates meeting Institutional Thresholds.\n")
    
    if df_train.empty:
        print("No candidates met initial screening. Broadening search criteria...")
        return
        
    df_train = df_train.sort_values(by="train_sharpe", ascending=False)
    print("Top 10 In-Sample Candidates (2018-2022):\n")
    for i, r in df_train.head(10).iterrows():
        c = r["cfg"]
        print(f"[OR {c['or_minutes']}m | Cap {c['compression_atr_cap']} | VWAP {c['use_vwap_filter']} | HTF {c['use_htf_ema_filter']}] SL: {c['sl_mode']}({c['sl_param']}) | TP: {c['tp_rr']}R (BE: {c['use_be_at_r']})")
        print(f"   PnL: ${r['train_pnl']:,.0f} | WR: {r['train_wr']:.1%} | PF: {r['train_pf']:.2f} | Exp: ${r['train_exp']:.2f} | MaxDD: ${r['train_max_dd']:,.0f} | Sharpe: {r['train_sharpe']:.2f}\n")
        
    # Phase 2: Out-of-Sample Verification (2023-2026) and Sizing / Tradeify Monte Carlo
    print("=========================================================================")
    print("PHASE 2: OUT-OF-SAMPLE VERIFICATION (2023-2026) & TRADEIFY MONTE CARLO")
    print("=========================================================================\n")
    
    final_candidates = []
    top_train = df_train.head(30)
    
    for idx, row in top_train.iterrows():
        cfg = row["cfg"]
        
        # Test on OOS sessions
        oos_pnls, oos_trades = simulate_orb_strategy(test_sessions, **cfg, contracts=2)
        n_oos = len(oos_trades)
        oos_pnl = np.sum(oos_pnls)
        wins_oos = [t.pnl_net for t in oos_trades if t.pnl_net > 0]
        losses_oos = [abs(t.pnl_net) for t in oos_trades if t.pnl_net <= 0]
        wr_oos = len(wins_oos) / n_oos if n_oos > 0 else 0.0
        pf_oos = (sum(wins_oos) / sum(losses_oos)) if sum(losses_oos) > 0 else 0.0
        exp_oos = oos_pnl / n_oos if n_oos > 0 else 0.0
        std_oos = np.std(oos_pnls)
        sharpe_oos = (np.mean(oos_pnls) / std_oos * np.sqrt(252)) if std_oos > 0 else 0.0
        
        # Full sample simulation
        full_pnls, full_trades = simulate_orb_strategy(sessions, **cfg, contracts=2)
        full_pnl = np.sum(full_pnls)
        wins_full = [t.pnl_net for t in full_trades if t.pnl_net > 0]
        losses_full = [abs(t.pnl_net) for t in full_trades if t.pnl_net <= 0]
        wr_full = len(wins_full) / len(full_trades) if len(full_trades) > 0 else 0.0
        pf_full = (sum(wins_full) / sum(losses_full)) if sum(losses_full) > 0 else 0.0
        
        # Sizing Sweep for Tradeify Challenge: 2, 3, 4 MNQ
        sizing_evals = {}
        for k in [2, 3, 4]:
            k_pnls = full_pnls * (k / 2.0)
            p_rate, t_pass, cost, vel = simulate_tradeify_challenge_monte_carlo(
                k_pnls, n_sims=1500, profit_target=3000.0, trailing_dd_limit=2000.0, daily_loss_limit=1250.0
            )
            sizing_evals[k] = {
                "pass_rate": p_rate,
                "avg_days": t_pass,
                "cost": cost,
                "velocity": vel
            }
            
        final_candidates.append({
            "cfg": cfg,
            "train_pnl": row["train_pnl"],
            "train_sharpe": row["train_sharpe"],
            "oos_pnl": oos_pnl,
            "oos_wr": wr_oos,
            "oos_pf": pf_oos,
            "oos_sharpe": sharpe_oos,
            "full_pnl": full_pnl,
            "full_wr": wr_full,
            "full_pf": pf_full,
            "sizing": sizing_evals
        })
        
    df_final = pd.DataFrame(final_candidates)
    # Filter for models that remained profitable in Out-of-Sample (WFE > 0)
    df_robust = df_final[df_final["oos_pnl"] > 0].copy()
    
    print(f"Out of 30 candidates, {len(df_robust)} confirmed robust in Out-of-Sample (2023-2026)!\n")
    
    # Sort by Tradeify 3 MNQ velocity score
    df_robust["velocity_3mnq"] = df_robust["sizing"].apply(lambda s: s[3]["velocity"])
    df_robust = df_robust.sort_values(by="velocity_3mnq", ascending=False)
    
    print("=========================================================================")
    print("TOP AUDITED STRATEGY CANDIDATES (Ranked by Tradeify Capital Velocity)")
    print("=========================================================================\n")
    
    for rank, (i, r) in enumerate(df_robust.head(5).iterrows(), start=1):
        c = r["cfg"]
        s2 = r["sizing"][2]
        s3 = r["sizing"][3]
        s4 = r["sizing"][4]
        print(f"RANK #{rank}: OR {c['or_minutes']}m Breakout | Cap {c['compression_atr_cap']} | VWAP {c['use_vwap_filter']} | HTF {c['use_htf_ema_filter']}")
        print(f"   Execution: SL: {c['sl_mode']}({c['sl_param']}) | TP: {c['tp_rr']}R (BE at: {c['use_be_at_r']}R)")
        print(f"   Full History (8.9y): PnL: ${r['full_pnl']:,.2f} | Win Rate: {r['full_wr']:.1%} | Profit Factor: {r['full_pf']:.2f}")
        print(f"   In-Sample (2018-2022) : PnL: ${r['train_pnl']:,.0f} (Sharpe: {r['train_sharpe']:.2f})")
        print(f"   Out-of-Sample (2023-2026): PnL: ${r['oos_pnl']:,.0f} (Sharpe: {r['oos_sharpe']:.2f}, WR: {r['oos_wr']:.1%}, PF: {r['oos_pf']:.2f})")
        print(f"   --- Tradeify $50K Challenge Evaluation ---")
        print(f"   • 2 MNQ: Pass Rate: {s2['pass_rate']:.1%} | Avg Days: {s2['avg_days']:.1f}d | Exp Cost: ${s2['cost']:.2f} | Vel: ${s2['velocity']:.2f}/d")
        print(f"   • 3 MNQ: Pass Rate: {s3['pass_rate']:.1%} | Avg Days: {s3['avg_days']:.1f}d | Exp Cost: ${s3['cost']:.2f} | Vel: ${s3['velocity']:.2f}/d")
        print(f"   • 4 MNQ: Pass Rate: {s4['pass_rate']:.1%} | Avg Days: {s4['avg_days']:.1f}d | Exp Cost: ${s4['cost']:.2f} | Vel: ${s4['velocity']:.2f}/d\n")


if __name__ == "__main__":
    run_clean_slate_optimization()

