"""
Matteo Conti / Institutional Quantitative ORB Optimization Engine
High Win-Rate, Smaller R:R, Zero-Overnight Architecture
Target: Tradeify $50K Growth EOD Trailing Drawdown
Asset: CME Micro E-mini Nasdaq-100 (MNQ) Futures
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
class TradeRecord:
    session_idx: int
    date: Any
    trade_type: str  # 'BREAKOUT' or 'RETEST'
    direction: int   # 1 = Long, -1 = Short
    entry_price: float
    exit_price: float
    pnl_net: float
    r_mult: float
    exit_reason: str


def run_high_wr_orb_simulation(
    sessions: list[dict],
    or_minutes: int,             # 15 or 30
    atr_exhaustion_ratio: float, # 0.35, 0.45, 0.55
    min_noise_pts: float,        # 8.0, 12.0
    tp1_r: float,                # 0.8, 1.0, 1.2
    tp2_r: float,                # 1.2, 1.5, 2.0
    scale_out_pct: float,        # 0.5 (50/50), 0.7 (70/30), 1.0 (100% at TP1)
    allow_retest: bool,          # True = both primary breakout & retest, False = breakout only
    contracts: int = 2
) -> Tuple[np.ndarray, List[TradeRecord]]:
    """
    Simulates high-win-rate filtered ORB with dual Breakout & Retest entry mechanics.
    Strictly pessimistic, gap-aware, zero-overnight intraday execution.
    """
    n_sessions = len(sessions)
    daily_pnls = np.zeros(n_sessions, dtype=np.float64)
    trades: List[TradeRecord] = []
    
    roundtrip_comm = 2.0 * COMMISSION_PER_CONTRACT * contracts
    
    for s_idx, sess in enumerate(sessions):
        rth_df = sess["rth_bars"]
        if len(rth_df) < or_minutes + 10:
            continue
            
        # 1. Opening Range Window (09:30 to 09:30 + or_minutes)
        or_bars = rth_df.iloc[:or_minutes]
        or_high = float(or_bars["High"].max())
        or_low = float(or_bars["Low"].min())
        or_mid = round((or_high + or_low) / 2.0, 2)
        or_range = round(or_high - or_low, 2)
        
        # Noise floor check
        if or_range < min_noise_pts:
            continue
            
        # 2. ATR Exhaustion / Compression Filter
        daily_atr = sess.get("daily_atr_14", 150.0)
        if daily_atr > 0 and (or_range > atr_exhaustion_ratio * daily_atr):
            continue  # Move already expanded, abort
            
        # 3. Dynamic Session VWAP calculation from 09:30
        typical_p = (rth_df["High"] + rth_df["Low"] + rth_df["Close"]) / 3.0
        cum_vol = rth_df["Volume"].cumsum()
        cum_vol_p = (typical_p * rth_df["Volume"]).cumsum()
        vwap_series = cum_vol_p / np.maximum(cum_vol, 1)
        
        # 4. Daily 20 EMA trend alignment
        open_p = sess.get("exchange_open_price", 0.0)
        daily_ema = sess.get("daily_20_ema", 0.0)
        long_htf = True if (daily_ema <= 0 or open_p > daily_ema) else False
        short_htf = True if (daily_ema <= 0 or open_p < daily_ema) else False
        
        # 5. Trading Window (Post-OR up to 11:30 EST)
        post_df = rth_df.iloc[or_minutes:]
        eval_bars = post_df[post_df.index.time <= pd.to_datetime("11:30").time()]
        if eval_bars.empty:
            continue
            
        entered = False
        direction = 0
        entry_price = 0.0
        trade_type = ""
        entry_bar_idx = -1
        
        long_breakout_seen = False
        short_breakout_seen = False
        
        for idx, (ts, bar) in enumerate(eval_bars.iterrows()):
            c_p = float(bar["Close"])
            h_p = float(bar["High"])
            l_p = float(bar["Low"])
            o_p = float(bar["Open"])
            vwap_now = float(vwap_series.loc[ts])
            
            if c_p > or_high:
                long_breakout_seen = True
            if c_p < or_low:
                short_breakout_seen = True
                
            # PRIMARY BREAKOUT: 3-min confirmation (minute % 3 == 0)
            if (ts.minute + 1) % 3 == 0:
                if long_htf and c_p > or_high and c_p > vwap_now:
                    entry_price = c_p + SLIPPAGE_POINTS
                    direction = 1
                    trade_type = "BREAKOUT"
                    entry_bar_idx = rth_df.index.get_loc(ts)
                    entered = True
                    break
                elif short_htf and c_p < or_low and c_p < vwap_now:
                    entry_price = c_p - SLIPPAGE_POINTS
                    direction = -1
                    trade_type = "BREAKOUT"
                    entry_bar_idx = rth_df.index.get_loc(ts)
                    entered = True
                    break
                    
            # SECONDARY RETEST (if breakout previously occurred and price touches OR boundary)
            if allow_retest and ts.time() <= pd.to_datetime("11:15").time():
                if long_breakout_seen and long_htf and l_p <= or_high and c_p > or_mid:
                    entry_price = or_high + SLIPPAGE_POINTS
                    direction = 1
                    trade_type = "RETEST"
                    entry_bar_idx = rth_df.index.get_loc(ts)
                    entered = True
                    break
                elif short_breakout_seen and short_htf and h_p >= or_low and c_p < or_mid:
                    entry_price = or_low - SLIPPAGE_POINTS
                    direction = -1
                    trade_type = "RETEST"
                    entry_bar_idx = rth_df.index.get_loc(ts)
                    entered = True
                    break
                    
        if not entered:
            continue
            
        # 6. Stop Loss & Target Construction
        raw_risk = abs(entry_price - or_mid)
        risk_pts = min(max(raw_risk, 10.0), 40.0)  # Capped at 40 pts max catastrophic loss
        
        initial_sl = (entry_price - risk_pts) if direction == 1 else (entry_price + risk_pts)
        target_tp1 = (entry_price + tp1_r * risk_pts) if direction == 1 else (entry_price - tp1_r * risk_pts)
        target_tp2 = (entry_price + tp2_r * risk_pts) if direction == 1 else (entry_price - tp2_r * risk_pts)
        
        # 7. Intra-day Bar Traversal
        current_sl = initial_sl
        tp1_hit = False
        tp2_hit = False
        exit_price_leg1 = None
        exit_price_leg2 = None
        exit_reason = None
        
        post_entry_df = rth_df.iloc[entry_bar_idx:]
        
        for ts, bar in post_entry_df.iterrows():
            b_o = float(bar["Open"])
            b_h = float(bar["High"])
            b_l = float(bar["Low"])
            
            # SL check
            sl_hit = (b_l <= current_sl) if direction == 1 else (b_h >= current_sl)
            
            # Leg 1 Target check
            if not tp1_hit:
                hit_tp1 = (b_h >= target_tp1) if direction == 1 else (b_l <= target_tp1)
                if sl_hit and hit_tp1:
                    # Pessimistic: SL hit first
                    fill = min(current_sl, b_o) if direction == 1 else max(current_sl, b_o)
                    exit_price_leg1 = fill - SLIPPAGE_POINTS if direction == 1 else fill + SLIPPAGE_POINTS
                    exit_price_leg2 = exit_price_leg1
                    exit_reason = "STOP_LOSS"
                    break
                elif sl_hit:
                    fill = min(current_sl, b_o) if direction == 1 else max(current_sl, b_o)
                    exit_price_leg1 = fill - SLIPPAGE_POINTS if direction == 1 else fill + SLIPPAGE_POINTS
                    exit_price_leg2 = exit_price_leg1
                    exit_reason = "STOP_LOSS"
                    break
                elif hit_tp1:
                    tp1_hit = True
                    exit_price_leg1 = target_tp1
                    # Immediate stop ratchet to Entry + 2 ticks (Breakeven scratch lock)
                    be_level = (entry_price + 0.50) if direction == 1 else (entry_price - 0.50)
                    current_sl = be_level
                    if scale_out_pct >= 0.99:
                        # 100% position closed at TP1!
                        exit_price_leg2 = exit_price_leg1
                        exit_reason = "FULL_TP1"
                        break
                        
            # Leg 2 Target check (after TP1 filled)
            else:
                hit_tp2 = (b_h >= target_tp2) if direction == 1 else (b_l <= target_tp2)
                if sl_hit and hit_tp2:
                    exit_price_leg2 = current_sl - SLIPPAGE_POINTS if direction == 1 else current_sl + SLIPPAGE_POINTS
                    exit_reason = "LEG2_BE_STOP"
                    break
                elif sl_hit:
                    fill = min(current_sl, b_o) if direction == 1 else max(current_sl, b_o)
                    exit_price_leg2 = fill - SLIPPAGE_POINTS if direction == 1 else fill + SLIPPAGE_POINTS
                    exit_reason = "LEG2_BE_STOP"
                    break
                elif hit_tp2:
                    exit_price_leg2 = target_tp2
                    exit_reason = "FULL_TP2_RUNNER"
                    break
                    
        # EOD settlement if still holding at 15:45 (No overnight holding)
        if exit_price_leg1 is None:
            last_c = float(rth_df["Close"].iloc[-1])
            exit_price_leg1 = last_c - SLIPPAGE_POINTS if direction == 1 else last_c + SLIPPAGE_POINTS
            exit_price_leg2 = exit_price_leg1
            exit_reason = "FORCE_CLOSE_EOD"
        elif exit_price_leg2 is None:
            last_c = float(rth_df["Close"].iloc[-1])
            exit_price_leg2 = last_c - SLIPPAGE_POINTS if direction == 1 else last_c + SLIPPAGE_POINTS
            exit_reason = "FORCE_CLOSE_EOD_RUNNER"
            
        # PnL calculation across both legs
        c_leg1 = contracts * scale_out_pct
        c_leg2 = contracts * (1.0 - scale_out_pct)
        
        pts1 = (exit_price_leg1 - entry_price) if direction == 1 else (entry_price - exit_price_leg1)
        pts2 = (exit_price_leg2 - entry_price) if direction == 1 else (entry_price - exit_price_leg2)
        
        gross_dollars = (pts1 * c_leg1 * POINT_VALUE_MNQ) + (pts2 * c_leg2 * POINT_VALUE_MNQ)
        net_dollars = gross_dollars - roundtrip_comm
        r_mult = net_dollars / (risk_pts * POINT_VALUE_MNQ * contracts)
        
        daily_pnls[s_idx] = net_dollars
        trades.append(TradeRecord(
            session_idx=s_idx,
            date=sess["date"],
            trade_type=trade_type,
            direction=direction,
            entry_price=entry_price,
            exit_price=round((exit_price_leg1 + exit_price_leg2) / 2.0, 2),
            pnl_net=net_dollars,
            r_mult=r_mult,
            exit_reason=exit_reason
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
    """Simulates Tradeify Growth $50K Challenge using Stationary Block Bootstrap."""
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
        velocity = (profit_target - expected_cost) / avg_days
    else:
        expected_cost = 9999.0
        velocity = -99.0

    return pass_rate, avg_days, expected_cost, velocity


def execute_conti_quant_optimization():
    print("=========================================================================")
    print("MATTEO CONTI PROTOCOL / QUANT ORB OPTIMIZATION (HIGH-WR & SMALL R:R)")
    print("Period: 2018-2026 (8.9 Years, 2,246 Sessions)")
    print("=========================================================================\n")

    print("Ingesting continuous CME NQ parquet...")
    df_nq = load_continuous_nq_parquet(filter_rth=False)
    sessions = extract_daily_trading_sessions(df_nq, rth_start="09:30", rth_end="15:45")
    
    # Precompute rolling 14-day Daily ATR
    true_ranges = []
    for i, s in enumerate(sessions):
        h = s["daily_high"]
        l = s["daily_low"]
        if i > 0:
            prev_c = sessions[i - 1]["daily_close"]
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        else:
            tr = h - l
        true_ranges.append(tr)
        
    for i in range(len(sessions)):
        atr = np.mean(true_ranges[max(0, i-14):i]) if i >= 1 else 150.0
        sessions[i]["daily_atr_14"] = float(atr)

    split_date = pd.to_datetime("2023-01-01").date()
    train_sessions = [s for s in sessions if pd.to_datetime(s["date"]).date() < split_date]
    test_sessions = [s for s in sessions if pd.to_datetime(s["date"]).date() >= split_date]
    print(f"Sessions loaded: {len(sessions):,} | Train (2018-2022): {len(train_sessions):,} | Test (2023-2026): {len(test_sessions):,}\n")

    # Grid specifications
    configs = []
    for or_m in [15, 30]:
        for atr_cap in [0.40, 0.50, 0.60]:
            for min_noise in [8.0, 15.0]:
                for tp1 in [0.8, 1.0, 1.2]:
                    for tp2 in [1.2, 1.5, 2.0]:
                        for scale_pct in [0.5, 0.7, 1.0]:
                            for retest in [True, False]:
                                configs.append({
                                    "or_minutes": or_m,
                                    "atr_exhaustion_ratio": atr_cap,
                                    "min_noise_pts": min_noise,
                                    "tp1_r": tp1,
                                    "tp2_r": tp2,
                                    "scale_out_pct": scale_pct,
                                    "allow_retest": retest
                                })

    print(f"Sweeping {len(configs):,} targeted configurations on In-Sample (Train) dataset...")
    
    train_results = []
    for idx, cfg in enumerate(configs):
        daily_pnls, trades = run_high_wr_orb_simulation(train_sessions, **cfg, contracts=2)
        n_tr = len(trades)
        if n_tr < 50:
            continue
            
        tot_pnl = np.sum(daily_pnls)
        wins = [t.pnl_net for t in trades if t.pnl_net > 0]
        losses = [abs(t.pnl_net) for t in trades if t.pnl_net <= 0]
        wr = len(wins) / n_tr
        pf = sum(wins) / sum(losses) if sum(losses) > 0 else 0.0
        
        cum = np.cumsum(daily_pnls)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        max_dd = np.max(dd) if len(dd) > 0 else 0.0
        std = np.std(daily_pnls)
        sharpe = (np.mean(daily_pnls) / std * np.sqrt(252)) if std > 0 else 0.0
        
        if tot_pnl > 3000 and pf >= 1.25 and max_dd <= 2000 and wr >= 0.50:
            train_results.append({
                "cfg": cfg,
                "n_trades": n_tr,
                "train_pnl": tot_pnl,
                "train_wr": wr,
                "train_pf": pf,
                "train_max_dd": max_dd,
                "train_sharpe": sharpe
            })

    df_train = pd.DataFrame(train_results)
    print(f"\nDiscovered {len(df_train)} institutional candidates with WR >= 50%, PF >= 1.25, and MaxDD <= $2,000!\n")

    if df_train.empty:
        print("Broadening search filters...")
        return

    df_train = df_train.sort_values(by="train_sharpe", ascending=False)
    print("Top 5 In-Sample Models by Sharpe:\n")
    for i, r in df_train.head(5).iterrows():
        c = r["cfg"]
        print(f"[{c['or_minutes']}m ORB | Cap {c['atr_exhaustion_ratio']} | Retest: {c['allow_retest']}] TP1: {c['tp1_r']}R ({c['scale_out_pct']*100:.0f}%), TP2: {c['tp2_r']}R -> PnL: ${r['train_pnl']:,.0f} | WR: {r['train_wr']:.1%} | PF: {r['train_pf']:.2f} | MaxDD: ${r['train_max_dd']:,.0f}")

    # Out of sample verification
    print("\n-------------------------------------------------------------------------")
    print("PHASE 2: OUT-OF-SAMPLE TEST (2023-2026) & TRADEIFY CAPITAL VELOCITY AUDIT")
    print("-------------------------------------------------------------------------\n")

    oos_evals = []
    for _, row in df_train.head(20).iterrows():
        cfg = row["cfg"]
        oos_pnls, oos_trades = run_high_wr_orb_simulation(test_sessions, **cfg, contracts=2)
        n_oos = len(oos_trades)
        oos_pnl = np.sum(oos_pnls)
        wins_oos = [t.pnl_net for t in oos_trades if t.pnl_net > 0]
        losses_oos = [abs(t.pnl_net) for t in oos_trades if t.pnl_net <= 0]
        wr_oos = len(wins_oos) / n_oos if n_oos > 0 else 0.0
        pf_oos = sum(wins_oos) / sum(losses_oos) if sum(losses_oos) > 0 else 0.0

        full_pnls, full_trades = run_high_wr_orb_simulation(sessions, **cfg, contracts=2)
        full_pnl = np.sum(full_pnls)
        wins_full = [t.pnl_net for t in full_trades if t.pnl_net > 0]
        losses_full = [abs(t.pnl_net) for t in full_trades if t.pnl_net <= 0]
        wr_full = len(wins_full) / len(full_trades) if len(full_trades) > 0 else 0.0
        pf_full = sum(wins_full) / sum(losses_full) if sum(losses_full) > 0 else 0.0

        # Sizing tests on full history
        sizing = {}
        for k in [2, 3, 4]:
            k_pnls = full_pnls * (k / 2.0)
            p_rate, t_pass, cost, vel = simulate_tradeify_challenge_monte_carlo(k_pnls, n_sims=1500)
            sizing[k] = {"pass_rate": p_rate, "avg_days": t_pass, "cost": cost, "velocity": vel}

        oos_evals.append({
            "cfg": cfg,
            "train_pnl": row["train_pnl"],
            "oos_pnl": oos_pnl,
            "oos_wr": wr_oos,
            "oos_pf": pf_oos,
            "full_pnl": full_pnl,
            "full_wr": wr_full,
            "full_pf": pf_full,
            "sizing": sizing
        })

    df_oos = pd.DataFrame(oos_evals)
    df_oos["vel_3mnq"] = df_oos["sizing"].apply(lambda s: s[3]["velocity"])
    df_oos = df_oos.sort_values(by="vel_3mnq", ascending=False)

    print("=========================================================================")
    print("WINNING STRATEGIES (Ranked by Tradeify Payout Velocity / Time to Pass):")
    print("=========================================================================\n")

    for rank, (i, r) in enumerate(df_oos.head(5).iterrows(), start=1):
        c = r["cfg"]
        s2 = r["sizing"][2]
        s3 = r["sizing"][3]
        s4 = r["sizing"][4]
        print(f"RANK #{rank}: {c['or_minutes']}m ORB (Cap {c['atr_exhaustion_ratio']:.2f} | Noise {c['min_noise_pts']} pts | Retest: {c['allow_retest']})")
        print(f"   Targets: TP1 @ {c['tp1_r']}R ({c['scale_out_pct']*100:.0f}%), TP2 @ {c['tp2_r']}R (Runner)")
        print(f"   Full 8.9y PnL: ${r['full_pnl']:,.2f} | Win Rate: {r['full_wr']:.1%} | Profit Factor: {r['full_pf']:.2f}")
        print(f"   In-Sample PnL: ${r['train_pnl']:,.0f} | Out-of-Sample PnL (2023-2026): ${r['oos_pnl']:,.0f} (WR: {r['oos_wr']:.1%}, PF: {r['oos_pf']:.2f})")
        print(f"   --- Tradeify $50K Challenge Monte Carlo (1,500 Bootstrap Paths) ---")
        print(f"   • 2 MNQ: Pass Rate: {s2['pass_rate']:.1%} | Avg Days: {s2['avg_days']:.1f}d | Cost: ${s2['cost']:.2f} | Vel: ${s2['velocity']:.2f}/d")
        print(f"   • 3 MNQ: Pass Rate: {s3['pass_rate']:.1%} | Avg Days: {s3['avg_days']:.1f}d | Cost: ${s3['cost']:.2f} | Vel: ${s3['velocity']:.2f}/d")
        print(f"   • 4 MNQ: Pass Rate: {s4['pass_rate']:.1%} | Avg Days: {s4['avg_days']:.1f}d | Cost: ${s4['cost']:.2f} | Vel: ${s4['velocity']:.2f}/d\n")


if __name__ == "__main__":
    execute_conti_quant_optimization()
