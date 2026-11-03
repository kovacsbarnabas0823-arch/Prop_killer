"""
High-Speed Precompiled Institutional Quant ORB Optimization Engine
Matteo Conti / Quantitative Research Protocol: High Win-Rate, Smaller R:R, Zero Overnight
Target: Tradeify $50K Growth EOD Trailing Drawdown ($3K Target, $2K Barrier)
"""

from __future__ import annotations
import sys
import time
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
class PrecompiledSession:
    idx: int
    date: Any
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    vwaps: np.ndarray
    minutes: np.ndarray
    is_before_1115: np.ndarray
    is_before_1130: np.ndarray
    daily_atr: float
    open_price: float
    daily_20_ema: float
    long_htf: bool
    short_htf: bool


def precompile_dataset() -> List[PrecompiledSession]:
    print("Loading continuous CME NQ parquet dataset...")
    df_nq = load_continuous_nq_parquet(filter_rth=False)
    raw_sessions = extract_daily_trading_sessions(df_nq, rth_start="09:30", rth_end="15:45")
    
    # 1. 14-day rolling ATR
    true_ranges = []
    for i, s in enumerate(raw_sessions):
        h = s["daily_high"]
        l = s["daily_low"]
        if i > 0:
            prev_c = raw_sessions[i - 1]["daily_close"]
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        else:
            tr = h - l
        true_ranges.append(tr)
        
    daily_atrs = []
    for i in range(len(raw_sessions)):
        atr = np.mean(true_ranges[max(0, i-14):i]) if i >= 1 else 150.0
        daily_atrs.append(float(atr))

    # 2. Precompile into high-speed numpy arrays
    precompiled = []
    for i, s in enumerate(raw_sessions):
        rth_df = s["rth_bars"]
        if len(rth_df) < 60:
            continue
            
        opens = rth_df["Open"].to_numpy(dtype=np.float64)
        highs = rth_df["High"].to_numpy(dtype=np.float64)
        lows = rth_df["Low"].to_numpy(dtype=np.float64)
        closes = rth_df["Close"].to_numpy(dtype=np.float64)
        volumes = rth_df["Volume"].to_numpy(dtype=np.float64)
        
        # Cumulative VWAP from 09:30
        typical = (highs + lows + closes) / 3.0
        cum_vol = np.cumsum(volumes)
        cum_vp = np.cumsum(typical * volumes)
        vwaps = cum_vp / np.maximum(cum_vol, 1.0)
        
        times = rth_df.index
        minutes = np.array([t.minute for t in times], dtype=np.int32)
        times_only = np.array([t.time() for t in times])
        t_1115 = pd.to_datetime("11:15").time()
        t_1130 = pd.to_datetime("11:30").time()
        is_before_1115 = times_only <= t_1115
        is_before_1130 = times_only <= t_1130
        
        open_p = float(s.get("exchange_open_price", 0.0))
        d_ema = float(s.get("daily_20_ema", 0.0))
        long_htf = True if (d_ema <= 0 or open_p > d_ema) else False
        short_htf = True if (d_ema <= 0 or open_p < d_ema) else False
        
        precompiled.append(PrecompiledSession(
            idx=i,
            date=s["date"],
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            vwaps=vwaps,
            minutes=minutes,
            is_before_1115=is_before_1115,
            is_before_1130=is_before_1130,
            daily_atr=daily_atrs[i],
            open_price=open_p,
            daily_20_ema=d_ema,
            long_htf=long_htf,
            short_htf=short_htf
        ))
        
    print(f"Precompiled {len(precompiled):,} sessions into high-speed memory buffers.")
    return precompiled


def fast_simulate_single_model(
    sessions: List[PrecompiledSession],
    or_minutes: int,             # 15 or 30
    atr_exhaustion_ratio: float, # 0.40, 0.50, 0.60
    min_noise_pts: float,        # 8.0, 15.0
    tp1_r: float,                # 0.8, 1.0, 1.2
    tp2_r: float,                # 1.2, 1.5, 2.0
    scale_out_pct: float,        # 0.5, 0.7, 1.0
    allow_retest: bool,          # True / False
    contracts: int = 2
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    n_sessions = len(sessions)
    daily_pnls = np.zeros(n_sessions, dtype=np.float64)
    trades = []
    roundtrip_comm = 2.0 * COMMISSION_PER_CONTRACT * contracts

    for s_idx, sess in enumerate(sessions):
        opens = sess.opens
        highs = sess.highs
        lows = sess.lows
        closes = sess.closes
        vwaps = sess.vwaps
        minutes = sess.minutes
        is_before_1115 = sess.is_before_1115
        is_before_1130 = sess.is_before_1130
        n_bars = len(opens)
        
        if n_bars < or_minutes + 10:
            continue
            
        # 1. Opening Range definition
        or_high = np.max(highs[:or_minutes])
        or_low = np.min(lows[:or_minutes])
        or_mid = round((or_high + or_low) / 2.0, 2)
        or_range = round(or_high - or_low, 2)
        
        if or_range < min_noise_pts:
            continue
        if sess.daily_atr > 0 and (or_range > atr_exhaustion_ratio * sess.daily_atr):
            continue
            
        # 2. Breakout and Retest evaluation
        entered = False
        direction = 0
        entry_price = 0.0
        trade_type = ""
        entry_bar_idx = -1
        
        long_breakout_seen = False
        short_breakout_seen = False
        
        for bar_i in range(or_minutes, n_bars):
            if not is_before_1130[bar_i]:
                break
                
            c_p = closes[bar_i]
            h_p = highs[bar_i]
            l_p = lows[bar_i]
            vwap_now = vwaps[bar_i]
            
            if c_p > or_high:
                long_breakout_seen = True
            if c_p < or_low:
                short_breakout_seen = True
                
            # PRIMARY BREAKOUT: 3-min confirmation
            if (minutes[bar_i] + 1) % 3 == 0:
                if sess.long_htf and c_p > or_high and c_p > vwap_now:
                    entry_price = c_p + SLIPPAGE_POINTS
                    direction = 1
                    trade_type = "BREAKOUT"
                    entry_bar_idx = bar_i
                    entered = True
                    break
                elif sess.short_htf and c_p < or_low and c_p < vwap_now:
                    entry_price = c_p - SLIPPAGE_POINTS
                    direction = -1
                    trade_type = "BREAKOUT"
                    entry_bar_idx = bar_i
                    entered = True
                    break
                    
            # SECONDARY RETEST
            if allow_retest and is_before_1115[bar_i]:
                if long_breakout_seen and sess.long_htf and l_p <= or_high and c_p > or_mid:
                    entry_price = or_high + SLIPPAGE_POINTS
                    direction = 1
                    trade_type = "RETEST"
                    entry_bar_idx = bar_i
                    entered = True
                    break
                elif short_breakout_seen and sess.short_htf and h_p >= or_low and c_p < or_mid:
                    entry_price = or_low - SLIPPAGE_POINTS
                    direction = -1
                    trade_type = "RETEST"
                    entry_bar_idx = bar_i
                    entered = True
                    break
                    
        if not entered:
            continue
            
        # 3. Stop loss & target construction
        raw_risk = abs(entry_price - or_mid)
        risk_pts = min(max(raw_risk, 10.0), 40.0)
        
        initial_sl = (entry_price - risk_pts) if direction == 1 else (entry_price + risk_pts)
        target_tp1 = (entry_price + tp1_r * risk_pts) if direction == 1 else (entry_price - tp1_r * risk_pts)
        target_tp2 = (entry_price + tp2_r * risk_pts) if direction == 1 else (entry_price - tp2_r * risk_pts)
        
        current_sl = initial_sl
        tp1_hit = False
        exit_price_leg1 = None
        exit_price_leg2 = None
        exit_reason = None
        
        for bar_i in range(entry_bar_idx, n_bars):
            b_o = opens[bar_i]
            b_h = highs[bar_i]
            b_l = lows[bar_i]
            
            sl_hit = (b_l <= current_sl) if direction == 1 else (b_h >= current_sl)
            
            if not tp1_hit:
                hit_tp1 = (b_h >= target_tp1) if direction == 1 else (b_l <= target_tp1)
                if sl_hit and hit_tp1:
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
                    be_level = (entry_price + 0.50) if direction == 1 else (entry_price - 0.50)
                    current_sl = be_level
                    if scale_out_pct >= 0.99:
                        exit_price_leg2 = exit_price_leg1
                        exit_reason = "FULL_TP1"
                        break
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
                    
        # EOD flatten at 15:45
        if exit_price_leg1 is None:
            last_c = closes[-1]
            exit_price_leg1 = last_c - SLIPPAGE_POINTS if direction == 1 else last_c + SLIPPAGE_POINTS
            exit_price_leg2 = exit_price_leg1
            exit_reason = "FORCE_CLOSE_EOD"
        elif exit_price_leg2 is None:
            last_c = closes[-1]
            exit_price_leg2 = last_c - SLIPPAGE_POINTS if direction == 1 else last_c + SLIPPAGE_POINTS
            exit_reason = "FORCE_CLOSE_EOD_RUNNER"
            
        c_leg1 = contracts * scale_out_pct
        c_leg2 = contracts * (1.0 - scale_out_pct)
        pts1 = (exit_price_leg1 - entry_price) if direction == 1 else (entry_price - exit_price_leg1)
        pts2 = (exit_price_leg2 - entry_price) if direction == 1 else (entry_price - exit_price_leg2)
        gross_dollars = (pts1 * c_leg1 * POINT_VALUE_MNQ) + (pts2 * c_leg2 * POINT_VALUE_MNQ)
        net_dollars = gross_dollars - roundtrip_comm
        
        daily_pnls[s_idx] = net_dollars
        trades.append({
            "session_idx": s_idx,
            "date": sess.date,
            "trade_type": trade_type,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": round((exit_price_leg1 + exit_price_leg2) / 2.0, 2),
            "pnl_net": net_dollars,
            "exit_reason": exit_reason
        })
        
    return daily_pnls, trades


def fast_monte_carlo(daily_pnls: np.ndarray, n_sims: int = 1500) -> Tuple[float, float, float, float]:
    n_samples = len(daily_pnls)
    if n_samples == 0 or np.sum(daily_pnls) <= 0:
        return 0.0, 999.0, 9999.0, -99.0

    passes = 0
    days_to_pass_list = []
    trailing_dd_limit = 2000.0
    daily_loss_limit = 1250.0
    profit_target = 3000.0
    max_days = 150
    block_size = 5

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
        floor = 48000.0
        passed = False

        for day_idx, pnl in enumerate(path_pnls):
            if pnl <= -daily_loss_limit:
                break
            balance += pnl
            if balance <= floor:
                break
            if balance > peak:
                peak = balance
                new_floor = min(peak - trailing_dd_limit, 50100.0)
                if new_floor > floor:
                    floor = new_floor
            if balance >= 53000.0:
                passed = True
                days_to_pass_list.append(day_idx + 1)
                break

        if passed:
            passes += 1

    pass_rate = passes / n_sims
    avg_days = float(np.mean(days_to_pass_list)) if len(days_to_pass_list) > 0 else float(max_days)
    if pass_rate > 0.01:
        expected_cost = 87.0 + ((1.0 / pass_rate) - 1.0) * 95.0
        velocity = (profit_target - expected_cost) / avg_days
    else:
        expected_cost = 9999.0
        velocity = -99.0

    return pass_rate, avg_days, expected_cost, velocity


def run():
    t0 = time.time()
    sessions = precompile_dataset()
    split_date = pd.to_datetime("2023-01-01").date()
    train_sessions = [s for s in sessions if pd.to_datetime(s.date).date() < split_date]
    test_sessions = [s for s in sessions if pd.to_datetime(s.date).date() >= split_date]
    
    print(f"Data split: Train (2018-2022) = {len(train_sessions)} sessions, Test (2023-2026) = {len(test_sessions)} sessions\n")

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

    print(f"Sweeping {len(configs)} configurations on Train dataset...")
    train_results = []
    for cfg in configs:
        daily_pnls, trades = fast_simulate_single_model(train_sessions, **cfg, contracts=2)
        n_tr = len(trades)
        if n_tr < 30: continue
        tot = np.sum(daily_pnls)
        wins = [t["pnl_net"] for t in trades if t["pnl_net"] > 0]
        losses = [abs(t["pnl_net"]) for t in trades if t["pnl_net"] <= 0]
        wr = len(wins) / n_tr
        pf = sum(wins) / sum(losses) if sum(losses) > 0 else 0.0
        cum = np.cumsum(daily_pnls)
        max_dd = np.max(np.maximum.accumulate(cum) - cum)
        std = np.std(daily_pnls)
        sharpe = (np.mean(daily_pnls) / std * np.sqrt(252)) if std > 0 else 0.0

        train_results.append({
            "cfg": cfg,
            "n_trades": n_tr,
            "train_pnl": tot,
            "train_wr": wr,
            "train_pf": pf,
            "train_max_dd": max_dd,
            "train_sharpe": sharpe
        })

    df_train = pd.DataFrame(train_results)
    df_train = df_train.sort_values(by="train_pnl", ascending=False)
    print(f"Top 10 In-Sample Models by Net PnL (Train: 2018-2022):\n")
    for i, r in df_train.head(10).iterrows():
        c = r["cfg"]
        print(f"[{c['or_minutes']}m ORB | Cap {c['atr_exhaustion_ratio']} | Noise {c['min_noise_pts']} | Retest: {c['allow_retest']}] TP1: {c['tp1_r']}R ({c['scale_out_pct']*100:.0f}%), TP2: {c['tp2_r']}R -> PnL: ${r['train_pnl']:,.0f} | WR: {r['train_wr']:.1%} | PF: {r['train_pf']:.2f} | MaxDD: ${r['train_max_dd']:,.0f} | Sharpe: {r['train_sharpe']:.2f}")

    if df_train.empty:
        return

    df_train = df_train.sort_values(by="train_sharpe", ascending=False)
    
    # Evaluate Out-of-sample and Tradeify Challenge
    final_evals = []
    for _, row in df_train.head(25).iterrows():
        cfg = row["cfg"]
        oos_pnls, oos_trades = fast_simulate_single_model(test_sessions, **cfg, contracts=2)
        n_oos = len(oos_trades)
        tot_oos = np.sum(oos_pnls)
        wins_oos = [t["pnl_net"] for t in oos_trades if t["pnl_net"] > 0]
        losses_oos = [abs(t["pnl_net"]) for t in oos_trades if t["pnl_net"] <= 0]
        wr_oos = len(wins_oos) / n_oos if n_oos > 0 else 0.0
        pf_oos = sum(wins_oos) / sum(losses_oos) if sum(losses_oos) > 0 else 0.0

        full_pnls, full_trades = fast_simulate_single_model(sessions, **cfg, contracts=2)
        tot_full = np.sum(full_pnls)
        wins_full = [t["pnl_net"] for t in full_trades if t["pnl_net"] > 0]
        losses_full = [abs(t["pnl_net"]) for t in full_trades if t["pnl_net"] <= 0]
        wr_full = len(wins_full) / len(full_trades) if len(full_trades) > 0 else 0.0
        pf_full = sum(wins_full) / sum(losses_full) if sum(losses_full) > 0 else 0.0

        cum_f = np.cumsum(full_pnls)
        full_max_dd = np.max(np.maximum.accumulate(cum_f) - cum_f)

        # Sizing sweep for Tradeify 2, 3, 4 MNQ
        sizing = {}
        for k in [2, 3, 4]:
            k_pnls = full_pnls * (k / 2.0)
            p_rate, t_pass, cost, vel = fast_monte_carlo(k_pnls, n_sims=1500)
            sizing[k] = {"pass_rate": p_rate, "avg_days": t_pass, "cost": cost, "velocity": vel}

        final_evals.append({
            "cfg": cfg,
            "train_pnl": row["train_pnl"],
            "oos_pnl": tot_oos,
            "oos_wr": wr_oos,
            "oos_pf": pf_oos,
            "full_pnl": tot_full,
            "full_wr": wr_full,
            "full_pf": pf_full,
            "full_max_dd": full_max_dd,
            "sizing": sizing
        })

    df_final = pd.DataFrame(final_evals)
    df_final["vel_3mnq"] = df_final["sizing"].apply(lambda s: s[3]["velocity"])
    df_final = df_final.sort_values(by="vel_3mnq", ascending=False)

    print("=========================================================================")
    print("TOP CANDIDATES RANKED BY CAPITAL VELOCITY (FAST PASS + HIGH WR):")
    print("=========================================================================\n")

    for rank, (i, r) in enumerate(df_final.head(5).iterrows(), start=1):
        c = r["cfg"]
        s2 = r["sizing"][2]
        s3 = r["sizing"][3]
        s4 = r["sizing"][4]
        print(f"RANK #{rank}: {c['or_minutes']}m ORB (Cap {c['atr_exhaustion_ratio']:.2f} | Noise {c['min_noise_pts']} pts | Retest: {c['allow_retest']})")
        print(f"   Targets: TP1 @ {c['tp1_r']}R ({c['scale_out_pct']*100:.0f}%), TP2 @ {c['tp2_r']}R (Runner)")
        print(f"   Full History (8.9y): PnL: ${r['full_pnl']:,.2f} | Win Rate: {r['full_wr']:.1%} | Profit Factor: {r['full_pf']:.2f} | MaxDD (2 MNQ): ${r['full_max_dd']:,.2f}")
        print(f"   In-Sample: ${r['train_pnl']:,.0f} | Out-of-Sample (2023-2026): ${r['oos_pnl']:,.0f} (WR: {r['oos_wr']:.1%}, PF: {r['oos_pf']:.2f})")
        print(f"   --- Tradeify $50K Challenge Evaluation (1,500 Bootstrap Paths) ---")
        print(f"   • 2 MNQ: Pass Rate: {s2['pass_rate']:.1%} | Avg Days: {s2['avg_days']:.1f}d | Cost: ${s2['cost']:.2f} | Vel: ${s2['velocity']:.2f}/d")
        print(f"   • 3 MNQ: Pass Rate: {s3['pass_rate']:.1%} | Avg Days: {s3['avg_days']:.1f}d | Cost: ${s3['cost']:.2f} | Vel: ${s3['velocity']:.2f}/d")
        print(f"   • 4 MNQ: Pass Rate: {s4['pass_rate']:.1%} | Avg Days: {s4['avg_days']:.1f}d | Cost: ${s4['cost']:.2f} | Vel: ${s4['velocity']:.2f}/d\n")

    print(f"Finished in {time.time() - t0:.1f} seconds.")


if __name__ == "__main__":
    run()
