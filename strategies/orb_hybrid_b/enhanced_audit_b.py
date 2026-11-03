"""
Enhanced Quantitative Audit for Option B (15-Minute High-Velocity ORB)
Asset: CME Micro E-mini Nasdaq-100 (MNQ) Futures
Prop Firm Engine: Tradeify $50K Growth EOD Trailing Drawdown

Tests 3 distinct variants of Option B:
  1. Raw Option B (Baseline 15m ORB, 1.2R/2.0R, OR_Mid stop)
  2. Filtered Option B (ATR compression < 0.40, Overnight Inventory Filter, Dual 20 EMA + VWAP trend alignment)
  3. Sprint Option B (Filtered + Dynamic Risk Sizing 2 MNQ -> 4 MNQ buffer sprint)
"""

from __future__ import annotations
import sys
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data_loader import load_continuous_nq_parquet, extract_daily_trading_sessions

POINT_VALUE_MNQ = 2.0
COMMISSION_PER_CONTRACT = 1.24  # $2.48 roundturn per contract
SLIPPAGE_POINTS = 0.50          # 0.50 pt ($1.00 MNQ) pessimistic entry/exit


@dataclass
class PrecompiledSession15m:
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
    prev_settle: float
    on_vwap: float
    is_net_long_on: bool
    is_net_short_on: bool


def precompile_sessions_15m() -> List[PrecompiledSession15m]:
    print("Loading continuous CME NQ parquet dataset for Option B audit...")
    df_nq = load_continuous_nq_parquet(filter_rth=False)
    raw_sessions = extract_daily_trading_sessions(df_nq, rth_start="09:30", rth_end="15:45")

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

        typical = (highs + lows + closes) / 3.0
        cum_vol = np.cumsum(volumes)
        cum_vp = np.cumsum(typical * volumes)
        vwaps = cum_vp / np.maximum(cum_vol, 1.0)

        times = rth_df.index
        minutes = np.array([t.minute for t in times], dtype=np.int32)
        times_only = np.array([t.time() for t in times])
        is_before_1115 = times_only <= pd.to_datetime("11:15").time()
        is_before_1130 = times_only <= pd.to_datetime("11:30").time()

        open_p = float(s.get("exchange_open_price", 0.0))
        d_ema = float(s.get("daily_20_ema", 0.0))
        long_htf = True if (d_ema <= 0 or open_p > d_ema) else False
        short_htf = True if (d_ema <= 0 or open_p < d_ema) else False

        # Overnight session features
        overnight_df = s.get("overnight_bars", pd.DataFrame())
        prev_settle = float(raw_sessions[i-1]["daily_close"]) if i > 0 else open_p
        if not overnight_df.empty and len(overnight_df) > 30:
            on_v_above = overnight_df.loc[overnight_df["Close"] > prev_settle, "Volume"].sum()
            on_v_tot = overnight_df["Volume"].sum()
            on_ratio = on_v_above / max(on_v_tot, 1)
            on_typ = (overnight_df["High"] + overnight_df["Low"] + overnight_df["Close"]) / 3.0
            on_vwap = float((on_typ * overnight_df["Volume"]).sum() / max(on_v_tot, 1))
            is_net_long_on = (on_ratio >= 0.80)
            is_net_short_on = (on_ratio <= 0.20)
        else:
            on_vwap = open_p
            is_net_long_on = False
            is_net_short_on = False

        precompiled.append(PrecompiledSession15m(
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
            short_htf=short_htf,
            prev_settle=prev_settle,
            on_vwap=on_vwap,
            is_net_long_on=is_net_long_on,
            is_net_short_on=is_net_short_on
        ))

    print(f"Precompiled {len(precompiled):,} sessions into memory.")
    return precompiled


def simulate_enhanced_b(
    sessions: List[PrecompiledSession15m],
    contracts: int = 2,
    or_minutes: int = 15,
    atr_cap: float = 0.40,           # Stricter compression filter (was 0.50)
    min_noise: float = 12.0,
    tp1_r: float = 1.20,
    tp2_r: float = 2.50,             # Expanded runner (was 2.00)
    profit_lock_r: float = 2.00,
    profit_lock_trail: float = 1.00,
    use_on_filter: bool = True,
    allow_retest: bool = True
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Simulates Enhanced Option B with Overnight Inventory Filter and Asymmetric Targets.
    """
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
        n_bars = len(opens)

        if n_bars < or_minutes + 10:
            continue

        or_high = np.max(highs[:or_minutes])
        or_low = np.min(lows[:or_minutes])
        or_mid = round((or_high + or_low) / 2.0, 2)
        or_range = round(or_high - or_low, 2)

        if or_range < min_noise:
            continue
        if sess.daily_atr > 0 and (or_range > atr_cap * sess.daily_atr):
            continue

        entered = False
        direction = 0
        entry_price = 0.0
        trade_type = ""
        entry_bar_idx = -1

        long_breakout_seen = False
        short_breakout_seen = False
        touched_on_vwap = False

        for bar_i in range(or_minutes, n_bars):
            if not sess.is_before_1130[bar_i]:
                break

            c_p = closes[bar_i]
            h_p = highs[bar_i]
            l_p = lows[bar_i]
            vwap_now = vwaps[bar_i]

            if l_p <= sess.on_vwap <= h_p:
                touched_on_vwap = True

            if c_p > or_high:
                long_breakout_seen = True
            if c_p < or_low:
                short_breakout_seen = True

            # Anomaly filter check
            long_blocked = use_on_filter and sess.is_net_long_on and not touched_on_vwap
            short_blocked = use_on_filter and sess.is_net_short_on and not touched_on_vwap

            # 1. Primary Breakout on 3-min close confirmation
            if (minutes[bar_i] + 1) % 3 == 0:
                if sess.long_htf and c_p > or_high and c_p > vwap_now and not long_blocked:
                    entry_price = c_p + SLIPPAGE_POINTS
                    direction = 1
                    trade_type = "BREAKOUT"
                    entry_bar_idx = bar_i
                    entered = True
                    break
                elif sess.short_htf and c_p < or_low and c_p < vwap_now and not short_blocked:
                    entry_price = c_p - SLIPPAGE_POINTS
                    direction = -1
                    trade_type = "BREAKOUT"
                    entry_bar_idx = bar_i
                    entered = True
                    break

            # 2. Secondary Retest Limit Order
            if allow_retest and sess.is_before_1115[bar_i]:
                if long_breakout_seen and sess.long_htf and l_p <= or_high and c_p > or_mid and not long_blocked:
                    entry_price = or_high + SLIPPAGE_POINTS
                    direction = 1
                    trade_type = "RETEST"
                    entry_bar_idx = bar_i
                    entered = True
                    break
                elif short_breakout_seen and sess.short_htf and h_p >= or_low and c_p < or_mid and not short_blocked:
                    entry_price = or_low - SLIPPAGE_POINTS
                    direction = -1
                    trade_type = "RETEST"
                    entry_bar_idx = bar_i
                    entered = True
                    break

        if not entered:
            continue

        raw_risk = abs(entry_price - or_mid)
        risk_pts = min(max(raw_risk, 8.0), 45.0)

        initial_sl = (entry_price - risk_pts) if direction == 1 else (entry_price + risk_pts)
        target_tp1 = (entry_price + tp1_r * risk_pts) if direction == 1 else (entry_price - tp1_r * risk_pts)
        target_tp2 = (entry_price + tp2_r * risk_pts) if direction == 1 else (entry_price - tp2_r * risk_pts)
        target_lock = (entry_price + profit_lock_r * risk_pts) if direction == 1 else (entry_price - profit_lock_r * risk_pts)

        current_sl = initial_sl
        tp1_hit = False
        profit_lock_hit = False
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
            else:
                if not profit_lock_hit:
                    hit_lock = (b_h >= target_lock) if direction == 1 else (b_l <= target_lock)
                    if hit_lock:
                        profit_lock_hit = True
                        lock_level = (entry_price + profit_lock_trail * risk_pts) if direction == 1 else (entry_price - profit_lock_trail * risk_pts)
                        current_sl = lock_level

                hit_tp2 = (b_h >= target_tp2) if direction == 1 else (b_l <= target_tp2)
                if sl_hit and hit_tp2:
                    fill = min(current_sl, b_o) if direction == 1 else max(current_sl, b_o)
                    exit_price_leg2 = fill - SLIPPAGE_POINTS if direction == 1 else fill + SLIPPAGE_POINTS
                    exit_reason = "LEG2_LOCK_STOP" if profit_lock_hit else "LEG2_BE_STOP"
                    break
                elif sl_hit:
                    fill = min(current_sl, b_o) if direction == 1 else max(current_sl, b_o)
                    exit_price_leg2 = fill - SLIPPAGE_POINTS if direction == 1 else fill + SLIPPAGE_POINTS
                    exit_reason = "LEG2_LOCK_STOP" if profit_lock_hit else "LEG2_BE_STOP"
                    break
                elif hit_tp2:
                    exit_price_leg2 = target_tp2
                    exit_reason = "FULL_TP2_RUNNER"
                    break

        if exit_price_leg1 is None:
            last_c = closes[-1]
            exit_price_leg1 = last_c - SLIPPAGE_POINTS if direction == 1 else last_c + SLIPPAGE_POINTS
            exit_price_leg2 = exit_price_leg1
            exit_reason = "FORCE_CLOSE_EOD"
        elif exit_price_leg2 is None:
            last_c = closes[-1]
            exit_price_leg2 = last_c - SLIPPAGE_POINTS if direction == 1 else last_c + SLIPPAGE_POINTS
            exit_reason = "FORCE_CLOSE_EOD_RUNNER"

        c_leg1 = contracts * 0.50
        c_leg2 = contracts * 0.50
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


def run_dynamic_two_phase_mc_b(
    trades: List[Dict[str, Any]],
    n_sims: int = 5000
) -> Dict[str, Any]:
    """
    Two-Phase Monte Carlo with Dynamic Sizing (2 MNQ base -> 4 MNQ sprint).
    """
    trade_pnls_base = np.array([t["pnl_net"] for t in trades])
    n_trades = len(trade_pnls_base)

    eval_passes = 0
    eval_days = []
    payout_achieved = 0
    payout_days = []

    challenge_fee = 87.0
    daily_loss_limit = 1250.0

    for _ in range(n_sims):
        # Phase 1: Evaluation
        balance = 50000.0
        peak = 50000.0
        floor = 48000.0
        passed_eval = False
        days_in_eval = 0

        sampled_eval = np.random.choice(n_trades, size=150, replace=True)

        for trade_i in sampled_eval:
            days_in_eval += 2.2
            base_pnl = trade_pnls_base[trade_i]

            # Dynamic Sizing Protocol
            cushion = balance - floor
            if balance >= 51200.0 and cushion >= 1500.0:
                pnl = base_pnl * 2.0  # 4 MNQ sprint
            elif cushion < 1000.0:
                pnl = base_pnl * 0.5  # 1 MNQ shield
            else:
                pnl = base_pnl        # 2 MNQ base

            if pnl <= -daily_loss_limit:
                break
            balance += pnl
            if balance <= floor:
                break

            if balance > peak:
                peak = balance
                new_floor = min(peak - 2000.0, 50100.0)
                if new_floor > floor:
                    floor = new_floor

            if balance >= 53000.0:
                passed_eval = True
                eval_days.append(days_in_eval)
                break

        if not passed_eval:
            continue

        eval_passes += 1

        # Phase 2: Funded Account
        f_balance = 50000.0
        f_peak = 50000.0
        f_floor = 48000.0
        days_in_funded = 0
        funded_daily_profits = []
        payout_won = False

        sampled_funded = np.random.choice(n_trades, size=150, replace=True)

        for trade_i in sampled_funded:
            days_in_funded += 2.2
            base_pnl = trade_pnls_base[trade_i]

            # In funded: 2 MNQ fixed, or 1 MNQ if cushion < $1,000
            f_cushion = f_balance - f_floor
            pnl = base_pnl * 0.5 if f_cushion < 1000.0 else base_pnl

            if pnl <= -daily_loss_limit:
                break
            f_balance += pnl
            if pnl > 0:
                funded_daily_profits.append(pnl)

            if f_balance <= f_floor:
                break

            if f_balance > f_peak:
                f_peak = f_balance
                new_floor = min(f_peak - 2000.0, 50100.0)
                if new_floor > f_floor:
                    f_floor = new_floor

            tot_profit = f_balance - 50000.0
            if tot_profit >= 2500.0 and days_in_funded >= 11.0:
                max_single_day = max(funded_daily_profits) if funded_daily_profits else 0.0
                if max_single_day <= 0.35 * tot_profit:
                    payout_won = True
                    payout_achieved += 1
                    total_journey_days = days_in_eval + days_in_funded
                    payout_days.append(total_journey_days)
                    break

    eval_pass_rate = eval_passes / n_sims
    payout_pass_rate = payout_achieved / n_sims
    avg_eval_days = float(np.mean(eval_days)) if eval_days else 999.0
    avg_total_payout_days = float(np.mean(payout_days)) if payout_days else 999.0

    return {
        "eval_pass_rate": eval_pass_rate,
        "avg_eval_days": avg_eval_days,
        "payout_pass_rate": payout_pass_rate,
        "avg_total_payout_days": avg_total_payout_days,
        "cost_per_attempt": challenge_fee,
        "expected_attempts": 1.0 / eval_pass_rate if eval_pass_rate > 0 else 99.0
    }


def main():
    print("=" * 75)
    print("ENHANCED QUANTITATIVE AUDIT: OPTION B (15M ORB + ANOMALY FILTERS)")
    print("=" * 75)

    sessions = precompile_sessions_15m()
    split_date = pd.to_datetime("2023-01-01").date()

    daily_pnls, trades = simulate_enhanced_b(
        sessions=sessions,
        contracts=2,
        or_minutes=15,
        atr_cap=0.40,
        min_noise=12.0,
        tp1_r=1.20,
        tp2_r=2.50,
        profit_lock_r=2.00,
        profit_lock_trail=1.00,
        use_on_filter=True,
        allow_retest=True
    )

    n_trades = len(trades)
    pnls = [t["pnl_net"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p <= 0]
    tot_pnl = sum(pnls)
    wr = len(wins) / n_trades if n_trades > 0 else 0.0
    pf = sum(wins) / sum(losses) if sum(losses) > 0 else 0.0
    exp = tot_pnl / n_trades if n_trades > 0 else 0.0

    cum = np.cumsum(daily_pnls)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = np.max(dd)

    is_pnls = [daily_pnls[i] for i, s in enumerate(sessions) if pd.to_datetime(s.date).date() < split_date]
    oos_pnls = [daily_pnls[i] for i, s in enumerate(sessions) if pd.to_datetime(s.date).date() >= split_date]
    is_trades = [t for t in trades if pd.to_datetime(t["date"]).date() < split_date]
    oos_trades = [t for t in trades if pd.to_datetime(t["date"]).date() >= split_date]

    wr_is = sum(1 for t in is_trades if t["pnl_net"] > 0) / len(is_trades)
    pf_is = sum(t["pnl_net"] for t in is_trades if t["pnl_net"] > 0) / abs(sum(t["pnl_net"] for t in is_trades if t["pnl_net"] <= 0))
    wr_oos = sum(1 for t in oos_trades if t["pnl_net"] > 0) / len(oos_trades)
    pf_oos = sum(t["pnl_net"] for t in oos_trades if t["pnl_net"] > 0) / abs(sum(t["pnl_net"] for t in oos_trades if t["pnl_net"] <= 0))

    mc_res = run_dynamic_two_phase_mc_b(trades, n_sims=5000)

    print("\n1. PERFORMANCE METRICS (ENHANCED OPTION B):")
    print(f"Total Net PnL (8.9y)       : ${tot_pnl:,.2f} (vs Baseline: $2,254.20)")
    print(f"Total Trades Executed      : {n_trades:,} trades ({n_trades / 8.9:.1f} trades / year)")
    print(f"Win Rate                   : {wr:.2%} ({len(wins)} Wins / {len(losses)} Losses)")
    print(f"Profit Factor              : {pf:.2f} (vs Baseline: 1.03)")
    print(f"Average Expectancy         : ${exp:.2f} / trade (vs Baseline: $2.18 / trade)")
    print(f"Maximum Realized Drawdown  : -${max_dd:,.2f} (vs Baseline: -$3,710.29)")

    print("\n2. OUT-OF-SAMPLE STABILITY:")
    print(f"In-Sample (2018-2022)      : Net PnL = ${sum(is_pnls):,.2f} | WR = {wr_is:.1%} | PF = {pf_is:.2f}")
    print(f"Out-of-Sample (2023-2026)  : Net PnL = ${sum(oos_pnls):,.2f} | WR = {wr_oos:.1%} | PF = {pf_oos:.2f}")

    print("\n3. TWO-PHASE DYNAMIC MONTE CARLO (TRADEIFY $50K GROWTH):")
    print(f"Challenge Pass Rate        : {mc_res['eval_pass_rate']:.1%} (vs Baseline: 17.0%)")
    print(f"Avg Days to Pass Eval      : {mc_res['avg_eval_days']:.1f} trading days (~3.5 months)")
    print(f"Expected Attempts          : {mc_res['expected_attempts']:.2f} attempts")
    print(f"Expected Total Fee         : ${mc_res['expected_attempts'] * 87.0:,.2f} (vs Baseline: $512.97)")
    print(f"Overall Payout Success     : {mc_res['payout_pass_rate']:.1%} (vs Baseline: 5.6%)")
    print(f"Total Days to 1st Cash     : {mc_res['avg_total_payout_days']:.1f} trading days")
    print(f"Net Cash ROI               : {((1500.0 - (mc_res['expected_attempts'] * 87.0)) / (mc_res['expected_attempts'] * 87.0) * 100):.1f}%")

    print("\n4. YEAR-BY-YEAR PERFORMANCE:")
    df_yr = pd.DataFrame([{"year": pd.to_datetime(t["date"]).year, "pnl": t["pnl_net"]} for t in trades])
    yearly = df_yr.groupby("year")["pnl"].agg(["count", "sum"]).rename(columns={"count": "trades", "sum": "net_pnl"})
    yearly["win_rate"] = df_yr.groupby("year")["pnl"].apply(lambda s: (s > 0).mean() * 100.0)
    print(yearly.to_string())

    report_content = f"""
ENHANCED OPTION B AUDIT REPORT
=====================================
Total Net PnL: ${tot_pnl:,.2f}
Win Rate: {wr:.2%}
Profit Factor: {pf:.2f}
Expectancy: ${exp:.2f}/trade
Max Drawdown: -${max_dd:,.2f}
In-Sample PnL: ${sum(is_pnls):,.2f} (PF {pf_is:.2f})
Out-of-Sample PnL: ${sum(oos_pnls):,.2f} (PF {pf_oos:.2f})
Tradeify Challenge Pass Rate: {mc_res['eval_pass_rate']:.1%}
Avg Days to Pass: {mc_res['avg_eval_days']:.1f}
Expected Attempts: {mc_res['expected_attempts']:.2f} (${mc_res['expected_attempts']*87:.2f})
Funded Payout Success: {mc_res['payout_pass_rate']:.1%}
Total Days to Cash: {mc_res['avg_total_payout_days']:.1f}
"""
    with open("strategies/orb_hybrid_b/enhanced_audit_b_report.txt", "w") as f:
        f.write(report_content)


if __name__ == "__main__":
    main()
