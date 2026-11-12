"""
Phase 6 Frequency Expansion Ablation Grid (experiments/ablation_phase6_frequency.py)
Author: Principal Quantitative Researcher & Lead Financial Engineer

Ablation Suite:
  Part 1: H4 Dual-Checkpoint (15:00 & 15:15 NY) & Threshold Calibration
  Part 2: H3 Frequency Unlocks:
          - Volume Filter (True vs False)
          - Seeck Short Filter (Strict vs Surgical Bull-Trap vs None)
          - Start Window (10:15 vs 10:30)
          - Noise-Band Multiplier K in {1.00, 1.05, 1.10, 1.15, 1.20, 1.25}
          - Opposite-Band V-Reversal Capture (max_entries_per_direction = 1)
  Part 3: Concurrent Same-Direction Pyramiding at 15:00/15:15 NY (Unlocking H4 trades)
"""

from __future__ import annotations
import sys
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.session_preprocessor import get_preprocessor
from strategies.base_strategy import TradeSignal, CompletedTrade
from strategies.h3_noise_band_momentum import H3NoiseBandMomentum
from strategies.h4_letf_1500_rebalance import H4LETF1500Rebalance
from engine.pessimistic_sim import PessimisticExecutionSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase6Ablation")


def compute_metrics(trades: List[CompletedTrade]) -> Dict[str, Any]:
    if not trades:
        return {"total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "total_pnl": 0.0, "expectancy": 0.0, "realized_rr": 0.0, "max_dd": 0.0}

    pnls = np.array([t.net_pnl for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    total_trades = len(pnls)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total_trades) * 100.0 if total_trades > 0 else 0.0

    total_pnl = float(np.sum(pnls))
    gross_win = float(np.sum(wins)) if win_count > 0 else 0.0
    gross_loss = float(abs(np.sum(losses))) if loss_count > 0 else 0.0

    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)
    expectancy = total_pnl / total_trades if total_trades > 0 else 0.0
    net_avg_win = float(np.mean(wins)) if win_count > 0 else 0.0
    net_avg_loss = float(np.mean(losses)) if loss_count > 0 else 0.0
    realized_rr = (net_avg_win / abs(net_avg_loss)) if abs(net_avg_loss) > 0 else 0.0

    cum_pnl = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = running_max - cum_pnl
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "total_pnl": total_pnl,
        "realized_rr": realized_rr,
        "max_dd": max_dd
    }


def simulate_h4_variant(
    prep,
    all_dates: List[date],
    checkpoints: List[str] = ["15:00"],
    rod_thresh: float = 0.0075,
    atr_move_thresh: float = 0.60,
    range_pos_long: float = 0.78,
    range_pos_short: float = 0.22,
    contracts: int = 2
) -> List[CompletedTrade]:
    sim = PessimisticExecutionSimulator()
    completed: List[CompletedTrade] = []

    for d in all_dates:
        df_day = prep.get_session_bars(d)
        if df_day.empty:
            continue
        daily_atr = prep.daily_atr_map.get(d)
        prev_close = prep.prev_day_close_map.get(d)
        if not daily_atr or not prev_close or pd.isna(daily_atr) or pd.isna(prev_close):
            continue

        rth_bars = df_day.between_time("09:30", "15:44")
        if len(rth_bars) < 200:
            continue

        # Bar at 10:00 (r_1)
        time_1000_mask = rth_bars.index.time == pd.Timestamp("10:00:00").time()
        if not time_1000_mask.any():
            continue
        close_1000 = float(rth_bars.loc[time_1000_mask].iloc[-1]["Close"])
        r_1 = close_1000 - prev_close

        # Bar at 13:30
        time_1330_mask = rth_bars.index.time == pd.Timestamp("13:30:00").time()
        close_1330 = float(rth_bars.loc[time_1330_mask].iloc[-1]["Close"]) if time_1330_mask.any() else None

        open_0930 = float(rth_bars.iloc[0]["Open"])

        # Running VWAP
        tp = (rth_bars["High"] + rth_bars["Low"] + rth_bars["Close"]) / 3.0
        cum_vol = rth_bars["Volume"].cumsum()
        vwap_series = (tp * rth_bars["Volume"]).cumsum() / cum_vol.replace(0, np.nan)

        df_ind = pd.DataFrame({"VWAP": vwap_series}, index=rth_bars.index)

        # Check each late-day checkpoint in order (max 1 trade per day)
        day_traded = False
        for cp in checkpoints:
            if day_traded:
                break
            cp_time_mask = rth_bars.index.time == pd.Timestamp(f"{cp}:00").time()
            if not cp_time_mask.any():
                continue
            bar_cp = rth_bars.loc[cp_time_mask].iloc[-1]
            t_cp = rth_bars.loc[cp_time_mask].index[-1]
            close_cp = float(bar_cp["Close"])

            bars_to_cp = rth_bars.loc[rth_bars.index <= t_cp]
            high_cp = float(bars_to_cp["High"].max())
            low_cp = float(bars_to_cp["Low"].min())
            range_cp = high_cp - low_cp
            if range_cp <= 0:
                continue

            r_rod = (close_cp - prev_close) / prev_close
            norm_move_atr = abs(close_cp - open_0930) / daily_atr
            range_pos = (close_cp - low_cp) / range_cp
            vwap_cp = float(vwap_series.loc[t_cp])

            # Gamma pin filter
            s_50 = 50.0 * round(close_cp / 50.0)
            if abs(close_cp - s_50) <= 8.0:
                t_pin_start = (pd.Timestamp(f"{cp}:00") - pd.Timedelta(minutes=15)).strftime("%H:%M")
                pin_bars = df_day.between_time(t_pin_start, cp)
                if not pin_bars.empty and float(pin_bars["High"].max() - pin_bars["Low"].min()) < 15.0:
                    continue

            # Displacement filter
            if not (abs(r_rod) >= rod_thresh or norm_move_atr >= atr_move_thresh):
                continue

            midday_ok_long = (close_1330 is not None and close_cp >= close_1330)
            midday_ok_short = (close_1330 is not None and close_cp <= close_1330)

            future_1m = df_day.loc[df_day.index > t_cp]
            if future_1m.empty:
                continue

            sl_dist = float(np.clip(0.14 * range_cp, 16.0, 32.0))
            open_exec = float(future_1m.iloc[0]["Open"])
            t_exec = future_1m.index[0]

            sig = None
            if r_rod > 0 and r_1 > 0 and midday_ok_long and close_cp > vwap_cp and range_pos >= range_pos_long:
                entry_price = open_exec + 0.50
                sl_price = entry_price - sl_dist
                sig = TradeSignal(
                    strategy_id="H4",
                    strategy_name="H4_LETF_Rebalance",
                    timestamp=t_cp,
                    execution_time=t_exec,
                    direction=1,
                    entry_price=entry_price,
                    stop_loss_price=sl_price,
                    tp1_price=None,
                    sl_dist=sl_dist,
                    scale_out_tp1=False,
                    be_trigger_r=None,
                    trail_rule="VWAP",
                    hard_time_exit="15:44:00"
                )
            elif r_rod < 0 and r_1 < 0 and midday_ok_short and close_cp < vwap_cp and range_pos <= range_pos_short:
                entry_price = open_exec - 0.50
                sl_price = entry_price + sl_dist
                sig = TradeSignal(
                    strategy_id="H4",
                    strategy_name="H4_LETF_Rebalance",
                    timestamp=t_cp,
                    execution_time=t_exec,
                    direction=-1,
                    entry_price=entry_price,
                    stop_loss_price=sl_price,
                    tp1_price=None,
                    sl_dist=sl_dist,
                    scale_out_tp1=False,
                    be_trigger_r=None,
                    trail_rule="VWAP",
                    hard_time_exit="15:44:00"
                )

            if sig:
                c_tr = sim.simulate_trade(sig, df_day, contract_qty=contracts, df_indicators=df_ind)
                if c_tr:
                    completed.extend(c_tr)
                    day_traded = True

    return completed


def simulate_h3_variant(
    prep,
    all_dates: List[date],
    start_time: str = "10:30",
    end_time: str = "13:30",
    kd: float = 1.25,
    vol_filter: bool = True,
    seeck_mode: str = "strict",  # "strict", "surgical", "none"
    allow_v_reversal: bool = False,
    contracts: int = 2
) -> List[CompletedTrade]:
    sim = PessimisticExecutionSimulator()
    completed: List[CompletedTrade] = []

    for d in all_dates:
        df_day = prep.get_session_bars(d)
        if df_day.empty:
            continue
        daily_atr = prep.daily_atr_map.get(d)
        prev_close = prep.prev_day_close_map.get(d)
        if not daily_atr or not prev_close or pd.isna(daily_atr) or pd.isna(prev_close):
            continue

        rth_bars = df_day.between_time("09:30", "15:44")
        if len(rth_bars) < 60:
            continue

        # Anchored RTH VWAP
        tp = (rth_bars["High"] + rth_bars["Low"] + rth_bars["Close"]) / 3.0
        cum_vol = rth_bars["Volume"].cumsum()
        vwap_series = (tp * rth_bars["Volume"]).cumsum() / cum_vol.replace(0, np.nan)

        # 5m bars
        vol_bars = df_day.between_time("08:00", "15:44")
        if vol_bars.empty:
            vol_bars = rth_bars

        resampled_5m = vol_bars.resample("5min", closed="left", label="left").agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
        }).dropna()

        ema9_5m = resampled_5m["Close"].ewm(span=9, adjust=False).mean()
        vol_sma_5m = resampled_5m["Volume"].rolling(window=12, min_periods=4).mean()

        eval_window_5m = resampled_5m.between_time(start_time, end_time)
        if eval_window_5m.empty:
            continue

        rth_sub = df_day.between_time("09:30", "09:30")
        d_open = float(rth_sub.iloc[0]["Open"]) if not rth_sub.empty else prev_close
        prev_high = prep.prev_day_high_map.get(d)
        prev_low = prep.prev_day_low_map.get(d)
        sma_atr20 = prep.daily_atr_sma20_map.get(d, daily_atr)

        # Precompute indicators for simulation
        df_ind = pd.DataFrame(index=rth_bars.index)
        df_ind["VWAP"] = vwap_series
        u_bands, l_bands = [], []
        for t in rth_bars.index:
            b = prep.get_noise_band(d, t.strftime("%H:%M"), kd=kd)
            u_bands.append(b[0] if b else np.nan)
            l_bands.append(b[1] if b else np.nan)
        df_ind["UpperBand"] = pd.Series(u_bands, index=rth_bars.index).ffill()
        df_ind["LowerBand"] = pd.Series(l_bands, index=rth_bars.index).ffill()

        day_trades: List[CompletedTrade] = []
        long_taken = False
        short_taken = False
        last_exit_time = pd.Timestamp.min.tz_localize(df_day.index.tz)
        running_day_pnl = 0.0

        for bar_start, bar_5m in eval_window_5m.iterrows():
            bar_end_time = bar_start + pd.Timedelta(minutes=5)
            if bar_end_time < last_exit_time:
                continue

            # Limit entries:
            if not allow_v_reversal:
                if long_taken or short_taken:
                    break
            else:
                # If v_reversal allowed, max 1 long and max 1 short per day, and daily pnl > -350
                if (long_taken and short_taken) or running_day_pnl <= -350.0:
                    break

            time_str = bar_end_time.strftime("%H:%M")
            band = prep.get_noise_band(d, time_str, kd=kd)
            if band is None:
                continue
            upper_band, lower_band = band
            close_5m = float(bar_5m["Close"])
            vol_5m = float(bar_5m["Volume"])
            vol_sma_val = float(vol_sma_5m.loc[bar_start]) if bar_start in vol_sma_5m.index else 0.0

            prior_vwap_mask = vwap_series.index < bar_end_time
            if not prior_vwap_mask.any():
                continue
            curr_vwap = float(vwap_series.loc[prior_vwap_mask].iloc[-1])
            curr_ema = float(ema9_5m.loc[bar_start])

            vol_ok = True
            if vol_filter:
                vol_ok = (vol_sma_val > 0 and vol_5m >= 1.10 * vol_sma_val)

            sig = None

            # LONG BREAKOUT
            if not long_taken and close_5m > upper_band and close_5m > curr_vwap and curr_ema > curr_vwap and vol_ok:
                future_1m = df_day.loc[df_day.index >= bar_end_time]
                if not future_1m.empty:
                    open_1m = float(future_1m.iloc[0]["Open"])
                    entry_price = open_1m + 0.50
                    raw_sl = abs(entry_price - curr_vwap) + 4.0
                    sl_dist = float(np.clip(raw_sl, 22.0, 48.0))
                    sl_price = entry_price - sl_dist
                    sig = TradeSignal(
                        strategy_id="H3",
                        strategy_name="H3_Zarattini_NoiseBand_Momentum",
                        timestamp=bar_end_time,
                        execution_time=bar_end_time,
                        direction=1,
                        entry_price=entry_price,
                        stop_loss_price=sl_price,
                        tp1_price=None,
                        sl_dist=sl_dist,
                        scale_out_tp1=False,
                        be_trigger_r=None,
                        trail_rule="VWAP_BAND",
                        trail_band_mode="WIDE",
                        trail_activation_r=0.0,
                        hard_time_exit="15:44:00"
                    )
                    long_taken = True

            # SHORT BREAKOUT
            elif not short_taken and close_5m < lower_band and close_5m < curr_vwap and curr_ema < curr_vwap and vol_ok:
                short_valid = True
                if seeck_mode == "strict":
                    is_bearish = (d_open <= prev_close) or (prev_low is not None and close_5m < prev_low) or (daily_atr >= sma_atr20)
                    if not is_bearish:
                        short_valid = False
                elif seeck_mode == "surgical":
                    # Only block shorts when gapping up above yesterday's high in low-volatility regime
                    is_bull_trap_regime = (prev_high is not None and d_open > prev_high) and (daily_atr < sma_atr20)
                    if is_bull_trap_regime:
                        short_valid = False
                elif seeck_mode == "none":
                    short_valid = True

                if short_valid:
                    future_1m = df_day.loc[df_day.index >= bar_end_time]
                    if not future_1m.empty:
                        open_1m = float(future_1m.iloc[0]["Open"])
                        entry_price = open_1m - 0.50
                        raw_sl = abs(curr_vwap - entry_price) + 4.0
                        sl_dist = float(np.clip(raw_sl, 22.0, 48.0))
                        sl_price = entry_price + sl_dist
                        sig = TradeSignal(
                            strategy_id="H3",
                            strategy_name="H3_Zarattini_NoiseBand_Momentum",
                            timestamp=bar_end_time,
                            execution_time=bar_end_time,
                            direction=-1,
                            entry_price=entry_price,
                            stop_loss_price=sl_price,
                            tp1_price=None,
                            sl_dist=sl_dist,
                            scale_out_tp1=False,
                            be_trigger_r=None,
                            trail_rule="VWAP_BAND",
                            trail_band_mode="WIDE",
                            trail_activation_r=0.0,
                            hard_time_exit="15:44:00"
                        )
                        short_taken = True

            if sig:
                c_tr = sim.simulate_trade(sig, df_day, contract_qty=contracts, df_indicators=df_ind)
                if c_tr:
                    day_trades.extend(c_tr)
                    last_exit_time = max(t.exit_time for t in c_tr)
                    running_day_pnl += sum(t.net_pnl for t in c_tr)
                    if not allow_v_reversal:
                        break

        completed.extend(day_trades)

    return completed


def run_full_ablation():
    prep = get_preprocessor()
    all_dates = sorted(list(prep.daily_atr_map.keys()))
    oos_dates = [d for d in all_dates if d >= date(2023, 1, 1)]

    print("================================================================================")
    print("PART 1: H4 DUAL-CHECKPOINT (15:00 & 15:15 NY) & THRESHOLD CALIBRATION")
    print("================================================================================")

    h4_variants = [
        ("H4 Baseline (15:00 only, 0.0075 rod / 0.60 ATR)", ["15:00"], 0.0075, 0.60, 0.78, 0.22),
        ("H4 Dual-CP (15:00 & 15:15, 0.0075 rod / 0.60 ATR)", ["15:00", "15:15"], 0.0075, 0.60, 0.78, 0.22),
        ("H4 Dual-CP (15:00 & 15:15, 0.0068 rod / 0.55 ATR)", ["15:00", "15:15"], 0.0068, 0.55, 0.76, 0.24),
        ("H4 Dual-CP (15:00 & 15:15, 0.0062 rod / 0.50 ATR)", ["15:00", "15:15"], 0.0062, 0.50, 0.75, 0.25),
        ("H4 Dual-CP (15:00 & 15:15, 0.0055 rod / 0.45 ATR)", ["15:00", "15:15"], 0.0055, 0.45, 0.72, 0.28),
    ]

    for name, cps, rod, atr_m, rp_l, rp_s in h4_variants:
        tr = simulate_h4_variant(prep, all_dates, checkpoints=cps, rod_thresh=rod, atr_move_thresh=atr_m, range_pos_long=rp_l, range_pos_short=rp_s)
        m = compute_metrics(tr)
        oos_tr = [t for t in tr if t.entry_time.date() >= date(2023, 1, 1)]
        oos_m = compute_metrics(oos_tr)
        print(f"{name:<55} | Trades: {m['total_trades']:<4} | WR: {m['win_rate']:>5.2f}% | PF: {m['profit_factor']:>4.2f} | Exp: ${m['expectancy']:>6.2f} | PnL: ${m['total_pnl']:>8.2f} | OOS: ${oos_m['total_pnl']:>8.2f}")

    print("\n================================================================================")
    print("PART 2: H3 FREQUENCY EXPANSION ABLATION GRID")
    print("================================================================================")

    h3_grid = [
        # Baseline Phase 5
        ("H3 Baseline (10:30-13:30, K=1.25, Vol=True, Seeck=Strict, 1 tr/d)", "10:30", "13:30", 1.25, True, "strict", False),
        # 1. Volume filter off
        ("H3 Vol=False (10:30-13:30, K=1.25, Seeck=Strict)", "10:30", "13:30", 1.25, False, "strict", False),
        # 2. Seeck relaxation
        ("H3 Seeck=Surgical (10:30-13:30, K=1.25, Vol=True)", "10:30", "13:30", 1.25, True, "surgical", False),
        ("H3 Seeck=None (10:30-13:30, K=1.25, Vol=True)", "10:30", "13:30", 1.25, True, "none", False),
        ("H3 Vol=False + Seeck=Surgical (10:30-13:30, K=1.25)", "10:30", "13:30", 1.25, False, "surgical", False),
        # 3. Start window 10:15
        ("H3 Window 10:15-13:30 (K=1.25, Vol=False, Seeck=Surgical)", "10:15", "13:30", 1.25, False, "surgical", False),
        # 4. K-factor sweep (10:15-13:30, Vol=False, Seeck=Surgical)
        ("H3 K=1.20 (10:15-13:30, Vol=False, Seeck=Surgical)", "10:15", "13:30", 1.20, False, "surgical", False),
        ("H3 K=1.15 (10:15-13:30, Vol=False, Seeck=Surgical)", "10:15", "13:30", 1.15, False, "surgical", False),
        ("H3 K=1.10 (10:15-13:30, Vol=False, Seeck=Surgical)", "10:15", "13:30", 1.10, False, "surgical", False),
        ("H3 K=1.05 (10:15-13:30, Vol=False, Seeck=Surgical)", "10:15", "13:30", 1.05, False, "surgical", False),
        ("H3 K=1.00 (10:15-13:30, Vol=False, Seeck=Surgical)", "10:15", "13:30", 1.00, False, "surgical", False),
        # 5. V-Reversal Capture (with optimal K and settings)
        ("H3 K=1.15 + V-Reversal (10:15-13:30, Vol=False, Seeck=Surgical)", "10:15", "13:30", 1.15, False, "surgical", True),
        ("H3 K=1.10 + V-Reversal (10:15-13:30, Vol=False, Seeck=Surgical)", "10:15", "13:30", 1.10, False, "surgical", True),
        ("H3 K=1.20 + V-Reversal (10:15-13:30, Vol=False, Seeck=Surgical)", "10:15", "13:30", 1.20, False, "surgical", True),
        ("H3 K=1.25 + V-Reversal (10:15-13:30, Vol=False, Seeck=Surgical)", "10:15", "13:30", 1.25, False, "surgical", True),
        ("H3 K=1.25 + V-Reversal (10:30-13:30, Vol=False, Seeck=Surgical)", "10:30", "13:30", 1.25, False, "surgical", True),
    ]

    for name, st, et, kd_val, vf, sm, vr in h3_grid:
        tr = simulate_h3_variant(prep, all_dates, start_time=st, end_time=et, kd=kd_val, vol_filter=vf, seeck_mode=sm, allow_v_reversal=vr)
        m = compute_metrics(tr)
        oos_tr = [t for t in tr if t.entry_time.date() >= date(2023, 1, 1)]
        oos_m = compute_metrics(oos_tr)
        print(f"{name:<64} | Tr: {m['total_trades']:<4} | WR: {m['win_rate']:>5.2f}% | PF: {m['profit_factor']:>4.2f} | Exp: ${m['expectancy']:>6.2f} | PnL: ${m['total_pnl']:>8.2f} | OOS: ${oos_m['total_pnl']:>8.2f}")


if __name__ == "__main__":
    run_full_ablation()
