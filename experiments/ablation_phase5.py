"""
Phase 5 Ablation Analysis: Safe Expansion of Trade Frequency & Sizing Parity
Author: Principal Quantitative Architect & Financial Engineer
"""

import sys
import logging
from pathlib import Path
from datetime import date
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.session_preprocessor import get_preprocessor
from strategies.base_strategy import TradeSignal, CompletedTrade
from engine.pessimistic_sim import PessimisticExecutionSimulator
from engine.run_phase3_audit import compute_comprehensive_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AblationPhase5")


def evaluate_h3_candidate(
    prep,
    all_dates,
    window_end: str = "13:30",  # "13:30" or "14:00"
    allow_breakdown_clause: bool = False,
    breakdown_atr_mult: float = 0.18
) -> List[CompletedTrade]:
    sim = PessimisticExecutionSimulator()
    all_trades = []

    for d in all_dates:
        df_day = prep.get_session_bars(d)
        if df_day.empty:
            continue
        atr = prep.daily_atr_map.get(d)
        prev_close = prep.prev_day_close_map.get(d)
        prev_low = prep.prev_day_low_map.get(d)
        sma_atr20 = prep.daily_atr_sma20_map.get(d, atr)
        if not atr or not prev_close or pd.isna(atr) or pd.isna(prev_close):
            continue

        rth_bars = df_day.between_time("09:30", "15:44")
        if len(rth_bars) < 60:
            continue

        tp = (rth_bars["High"] + rth_bars["Low"] + rth_bars["Close"]) / 3.0
        cum_vol = rth_bars["Volume"].cumsum()
        vwap_series = (tp * rth_bars["Volume"]).cumsum() / cum_vol.replace(0, np.nan)

        vol_bars = df_day.between_time("08:00", "15:44")
        if vol_bars.empty:
            vol_bars = rth_bars

        resampled_5m = vol_bars.resample("5min", closed="left", label="left").agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
        }).dropna()

        ema9_5m = resampled_5m["Close"].ewm(span=9, adjust=False).mean()
        vol_sma_5m = resampled_5m["Volume"].rolling(window=12, min_periods=4).mean()

        eval_window = resampled_5m.between_time("10:30", window_end)
        if eval_window.empty:
            continue

        rth_sub = df_day.between_time("09:30", "09:30")
        d_open = float(rth_sub.iloc[0]["Open"]) if not rth_sub.empty else prev_close

        for bar_start, bar_5m in eval_window.iterrows():
            bar_end_time = bar_start + pd.Timedelta(minutes=5)
            time_str = bar_end_time.strftime("%H:%M")

            band = prep.get_noise_band(d, time_str)
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

            vol_ok = (vol_sma_val > 0 and vol_5m >= 1.10 * vol_sma_val)

            # Long Breakout
            if close_5m > upper_band and close_5m > curr_vwap and curr_ema > curr_vwap and vol_ok:
                future_1m = df_day.loc[df_day.index >= bar_end_time]
                if not future_1m.empty:
                    open_1m = float(future_1m.iloc[0]["Open"])
                    entry_price = open_1m + 0.50
                    raw_sl = abs(entry_price - curr_vwap) + 4.0
                    sl_dist = float(np.clip(raw_sl, 22.0, 48.0))
                    sl_price = entry_price - sl_dist

                    sig = TradeSignal(
                        strategy_id="H3",
                        strategy_name="H3_NoiseBand",
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
                        hard_time_exit="15:44:00"
                    )
                    df_ind = pd.DataFrame(index=rth_bars.index)
                    df_ind["VWAP"] = vwap_series
                    u_bands, l_bands = [], []
                    for t in rth_bars.index:
                        b = prep.get_noise_band(d, t.strftime("%H:%M"))
                        u_bands.append(b[0] if b else np.nan)
                        l_bands.append(b[1] if b else np.nan)
                    df_ind["UpperBand"] = pd.Series(u_bands, index=rth_bars.index).ffill()
                    df_ind["LowerBand"] = pd.Series(l_bands, index=rth_bars.index).ffill()

                    c_tr = sim.simulate_trade(sig, df_day, contract_qty=2, df_indicators=df_ind)
                    if c_tr:
                        all_trades.extend(c_tr)
                    break

            # Short Breakout
            elif close_5m < lower_band and close_5m < curr_vwap and curr_ema < curr_vwap and vol_ok:
                is_bearish = (d_open <= prev_close) or (prev_low is not None and close_5m < prev_low) or (atr >= sma_atr20)
                if allow_breakdown_clause:
                    # Intraday institutional breakdown clause: Close_5m < VWAP - breakdown_atr_mult * ATR_14
                    is_bearish = is_bearish or (close_5m < curr_vwap - (breakdown_atr_mult * atr))

                if is_bearish:
                    future_1m = df_day.loc[df_day.index >= bar_end_time]
                    if not future_1m.empty:
                        open_1m = float(future_1m.iloc[0]["Open"])
                        entry_price = open_1m - 0.50
                        raw_sl = abs(curr_vwap - entry_price) + 4.0
                        sl_dist = float(np.clip(raw_sl, 22.0, 48.0))
                        sl_price = entry_price + sl_dist

                        sig = TradeSignal(
                            strategy_id="H3",
                            strategy_name="H3_NoiseBand",
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
                            hard_time_exit="15:44:00"
                        )
                        df_ind = pd.DataFrame(index=rth_bars.index)
                        df_ind["VWAP"] = vwap_series
                        u_bands, l_bands = [], []
                        for t in rth_bars.index:
                            b = prep.get_noise_band(d, t.strftime("%H:%M"))
                            u_bands.append(b[0] if b else np.nan)
                            l_bands.append(b[1] if b else np.nan)
                        df_ind["UpperBand"] = pd.Series(u_bands, index=rth_bars.index).ffill()
                        df_ind["LowerBand"] = pd.Series(l_bands, index=rth_bars.index).ffill()

                        c_tr = sim.simulate_trade(sig, df_day, contract_qty=2, df_indicators=df_ind)
                        if c_tr:
                            all_trades.extend(c_tr)
                        break

    return all_trades


def evaluate_h4_candidate(
    prep,
    all_dates,
    rod_thresh: float = 0.0085,
    atr_open_thresh: float = 0.70,
    atr_close_thresh: float = 0.70,
    range_pos_long: float = 0.80,
    range_pos_short: float = 0.20
) -> List[CompletedTrade]:
    sim = PessimisticExecutionSimulator()
    all_trades = []

    for d in all_dates:
        df_day = prep.get_session_bars(d)
        if df_day.empty:
            continue
        atr = prep.daily_atr_map.get(d)
        prev_close = prep.prev_day_close_map.get(d)
        if not atr or not prev_close or pd.isna(atr) or pd.isna(prev_close):
            continue

        rth_bars = df_day.between_time("09:30", "15:44")
        if len(rth_bars) < 200:
            continue

        m_1000 = rth_bars.index.time == pd.Timestamp("10:00:00").time()
        if not m_1000.any():
            continue
        c_1000 = float(rth_bars.loc[m_1000].iloc[-1]["Close"])
        r_1 = c_1000 - prev_close

        m_1330 = rth_bars.index.time == pd.Timestamp("13:30:00").time()
        c_1330 = float(rth_bars.loc[m_1330].iloc[-1]["Close"]) if m_1330.any() else None

        m_1500 = rth_bars.index.time == pd.Timestamp("15:00:00").time()
        if not m_1500.any():
            continue
        bar_1500 = rth_bars.loc[m_1500].iloc[-1]
        t_1500 = rth_bars.loc[m_1500].index[-1]
        c_1500 = float(bar_1500["Close"])

        bars_to_1500 = rth_bars.loc[rth_bars.index <= t_1500]
        h_1500 = float(bars_to_1500["High"].max())
        l_1500 = float(bars_to_1500["Low"].min())
        rng_1500 = h_1500 - l_1500
        if rng_1500 <= 0:
            continue

        disp_rod = (c_1500 - prev_close) / prev_close
        open_0930 = float(rth_bars.iloc[0]["Open"])
        norm_move_open = abs(c_1500 - open_0930) / atr
        norm_move_close = abs(c_1500 - prev_close) / atr
        range_pos = (c_1500 - l_1500) / rng_1500

        tp = (rth_bars["High"] + rth_bars["Low"] + rth_bars["Close"]) / 3.0
        cum_vol = rth_bars["Volume"].cumsum()
        vwap_series = (tp * rth_bars["Volume"]).cumsum() / cum_vol.replace(0, np.nan)
        vwap_1500 = float(vwap_series.loc[t_1500])

        # 0DTE Pin check
        s_50 = 50.0 * round(c_1500 / 50.0)
        if abs(c_1500 - s_50) <= 8.0:
            pin_bars = df_day.between_time("14:45", "15:00")
            if not pin_bars.empty and (pin_bars["High"].max() - pin_bars["Low"].min()) < 15.0:
                continue

        # Displacement threshold test
        has_disp = (abs(disp_rod) >= rod_thresh) or (norm_move_open >= atr_open_thresh) or (norm_move_close >= atr_close_thresh)
        if not has_disp:
            continue

        # Triple Trend Alignment
        midday_ok_long = (c_1330 is not None and c_1500 >= c_1330)
        midday_ok_short = (c_1330 is not None and c_1500 <= c_1330)

        future_1m = df_day.loc[df_day.index > t_1500]
        if future_1m.empty:
            continue

        sl_dist = float(np.clip(0.14 * rng_1500, 16.0, 32.0))
        open_exec = float(future_1m.iloc[0]["Open"])
        t_exec = future_1m.index[0]

        if disp_rod > 0 and r_1 > 0 and midday_ok_long and c_1500 > vwap_1500 and range_pos >= range_pos_long:
            entry_p = open_exec + 0.50
            sl_p = entry_p - sl_dist
            sig = TradeSignal(
                strategy_id="H4",
                strategy_name="H4_LETF",
                timestamp=t_1500,
                execution_time=t_exec,
                direction=1,
                entry_price=entry_p,
                stop_loss_price=sl_p,
                sl_dist=sl_dist,
                scale_out_tp1=False,
                be_trigger_r=None,
                trail_rule="VWAP",
                hard_time_exit="15:44:00"
            )
            df_ind = pd.DataFrame(index=rth_bars.index)
            df_ind["VWAP"] = vwap_series
            c_tr = sim.simulate_trade(sig, df_day, contract_qty=2, df_indicators=df_ind)
            if c_tr:
                all_trades.extend(c_tr)

        elif disp_rod < 0 and r_1 < 0 and midday_ok_short and c_1500 < vwap_1500 and range_pos <= range_pos_short:
            entry_p = open_exec - 0.50
            sl_p = entry_p + sl_dist
            sig = TradeSignal(
                strategy_id="H4",
                strategy_name="H4_LETF",
                timestamp=t_1500,
                execution_time=t_exec,
                direction=-1,
                entry_price=entry_p,
                stop_loss_price=sl_p,
                sl_dist=sl_dist,
                scale_out_tp1=False,
                be_trigger_r=None,
                trail_rule="VWAP",
                hard_time_exit="15:44:00"
            )
            df_ind = pd.DataFrame(index=rth_bars.index)
            df_ind["VWAP"] = vwap_series
            c_tr = sim.simulate_trade(sig, df_day, contract_qty=2, df_indicators=df_ind)
            if c_tr:
                all_trades.extend(c_tr)

    return all_trades


def run_ablations():
    logger.info("Initializing Phase 5 Controlled Ablation Experiments...")
    prep = get_preprocessor()
    all_dates = sorted(list(prep.daily_atr_map.keys()))

    print("\n" + "="*115)
    print("PART 1: H3 FREQUENCY EXPANSION ABLATION (Static 2 MNQ)")
    print("="*115)
    print(f"{'H3 Configuration':<45} | {'Trades':<6} | {'WR%':<6} | {'AvgWin':<8} | {'AvgLoss':<8} | {'R:R':<5} | {'PF':<5} | {'Exp($)':<7} | {'NetPnL($)':<10} | {'OOS PnL($)':<10}")
    print("-" * 115)

    h3_variants = [
        ("Phase 4 Baseline (10:30-13:30, Seeck)", "13:30", False, 0.18),
        ("H3 Opt 1: Add Breakdown Clause (0.18 ATR)", "13:30", True, 0.18),
        ("H3 Opt 2: Add Breakdown Clause (0.15 ATR)", "13:30", True, 0.15),
        ("H3 Opt 3: Window to 14:00 NY", "14:00", False, 0.18),
        ("H3 Opt 4: Window to 14:00 + Breakdown (0.18 ATR)", "14:00", True, 0.18),
    ]

    for name, w_end, b_clause, b_mult in h3_variants:
        tr = evaluate_h3_candidate(prep, all_dates, window_end=w_end, allow_breakdown_clause=b_clause, breakdown_atr_mult=b_mult)
        m = compute_comprehensive_metrics(tr)
        oos_tr = [t for t in tr if t.entry_time.date() >= date(2023, 1, 1)]
        oos_m = compute_comprehensive_metrics(oos_tr)
        print(f"{name:<45} | {m['total_trades']:<6} | {m['win_rate']:<6.2f} | {m['net_avg_win']:<8.2f} | {m['net_avg_loss']:<8.2f} | {m['realized_rr']:<5.2f} | {m['profit_factor']:<5.2f} | {m['expectancy']:<7.2f} | {m['total_pnl']:<10.2f} | {oos_m['total_pnl']:<10.2f}")

    print("\n" + "="*115)
    print("PART 2: H4 VOLATILITY-ADAPTIVE THRESHOLD EXPANSION (Static 2 MNQ)")
    print("="*115)
    print(f"{'H4 Configuration':<45} | {'Trades':<6} | {'WR%':<6} | {'AvgWin':<8} | {'AvgLoss':<8} | {'R:R':<5} | {'PF':<5} | {'Exp($)':<7} | {'NetPnL($)':<10} | {'OOS PnL($)':<10}")
    print("-" * 115)

    h4_variants = [
        ("Phase 4 Baseline (0.0085 rod / 0.70 ATR)", 0.0085, 0.70, 0.70, 0.80, 0.20),
        ("H4 Opt 1 (0.52 close / 0.55 open / 0.0068 rod, 0.75/0.25)", 0.0068, 0.55, 0.52, 0.75, 0.25),
        ("H4 Opt 2 (0.55 close / 0.58 open / 0.0072 rod, 0.76/0.24)", 0.0072, 0.58, 0.55, 0.76, 0.24),
        ("H4 Opt 3 (0.58 close / 0.60 open / 0.0075 rod, 0.78/0.22)", 0.0075, 0.60, 0.58, 0.78, 0.22),
    ]

    for name, r_th, a_op, a_cl, rp_l, rp_s in h4_variants:
        tr = evaluate_h4_candidate(prep, all_dates, rod_thresh=r_th, atr_open_thresh=a_op, atr_close_thresh=a_cl, range_pos_long=rp_l, range_pos_short=rp_s)
        m = compute_comprehensive_metrics(tr)
        oos_tr = [t for t in tr if t.entry_time.date() >= date(2023, 1, 1)]
        oos_m = compute_comprehensive_metrics(oos_tr)
        print(f"{name:<45} | {m['total_trades']:<6} | {m['win_rate']:<6.2f} | {m['net_avg_win']:<8.2f} | {m['net_avg_loss']:<8.2f} | {m['realized_rr']:<5.2f} | {m['profit_factor']:<5.2f} | {m['expectancy']:<7.2f} | {m['total_pnl']:<10.2f} | {oos_m['total_pnl']:<10.2f}")


if __name__ == "__main__":
    run_ablations()
