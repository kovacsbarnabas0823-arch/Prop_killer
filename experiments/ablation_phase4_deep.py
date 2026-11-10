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
logger = logging.getLogger("AblationDeep")


def precompute_noise_bands_dual_k(df: pd.DataFrame, daily_dict: dict, prev_close_map: dict, checkpoints: list, mode: str = "pts", k_long: float = 1.25, k_short: float = 1.25):
    rth_1m = df.between_time("09:30", "14:00")
    opens_0930 = rth_1m.between_time("09:30", "09:30").groupby(rth_1m.between_time("09:30", "09:30").index.date)["Open"].first()

    moves_by_cp = {}
    for cp in checkpoints:
        closes_cp = rth_1m.between_time(cp, cp).groupby(rth_1m.between_time(cp, cp).index.date)["Close"].last()
        df_cp = pd.DataFrame({"Open_0930": opens_0930, "Close_cp": closes_cp}).dropna()
        if mode == "pct":
            df_cp["move"] = (df_cp["Close_cp"] - df_cp["Open_0930"]).abs() / df_cp["Open_0930"]
        else:
            df_cp["move"] = (df_cp["Close_cp"] - df_cp["Open_0930"]).abs()
        sigma_cp = df_cp["move"].rolling(window=14).mean().shift(1)
        moves_by_cp[cp] = sigma_cp.to_dict()

    dates_list = sorted(list(daily_dict.keys()))
    bands_map = {}
    for d in dates_list:
        if d not in prev_close_map or pd.isna(prev_close_map[d]):
            continue
        p_close = prev_close_map[d]
        day_df = daily_dict.get(d)
        if day_df is None or day_df.empty:
            continue
        rth_sub = day_df.between_time("09:30", "09:30")
        if rth_sub.empty:
            continue
        d_open = float(rth_sub.iloc[0]["Open"])
        ref_high = max(d_open, p_close)
        ref_low = min(d_open, p_close)

        bands_map[d] = {}
        for cp in checkpoints:
            sig = moves_by_cp[cp].get(d)
            if sig is not None and not pd.isna(sig) and sig > 0:
                if mode == "pct":
                    upper = ref_high * (1.0 + (k_long * sig))
                    lower = ref_low * (1.0 - (k_short * sig))
                else:
                    upper = ref_high + (k_long * sig)
                    lower = ref_low - (k_short * sig)
                bands_map[d][cp] = (upper, lower)
    return bands_map


def get_band(bands_map, session_date, time_str: str):
    day_bands = bands_map.get(session_date)
    if not day_bands:
        return None
    valid_cps = [cp for cp in day_bands.keys() if cp <= time_str]
    if not valid_cps:
        return None
    return day_bands[max(valid_cps)]


def test_h3_asymmetry(
    prep,
    all_dates,
    bands_map,
    k_short: float = 1.25,
    short_filter_mode: str = "none",  # "none", "strict_seeck"
    ratchet_tiers: Optional[List[Tuple[float, float]]] = None,
    trail_band_mode: str = "WIDE"
):
    sim = PessimisticExecutionSimulator()
    all_trades = []
    atr_series = pd.Series(prep.daily_atr_map).sort_index()
    atr_sma20 = atr_series.rolling(20).mean().shift(1).to_dict()

    for d in all_dates:
        df_day = prep.get_session_bars(d)
        if df_day.empty:
            continue
        atr = prep.daily_atr_map.get(d)
        prev_close = prep.prev_day_close_map.get(d)
        prev_low = prep.prev_day_low_map.get(d)
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

        eval_window = resampled_5m.between_time("10:30", "13:30")
        if eval_window.empty:
            continue

        rth_sub = df_day.between_time("09:30", "09:30")
        d_open = float(rth_sub.iloc[0]["Open"]) if not rth_sub.empty else prev_close
        sma_atr = atr_sma20.get(d, 0.0)

        for bar_start, bar_5m in eval_window.iterrows():
            bar_end_time = bar_start + pd.Timedelta(minutes=5)
            time_str = bar_end_time.strftime("%H:%M")

            band = get_band(bands_map, d, time_str)
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
                        trail_band_mode=trail_band_mode,
                        ratchet_tiers=ratchet_tiers,
                        hard_time_exit="15:44:00"
                    )
                    df_ind = pd.DataFrame(index=rth_bars.index)
                    df_ind["VWAP"] = vwap_series
                    u_bands, l_bands = [], []
                    for t in rth_bars.index:
                        b = get_band(bands_map, d, t.strftime("%H:%M"))
                        if b:
                            u_bands.append(b[0])
                            l_bands.append(b[1])
                        else:
                            u_bands.append(np.nan)
                            l_bands.append(np.nan)
                    df_ind["UpperBand"] = pd.Series(u_bands, index=rth_bars.index).ffill()
                    df_ind["LowerBand"] = pd.Series(l_bands, index=rth_bars.index).ffill()

                    c_tr = sim.simulate_trade(sig, df_day, contract_qty=2, df_indicators=df_ind)
                    if c_tr:
                        all_trades.extend(c_tr)
                    break

            # Short Breakout with Seeck (2026) filter
            elif close_5m < lower_band and close_5m < curr_vwap and curr_ema < curr_vwap and vol_ok:
                short_valid = True
                if short_filter_mode == "strict_seeck":
                    # Non-gap-up day OR close < prev_low OR expanding volatility ATR >= SMA(ATR, 20)
                    is_bearish = (d_open <= prev_close) or (prev_low is not None and close_5m < prev_low) or (atr >= sma_atr)
                    if not is_bearish:
                        short_valid = False

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
                            trail_band_mode=trail_band_mode,
                            ratchet_tiers=ratchet_tiers,
                            hard_time_exit="15:44:00"
                        )
                        df_ind = pd.DataFrame(index=rth_bars.index)
                        df_ind["VWAP"] = vwap_series
                        u_bands, l_bands = [], []
                        for t in rth_bars.index:
                            b = get_band(bands_map, d, t.strftime("%H:%M"))
                            if b:
                                u_bands.append(b[0])
                                l_bands.append(b[1])
                            else:
                                u_bands.append(np.nan)
                                l_bands.append(np.nan)
                        df_ind["UpperBand"] = pd.Series(u_bands, index=rth_bars.index).ffill()
                        df_ind["LowerBand"] = pd.Series(l_bands, index=rth_bars.index).ffill()

                        c_tr = sim.simulate_trade(sig, df_day, contract_qty=2, df_indicators=df_ind)
                        if c_tr:
                            all_trades.extend(c_tr)
                        break

    return all_trades


def test_h4_variants(
    prep,
    all_dates,
    disp_thresh_rod: float = 0.0085,
    disp_thresh_atr: float = 0.70,
    use_midday_continuation: bool = True,
    range_pos_long: float = 0.80,
    range_pos_short: float = 0.20,
    sl_factor: float = 0.14,
    min_sl: float = 16.0,
    max_sl: float = 32.0,
    exit_mode: str = "VWAP_TRAIL",  # "VWAP_TRAIL", "HOLD_1544", "HOLD_WITH_RATCHET"
    ratchet_tiers: Optional[List[Tuple[float, float]]] = None
):
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

        # Bar at 10:00:00 (r_1)
        m_1000 = rth_bars.index.time == pd.Timestamp("10:00:00").time()
        if not m_1000.any():
            continue
        c_1000 = float(rth_bars.loc[m_1000].iloc[-1]["Close"])
        r_1 = c_1000 - prev_close

        # Bar at 13:30:00
        m_1330 = rth_bars.index.time == pd.Timestamp("13:30:00").time()
        c_1330 = float(rth_bars.loc[m_1330].iloc[-1]["Close"]) if m_1330.any() else None

        # Bar at 15:00:00
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
        disp_pts = c_1500 - prev_close
        disp_from_open = c_1500 - float(rth_bars.iloc[0]["Open"])
        norm_move_atr = abs(disp_from_open) / atr

        range_pos = (c_1500 - l_1500) / rng_1500

        # VWAP at 15:00
        tp = (rth_bars["High"] + rth_bars["Low"] + rth_bars["Close"]) / 3.0
        cum_vol = rth_bars["Volume"].cumsum()
        vwap_series = (tp * rth_bars["Volume"]).cumsum() / cum_vol.replace(0, np.nan)
        vwap_1500 = float(vwap_series.loc[t_1500])

        # 0DTE Pin check
        s_50 = 50.0 * round(c_1500 / 50.0)
        is_pinned = False
        if abs(c_1500 - s_50) <= 8.0:
            pin_bars = df_day.between_time("14:45", "15:00")
            if not pin_bars.empty and (pin_bars["High"].max() - pin_bars["Low"].min()) < 15.0:
                is_pinned = True
        if is_pinned:
            continue

        # Check displacement threshold
        has_disp = (abs(disp_rod) >= disp_thresh_rod) or (norm_move_atr >= disp_thresh_atr)
        if not has_disp:
            continue

        # Midday continuation check
        midday_ok_long = True
        midday_ok_short = True
        if use_midday_continuation and c_1330 is not None:
            midday_ok_long = (c_1500 >= c_1330)
            midday_ok_short = (c_1500 <= c_1330)

        future_1m = df_day.loc[df_day.index > t_1500]
        if future_1m.empty:
            continue

        sl_dist = float(np.clip(sl_factor * rng_1500, min_sl, max_sl))
        open_exec = float(future_1m.iloc[0]["Open"])
        t_exec = future_1m.index[0]

        trail_rule = "VWAP" if exit_mode == "VWAP_TRAIL" else None

        # Long Signal
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
                trail_rule=trail_rule,
                ratchet_tiers=ratchet_tiers if exit_mode == "HOLD_WITH_RATCHET" else None,
                hard_time_exit="15:44:00"
            )
            df_ind = pd.DataFrame(index=rth_bars.index)
            df_ind["VWAP"] = vwap_series
            c_tr = sim.simulate_trade(sig, df_day, contract_qty=2, df_indicators=df_ind)
            if c_tr:
                all_trades.extend(c_tr)

        # Short Signal
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
                trail_rule=trail_rule,
                ratchet_tiers=ratchet_tiers if exit_mode == "HOLD_WITH_RATCHET" else None,
                hard_time_exit="15:44:00"
            )
            df_ind = pd.DataFrame(index=rth_bars.index)
            df_ind["VWAP"] = vwap_series
            c_tr = sim.simulate_trade(sig, df_day, contract_qty=2, df_indicators=df_ind)
            if c_tr:
                all_trades.extend(c_tr)

    return all_trades


def run_deep_experiments():
    prep = get_preprocessor()
    all_dates = sorted(list(prep.daily_atr_map.keys()))
    cps_15m = ["10:30", "10:45", "11:00", "11:15", "11:30", "11:45", "12:00", "12:15", "12:30", "12:45", "13:00", "13:15", "13:30"]

    print("\n--- ABLATION: H3 ASYMMETRY & K_SHORT ---")
    b_add = precompute_noise_bands_dual_k(prep.df, prep.daily_dict, prep.prev_day_close_map, cps_15m, mode="pts", k_long=1.25, k_short=1.25)
    b_add_k132 = precompute_noise_bands_dual_k(prep.df, prep.daily_dict, prep.prev_day_close_map, cps_15m, mode="pts", k_long=1.25, k_short=1.32)

    h3_exps = [
        ("H3 Baseline Additive (k=1.25/1.25)", b_add, 1.25, "none"),
        ("H3 Additive + Strict Seeck Asym (k=1.25/1.25)", b_add, 1.25, "strict_seeck"),
        ("H3 Additive + K_short=1.32", b_add_k132, 1.32, "none"),
        ("H3 Additive + K_short=1.32 + Strict Seeck", b_add_k132, 1.32, "strict_seeck"),
    ]

    print(f"{'H3 Configuration':<45} | {'Trades':<6} | {'WR%':<6} | {'AvgWin':<8} | {'AvgLoss':<8} | {'R:R':<5} | {'PF':<5} | {'Exp($)':<7} | {'NetPnL($)':<10} | {'OOS PnL($)':<10}")
    print("-" * 120)
    for name, bmap, ks, filt in h3_exps:
        tr = test_h3_asymmetry(prep, all_dates, bmap, k_short=ks, short_filter_mode=filt, trail_band_mode="WIDE")
        m = compute_comprehensive_metrics(tr)
        oos_tr = [t for t in tr if t.entry_time.date() >= date(2023, 1, 1)]
        oos_m = compute_comprehensive_metrics(oos_tr)
        print(f"{name:<45} | {m['total_trades']:<6} | {m['win_rate']:<6.2f} | {m['net_avg_win']:<8.2f} | {m['net_avg_loss']:<8.2f} | {m['realized_rr']:<5.2f} | {m['profit_factor']:<5.2f} | {m['expectancy']:<7.2f} | {m['total_pnl']:<10.2f} | {oos_m['total_pnl']:<10.2f}")

    print("\n--- ABLATION: H4 RESTORATION & EXITS ---")
    h4_exps = [
        ("H4 Phase 3 (Loose 0.58 ATR, VWAP Trail)", 0.0070, 0.58, False, 0.78, 0.22, 0.15, 18.0, 36.0, "VWAP_TRAIL", None),
        ("H4 Phase 2 Baseline (0.0080 rod, VWAP Trail)", 0.0080, 0.65, False, 0.78, 0.22, 0.15, 18.0, 36.0, "VWAP_TRAIL", None),
        ("H4 Exp 1: High Conv (0.0085 rod / 0.70 ATR + Triple Align)", 0.0085, 0.70, True, 0.80, 0.20, 0.14, 16.0, 32.0, "VWAP_TRAIL", None),
        ("H4 Exp 2: High Conv + Hold to 15:44", 0.0085, 0.70, True, 0.80, 0.20, 0.14, 16.0, 32.0, "HOLD_1544", None),
        ("H4 Exp 3: High Conv + Hold with Ratchet (+1.5R->+0.8R)", 0.0085, 0.70, True, 0.80, 0.20, 0.14, 16.0, 32.0, "HOLD_WITH_RATCHET", [(1.5, 0.8)]),
    ]

    print(f"{'H4 Configuration':<45} | {'Trades':<6} | {'WR%':<6} | {'AvgWin':<8} | {'AvgLoss':<8} | {'R:R':<5} | {'PF':<5} | {'Exp($)':<7} | {'NetPnL($)':<10} | {'OOS PnL($)':<10}")
    print("-" * 120)
    for name, dr, da, mc, rpl, rps, sf, msl, xsl, em, rt in h4_exps:
        tr = test_h4_variants(prep, all_dates, dr, da, mc, rpl, rps, sf, msl, xsl, em, rt)
        m = compute_comprehensive_metrics(tr)
        oos_tr = [t for t in tr if t.entry_time.date() >= date(2023, 1, 1)]
        oos_m = compute_comprehensive_metrics(oos_tr)
        print(f"{name:<45} | {m['total_trades']:<6} | {m['win_rate']:<6.2f} | {m['net_avg_win']:<8.2f} | {m['net_avg_loss']:<8.2f} | {m['realized_rr']:<5.2f} | {m['profit_factor']:<5.2f} | {m['expectancy']:<7.2f} | {m['total_pnl']:<10.2f} | {oos_m['total_pnl']:<10.2f}")


if __name__ == "__main__":
    run_deep_experiments()
