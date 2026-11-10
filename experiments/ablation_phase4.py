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
logger = logging.getLogger("AblationPhase4")


def precompute_noise_bands(df: pd.DataFrame, daily_dict: dict, prev_close_map: dict, checkpoints: list, mode: str = "pct", k_mult: float = 1.25):
    """
    Computes time-of-day noise envelopes:
    mode='pct' (multiplicative percentage move per Zarattini 2024):
        UpperBand = ref_high * (1.0 + k * sigma_pct)
        LowerBand = ref_low * (1.0 - k * sigma_pct)
    mode='pts' (additive point move):
        UpperBand = ref_high + (k * sigma_pts)
        LowerBand = ref_low - (k * sigma_pts)
    """
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
                    upper = ref_high * (1.0 + (k_mult * sig))
                    lower = ref_low * (1.0 - (k_mult * sig))
                else:
                    upper = ref_high + (k_mult * sig)
                    lower = ref_low - (k_mult * sig)
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


def simulate_h3_variant(
    prep,
    all_dates,
    bands_map,
    window_start: str = "10:30",
    window_end: str = "13:30",
    use_asymmetry_filter: bool = False,
    short_k_factor: float = 1.25,
    trail_band_mode: str = "WIDE",
    ratchet_tiers: Optional[List[Tuple[float, float]]] = None,
    contracts: int = 2
):
    sim = PessimisticExecutionSimulator()
    all_trades = []
    
    # Precompute 20-day SMA of ATR_14 for asymmetry filter
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

        eval_window = resampled_5m.between_time(window_start, window_end)
        if eval_window.empty:
            continue

        rth_sub = df_day.between_time("09:30", "09:30")
        d_open = float(rth_sub.iloc[0]["Open"]) if not rth_sub.empty else prev_close
        sma_atr = atr_sma20.get(d, 0.0)

        # Asymmetry filter condition for Short
        short_allowed = True
        if use_asymmetry_filter:
            # Bearish Intraday/Daily Structure:
            # 1. Non-gap-up day: d_open <= prev_close
            # 2. Or close breaks below prior day low
            # 3. Or expanding volatility regime: ATR_14 >= SMA(ATR_14, 20)
            bearish_structure = (d_open <= prev_close) or (prev_low is not None and not pd.isna(prev_low)) or (atr >= sma_atr)
            short_allowed = bearish_structure

        signal_emitted = False
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
                    # Prepare indicators dataframe
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

                    c_tr = sim.simulate_trade(sig, df_day, contract_qty=contracts, df_indicators=df_ind)
                    if c_tr:
                        all_trades.extend(c_tr)
                    signal_emitted = True
                    break

            # Short Breakout
            elif short_allowed and close_5m < lower_band and close_5m < curr_vwap and curr_ema < curr_vwap and vol_ok:
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

                    c_tr = sim.simulate_trade(sig, df_day, contract_qty=contracts, df_indicators=df_ind)
                    if c_tr:
                        all_trades.extend(c_tr)
                    signal_emitted = True
                    break

    return all_trades


def run_all_ablations():
    logger.info("Initializing Phase 4 Quantitative Ablations...")
    prep = get_preprocessor()
    all_dates = sorted(list(prep.daily_atr_map.keys()))

    cps_15m = ["10:30", "10:45", "11:00", "11:15", "11:30", "11:45", "12:00", "12:15", "12:30", "12:45", "13:00", "13:15", "13:30"]
    cps_30m = ["10:30", "11:00", "11:30", "12:00", "12:30", "13:00", "13:30"]

    # 1. Precompute Noise Band variants
    logger.info("Precomputing Additive Bands (15m)...")
    bands_add_15m = precompute_noise_bands(prep.df, prep.daily_dict, prep.prev_day_close_map, cps_15m, mode="pts", k_mult=1.25)
    logger.info("Precomputing Multiplicative Bands (15m)...")
    bands_pct_15m = precompute_noise_bands(prep.df, prep.daily_dict, prep.prev_day_close_map, cps_15m, mode="pct", k_mult=1.25)
    logger.info("Precomputing Multiplicative Bands (30m Zarattini)...")
    bands_pct_30m = precompute_noise_bands(prep.df, prep.daily_dict, prep.prev_day_close_map, cps_30m, mode="pct", k_mult=1.25)

    experiments = [
        ("Phase 2 Baseline (Additive, 10:30-13:30, Wide Trail)", bands_add_15m, "10:30", "13:30", False, "WIDE", None),
        ("Exp 1: Multiplicative (15m, 10:30-13:30, Wide Trail)", bands_pct_15m, "10:30", "13:30", False, "WIDE", None),
        ("Exp 2: Multiplicative (30m Zarattini, 10:30-13:30, Wide Trail)", bands_pct_30m, "10:30", "13:30", False, "WIDE", None),
        ("Exp 3: Additive + Asymmetry Filter", bands_add_15m, "10:30", "13:30", True, "WIDE", None),
        ("Exp 4: Multiplicative + Asymmetry Filter", bands_pct_15m, "10:30", "13:30", True, "WIDE", None),
        ("Exp 5: Multiplicative + Tight Trail (Mode A: max UB/VWAP)", bands_pct_15m, "10:30", "13:30", False, "TIGHT", None),
        ("Exp 6: Multiplicative + Maroy Ratchet Floor (Mode B: Wide+Ratchet)", bands_pct_15m, "10:30", "13:30", False, "WIDE", [(2.5, 1.2), (4.0, 2.5)]),
        ("Exp 7: Multiplicative + Asymmetry + Maroy Ratchet Floor", bands_pct_15m, "10:30", "13:30", True, "WIDE", [(2.5, 1.2), (4.0, 2.5)]),
        ("Exp 8: Additive + Asymmetry + Maroy Ratchet Floor", bands_add_15m, "10:30", "13:30", True, "WIDE", [(2.5, 1.2), (4.0, 2.5)]),
        ("Exp 9: Additive + Maroy Ratchet Floor", bands_add_15m, "10:30", "13:30", False, "WIDE", [(2.5, 1.2), (4.0, 2.5)]),
        ("Exp 10: Multiplicative + Tight Trail + Maroy Ratchet (Mode C)", bands_pct_15m, "10:30", "13:30", False, "TIGHT", [(2.5, 1.2), (4.0, 2.5)]),
    ]

    print("\n" + "="*120)
    print(f"{'Experiment Name':<45} | {'Trades':<6} | {'WR%':<6} | {'AvgWin':<8} | {'AvgLoss':<8} | {'R:R':<5} | {'PF':<5} | {'Exp($)':<7} | {'NetPnL($)':<10} | {'OOS PnL($)':<10}")
    print("="*120)

    for name, bmap, w_s, w_e, asym, t_mode, r_tiers in experiments:
        tr = simulate_h3_variant(prep, all_dates, bmap, w_s, w_e, asym, 1.25, t_mode, r_tiers)
        m = compute_comprehensive_metrics(tr)
        oos_tr = [t for t in tr if t.entry_time.date() >= date(2023, 1, 1)]
        oos_m = compute_comprehensive_metrics(oos_tr)
        print(f"{name:<45} | {m['total_trades']:<6} | {m['win_rate']:<6.2f} | {m['net_avg_win']:<8.2f} | {m['net_avg_loss']:<8.2f} | {m['realized_rr']:<5.2f} | {m['profit_factor']:<5.2f} | {m['expectancy']:<7.2f} | {m['total_pnl']:<10.2f} | {oos_m['total_pnl']:<10.2f}")


if __name__ == "__main__":
    run_all_ablations()
