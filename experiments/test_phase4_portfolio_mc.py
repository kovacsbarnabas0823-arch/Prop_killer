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
from engine.tradeify_rules import TradeifyAccountState, AccountPhase, TradeifyAccountConfig
from engine.monte_carlo import TradeifyChallengeSimulator
from engine.run_phase3_audit import compute_comprehensive_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase4PortfolioMC")


def precompute_noise_bands_phase4(df: pd.DataFrame, daily_dict: dict, prev_close_map: dict):
    cps_15m = ["10:30", "10:45", "11:00", "11:15", "11:30", "11:45", "12:00", "12:15", "12:30", "12:45", "13:00", "13:15", "13:30"]
    rth_1m = df.between_time("09:30", "14:00")
    opens_0930 = rth_1m.between_time("09:30", "09:30").groupby(rth_1m.between_time("09:30", "09:30").index.date)["Open"].first()

    moves_by_cp = {}
    for cp in cps_15m:
        closes_cp = rth_1m.between_time(cp, cp).groupby(rth_1m.between_time(cp, cp).index.date)["Close"].last()
        df_cp = pd.DataFrame({"Open_0930": opens_0930, "Close_cp": closes_cp}).dropna()
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
        for cp in cps_15m:
            sig = moves_by_cp[cp].get(d)
            if sig is not None and not pd.isna(sig) and sig > 0:
                upper = ref_high + (1.25 * sig)
                lower = ref_low - (1.25 * sig)
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


def generate_h3_phase4_signals(prep, bands_map, d, df_day, atr, prev_close, prev_low, atr_sma20):
    rth_bars = df_day.between_time("09:30", "15:44")
    if len(rth_bars) < 60:
        return []

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
        return []

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

                return [TradeSignal(
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
                )]

        # Short Breakout with Seeck (2026) Asymmetry
        elif close_5m < lower_band and close_5m < curr_vwap and curr_ema < curr_vwap and vol_ok:
            is_bearish = (d_open <= prev_close) or (prev_low is not None and close_5m < prev_low) or (atr >= sma_atr)
            if is_bearish:
                future_1m = df_day.loc[df_day.index >= bar_end_time]
                if not future_1m.empty:
                    open_1m = float(future_1m.iloc[0]["Open"])
                    entry_price = open_1m - 0.50
                    raw_sl = abs(curr_vwap - entry_price) + 4.0
                    sl_dist = float(np.clip(raw_sl, 22.0, 48.0))
                    sl_price = entry_price + sl_dist

                    return [TradeSignal(
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
                    )]
    return []


def generate_h4_phase4_signals(d, df_day, atr, prev_close):
    rth_bars = df_day.between_time("09:30", "15:44")
    if len(rth_bars) < 200:
        return []

    # Bar at 10:00:00 (r_1)
    m_1000 = rth_bars.index.time == pd.Timestamp("10:00:00").time()
    if not m_1000.any():
        return []
    c_1000 = float(rth_bars.loc[m_1000].iloc[-1]["Close"])
    r_1 = c_1000 - prev_close

    # Bar at 13:30:00
    m_1330 = rth_bars.index.time == pd.Timestamp("13:30:00").time()
    c_1330 = float(rth_bars.loc[m_1330].iloc[-1]["Close"]) if m_1330.any() else None

    # Bar at 15:00:00
    m_1500 = rth_bars.index.time == pd.Timestamp("15:00:00").time()
    if not m_1500.any():
        return []
    bar_1500 = rth_bars.loc[m_1500].iloc[-1]
    t_1500 = rth_bars.loc[m_1500].index[-1]
    c_1500 = float(bar_1500["Close"])

    bars_to_1500 = rth_bars.loc[rth_bars.index <= t_1500]
    h_1500 = float(bars_to_1500["High"].max())
    l_1500 = float(bars_to_1500["Low"].min())
    rng_1500 = h_1500 - l_1500
    if rng_1500 <= 0:
        return []

    disp_rod = (c_1500 - prev_close) / prev_close
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
    if abs(c_1500 - s_50) <= 8.0:
        pin_bars = df_day.between_time("14:45", "15:00")
        if not pin_bars.empty and (pin_bars["High"].max() - pin_bars["Low"].min()) < 15.0:
            return []

    # High-Conviction Displacement (0.0085 rod or 0.70 ATR)
    if not ((abs(disp_rod) >= 0.0085) or (norm_move_atr >= 0.70)):
        return []

    # Triple Trend Alignment
    midday_ok_long = (c_1330 is not None and c_1500 >= c_1330)
    midday_ok_short = (c_1330 is not None and c_1500 <= c_1330)

    future_1m = df_day.loc[df_day.index > t_1500]
    if future_1m.empty:
        return []

    sl_dist = float(np.clip(0.14 * rng_1500, 16.0, 32.0))
    open_exec = float(future_1m.iloc[0]["Open"])
    t_exec = future_1m.index[0]

    if disp_rod > 0 and r_1 > 0 and midday_ok_long and c_1500 > vwap_1500 and range_pos >= 0.80:
        entry_p = open_exec + 0.50
        sl_p = entry_p - sl_dist
        return [TradeSignal(
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
        )]
    elif disp_rod < 0 and r_1 < 0 and midday_ok_short and c_1500 < vwap_1500 and range_pos <= 0.20:
        entry_p = open_exec - 0.50
        sl_p = entry_p + sl_dist
        return [TradeSignal(
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
        )]
    return []


def calculate_phase4_position_size(account: TradeifyAccountState, sl_dist_pts: float, atr_14: float) -> int:
    """
    Zarattini ATR_14 Volatility Normalization + Pre-Lock Safe-Harbor Protocol:
    ref_vol_pts = 0.22 * atr_14
    unit_vol_risk_usd = (0.22 * atr_14 * 2.00) + 4.48
    base_contracts = int(np.clip(np.round(R_target / unit_vol_risk_usd), 1, 12))
    max_safe_contracts = int(np.max([1, np.floor((1.35 * R_target) / (sl_dist_pts * 2.00 + 4.48))]))
    contracts = min(base_contracts, max_safe_contracts)
    """
    if account.is_breached or account.is_daily_stopped or account.is_governor_halted:
        return 0

    buffer = account.usable_buffer
    profit = account.net_profit
    current_equity = account.current_equity

    if account.phase == AccountPhase.EVALUATION:
        # Evaluation Sprint
        if current_equity >= 51900.0:
            r_target = 440.0
        elif profit >= 700.0 and buffer >= 1300.0:
            r_target = 320.0
        elif buffer >= 850.0:
            r_target = 225.0
        else:
            r_target = 140.0
    else:
        # Funded Phase: Pre-Lock vs Post-Lock Two-Zone Escalation
        if not account.is_floor_locked and account.eod_hwm < 52100.0:
            # Zone 1: Trailing Danger Zone
            if buffer < 1100.0:
                r_target = 100.0
            else:
                r_target = 135.0
        else:
            # Zone 2: Static Locked Zone
            r_target = 230.0

    ref_vol_pts = 0.22 * atr_14 if (atr_14 and atr_14 > 0) else max(sl_dist_pts, 30.0)
    unit_vol_risk_usd = (ref_vol_pts * 2.00) + 4.48
    base_contracts = int(np.clip(np.round(r_target / unit_vol_risk_usd), 1, 12))

    per_contract_sl_risk = (sl_dist_pts * 2.00) + 4.48
    max_safe_contracts = int(np.max([1, np.floor((1.35 * r_target) / per_contract_sl_risk)]))

    return int(np.clip(min(base_contracts, max_safe_contracts), 1, 12))


def run_phase4_simulation():
    prep = get_preprocessor()
    all_dates = sorted(list(prep.daily_atr_map.keys()))
    bands_map = precompute_noise_bands_phase4(prep.df, prep.daily_dict, prep.prev_day_close_map)

    atr_series = pd.Series(prep.daily_atr_map).sort_index()
    atr_sma20 = atr_series.rolling(20).mean().shift(1).to_dict()

    sim = PessimisticExecutionSimulator()

    for test_mode in ["EVALUATION", "FUNDED"]:
        phase_enum = AccountPhase.EVALUATION if test_mode == "EVALUATION" else AccountPhase.FUNDED
        account = TradeifyAccountState(phase=phase_enum)
        portfolio_trades = []
        daily_pnls = {}

        for d in all_dates:
            df_day = prep.get_session_bars(d)
            if df_day.empty:
                continue
            atr = prep.daily_atr_map.get(d)
            prev_close = prep.prev_day_close_map.get(d)
            prev_low = prep.prev_day_low_map.get(d)
            if not atr or not prev_close or pd.isna(atr) or pd.isna(prev_close):
                continue

            account.start_session(d)
            session_trades = []
            day_pnl = 0.0

            rth_bars = df_day.between_time("09:30", "15:44")
            df_ind = pd.DataFrame(index=rth_bars.index)
            if not rth_bars.empty:
                tp = (rth_bars["High"] + rth_bars["Low"] + rth_bars["Close"]) / 3.0
                cum_vol = rth_bars["Volume"].cumsum()
                df_ind["VWAP"] = (tp * rth_bars["Volume"]).cumsum() / cum_vol.replace(0, np.nan)
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

            # 1. Evaluate H3
            h3_sigs = generate_h3_phase4_signals(prep, bands_map, d, df_day, atr, prev_close, prev_low, atr_sma20)
            for s in h3_sigs:
                contracts = calculate_phase4_position_size(account, s.sl_dist, atr)
                c_tr = sim.simulate_trade(s, df_day, contract_qty=contracts, df_indicators=df_ind)
                if c_tr:
                    session_trades.extend(c_tr)
                    t_pnl = sum(t.net_pnl for t in c_tr)
                    day_pnl += t_pnl
                    account.update_intraday_pnl(t_pnl)
                    account.close_trade(t_pnl)

            # 2. Evaluate H4 with Pre-Lock Safe-Harbor Protocol (Rule A)
            skip_h4 = False
            if phase_enum == AccountPhase.FUNDED:
                # If morning H3 secured >= +$220.0 and still pre-lock, bank the qualifying day and skip H4!
                if day_pnl >= 220.0 and not account.is_floor_locked and account.eod_hwm < 52100.0:
                    skip_h4 = True

            if not skip_h4 and not account.is_daily_stopped and not account.is_governor_halted:
                h4_sigs = generate_h4_phase4_signals(d, df_day, atr, prev_close)
                for s in h4_sigs:
                    contracts = calculate_phase4_position_size(account, s.sl_dist, atr)
                    c_tr = sim.simulate_trade(s, df_day, contract_qty=contracts, df_indicators=df_ind)
                    if c_tr:
                        session_trades.extend(c_tr)
                        t_pnl = sum(t.net_pnl for t in c_tr)
                        day_pnl += t_pnl
                        account.update_intraday_pnl(t_pnl)
                        account.close_trade(t_pnl)

            account.end_session(d)
            portfolio_trades.extend(session_trades)
            daily_pnls[d] = day_pnl

        m = compute_comprehensive_metrics(portfolio_trades)
        oos_tr = [t for t in portfolio_trades if t.entry_time.date() >= date(2023, 1, 1)]
        oos_m = compute_comprehensive_metrics(oos_tr)

        print(f"\n--- {test_mode} PHASE PORTFOLIO METRICS ---")
        print(f"Trades: {m['total_trades']}, WR: {m['win_rate']:.2f}%, PF: {m['profit_factor']:.2f}, Exp: ${m['expectancy']:.2f}, TotalPnL: ${m['total_pnl']:.2f}, OOS: ${oos_m['total_pnl']:.2f}, MaxDD: ${m['max_drawdown']:.2f}")

    print("\nSimulation complete!")


if __name__ == "__main__":
    run_phase4_simulation()
