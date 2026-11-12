import sys
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.session_preprocessor import get_preprocessor
from strategies.base_strategy import TradeSignal, CompletedTrade
from engine.pessimistic_sim import PessimisticExecutionSimulator
from engine.tradeify_rules import TradeifyAccountState, AccountPhase, TradeifyAccountConfig, get_strategy_unit_risk_usd
from engine.monte_carlo import TradeifyChallengeSimulator
from engine.run_phase5_audit import compute_comprehensive_metrics


def test_portfolio_config(
    h3_kd=1.25,
    h3_start="10:15",
    allow_v_rev=False,
    funded_r_prelock=150.0,
    funded_shield_buffer=1150.0,
    eval_tier1_r=160.0
):
    prep = get_preprocessor()
    all_dates = sorted(list(prep.daily_atr_map.keys()))
    sim = PessimisticExecutionSimulator()

    results = {}
    for phase_name in ["EVALUATION", "FUNDED"]:
        phase_enum = AccountPhase.EVALUATION if phase_name == "EVALUATION" else AccountPhase.FUNDED
        account = TradeifyAccountState(phase=phase_enum)
        portfolio_trades = []
        daily_pnls = {}

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

            account.start_session(d)
            session_trades = []
            day_pnl = 0.0

            tp = (rth_bars["High"] + rth_bars["Low"] + rth_bars["Close"]) / 3.0
            cum_vol = rth_bars["Volume"].cumsum()
            vwap_series = (tp * rth_bars["Volume"]).cumsum() / cum_vol.replace(0, np.nan)

            df_ind = pd.DataFrame(index=rth_bars.index)
            df_ind["VWAP"] = vwap_series
            u_bands, l_bands = [], []
            for t in rth_bars.index:
                b = prep.get_noise_band(d, t.strftime("%H:%M"), kd=h3_kd)
                u_bands.append(b[0] if b else np.nan)
                l_bands.append(b[1] if b else np.nan)
            df_ind["UpperBand"] = pd.Series(u_bands, index=rth_bars.index).ffill()
            df_ind["LowerBand"] = pd.Series(l_bands, index=rth_bars.index).ffill()

            # H3
            vol_bars = df_day.between_time("08:00", "15:44")
            if vol_bars.empty:
                vol_bars = rth_bars
            resampled_5m = vol_bars.resample("5min", closed="left", label="left").agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
            }).dropna()
            ema9_5m = resampled_5m["Close"].ewm(span=9, adjust=False).mean()

            eval_window_5m = resampled_5m.between_time(h3_start, "13:30")
            rth_sub = df_day.between_time("09:30", "09:30")
            d_open = float(rth_sub.iloc[0]["Open"]) if not rth_sub.empty else prev_close
            prev_high = prep.prev_day_high_map.get(d)
            prev_low = prep.prev_day_low_map.get(d)
            sma_atr20 = prep.daily_atr_sma20_map.get(d, daily_atr)

            h3_day_trades = []
            long_taken = False
            short_taken = False
            last_exit_time = pd.Timestamp.min.tz_localize(df_day.index.tz)
            running_day_pnl = 0.0

            for bar_start, bar_5m in eval_window_5m.iterrows():
                bar_end_time = bar_start + pd.Timedelta(minutes=5)
                if bar_end_time < last_exit_time:
                    continue

                if not allow_v_rev:
                    if long_taken or short_taken:
                        break
                else:
                    if (long_taken and short_taken) or running_day_pnl <= -350.0:
                        break

                time_str = bar_end_time.strftime("%H:%M")
                band = prep.get_noise_band(d, time_str, kd=h3_kd)
                if band is None:
                    continue
                upper_band, lower_band = band
                close_5m = float(bar_5m["Close"])
                prior_vwap_mask = vwap_series.index < bar_end_time
                if not prior_vwap_mask.any():
                    continue
                curr_vwap = float(vwap_series.loc[prior_vwap_mask].iloc[-1])
                curr_ema = float(ema9_5m.loc[bar_start])

                sig = None
                if not long_taken and close_5m > upper_band and close_5m > curr_vwap and curr_ema > curr_vwap:
                    future_1m = df_day.loc[df_day.index >= bar_end_time]
                    if not future_1m.empty:
                        open_1m = float(future_1m.iloc[0]["Open"])
                        entry_price = open_1m + 0.50
                        raw_sl = abs(entry_price - curr_vwap) + 4.0
                        sl_dist = float(np.clip(raw_sl, 22.0, 48.0))
                        sl_price = entry_price - sl_dist
                        sig = TradeSignal(
                            strategy_id="H3", strategy_name="H3_NoiseBand", timestamp=bar_end_time,
                            execution_time=bar_end_time, direction=1, entry_price=entry_price,
                            stop_loss_price=sl_price, tp1_price=None, sl_dist=sl_dist,
                            scale_out_tp1=False, be_trigger_r=None, trail_rule="VWAP_BAND",
                            trail_band_mode="WIDE", trail_activation_r=0.0, hard_time_exit="15:44:00"
                        )
                        long_taken = True
                elif not short_taken and close_5m < lower_band and close_5m < curr_vwap and curr_ema < curr_vwap:
                    is_bull_trap = (prev_high is not None and d_open > prev_high) and (daily_atr < sma_atr20)
                    if not is_bull_trap:
                        future_1m = df_day.loc[df_day.index >= bar_end_time]
                        if not future_1m.empty:
                            open_1m = float(future_1m.iloc[0]["Open"])
                            entry_price = open_1m - 0.50
                            raw_sl = abs(curr_vwap - entry_price) + 4.0
                            sl_dist = float(np.clip(raw_sl, 22.0, 48.0))
                            sl_price = entry_price + sl_dist
                            sig = TradeSignal(
                                strategy_id="H3", strategy_name="H3_NoiseBand", timestamp=bar_end_time,
                                execution_time=bar_end_time, direction=-1, entry_price=entry_price,
                                stop_loss_price=sl_price, tp1_price=None, sl_dist=sl_dist,
                                scale_out_tp1=False, be_trigger_r=None, trail_rule="VWAP_BAND",
                                trail_band_mode="WIDE", trail_activation_r=0.0, hard_time_exit="15:44:00"
                            )
                            short_taken = True

                if sig:
                    # Sizing
                    buffer = account.usable_buffer
                    profit = account.net_profit
                    equity = account.current_equity
                    if account.phase == AccountPhase.EVALUATION:
                        if equity >= 51900.0:
                            r_t, min_c, max_c = 330.0, 4, 8
                        elif profit >= 800.0 and buffer >= 1350.0:
                            r_t, min_c, max_c = 250.0, 3, 6
                        elif buffer >= 950.0:
                            r_t, min_c, max_c = eval_tier1_r, 2, 4
                        else:
                            r_t, min_c, max_c = 120.0, 1, 2
                    else:
                        if buffer < funded_shield_buffer:
                            r_t, min_c, max_c = 90.0, 1, 2
                        elif not account.is_floor_locked and account.eod_hwm < 52100.0:
                            r_t, min_c, max_c = funded_r_prelock, 2, 4
                        else:
                            r_t, min_c, max_c = 230.0, 3, 6

                    unit_r = get_strategy_unit_risk_usd("H3", daily_atr)
                    raw_c = int(np.round(r_t / unit_r))
                    max_safe = int(np.max([1, np.floor((1.35 * r_t) / ((sig.sl_dist * 2.0) + 4.48))]))
                    contracts = int(np.clip(min(raw_c, max_safe), min_c, max_c))

                    c_tr = sim.simulate_trade(sig, df_day, contract_qty=contracts, df_indicators=df_ind)
                    if c_tr:
                        h3_day_trades.extend(c_tr)
                        session_trades.extend(c_tr)
                        t_pnl = sum(t.net_pnl for t in c_tr)
                        running_day_pnl += t_pnl
                        day_pnl += t_pnl
                        last_exit_time = max(t.exit_time for t in c_tr)
                        account.update_intraday_pnl(t_pnl)
                        account.close_trade(t_pnl)
                        if not allow_v_rev:
                            break

            # H4 Late-day
            if not account.is_daily_stopped and not account.is_governor_halted and not account.is_breached:
                time_1000_mask = rth_bars.index.time == pd.Timestamp("10:00:00").time()
                time_1330_mask = rth_bars.index.time == pd.Timestamp("13:30:00").time()
                time_1500_mask = rth_bars.index.time == pd.Timestamp("15:00:00").time()

                if time_1000_mask.any() and time_1500_mask.any():
                    close_1000 = float(rth_bars.loc[time_1000_mask].iloc[-1]["Close"])
                    r_1 = close_1000 - prev_close
                    close_1330 = float(rth_bars.loc[time_1330_mask].iloc[-1]["Close"]) if time_1330_mask.any() else None

                    bar_1500 = rth_bars.loc[time_1500_mask].iloc[-1]
                    t_1500 = rth_bars.loc[time_1500_mask].index[-1]
                    close_1500 = float(bar_1500["Close"])

                    bars_to_1500 = rth_bars.loc[rth_bars.index <= t_1500]
                    range_1500 = float(bars_to_1500["High"].max() - bars_to_1500["Low"].min())

                    if range_1500 > 0:
                        r_rod = (close_1500 - prev_close) / prev_close
                        open_0930 = float(rth_bars.iloc[0]["Open"])
                        norm_move_atr = abs(close_1500 - open_0930) / daily_atr
                        range_pos = (close_1500 - float(bars_to_1500["Low"].min())) / range_1500
                        vwap_1500 = float(vwap_series.loc[t_1500])

                        s_50 = 50.0 * round(close_1500 / 50.0)
                        pin_ok = True
                        if abs(close_1500 - s_50) <= 8.0:
                            pin_bars = df_day.between_time("14:45", "15:00")
                            if not pin_bars.empty and float(pin_bars["High"].max() - pin_bars["Low"].min()) < 15.0:
                                pin_ok = False

                        disp_ok = (abs(r_rod) >= 0.0075 or norm_move_atr >= 0.60)
                        midday_ok_long = (close_1330 is not None and close_1500 >= close_1330)
                        midday_ok_short = (close_1330 is not None and close_1500 <= close_1330)

                        h4_sig = None
                        if pin_ok and disp_ok:
                            future_1m = df_day.loc[df_day.index > t_1500]
                            if not future_1m.empty:
                                sl_dist = float(np.clip(0.14 * range_1500, 16.0, 32.0))
                                open_exec = float(future_1m.iloc[0]["Open"])
                                t_exec = future_1m.index[0]
                                if r_rod > 0 and r_1 > 0 and midday_ok_long and close_1500 > vwap_1500 and range_pos >= 0.78:
                                    h4_sig = TradeSignal(
                                        strategy_id="H4", strategy_name="H4_LETF_Rebalance", timestamp=t_1500,
                                        execution_time=t_exec, direction=1, entry_price=open_exec + 0.50,
                                        stop_loss_price=open_exec + 0.50 - sl_dist, tp1_price=None, sl_dist=sl_dist,
                                        scale_out_tp1=False, be_trigger_r=None, trail_rule="VWAP", hard_time_exit="15:44:00"
                                    )
                                elif r_rod < 0 and r_1 < 0 and midday_ok_short and close_1500 < vwap_1500 and range_pos <= 0.22:
                                    h4_sig = TradeSignal(
                                        strategy_id="H4", strategy_name="H4_LETF_Rebalance", timestamp=t_1500,
                                        execution_time=t_exec, direction=-1, entry_price=open_exec - 0.50,
                                        stop_loss_price=open_exec - 0.50 + sl_dist, tp1_price=None, sl_dist=sl_dist,
                                        scale_out_tp1=False, be_trigger_r=None, trail_rule="VWAP", hard_time_exit="15:44:00"
                                    )

                        if h4_sig:
                            can_execute_h4 = False
                            effective_am_pnl = running_day_pnl
                            h3_still_open = any(t.exit_time > h4_sig.execution_time for t in h3_day_trades)

                            if not h3_still_open:
                                can_execute_h4 = True
                            else:
                                open_h3_tr = [t for t in h3_day_trades if t.exit_time > h4_sig.execution_time][0]
                                if open_h3_tr.direction == h4_sig.direction:
                                    h3_unreal = (close_1500 - open_h3_tr.entry_price) * open_h3_tr.direction * 2.00 * open_h3_tr.contracts - (2.48 * open_h3_tr.contracts)
                                    if account.phase == AccountPhase.EVALUATION:
                                        if h3_unreal >= 100.0:
                                            can_execute_h4 = True
                                    else:
                                        effective_am_pnl = running_day_pnl + max(0.0, 0.70 * h3_unreal)
                                        if effective_am_pnl >= 165.0:
                                            can_execute_h4 = True

                            if can_execute_h4:
                                buffer = account.usable_buffer
                                profit = account.net_profit
                                equity = account.current_equity
                                if account.phase == AccountPhase.EVALUATION:
                                    if equity >= 51900.0:
                                        r_t, min_c, max_c = 330.0, 4, 8
                                    elif profit >= 800.0 and buffer >= 1350.0:
                                        r_t, min_c, max_c = 250.0, 3, 6
                                    elif buffer >= 950.0:
                                        r_t, min_c, max_c = eval_tier1_r, 2, 4
                                    else:
                                        r_t, min_c, max_c = 120.0, 1, 2
                                else:
                                    if buffer < funded_shield_buffer:
                                        r_t, min_c, max_c = 90.0, 1, 2
                                    elif not account.is_floor_locked and account.eod_hwm < 52100.0:
                                        r_t, min_c, max_c = funded_r_prelock, 2, 4
                                    else:
                                        r_t, min_c, max_c = 230.0, 3, 6

                                    if not account.is_floor_locked and account.eod_hwm < 52100.0 and effective_am_pnl >= 165.0:
                                        house_money = effective_am_pnl - 160.0
                                        r_eff = min(r_t, house_money)
                                        h4_c = int(house_money // ((h4_sig.sl_dist * 2.0) + 4.48))
                                        contracts = max(0, min(max_c, h4_c))
                                    else:
                                        contracts = -1

                                if contracts == -1:
                                    unit_r = get_strategy_unit_risk_usd("H4", daily_atr)
                                    raw_c = int(np.round(r_t / unit_r))
                                    max_safe = int(np.max([1, np.floor((1.35 * r_t) / ((h4_sig.sl_dist * 2.0) + 4.48))]))
                                    contracts = int(np.clip(min(raw_c, max_safe), min_c, max_c))

                                if contracts > 0:
                                    c_tr = sim.simulate_trade(h4_sig, df_day, contract_qty=contracts, df_indicators=df_ind)
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
        results[phase_name] = {"metrics": m, "oos": oos_m, "daily_pnls": daily_pnls}

    mc = TradeifyChallengeSimulator(max_evaluation_days=250, max_funded_days=250)
    arr_e = np.array(list(results["EVALUATION"]["daily_pnls"].values()))
    arr_f = np.array(list(results["FUNDED"]["daily_pnls"].values()))
    mc_res = mc.run_monte_carlo("Portfolio Tuning", arr_e, daily_pnl_array_funded=arr_f)

    return results, mc_res


if __name__ == "__main__":
    configs = [
        # (name, kd, start, vrev, funded_r, shield_buf, eval_r)
        ("Conf A (K=1.25, 10:15, PreLock $145, Shield $1150)", 1.25, "10:15", False, 145.0, 1150.0, 160.0),
        ("Conf B (K=1.25, 10:15, VRev=True, PreLock $145, Shield $1150)", 1.25, "10:15", True, 145.0, 1150.0, 160.0),
        ("Conf C (K=1.20, 10:15, PreLock $145, Shield $1150)", 1.20, "10:15", False, 145.0, 1150.0, 160.0),
        ("Conf D (K=1.15, 10:15, PreLock $145, Shield $1150)", 1.15, "10:15", False, 145.0, 1150.0, 160.0),
        ("Conf E (K=1.10, 10:15, PreLock $145, Shield $1150)", 1.10, "10:15", False, 145.0, 1150.0, 160.0),
        ("Conf F (K=1.10, 10:15, PreLock $135, Shield $1200)", 1.10, "10:15", False, 135.0, 1200.0, 160.0),
    ]

    for name, kd, st, vr, fr, sb, er in configs:
        res, mc = test_portfolio_config(h3_kd=kd, h3_start=st, allow_v_rev=vr, funded_r_prelock=fr, funded_shield_buffer=sb, eval_tier1_r=er)
        em = res["EVALUATION"]["metrics"]
        fm = res["FUNDED"]["metrics"]
        print(f"\n=== {name} ===")
        print(f"Eval:   Trades={em['total_trades']}, WR={em['win_rate']:.2f}%, PF={em['profit_factor']:.2f}, PnL=${em['total_pnl']:.2f}, MaxDD=${em['max_drawdown']:.2f}")
        print(f"Funded: Trades={fm['total_trades']}, WR={fm['win_rate']:.2f}%, PF={fm['profit_factor']:.2f}, PnL=${fm['total_pnl']:.2f}, MaxDD=${fm['max_drawdown']:.2f}")
        print(f"MC: 90d Pass={mc.pass_rate_90d:.1f}%, 250d Pass={mc.pass_rate_pct:.1f}%, Breach={mc.breach_rate_pct:.1f}%, MedianDays={mc.median_days_to_pass:.0f}, CondPayout={mc.conditional_payout_rate_pct:.1f}%, UncondPayout={mc.unconditional_payout_rate_pct:.1f}%, NetEV=${mc.expected_net_value:.2f}")
