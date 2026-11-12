import sys
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
from engine.tradeify_rules import TradeifyAccountState, AccountPhase, TradeifyAccountConfig, get_strategy_unit_risk_usd
from engine.monte_carlo import TradeifyChallengeSimulator
from engine.run_phase5_audit import compute_comprehensive_metrics
from experiments.ablation_phase6_frequency import simulate_h3_variant, simulate_h4_variant, compute_metrics


def calculate_phase6_contract_size(
    account: TradeifyAccountState,
    strategy_id: str,
    sl_dist_pts: float,
    atr_14: float,
    morning_pnl: float = 0.0
) -> int:
    if account.is_breached or account.is_daily_stopped or account.is_governor_halted:
        return 0

    buffer = account.usable_buffer
    profit = account.net_profit
    current_equity = account.current_equity
    atr = atr_14 if (atr_14 is not None and atr_14 > 0) else 200.0

    if account.phase == AccountPhase.EVALUATION:
        # Regime 1: Evaluation Sprint Tiers
        if current_equity >= 51900.0:
            r_target, min_c, max_c = 330.0, 4, 8
        elif profit >= 800.0 and buffer >= 1350.0:
            r_target, min_c, max_c = 250.0, 3, 6
        elif buffer >= 950.0:
            r_target, min_c, max_c = 170.0, 2, 4
        else:
            r_target, min_c, max_c = 120.0, 1, 2
    else:
        # Regime 2: Funded Capital Preservation Tiers
        if buffer < 1050.0:
            r_target, min_c, max_c = 90.0, 1, 2
        elif not account.is_floor_locked and account.eod_hwm < account.config.FLOOR_LOCK_TRIGGER:
            r_target, min_c, max_c = 165.0, 2, 4
        else:
            r_target, min_c, max_c = 245.0, 3, 6

        # House-Money Floor Guard for H4 in Funded Pre-Lock Mode
        if strategy_id == "H4" and not account.is_floor_locked and account.eod_hwm < account.config.FLOOR_LOCK_TRIGGER:
            if morning_pnl >= 165.0:
                house_money_budget = morning_pnl - 160.0
                r_effective = min(r_target, house_money_budget)
                per_c_risk = (sl_dist_pts * 2.00) + 4.48
                h4_contracts = int(np.floor(r_effective / per_c_risk))
                return max(0, min(max_c, h4_contracts))

    unit_risk_usd = get_strategy_unit_risk_usd(strategy_id, atr)
    raw_contracts = int(np.round(r_target / unit_risk_usd))

    per_contract_sl_risk = (sl_dist_pts * 2.00) + 4.48
    max_safe_contracts = int(np.max([1, np.floor((1.35 * r_target) / per_contract_sl_risk)]))

    contracts = int(np.clip(min(raw_contracts, max_safe_contracts), min_c, max_c))
    return contracts


def run_portfolio_simulation(h3_kd=1.10, allow_v_rev=False, h4_rod=0.0075, h4_atr=0.60, h4_rp_l=0.78, h4_rp_s=0.22):
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

            # Compute VWAP and Noise Bands
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

            # 1. Evaluate H3 signals
            vol_bars = df_day.between_time("08:00", "15:44")
            if vol_bars.empty:
                vol_bars = rth_bars
            resampled_5m = vol_bars.resample("5min", closed="left", label="left").agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
            }).dropna()
            ema9_5m = resampled_5m["Close"].ewm(span=9, adjust=False).mean()
            vol_sma_5m = resampled_5m["Volume"].rolling(window=12, min_periods=4).mean()

            eval_window_5m = resampled_5m.between_time("10:15", "13:30")
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
                            trail_activation_r=0.0,
                            hard_time_exit="15:44:00"
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
                                trail_activation_r=0.0,
                                hard_time_exit="15:44:00"
                            )
                            short_taken = True

                if sig:
                    contracts = calculate_phase6_contract_size(account, "H3", sig.sl_dist, daily_atr)
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

            # 2. Evaluate H4 late-day signal (15:00:00 NY)
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
                    high_1500 = float(bars_to_1500["High"].max())
                    low_1500 = float(bars_to_1500["Low"].min())
                    range_1500 = high_1500 - low_1500

                    if range_1500 > 0:
                        r_rod = (close_1500 - prev_close) / prev_close
                        open_0930 = float(rth_bars.iloc[0]["Open"])
                        norm_move_atr = abs(close_1500 - open_0930) / daily_atr
                        range_pos = (close_1500 - low_1500) / range_1500
                        vwap_1500 = float(vwap_series.loc[t_1500])

                        s_50 = 50.0 * round(close_1500 / 50.0)
                        pin_ok = True
                        if abs(close_1500 - s_50) <= 8.0:
                            pin_bars = df_day.between_time("14:45", "15:00")
                            if not pin_bars.empty and float(pin_bars["High"].max() - pin_bars["Low"].min()) < 15.0:
                                pin_ok = False

                        disp_ok = (abs(r_rod) >= h4_rod or norm_move_atr >= h4_atr)

                        midday_ok_long = (close_1330 is not None and close_1500 >= close_1330)
                        midday_ok_short = (close_1330 is not None and close_1500 <= close_1330)

                        h4_sig = None
                        if pin_ok and disp_ok:
                            future_1m = df_day.loc[df_day.index > t_1500]
                            if not future_1m.empty:
                                sl_dist = float(np.clip(0.14 * range_1500, 16.0, 32.0))
                                open_exec = float(future_1m.iloc[0]["Open"])
                                t_exec = future_1m.index[0]

                                if r_rod > 0 and r_1 > 0 and midday_ok_long and close_1500 > vwap_1500 and range_pos >= h4_rp_l:
                                    entry_price = open_exec + 0.50
                                    h4_sig = TradeSignal(
                                        strategy_id="H4",
                                        strategy_name="H4_LETF_Rebalance",
                                        timestamp=t_1500,
                                        execution_time=t_exec,
                                        direction=1,
                                        entry_price=entry_price,
                                        stop_loss_price=entry_price - sl_dist,
                                        tp1_price=None,
                                        sl_dist=sl_dist,
                                        scale_out_tp1=False,
                                        be_trigger_r=None,
                                        trail_rule="VWAP",
                                        hard_time_exit="15:44:00"
                                    )
                                elif r_rod < 0 and r_1 < 0 and midday_ok_short and close_1500 < vwap_1500 and range_pos <= h4_rp_s:
                                    entry_price = open_exec - 0.50
                                    h4_sig = TradeSignal(
                                        strategy_id="H4",
                                        strategy_name="H4_LETF_Rebalance",
                                        timestamp=t_1500,
                                        execution_time=t_exec,
                                        direction=-1,
                                        entry_price=entry_price,
                                        stop_loss_price=entry_price + sl_dist,
                                        tp1_price=None,
                                        sl_dist=sl_dist,
                                        scale_out_tp1=False,
                                        be_trigger_r=None,
                                        trail_rule="VWAP",
                                        hard_time_exit="15:44:00"
                                    )

                        if h4_sig:
                            # STEP 1: CONCURRENT SAME-DIRECTION PYRAMIDING CHECK
                            can_execute_h4 = False
                            effective_am_pnl = running_day_pnl

                            # Check if an H3 trade is still open at 15:00
                            h3_still_open = any(t.exit_time > h4_sig.execution_time for t in h3_day_trades)

                            if not h3_still_open:
                                # Normal execution: H3 has already closed or didn't trade
                                can_execute_h4 = True
                            else:
                                # H3 is currently open
                                open_h3_tr = [t for t in h3_day_trades if t.exit_time > h4_sig.execution_time][0]
                                if open_h3_tr.direction == h4_sig.direction:
                                    # Same direction pyramiding!
                                    h3_unreal = (close_1500 - open_h3_tr.entry_price) * open_h3_tr.direction * 2.00 * open_h3_tr.contracts - (2.48 * open_h3_tr.contracts)
                                    if account.phase == AccountPhase.EVALUATION:
                                        if h3_unreal >= 100.0:
                                            can_execute_h4 = True
                                    else: # FUNDED
                                        effective_am_pnl = running_day_pnl + max(0.0, 0.70 * h3_unreal)
                                        if effective_am_pnl >= 165.0:
                                            can_execute_h4 = True

                            if can_execute_h4:
                                contracts = calculate_phase6_contract_size(
                                    account, "H4", h4_sig.sl_dist, daily_atr, morning_pnl=effective_am_pnl
                                )
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
        print(f"--- {phase_name} --- | Trades: {m['total_trades']:<4} | WR: {m['win_rate']:>5.2f}% | PF: {m['profit_factor']:>4.2f} | Exp: ${m['expectancy']:>6.2f} | Total PnL: ${m['total_pnl']:>9.2f} | OOS: ${oos_m['total_pnl']:>8.2f} | MaxDD: ${m['max_drawdown']:>7.2f}")

    # Monte Carlo
    print("\nRunning 10,000 Monte Carlo simulations...")
    mc = TradeifyChallengeSimulator(max_evaluation_days=250, max_funded_days=250)
    arr_e = np.array(list(results["EVALUATION"]["daily_pnls"].values()))
    arr_f = np.array(list(results["FUNDED"]["daily_pnls"].values()))
    mc_res = mc.run_monte_carlo("Phase 6 Master Portfolio", arr_e, daily_pnl_array_funded=arr_f)

    print(f"Pass Rate (90d): {mc_res.pass_rate_90d:.1f}%")
    print(f"Pass Rate (180d): {mc_res.pass_rate_180d:.1f}%")
    print(f"Pass Rate (250d): {mc_res.pass_rate_pct:.1f}%")
    print(f"Breach Rate: {mc_res.breach_rate_pct:.1f}%")
    print(f"Median Days to Pass: {mc_res.median_days_to_pass:.0f}")
    print(f"Conditional Funded Payout Rate: {mc_res.conditional_payout_rate_pct:.1f}%")
    print(f"Unconditional Payout Rate: {mc_res.unconditional_payout_rate_pct:.1f}%")
    print(f"Mean Days to Payout: {mc_res.mean_days_to_payout:.1f}")
    print(f"Net EV per $87 Ticket: ${mc_res.expected_net_value:.2f}")


if __name__ == "__main__":
    print("Testing Configuration 1: H3 (K=1.10, 10:15-13:30, No VRev) + H4 (15:00 Calibrated)")
    run_portfolio_simulation(h3_kd=1.10, allow_v_rev=False)
