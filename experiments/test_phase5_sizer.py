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
from strategies.h3_noise_band_momentum import H3NoiseBandMomentum
from strategies.h4_letf_1500_rebalance import H4LETF1500Rebalance
from engine.pessimistic_sim import PessimisticExecutionSimulator
from engine.tradeify_rules import TradeifyAccountState, AccountPhase, TradeifyAccountConfig
from engine.monte_carlo import TradeifyChallengeSimulator
from engine.run_phase3_audit import compute_comprehensive_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestPhase5Sizer")


def get_strategy_unit_risk_usd(strategy_id: str, atr_14: float) -> float:
    if strategy_id == "H4":
        ref_pts_h4 = float(np.clip(0.09 * atr_14, 16.0, 22.0))
        return (ref_pts_h4 * 2.00) + 4.48
    else:  # H3
        ref_pts_h3 = float(np.clip(0.18 * atr_14, 30.0, 40.0))
        return (ref_pts_h3 * 2.00) + 4.48


def calculate_phase5_contract_size(
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
        elif not account.is_floor_locked and account.eod_hwm < 52100.0:
            r_target, min_c, max_c = 165.0, 2, 4
        else:
            r_target, min_c, max_c = 245.0, 3, 6

        # Step 2: House-Money Floor Guard for H4 in Funded Mode
        if strategy_id == "H4" and not account.is_floor_locked and account.eod_hwm < 52100.0:
            if morning_pnl >= 165.0:
                house_money_budget = morning_pnl - 160.0
                r_effective = min(r_target, house_money_budget)
                per_c_risk = (sl_dist_pts * 2.00) + 4.48
                h4_contracts = int(np.floor(r_effective / per_c_risk))
                return max(0, min(max_c, h4_contracts))

    unit_risk_usd = get_strategy_unit_risk_usd(strategy_id, atr_14)
    raw_contracts = int(np.round(r_target / unit_risk_usd))

    per_contract_sl_risk = (sl_dist_pts * 2.00) + 4.48
    max_safe_contracts = int(np.max([1, np.floor((1.35 * r_target) / per_contract_sl_risk)]))

    contracts = int(np.clip(min(raw_contracts, max_safe_contracts), min_c, max_c))
    return contracts


def simulate_phase5():
    prep = get_preprocessor()
    all_dates = sorted(list(prep.daily_atr_map.keys()))

    # Initialize strategies
    h3 = H3NoiseBandMomentum()
    # H4 with Opt 3 calibrated settings (0.58 close / 0.60 open / 0.0075 rod, 0.78/0.22)
    h4 = H4LETF1500Rebalance(
        rod_thresh=0.0075,
        atr_move_thresh=0.60,
        range_pos_long=0.78,
        range_pos_short=0.22
    )

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
            atr = prep.daily_atr_map.get(d)
            prev_close = prep.prev_day_close_map.get(d)
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
                if d in prep.noise_band_map:
                    u_bands, l_bands = [], []
                    for t in rth_bars.index:
                        b = prep.get_noise_band(d, t.strftime("%H:%M"))
                        u_bands.append(b[0] if b else np.nan)
                        l_bands.append(b[1] if b else np.nan)
                    df_ind["UpperBand"] = pd.Series(u_bands, index=rth_bars.index).ffill()
                    df_ind["LowerBand"] = pd.Series(l_bands, index=rth_bars.index).ffill()

            # 1. H3 morning signal
            morning_trade_pnl = 0.0
            h3_sigs = h3.evaluate_session(d, df_day, atr, prev_close, preprocessor=prep)
            for s in h3_sigs:
                contracts = calculate_phase5_contract_size(account, "H3", s.sl_dist, atr)
                c_tr = sim.simulate_trade(s, df_day, contract_qty=contracts, df_indicators=df_ind)
                if c_tr:
                    session_trades.extend(c_tr)
                    t_pnl = sum(t.net_pnl for t in c_tr)
                    day_pnl += t_pnl
                    morning_trade_pnl += t_pnl
                    account.update_intraday_pnl(t_pnl)
                    account.close_trade(t_pnl)

            # 2. H4 late-day signal (with House-Money Floor Guard)
            if not account.is_daily_stopped and not account.is_governor_halted:
                h4_sigs = h4.evaluate_session(d, df_day, atr, prev_close, preprocessor=prep)
                for s in h4_sigs:
                    contracts = calculate_phase5_contract_size(account, "H4", s.sl_dist, atr, morning_pnl=morning_trade_pnl)
                    if contracts <= 0:
                        continue
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
        results[phase_name] = {"metrics": m, "oos": oos_m, "daily_pnls": daily_pnls}

        print(f"\n--- PHASE 5 {phase_name} PORTFOLIO ---")
        print(f"Trades: {m['total_trades']}, WR: {m['win_rate']:.2f}%, PF: {m['profit_factor']:.2f}, Exp: ${m['expectancy']:.2f}/tr, TotalPnL: ${m['total_pnl']:.2f}, OOS: ${oos_m['total_pnl']:.2f}, MaxDD: ${m['max_drawdown']:.2f}")

    # Run Monte Carlo
    print("\nRunning 10,000 Monte Carlo simulations...")
    mc = TradeifyChallengeSimulator(max_evaluation_days=250, max_funded_days=250)
    arr_e = np.array(list(results["EVALUATION"]["daily_pnls"].values()))
    arr_f = np.array(list(results["FUNDED"]["daily_pnls"].values()))
    mc_res = mc.run_monte_carlo("Phase 5 Master Portfolio", arr_e, daily_pnl_array_funded=arr_f)

    print("\n--- PHASE 5 MONTE CARLO RESULTS ---")
    print(f"Pass Rate (90d): {mc_res.pass_rate_90d:.1f}%")
    print(f"Pass Rate (180d): {mc_res.pass_rate_180d:.1f}%")
    print(f"Pass Rate (250d): {mc_res.pass_rate_pct:.1f}%")
    print(f"Breach Rate: {mc_res.breach_rate_pct:.1f}%")
    print(f"Median Days to Pass: {mc_res.median_days_to_pass:.0f}")
    print(f"Expected Cost per Pass: ${mc_res.avg_fee_cost_per_pass:.2f}")
    print(f"Conditional Funded Payout Rate: {mc_res.conditional_payout_rate_pct:.1f}%")
    print(f"Unconditional Payout Rate: {mc_res.unconditional_payout_rate_pct:.1f}%")
    print(f"Mean Days to Payout: {mc_res.mean_days_to_payout:.1f}")
    print(f"Net EV per $87 Ticket: ${mc_res.expected_net_value:.2f}")


if __name__ == "__main__":
    simulate_phase5()
