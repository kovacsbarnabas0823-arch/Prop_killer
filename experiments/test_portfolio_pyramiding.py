import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datetime import date
import numpy as np
import pandas as pd
from data.session_preprocessor import get_preprocessor
from strategies.base_strategy import TradeSignal, CompletedTrade
from strategies.h3_noise_band_momentum import H3NoiseBandMomentum
from strategies.h4_letf_1500_rebalance import H4LETF1500Rebalance
from engine.pessimistic_sim import PessimisticExecutionSimulator
from engine.tradeify_rules import TradeifyAccountState, AccountPhase, TradeifyAccountConfig, get_strategy_unit_risk_usd
from engine.monte_carlo import TradeifyChallengeSimulator
from engine.run_phase5_audit import compute_comprehensive_metrics

prep = get_preprocessor()
all_dates = sorted(list(prep.daily_atr_map.keys()))

print("Testing H3 with 10:15 - 14:00 NY and V-reversal...")
from experiments.ablation_phase6_frequency import simulate_h3_variant, compute_metrics

for kd in [1.25, 1.15, 1.10, 1.05, 1.00]:
    for et in ["13:30", "14:00"]:
        for vr in [False, True]:
            name = f"K={kd:.2f} | End={et} | VRev={vr}"
            tr = simulate_h3_variant(prep, all_dates, start_time="10:15", end_time=et, kd=kd, vol_filter=False, seeck_mode="surgical", allow_v_reversal=vr)
            m = compute_metrics(tr)
            oos_tr = [t for t in tr if t.entry_time.date() >= date(2023, 1, 1)]
            oos_m = compute_metrics(oos_tr)
            print(f"{name:<35} | Tr: {m['total_trades']:<4} | WR: {m['win_rate']:>5.2f}% | PF: {m['profit_factor']:>4.2f} | Exp: ${m['expectancy']:>5.2f} | PnL: ${m['total_pnl']:>7.2f} | OOS: ${oos_m['total_pnl']:>7.2f}")
