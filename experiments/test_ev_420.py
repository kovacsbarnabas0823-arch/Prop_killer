import sys
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.optimize_tiers_ev import run_tier_opt

tests = [
    ("Candidate 1 (Eval: 120/170/230/280 max 6c | Funded: 75/125/180 max 4c)",
     (120.0, 170.0, 230.0, 280.0), ((1,2), (2,3), (3,4), (3,6)),
     (75.0, 125.0, 180.0), ((1,2), (2,3), (2,4)), 1150.0, 1.10),
    ("Candidate 2 (Eval: 120/170/240/300 max 6c | Funded: 75/120/180 max 4c)",
     (120.0, 170.0, 240.0, 300.0), ((1,2), (2,3), (3,4), (3,6)),
     (75.0, 120.0, 180.0), ((1,2), (2,3), (2,4)), 1150.0, 1.10),
    ("Candidate 3 (Eval: 120/170/240/300 max 6c | Funded: 80/130/190 max 4c, Shield $1150)",
     (120.0, 170.0, 240.0, 300.0), ((1,2), (2,3), (3,4), (3,6)),
     (80.0, 130.0, 190.0), ((1,2), (2,3), (2,4)), 1150.0, 1.10),
    ("Candidate 4 (Eval: 120/165/225/270 max 6c | Funded: 75/125/185 max 4c)",
     (120.0, 165.0, 225.0, 270.0), ((1,2), (2,3), (3,4), (3,6)),
     (75.0, 125.0, 185.0), ((1,2), (2,3), (2,4)), 1150.0, 1.10),
]

for name, et, ec, ft, fc, sb, kd in tests:
    res, mc = run_tier_opt(eval_tiers=et, eval_tier_caps=ec, funded_tiers=ft, funded_tier_caps=fc, funded_shield_buf=sb, h3_kd=kd)
    em = res["EVALUATION"]["metrics"]
    fm = res["FUNDED"]["metrics"]
    print(f"\n=== {name} ===")
    print(f"Eval:   Trades={em['total_trades']}, WR={em['win_rate']:.2f}%, PF={em['profit_factor']:.2f}, PnL=${em['total_pnl']:.2f}, MaxDD=${em['max_drawdown']:.2f}")
    print(f"Funded: Trades={fm['total_trades']}, WR={fm['win_rate']:.2f}%, PF={fm['profit_factor']:.2f}, PnL=${fm['total_pnl']:.2f}, MaxDD=${fm['max_drawdown']:.2f}")
    print(f"MC: 90d Pass={mc.pass_rate_90d:.1f}%, 250d Pass={mc.pass_rate_pct:.1f}%, Breach={mc.breach_rate_pct:.1f}%, MedianDays={mc.median_days_to_pass:.0f}, CondPayout={mc.conditional_payout_rate_pct:.1f}%, UncondPayout={mc.unconditional_payout_rate_pct:.1f}%, NetEV=${mc.expected_net_value:.2f}")
