import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.session_preprocessor import get_preprocessor
from experiments.ablation_phase6_frequency import simulate_h4_variant, compute_metrics
from datetime import date

prep = get_preprocessor()
all_dates = sorted(list(prep.daily_atr_map.keys()))

grid = [
    ('15:00 (0.0075 / 0.60 / 0.78)', ['15:00'], 0.0075, 0.60, 0.78, 0.22),
    ('15:00 (0.0070 / 0.55 / 0.77)', ['15:00'], 0.0070, 0.55, 0.77, 0.23),
    ('15:00 (0.0065 / 0.52 / 0.76)', ['15:00'], 0.0065, 0.52, 0.76, 0.24),
    ('15:00 (0.0062 / 0.50 / 0.75)', ['15:00'], 0.0062, 0.50, 0.75, 0.25),
    ('15:00 (0.0058 / 0.48 / 0.74)', ['15:00'], 0.0058, 0.48, 0.74, 0.26),
    ('15:00 (0.0055 / 0.45 / 0.72)', ['15:00'], 0.0055, 0.45, 0.72, 0.28),
    # Also test if 15:15 works with tighter displacement (e.g. accelerating trend)
    ('Dual-CP (15:00 0.0075, 15:15 strict 0.010)', ['15:00', '15:15'], 0.0075, 0.60, 0.80, 0.20),
    ('14:45 checkpoint test', ['14:45'], 0.0075, 0.60, 0.78, 0.22),
]

for name, cps, rod, atr_m, rp_l, rp_s in grid:
    tr = simulate_h4_variant(prep, all_dates, checkpoints=cps, rod_thresh=rod, atr_move_thresh=atr_m, range_pos_long=rp_l, range_pos_short=rp_s)
    m = compute_metrics(tr)
    oos_tr = [t for t in tr if t.entry_time.date() >= date(2023, 1, 1)]
    oos_m = compute_metrics(oos_tr)
    print(f"{name:<45} | Tr: {m['total_trades']:<4} | WR: {m['win_rate']:>5.2f}% | PF: {m['profit_factor']:>4.2f} | Exp: ${m['expectancy']:>5.2f} | PnL: ${m['total_pnl']:>7.2f} | OOS: ${oos_m['total_pnl']:>7.2f}")
