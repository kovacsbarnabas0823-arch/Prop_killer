import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from strategies.orb_5m.backtest.sweep_engine_5m import SessionPrecompiler
import numpy as np
import pandas as pd

sessions = SessionPrecompiler.load_sessions()

records = []
for s in sessions:
    direction = 1 if s['c_close'] > s['c_ema'] else -1
    c_open = s['c_open']
    c_high = s['c_high']
    c_low = s['c_low']
    c_close = s['c_close']
    c_ema = s['c_ema']
    c_range = max(c_high - c_low, 0.25)
    
    # Feature 1: Candle color agreement
    color_agree = (c_close > c_open) if direction == 1 else (c_close < c_open)
    
    # Feature 2: Close location within candle (0 = low, 1 = high)
    close_loc = (c_close - c_low) / c_range
    strong_close = (close_loc >= 0.65) if direction == 1 else (close_loc <= 0.35)
    
    dist_ema = abs(c_close - c_ema)
    
    opens = s['opens']
    closes = s['closes']
    highs = s['highs']
    lows = s['lows']
    if len(opens) < 60:
        continue
    
    # 60 min return
    ret_60 = (closes[59] - opens[0]) if direction == 1 else (opens[0] - closes[59])
    mfe_60 = (np.max(highs[:60]) - opens[0]) if direction == 1 else (opens[0] - np.min(lows[:60]))
    mae_60 = (opens[0] - np.min(lows[:60])) if direction == 1 else (np.max(highs[:60]) - opens[0])
    
    # Retest of 5m midpoint or open within first 15 mins
    midpoint = (c_high + c_low) / 2.0
    if direction == 1:
        retested_mid = np.min(lows[:15]) <= midpoint
        retested_ema = np.min(lows[:15]) <= c_ema
    else:
        retested_mid = np.max(highs[:15]) >= midpoint
        retested_ema = np.max(highs[:15]) >= c_ema

    records.append({
        'direction': direction,
        'color_agree': color_agree,
        'strong_close': strong_close,
        'retested_mid': retested_mid,
        'retested_ema': retested_ema,
        'range': c_range,
        'dist_ema': dist_ema,
        'ret_60': ret_60,
        'mfe_60': mfe_60,
        'mae_60': mae_60
    })

df = pd.DataFrame(records)

print('=== STATISTICAL BREAKDOWN OF 12 EMA SIGNAL (60 MIN HORIZON) ===')
mean_ret = df['ret_60'].mean()
wr = (df['ret_60'] > 0).mean()
print(f'Overall Baseline ({len(df)} days): Mean Ret = {mean_ret:+6.2f} pts | Win Rate = {wr:.1%}')

for col in ['color_agree', 'strong_close', 'retested_mid', 'retested_ema']:
    for val in [True, False]:
        sub = df[df[col] == val]
        m = sub['ret_60'].mean()
        w = (sub['ret_60'] > 0).mean()
        print(f'{col:13s} == {str(val):5s}: Count = {len(sub):4d} ({len(sub)/len(df):.1%}) | Mean Ret = {m:+6.2f} pts | Win Rate = {w:.1%} | Avg MFE = {sub["mfe_60"].mean():.1f} | Avg MAE = {sub["mae_60"].mean():.1f}')

# Range quartiles
df['range_bin'] = pd.qcut(df['range'], 4, labels=['Q1_low', 'Q2_med_low', 'Q3_med_high', 'Q4_high'])
print('\nBy 5m Candle Range Quartile:')
for rbin, sub in df.groupby('range_bin', observed=False):
    print(f'Range {rbin:11s}: Range [{sub["range"].min():.1f} - {sub["range"].max():.1f}] | Count = {len(sub):4d} | Mean Ret = {sub["ret_60"].mean():+6.2f} pts | Win Rate = {(sub["ret_60"] > 0).mean():.1%}')

# Distance from EMA quartiles
df['dist_bin'] = pd.qcut(df['dist_ema'], 4, labels=['Q1_near', 'Q2_mid_near', 'Q3_mid_far', 'Q4_far'])
print('\nBy Distance from 12 EMA Quartile:')
for dbin, sub in df.groupby('dist_bin', observed=False):
    print(f'Dist {dbin:11s}: Dist [{sub["dist_ema"].min():.1f} - {sub["dist_ema"].max():.1f}] | Count = {len(sub):4d} | Mean Ret = {sub["ret_60"].mean():+6.2f} pts | Win Rate = {(sub["dist_ema"] > 0).mean():.1%}')
