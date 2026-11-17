import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd
import numpy as np
from analytics import fetch_stock_data, add_technical_indicators, generate_signals

df, used_symbol, _ = fetch_stock_data("TCS", period="60d", interval="5m")
df = add_technical_indicators(df)
signals = generate_signals(df)

# Look at all BUY/SELL signals between 22 Sep and 09 Oct
# What was the MAXIMUM favorable excursion (MFE) in the next 15-30 candles?
sept22_oct9_indices = [
    i for i in range(len(df))
    if "2026-09-22" <= df.index[i].strftime("%Y-%m-%d") <= "2026-10-09"
    and signals.iloc[i]['Signal'] in ['BUY', 'STRONG BUY', 'SELL', 'STRONG SELL']
]

print(f"Total signals in window: {len(sept22_oct9_indices)}")
results = []
for idx in sept22_oct9_indices:
    row = df.iloc[idx]
    sig = signals.iloc[idx]['Signal']
    is_buy = 'BUY' in sig
    entry = float(row['Close'])
    atr = float(row['ATR'])
    
    # Check max favorable excursion (MFE) and max adverse excursion (MAE) over next 25 candles
    future = df.iloc[idx+1 : idx+26]
    if len(future) == 0: continue
    
    if is_buy:
        max_fav = future['High'].max() - entry
        max_adv = entry - future['Low'].min()
    else:
        max_fav = entry - future['Low'].min()
        max_adv = future['High'].max() - entry
        
    results.append({
        "time": str(df.index[idx])[:16],
        "sig": "BUY" if is_buy else "SELL",
        "entry": entry,
        "atr": atr,
        "mfe_pts": max_fav,
        "mfe_pct": (max_fav / entry) * 100,
        "mae_pts": max_adv,
        "mae_pct": (max_adv / entry) * 100,
    })

res_df = pd.DataFrame(results)
print(res_df.to_string())
print("\n" + "="*50)
print(f"Average Maximum Favorable Move (MFE): {res_df['mfe_pts'].mean():.2f} pts ({res_df['mfe_pct'].mean():.2f}%)")
print(f"Median Maximum Favorable Move: {res_df['mfe_pts'].median():.2f} pts")
print(f"Max Favorable Move: {res_df['mfe_pts'].max():.2f} pts")
print(f"Average Adverse Move (MAE): {res_df['mae_pts'].mean():.2f} pts ({res_df['mae_pct'].mean():.2f}%)")
print("="*50)
