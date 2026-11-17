import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd
import numpy as np
from analytics import fetch_stock_data, add_technical_indicators, generate_signals

for tf in ["15m", "1h"]:
    df, used_symbol, _ = fetch_stock_data("TCS", period="60d", interval=tf)
    df = add_technical_indicators(df)
    signals = generate_signals(df)
    
    # 22 Sep to 09 Oct window
    sept22_oct9_indices = [
        i for i in range(len(df))
        if "2026-09-22" <= df.index[i].strftime("%Y-%m-%d") <= "2026-10-09"
        and signals.iloc[i]['Signal'] in ['BUY', 'STRONG BUY', 'SELL', 'STRONG SELL']
    ]
    
    print(f"\n{'='*70}")
    print(f"TCS ON {tf.upper()} TIMEFRAME (22 Sep - 09 Oct): Total Signals = {len(sept22_oct9_indices)}")
    print(f"{'='*70}")
    
    results = []
    for idx in sept22_oct9_indices:
        row = df.iloc[idx]
        sig = signals.iloc[idx]['Signal']
        is_buy = 'BUY' in sig
        c = float(row['Close'])
        atr = float(row['ATR'])
        
        # On 15m/1h, targets are naturally larger in points!
        # T1 = 1.2x ATR
        # T2 = 2.4x ATR
        # SL = 1.0x ATR
        t1_pts = round(1.2 * atr, 2)
        t2_pts = round(2.4 * atr, 2)
        sl_pts = round(1.0 * atr, 2)
        
        t1 = round(c + t1_pts if is_buy else c - t1_pts, 2)
        t2 = round(c + t2_pts if is_buy else c - t2_pts, 2)
        sl = round(c - sl_pts if is_buy else c + sl_pts, 2)
        
        future = df.iloc[idx+1 : idx+25]
        outcome = "Pending"
        pnl = 0
        curr_sl = sl
        t1_hit = False
        
        for _, f_row in future.iterrows():
            h = float(f_row['High'])
            l = float(f_row['Low'])
            
            if is_buy:
                if not t1_hit and h >= t1:
                    t1_hit = True
                    curr_sl = c + (0.3 * atr)
                if t1_hit and h >= t2:
                    outcome = "T2 Full Hit"
                    pnl = (0.5 * t1_pts) + (0.5 * t2_pts)
                    break
                if l <= curr_sl:
                    if t1_hit:
                        outcome = "T1 Hit + Trail"
                        pnl = (0.5 * t1_pts) + (0.5 * (curr_sl - c))
                    else:
                        outcome = "SL Hit"
                        pnl = -sl_pts
                    break
            else:
                if not t1_hit and l <= t1:
                    t1_hit = True
                    curr_sl = c - (0.3 * atr)
                if t1_hit and l <= t2:
                    outcome = "T2 Full Hit"
                    pnl = (0.5 * t1_pts) + (0.5 * t2_pts)
                    break
                if h >= curr_sl:
                    if t1_hit:
                        outcome = "T1 Hit + Trail"
                        pnl = (0.5 * t1_pts) + (0.5 * (c - curr_sl))
                    else:
                        outcome = "SL Hit"
                        pnl = -sl_pts
                    break
                    
        if outcome != "Pending":
            results.append({
                "time": str(df.index[idx])[:16],
                "sig": "BUY" if is_buy else "SELL",
                "entry": c,
                "t1_pts": t1_pts,
                "t2_pts": t2_pts,
                "sl_pts": sl_pts,
                "outcome": outcome,
                "pnl": round(pnl, 2)
            })
            
    for r in results:
        print(f"{r['time']} | {r['sig']:4s} @ Rs.{r['entry']:.2f} | T1: +{r['t1_pts']:.2f} pts | T2: +{r['t2_pts']:.2f} pts | SL: -{r['sl_pts']:.2f} pts | {r['outcome']} -> PnL: {r['pnl']:+.2f} pts")
        
    wins = [r for r in results if "Hit" in r['outcome'] and "SL" not in r['outcome']]
    sls = [r for r in results if r['outcome'] == "SL Hit"]
    if results:
        print(f"\nWin Rate: {len(wins)/len(results)*100:.1f}% ({len(wins)}/{len(results)}) | Net PnL: {sum(r['pnl'] for r in results):+.2f} pts")
