import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd
import numpy as np
from analytics import fetch_stock_data, add_technical_indicators, generate_signals

def run_filtered_test():
    df, used_symbol, _ = fetch_stock_data("TCS", period="60d", interval="5m")
    df = add_technical_indicators(df)
    signals = generate_signals(df)
    
    # 22 Sep to 09 Oct window
    sept22_oct9_indices = [
        i for i in range(len(df))
        if "2026-09-22" <= df.index[i].strftime("%Y-%m-%d") <= "2026-10-09"
        and signals.iloc[i]['Signal'] in ['BUY', 'STRONG BUY', 'SELL', 'STRONG SELL']
    ]
    
    results = []
    for idx in sept22_oct9_indices:
        t_dt = df.index[idx]
        hour = t_dt.hour
        minute = t_dt.minute
        row = df.iloc[idx]
        sig = signals.iloc[idx]['Signal']
        is_buy = 'BUY' in sig
        c = float(row['Close'])
        atr = float(row['ATR'])
        div = int(row.get('Divergence_Signal', 0))
        vwap = float(row.get('VWAP', c))
        ema9 = float(row.get('EMA_9', c))
        ema21 = float(row.get('EMA_21', c))
        ema50 = float(row.get('EMA_50', c))
        rsi = float(row.get('RSI', 50))
        adx = float(row.get('ADX', 20))
        
        # 1. Confluence Filter
        if is_buy:
            if div == -1 or c < vwap or ema9 < ema21:
                continue
            # Avoid buy when RSI is weak or severely overbought
            if rsi < 50 or rsi > 78:
                continue
        else:
            if div == 1 or c > vwap or ema9 > ema21:
                continue
            if rsi > 50 or rsi < 22:
                continue
                
        # 2. Time-of-day filter: Skip extreme opening 15m whipsaw (09:15 - 09:30)
        if hour == 9 and minute < 30:
            continue
            
        # Realistic Intraday Targets:
        # Target 1 = 1.4x ATR (Books 50% quick profit, approx 6-10 pts)
        # Target 2 = 2.8x ATR (Captures full swing, approx 12-22 pts)
        # Stop Loss = 1.3x ATR (approx 5-8 pts)
        t1_pts = round(1.4 * atr, 2)
        t2_pts = round(2.8 * atr, 2)
        sl_pts = round(1.3 * atr, 2)
        
        t1_price = round(c + t1_pts if is_buy else c - t1_pts, 2)
        t2_price = round(c + t2_pts if is_buy else c - t2_pts, 2)
        sl_price = round(c - sl_pts if is_buy else c + sl_pts, 2)
        
        curr_sl = sl_price
        t1_hit = False
        outcome = "Pending"
        pnl = 0
        
        future = df.iloc[idx+1 : idx+40]
        for _, f_row in future.iterrows():
            h = float(f_row['High'])
            l = float(f_row['Low'])
            
            if is_buy:
                # Pre-trail: If price moves +0.8x ATR, move SL to Entry + 0.25x ATR
                if not t1_hit and h >= (c + 0.8 * atr):
                    curr_sl = c + (0.25 * atr)
                    
                if not t1_hit and h >= t1_price:
                    t1_hit = True
                    curr_sl = c + (0.60 * atr) # Lock +3 pts profit
                    
                if t1_hit and h >= t2_price:
                    outcome = "Target 2 Hit (Big Win)"
                    pnl = (0.5 * t1_pts) + (0.5 * t2_pts) # ~2.1x ATR (~9-16 pts)
                    break
                    
                if l <= curr_sl:
                    if t1_hit:
                        outcome = "Target 1 Hit + Trailed Exit"
                        pnl = (0.5 * t1_pts) + (0.5 * (curr_sl - c)) # ~1.0x ATR (~5-8 pts)
                    elif curr_sl > c:
                        outcome = "Pre-Trail Secured"
                        pnl = curr_sl - c # ~0.25x ATR (~1.5 pts)
                    else:
                        outcome = "Stop Loss Hit"
                        pnl = -sl_pts
                    break
            else:
                if not t1_hit and l <= (c - 0.8 * atr):
                    curr_sl = c - (0.25 * atr)
                    
                if not t1_hit and l <= t1_price:
                    t1_hit = True
                    curr_sl = c - (0.60 * atr)
                    
                if t1_hit and l <= t2_price:
                    outcome = "Target 2 Hit (Big Win)"
                    pnl = (0.5 * t1_pts) + (0.5 * t2_pts)
                    break
                    
                if h >= curr_sl:
                    if t1_hit:
                        outcome = "Target 1 Hit + Trailed Exit"
                        pnl = (0.5 * t1_pts) + (0.5 * (c - curr_sl))
                    elif curr_sl < c:
                        outcome = "Pre-Trail Secured"
                        pnl = c - curr_sl
                    else:
                        outcome = "Stop Loss Hit"
                        pnl = -sl_pts
                    break
                    
        if outcome != "Pending":
            results.append({
                "time": str(df.index[idx])[:16],
                "sig": "BUY" if is_buy else "SELL",
                "entry": c,
                "t1": t1_price,
                "t1_pts": t1_pts,
                "t2": t2_price,
                "t2_pts": t2_pts,
                "sl": sl_price,
                "sl_pts": sl_pts,
                "outcome": outcome,
                "pnl": round(pnl, 2)
            })
            
    wins = [r for r in results if "Hit" in r['outcome'] or "Secured" in r['outcome']]
    sls = [r for r in results if r['outcome'] == "Stop Loss Hit"]
    t2_hits = [r for r in results if "Target 2" in r['outcome']]
    total = len(results)
    
    win_rate = len(wins) / total * 100 if total else 0
    total_profit = sum(r['pnl'] for r in results)
    avg_win = np.mean([r['pnl'] for r in wins]) if wins else 0
    avg_loss = np.mean([abs(r['pnl']) for r in sls]) if sls else 0
    
    print("\n" + "="*70)
    print("RESULTS WITH INSTITUTIONAL TARGET EXPANSION & TRAILING RUNNERS:")
    print(f"Total Signals: {total} | T2 Big Wins: {len(t2_hits)} | T1/Secured Wins: {len(wins)-len(t2_hits)} | SL Hits: {len(sls)}")
    print(f"WIN RATE: {win_rate:.1f}% ({len(wins)}/{total}) | SL RATE: {len(sls)/total*100:.1f}%")
    print(f"Average Win: +{avg_win:.2f} pts | Average Loss: -{avg_loss:.2f} pts")
    print(f"TOTAL NET POINTS GAINED: {total_profit:+.2f} pts")
    print("="*70 + "\n")
    
    for idx, r in enumerate(results, 1):
        print(f"#{idx:02d} | {r['time']} | {r['sig']:4s} @ Rs.{r['entry']:.2f} | T1: Rs.{r['t1']:.2f} (+{r['t1_pts']:.2f} pts) | T2: Rs.{r['t2']:.2f} (+{r['t2_pts']:.2f} pts) | SL: Rs.{r['sl']:.2f} (-{r['sl_pts']:.2f} pts) | {r['outcome']} -> PnL: {r['pnl']:+.2f} pts")

if __name__ == "__main__":
    run_filtered_test()
