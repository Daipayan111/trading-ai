import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd
import numpy as np
from analytics import fetch_stock_data, add_technical_indicators, generate_signals

def test_big_points_engine():
    df, used_symbol, _ = fetch_stock_data("TCS", period="60d", interval="5m")
    df = add_technical_indicators(df)
    signals = generate_signals(df)
    
    # Filter signals between 22 Sep and 09 Oct
    sept22_oct9_indices = [
        i for i in range(len(df))
        if "2026-09-22" <= df.index[i].strftime("%Y-%m-%d") <= "2026-10-09"
        and signals.iloc[i]['Signal'] in ['BUY', 'STRONG BUY', 'SELL', 'STRONG SELL']
    ]
    
    print(f"--- TESTING EXPANDED TARGET POINTS ON TCS (22 Sep - 09 Oct) ---")
    
    # We will test 3 target point levels:
    # Option A: 1.5x ATR (~7 - 10 pts)
    # Option B: 2.2x ATR (~10 - 16 pts)
    # Option C: 3.0x ATR (~15 - 25 pts)
    # With intelligent trailing:
    # When price advances >= 1.0x ATR, trail SL to Entry + 0.3x ATR (Locks profit, eliminates risk)
    # When price advances >= 1.8x ATR, trail SL to Entry + 1.0x ATR (Locks substantial profit)
    
    for mult, name in [(1.6, "Level 1: 1.6x ATR (~7-10 pts)"), 
                       (2.2, "Level 2: 2.2x ATR (~10-15 pts)"), 
                       (2.8, "Level 3: 2.8x ATR (~14-22 pts)")]:
        
        results = []
        for idx in sept22_oct9_indices:
            row = df.iloc[idx]
            sig = signals.iloc[idx]['Signal']
            is_buy = 'BUY' in sig
            c = float(row['Close'])
            atr = float(row['ATR'])
            div = int(row.get('Divergence_Signal', 0))
            vwap = float(row.get('VWAP', c))
            ema9 = float(row.get('EMA_9', c))
            ema21 = float(row.get('EMA_21', c))
            
            # Confluence filter (avoid false breakout)
            if is_buy and (div == -1 or (c < vwap and ema9 < ema21)):
                continue
            if not is_buy and (div == 1 or (c > vwap and ema9 > ema21)):
                continue
                
            tgt_pts = round(mult * atr, 2)
            sl_pts = round(1.4 * atr, 2)  # Reasonable stop, not 2.2x ATR
            
            target = round(c + tgt_pts if is_buy else c - tgt_pts, 2)
            stop = round(c - sl_pts if is_buy else c + sl_pts, 2)
            
            curr_sl = stop
            trail_stage = 0
            outcome = "Pending"
            pnl = 0
            
            future = df.iloc[idx+1 : idx+40]
            for _, f_row in future.iterrows():
                h = float(f_row['High'])
                l = float(f_row['Low'])
                
                if is_buy:
                    # Stage 1: Profit lock at 0.9x ATR move
                    if h >= (c + 0.9 * atr) and trail_stage < 1:
                        curr_sl = c + (0.35 * atr)  # Lock +1.5 to +2.5 pts minimum
                        trail_stage = 1
                    # Stage 2: Profit lock at 1.6x ATR move
                    if h >= (c + 1.6 * atr) and trail_stage < 2:
                        curr_sl = c + (1.0 * atr)   # Lock +4 to +6 pts
                        trail_stage = 2
                    # Target hit
                    if h >= target:
                        outcome = "Target Hit"
                        pnl = tgt_pts
                        break
                    # Stop hit
                    if l <= curr_sl:
                        if trail_stage > 0:
                            outcome = "Trailed Profit"
                            pnl = round(curr_sl - c, 2)
                        else:
                            outcome = "Stop Loss Hit"
                            pnl = -sl_pts
                        break
                else:
                    if l <= (c - 0.9 * atr) and trail_stage < 1:
                        curr_sl = c - (0.35 * atr)
                        trail_stage = 1
                    if l <= (c - 1.6 * atr) and trail_stage < 2:
                        curr_sl = c - (1.0 * atr)
                        trail_stage = 2
                    if l <= target:
                        outcome = "Target Hit"
                        pnl = tgt_pts
                        break
                    if h >= curr_sl:
                        if trail_stage > 0:
                            outcome = "Trailed Profit"
                            pnl = round(c - curr_sl, 2)
                        else:
                            outcome = "Stop Loss Hit"
                            pnl = -sl_pts
                        break
                        
            if outcome != "Pending":
                results.append({
                    "time": str(df.index[idx])[:16],
                    "sig": "BUY" if is_buy else "SELL",
                    "entry": c,
                    "target": target,
                    "target_pts": tgt_pts,
                    "stop": stop,
                    "sl_pts": sl_pts,
                    "outcome": outcome,
                    "pnl": pnl
                })
                
        wins = [r for r in results if r['outcome'] in ("Target Hit", "Trailed Profit")]
        full_tgt = [r for r in results if r['outcome'] == "Target Hit"]
        sls = [r for r in results if r['outcome'] == "Stop Loss Hit"]
        total = len(results)
        
        win_rate = len(wins) / total * 100 if total else 0
        total_pts = sum(r['pnl'] for r in results)
        avg_win = np.mean([r['pnl'] for r in wins]) if wins else 0
        avg_loss = np.mean([abs(r['pnl']) for r in sls]) if sls else 0
        
        print("\n" + "="*70)
        print(f"CONFIGURATION: {name}")
        print(f"Total Signals: {total} | Full Target Hit: {len(full_tgt)} | Trailed Profit: {len(wins)-len(full_tgt)} | SL Hit: {len(sls)}")
        print(f"Win/Secured Rate: {win_rate:.1f}% | SL Rate: {len(sls)/total*100:.1f}%")
        print(f"Avg Win Points: +{avg_win:.2f} pts | Avg Loss: -{avg_loss:.2f} pts")
        print(f"TOTAL NET POINTS PROFIT: {total_pts:+.2f} pts")
        print("="*70)
        
        # Print trades for Level 2
        if mult == 2.2:
            print("\nDetailed Trades for Level 2 (10 - 15 pts target):")
            for r in results:
                print(f"{r['time']} | {r['sig']:4s} @ Rs.{r['entry']:.2f} | Tgt: Rs.{r['target']:.2f} (+{r['target_pts']:.2f} pts) | SL: Rs.{r['stop']:.2f} (-{r['sl_pts']:.2f} pts) | {r['outcome']} -> PnL: {r['pnl']:+.2f} pts")

if __name__ == "__main__":
    test_big_points_engine()
