import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd
import numpy as np
from analytics import fetch_stock_data, add_technical_indicators, generate_signals

def evaluate_stock(symbol, interval="5m"):
    df, used_symbol, _ = fetch_stock_data(symbol, period="60d", interval=interval)
    df = add_technical_indicators(df)
    signals = generate_signals(df)
    
    # Calculate EMA 200 / Trend
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['Vol_MA'] = df['Volume'].rolling(window=20).mean()
    
    trades = []
    
    # We want TARGET POINTS TO BE SIGNIFICANT (8 - 25+ pts on TCS, 0.4% - 1.2%)
    # Let's test:
    # T1 = 1.5 * ATR (7 - 12 pts) -> Books 50%, moves SL to Entry + 0.3*ATR (safe breakeven lock)
    # T2 = 3.0 * ATR (15 - 25 pts) -> Full target
    # SL = 1.2 * ATR (6 - 8 pts)
    # Window = 45 candles (~3.75 hours intraday)
    
    for i in range(50, len(df) - 45):
        row = df.iloc[i]
        sig = signals.iloc[i]['Signal']
        if sig not in ['BUY', 'STRONG BUY', 'SELL', 'STRONG SELL']:
            continue
            
        is_buy = 'BUY' in sig
        c = float(row['Close'])
        atr = float(row.get('ATR', c * 0.005))
        if pd.isna(atr) or atr <= 0: atr = c * 0.005
        
        vwap = float(row.get('VWAP', c))
        ema9 = float(row.get('EMA_9', c))
        ema21 = float(row.get('EMA_21', c))
        ema50 = float(row.get('EMA_50', c))
        rsi = float(row.get('RSI', 50))
        div = int(row.get('Divergence_Signal', 0))
        adx = float(row.get('ADX', 20))
        
        # Confluence Entry Filter
        if is_buy:
            # Momentum buy: Price above VWAP, EMA9 > EMA21, RSI not overbought, ADX trending or RSI > 52
            if div == -1 or c < vwap or ema9 < ema21 or rsi < 52 or rsi > 78:
                continue
        else:
            if div == 1 or c > vwap or ema9 > ema21 or rsi > 48 or rsi < 22:
                continue
                
        # Points Targets:
        # T1 = 1.4 * ATR (Quick book)
        # T2 = 2.8 * ATR (Big profit runner)
        t1_pts = 1.4 * atr
        t2_pts = 2.8 * atr
        sl_pts = 1.2 * atr
        
        t1_price = c + t1_pts if is_buy else c - t1_pts
        t2_price = c + t2_pts if is_buy else c - t2_pts
        sl_price = c - sl_pts if is_buy else c + sl_pts
        
        # Simulation
        t1_hit = False
        t2_hit = False
        sl_hit = False
        curr_sl = sl_price
        realized_pnl = 0
        outcome = "PENDING"
        
        for j in range(i + 1, min(i + 1 + 50, len(df))):
            h = float(df.iloc[j]['High'])
            l = float(df.iloc[j]['Low'])
            
            if is_buy:
                # Check T1
                if not t1_hit and h >= t1_price:
                    t1_hit = True
                    # Book 50% at T1, and move SL to Entry + 0.3*ATR
                    curr_sl = c + (0.35 * atr)
                    
                # Check T2
                if t1_hit and h >= t2_price:
                    t2_hit = True
                    outcome = "FULL_TARGET_2_HIT"
                    realized_pnl = (0.5 * t1_pts) + (0.5 * t2_pts)  # 2.1 * ATR (~10-18 pts!)
                    break
                    
                # Check SL
                if l <= curr_sl:
                    if t1_hit:
                        outcome = "T1_HIT_TRAIL_EXIT"
                        realized_pnl = (0.5 * t1_pts) + (0.5 * (curr_sl - c)) # ~0.875 * ATR (~4-7 pts profit!)
                    else:
                        outcome = "SL_HIT"
                        realized_pnl = -sl_pts
                        sl_hit = True
                    break
            else:
                if not t1_hit and l <= t1_price:
                    t1_hit = True
                    curr_sl = c - (0.35 * atr)
                    
                if t1_hit and l <= t2_price:
                    t2_hit = True
                    outcome = "FULL_TARGET_2_HIT"
                    realized_pnl = (0.5 * t1_pts) + (0.5 * t2_pts)
                    break
                    
                if h >= curr_sl:
                    if t1_hit:
                        outcome = "T1_HIT_TRAIL_EXIT"
                        realized_pnl = (0.5 * t1_pts) + (0.5 * (c - curr_sl))
                    else:
                        outcome = "SL_HIT"
                        realized_pnl = -sl_pts
                        sl_hit = True
                    break
                    
        if outcome != "PENDING":
            trades.append({
                "time": str(df.index[i]),
                "type": "BUY" if is_buy else "SELL",
                "price": c,
                "t1_pts": t1_pts,
                "t2_pts": t2_pts,
                "sl_pts": sl_pts,
                "outcome": outcome,
                "pnl": realized_pnl
            })
            
    return trades

def main():
    symbols = ["TCS", "RELIANCE", "INFY", "HDFCBANK", "ICICIBANK"]
    for sym in symbols:
        trades = evaluate_stock(sym, interval="5m")
        if not trades:
            continue
        wins = [t for t in trades if t["outcome"] in ("FULL_TARGET_2_HIT", "T1_HIT_TRAIL_EXIT")]
        t2_hits = [t for t in trades if t["outcome"] == "FULL_TARGET_2_HIT"]
        t1_trail = [t for t in trades if t["outcome"] == "T1_HIT_TRAIL_EXIT"]
        sls = [t for t in trades if t["outcome"] == "SL_HIT"]
        total = len(trades)
        win_rate = (len(wins) / total) * 100
        avg_win_pts = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss_pts = np.mean([abs(t["pnl"]) for t in sls]) if sls else 0
        avg_t2_pts = np.mean([t["pnl"] for t in t2_hits]) if t2_hits else 0
        total_pnl = sum(t["pnl"] for t in trades)
        
        print("="*65)
        print(f"STOCK: {sym} (5m Intraday, 60 Days)")
        print(f"Total Trades: {total} | T2 Big Win: {len(t2_hits)} | T1 Trail Win: {len(t1_trail)} | SL Hits: {len(sls)}")
        print(f"WIN RATE: {win_rate:.1f}% | SL RATE: {len(sls)/total*100:.1f}%")
        print(f"Avg T2 Points: +{avg_t2_pts:.2f} pts | Avg Win: +{avg_win_pts:.2f} pts | Avg Loss: -{avg_loss_pts:.2f} pts")
        print(f"NET PROFIT POINTS: {total_pnl:+.2f} pts")
        print("="*65)

if __name__ == "__main__":
    main()
