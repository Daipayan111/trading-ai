import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from analytics import fetch_stock_data, add_technical_indicators, generate_signals

def run_test():
    symbol = "TCS"
    period = "60d"
    interval = "5m"
    
    df, used_symbol, _ = fetch_stock_data(symbol, period=period, interval=interval)
    df = add_technical_indicators(df)
    signals = generate_signals(df)
    
    # Calculate volume moving average
    df['Vol_MA'] = df['Volume'].rolling(window=20).mean()
    df['Body'] = (df['Close'] - df['Open']).abs()
    df['Range'] = df['High'] - df['Low']
    
    # Let's test High-Conviction Filters with Big Points
    # Target: 1.5x - 2.5x ATR (10 - 25 pts)
    # SL: 1.0x - 1.2x ATR (6 - 9 pts)
    # Trailing: As soon as price reaches +0.8x ATR (~5-6 pts), trail SL to +0.3x ATR (~2-3 pts)
    # At +1.2x ATR, trail SL to +0.7x ATR (~5 pts)
    
    experiments = [
        {"name": "Filter 1: ADX > 22 + Volume Surge (>1.1x)", "adx_thresh": 22, "vol_mult": 1.1, "tgt_atr": 1.6, "sl_atr": 1.2},
        {"name": "Filter 2: ADX > 24 + Volume Surge (>1.25x)", "adx_thresh": 24, "vol_mult": 1.25, "tgt_atr": 1.8, "sl_atr": 1.1},
        {"name": "Filter 3: Trend & Momentum Confirmed (ADX>20 + EMA Ribbon)", "adx_thresh": 20, "vol_mult": 1.05, "tgt_atr": 1.5, "sl_atr": 1.0},
        {"name": "Filter 4: Institutional 2-Tier Target (T1=1.0x ATR, T2=2.0x ATR)", "adx_thresh": 22, "vol_mult": 1.15, "tgt_atr": 2.0, "sl_atr": 1.1, "tiered": True},
    ]
    
    for exp in experiments:
        trades = []
        for i in range(50, len(df) - 60):
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
            div = int(row.get('Divergence_Signal', 0))
            rsi = float(row.get('RSI', 50))
            adx = float(row.get('ADX', 20))
            vol = float(row.get('Volume', 0))
            vol_ma = float(row.get('Vol_MA', vol))
            
            # 1. Trend & Volatility Filter
            if adx < exp["adx_thresh"]:
                continue
            if vol < (vol_ma * exp["vol_mult"]):
                continue
                
            # 2. Confluence alignment
            if is_buy:
                if div == -1 or c < vwap or ema9 < ema21 or rsi < 52 or ema21 < ema50:
                    continue
            else:
                if div == 1 or c > vwap or ema9 > ema21 or rsi > 48 or ema21 > ema50:
                    continue
                    
            tgt_pts = exp["tgt_atr"] * atr
            sl_pts = exp["sl_atr"] * atr
            
            target = c + tgt_pts if is_buy else c - tgt_pts
            stop = c - sl_pts if is_buy else c + sl_pts
            
            # Dynamic Step-Trailing:
            # Step 1: When price hits +0.8x ATR, lock +0.3x ATR
            # Step 2: When price hits +1.3x ATR, lock +0.8x ATR
            curr_sl = stop
            trail_step = 0
            outcome = "EXPIRED"
            pnl = 0
            
            for j in range(i + 1, min(i + 60, len(df))):
                h = float(df.iloc[j]['High'])
                l = float(df.iloc[j]['Low'])
                
                if is_buy:
                    # Trailing step 1
                    if h >= (c + 0.75 * atr) and trail_step < 1:
                        curr_sl = c + (0.30 * atr)
                        trail_step = 1
                    # Trailing step 2
                    if h >= (c + 1.30 * atr) and trail_step < 2:
                        curr_sl = c + (0.80 * atr)
                        trail_step = 2
                    # Target Hit
                    if h >= target:
                        outcome = "TARGET_HIT"
                        pnl = tgt_pts
                        break
                    # Stop Hit
                    if l <= curr_sl:
                        if trail_step > 0:
                            outcome = "TRAILED_WIN"
                            pnl = curr_sl - c
                        else:
                            outcome = "SL_HIT"
                            pnl = -sl_pts
                        break
                else:
                    if l <= (c - 0.75 * atr) and trail_step < 1:
                        curr_sl = c - (0.30 * atr)
                        trail_step = 1
                    if l <= (c - 1.30 * atr) and trail_step < 2:
                        curr_sl = c - (0.80 * atr)
                        trail_step = 2
                    if l <= target:
                        outcome = "TARGET_HIT"
                        pnl = tgt_pts
                        break
                    if h >= curr_sl:
                        if trail_step > 0:
                            outcome = "TRAILED_WIN"
                            pnl = c - curr_sl
                        else:
                            outcome = "SL_HIT"
                            pnl = -sl_pts
                        break
                        
            if outcome != "EXPIRED":
                trades.append({
                    "date": str(df.index[i]),
                    "type": "BUY" if is_buy else "SELL",
                    "price": c,
                    "target_pts": tgt_pts,
                    "sl_pts": sl_pts,
                    "pnl": pnl,
                    "outcome": outcome
                })
                
        wins = [t for t in trades if t["outcome"] in ("TARGET_HIT", "TRAILED_WIN")]
        full_tgt = [t for t in trades if t["outcome"] == "TARGET_HIT"]
        trailed_w = [t for t in trades if t["outcome"] == "TRAILED_WIN"]
        sls = [t for t in trades if t["outcome"] == "SL_HIT"]
        total = len(trades)
        
        if total > 0:
            win_rate = (len(wins) / total) * 100
            total_profit = sum(t["pnl"] for t in trades)
            avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
            avg_loss = np.mean([abs(t["pnl"]) for t in sls]) if sls else 0
            avg_full = np.mean([t["target_pts"] for t in full_tgt]) if full_tgt else 0
            
            print(f"\n{'='*70}")
            print(f"{exp['name']}")
            print(f"Total Trades: {total} | Full Target: {len(full_tgt)} | Trailed Wins: {len(trailed_w)} | SL Hits: {len(sls)}")
            print(f"WIN RATE: {win_rate:.1f}% (Target/Trailed Wins: {len(wins)}/{total}) | SL Rate: {len(sls)/total*100:.1f}%")
            print(f"Avg Full Target Points: +{avg_full:.2f} pts | Avg Win: +{avg_win:.2f} pts | Avg SL: -{avg_loss:.2f} pts")
            print(f"NET PROFIT POINTS: {total_profit:+.2f} pts")
            print(f"{'='*70}")

if __name__ == "__main__":
    run_test()
