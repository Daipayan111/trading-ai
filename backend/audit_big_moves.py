import sys, os
import pandas as pd
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytics import fetch_stock_data, add_technical_indicators, generate_signals

STOCKS = [
    'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK',
    'SBIN', 'BHARTIARTL', 'ITC', 'LT', 'KOTAKBANK',
    'AXISBANK', 'MARUTI', 'BAJFINANCE', '^NSEBANK'
]

def audit_big_moves():
    print("=" * 95)
    print("PRO TRADER AUDIT: DID THE AI CATCH THE 5-YEAR MEGA MOVES? (2021 - 2026)")
    print("=" * 95)
    
    results = []
    for sym in STOCKS:
        try:
            df, used_symbol, _ = fetch_stock_data(sym, period="5y", interval="1d")
            if df.empty or len(df) < 200:
                continue
                
            df = add_technical_indicators(df)
            signals = generate_signals(df)
            
            # Find the top 3 biggest multi-month swing runs in 5 years (60 bars forward)
            runs = []
            for i in range(10, len(df) - 60, 20):
                p0 = float(df['Close'].iloc[i])
                forward_high = float(df['High'].iloc[i+1 : i+61].max())
                gain_pct = (forward_high - p0) / p0 * 100
                gain_pts = forward_high - p0
                date_str = df.index[i].strftime("%Y-%m-%d")
                
                # Check if AI had a BUY signal within 3 days of run breakout
                window = signals.iloc[max(0, i-2) : min(len(signals), i+4)]
                caught = (window['Signal'].isin(['BUY', 'STRONG BUY'])).any()
                
                runs.append({
                    "date": date_str,
                    "entry_idx": i,
                    "p0": p0,
                    "high": forward_high,
                    "gain_pct": gain_pct,
                    "gain_pts": gain_pts,
                    "caught": caught
                })
                
            # Sort to find the biggest move
            top_run = max(runs, key=lambda x: x['gain_pct'])
            
            # Count overall capture rate of top 5 moves
            sorted_runs = sorted(runs, key=lambda x: x['gain_pct'], reverse=True)[:5]
            caught_count = sum(1 for r in sorted_runs if r['caught'])
            capture_rate = (caught_count / len(sorted_runs)) * 100
            
            status_tag = "CAUGHT [YES]" if top_run['caught'] else "MISSED [NO]"
            print(f"{sym:12} | Top Move: +{top_run['gain_pct']:5.1f}% ({top_run['p0']:6.1f} -> {top_run['high']:6.1f}) on {top_run['date']} | Top: {status_tag:13} | Top-5 Capture: {caught_count}/5 ({capture_rate:3.0f}%)")
            
            results.append({
                "symbol": sym,
                "top_move_pct": round(top_run['gain_pct'], 1),
                "top_move_pts": round(top_run['gain_pts'], 1),
                "date": top_run['date'],
                "top_caught": top_run['caught'],
                "top5_capture_rate": capture_rate
            })
        except Exception as e:
            print(f"{sym:12} | Error: {e}")
            
    print("=" * 95)
    
if __name__ == '__main__':
    audit_big_moves()
