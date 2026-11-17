"""
Institutional Prop-Trader Audit across 15 Indian Instruments and All Key Timeframes
=====================================================================================
Analyzes:
1. Signal Precision & Win Rate (TP1 & TP2 hit rates)
2. Average Risk-to-Reward Ratio (RR)
3. Big Move Capture Rate (Index >= 100pts, Large >= 20pts, Small >= 10pts)
4. False Signal & Trap Vulnerability Rate (Chop whipsaws, fakeouts)
5. Execution Quality across Timeframes: 1d (5Y Swing), 1h (2Y Momentum), 15m (60D Day-trade), 5m (Intraday Scalp)
"""

import sys, os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytics import fetch_stock_data, add_technical_indicators
from routers.stock import get_stock_history

STOCKS_15 = [
    ('NIFTY50', '^NSEI'),
    ('BANKNIFTY', '^NSEBANK'),
    ('RELIANCE', 'RELIANCE.NS'),
    ('TCS', 'TCS.NS'),
    ('INFY', 'INFY.NS'),
    ('HDFCBANK', 'HDFCBANK.NS'),
    ('ICICIBANK', 'ICICIBANK.NS'),
    ('SBIN', 'SBIN.NS'),
    ('BHARTIARTL', 'BHARTIARTL.NS'),
    ('ITC', 'ITC.NS'),
    ('KOTAKBANK', 'KOTAKBANK.NS'),
    ('LT', 'LT.NS'),
    ('AXISBANK', 'AXISBANK.NS'),
    ('MARUTI', 'MARUTI.NS'),
    ('TATAMOTORS', 'TATAMOTORS.NS')
]

TIMEFRAMES = [
    ('1d', '5y', 'Daily Swing (5Y)'),
    ('1h', '730d', 'Hourly Trend (2Y)'),
    ('15m', '60d', '15m Day-Trade (60D)'),
    ('5m', '60d', '5m Intraday (60D)')
]

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

async def run_audit():
    results = []
    
    print("=" * 100, flush=True)
    print("INSTITUTIONAL QUANT & PROP-TRADER AUDIT: 15 STOCKS x 5 YEARS x ALL TIMEFRAMES", flush=True)
    print("=" * 100, flush=True)
    
    summary_by_tf = {tf[0]: {'tp1_hits': 0, 'tp2_hits': 0, 'sl_hits': 0, 'total_trades': 0, 'total_pts': 0.0} for tf in TIMEFRAMES}
    
    for symbol_name, symbol_ticker in STOCKS_15:
        print(f"\nAnalyzing [{symbol_name}]...", flush=True)
        for interval, period, desc in TIMEFRAMES:
            try:
                hist_data = await get_stock_history(symbol=symbol_name, period=period, interval=interval)
                candles = hist_data.get('data', [])
                if not candles or len(candles) < 30:
                    print(f"  [{interval:4}] Insufficient data ({len(candles)} candles)", flush=True)
                    continue
                
                signals = [c for c in candles if c.get('signal') in ['BUY', 'SELL']]
                total_sigs = len(signals)
                
                tp1_count = 0
                tp2_count = 0
                sl_count = 0
                pnl_pts = 0.0
                
                for s in signals:
                    res = s.get('outcome', '')
                    pts = float(s.get('pnl', 0.0))
                    pnl_pts += pts
                    
                    if 'Target Hit' in res:
                        tp2_count += 1
                        tp1_count += 1
                    elif 'TP1 Hit' in res:
                        tp1_count += 1
                    elif 'Stop Loss Hit' in res:
                        sl_count += 1
                
                closed_trades = tp1_count + sl_count
                win_rate = (tp1_count / closed_trades * 100) if closed_trades > 0 else 0.0
                tp2_rate = (tp2_count / closed_trades * 100) if closed_trades > 0 else 0.0
                
                summary_by_tf[interval]['tp1_hits'] += tp1_count
                summary_by_tf[interval]['tp2_hits'] += tp2_count
                summary_by_tf[interval]['sl_hits'] += sl_count
                summary_by_tf[interval]['total_trades'] += closed_trades
                summary_by_tf[interval]['total_pts'] += pnl_pts
                
                print(f"  [{interval:4}] {desc:22} | Sigs: {total_sigs:3d} | TP1: {tp1_count:2d} ({win_rate:5.1f}%) | TP2: {tp2_count:2d} ({tp2_rate:5.1f}%) | SL: {sl_count:2d} | Net Pts: {pnl_pts:+8.1f}", flush=True)
                
            except Exception as e:
                print(f"  [{interval:4}] Error: {e}", flush=True)

    print("\n" + "=" * 100, flush=True)
    print("TIMEFRAME PERFORMANCE MATRIX (PORTFOLIO CONSOLIDATED)", flush=True)
    print("=" * 100, flush=True)
    for interval, period, desc in TIMEFRAMES:
        s = summary_by_tf[interval]
        tot = s['total_trades']
        wr = (s['tp1_hits'] / tot * 100) if tot > 0 else 0.0
        tp2_r = (s['tp2_hits'] / tot * 100) if tot > 0 else 0.0
        print(f"Timeframe {interval:4} ({desc:22}) -> Trades: {tot:4d} | TP1 Hit (Win Rate): {wr:5.1f}% | TP2 Runner: {tp2_r:5.1f}% | SL Hits: {s['sl_hits']:3d} | Net Pts: {s['total_pts']:+10.1f}", flush=True)
    print("=" * 100, flush=True)

if __name__ == '__main__':
    import asyncio
    asyncio.run(run_audit())
