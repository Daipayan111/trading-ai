"""
Generate Complete Quantitative Audit Stats for 15 Top Instruments across all Timeframes:
- Target 1 (TP1) Hit %
- Target 2 (TP2) Runner %
- Stop Loss (SL) Hit %
- Average Points per Trade (Gor Point)
- Total Net Points
"""

import sys, os
import json
import asyncio

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from routers.stock import get_stock_history

STOCKS_15 = [
    'NIFTY50', 'BANKNIFTY', 'RELIANCE', 'TCS', 'INFY',
    'HDFCBANK', 'ICICIBANK', 'SBIN', 'BHARTIARTL', 'ITC',
    'KOTAKBANK', 'LT', 'AXISBANK', 'MARUTI', 'BAJFINANCE'
]

TIMEFRAMES = [
    ('1d', '5y', 'Daily (5Y Swing)'),
    ('1h', '730d', '1-Hour (2Y Trend)'),
    ('15m', '60d', '15-Min (60D Day-Trade)'),
    ('5m', '60d', '5-Min (60D Intraday)')
]

async def compute_all_stats():
    full_stats = []
    
    print("=" * 110, flush=True)
    print("🚀 COMPUTING FULL STATS: 15 STOCKS x ALL TIMEFRAMES (TARGET HIT %, SL HIT %, GOR POINT)", flush=True)
    print("=" * 110, flush=True)
    
    for sym in STOCKS_15:
        print(f"\nProcessing [{sym}]...", flush=True)
        for interval, period, name in TIMEFRAMES:
            try:
                hist = await get_stock_history(symbol=sym, period=period, interval=interval)
                candles = hist.get('data', [])
                if not candles:
                    continue
                
                signals = [c for c in candles if c.get('signal') in ['BUY', 'SELL']]
                tp1_hits = 0
                tp2_hits = 0
                sl_hits = 0
                net_pts = 0.0
                win_pts_list = []
                loss_pts_list = []
                
                for s in signals:
                    res = s.get('outcome', '')
                    pts = float(s.get('pnl', 0.0))
                    
                    if 'Target Hit' in res:
                        tp2_hits += 1
                        tp1_hits += 1
                        net_pts += pts
                        win_pts_list.append(pts)
                    elif 'TP1 Hit' in res:
                        tp1_hits += 1
                        net_pts += pts
                        win_pts_list.append(pts)
                    elif 'Stop Loss Hit' in res:
                        sl_hits += 1
                        net_pts += pts
                        loss_pts_list.append(pts)
                
                closed = tp1_hits + sl_hits
                tp1_pct = (tp1_hits / closed * 100) if closed > 0 else 0.0
                tp2_pct = (tp2_hits / closed * 100) if closed > 0 else 0.0
                sl_pct  = (sl_hits / closed * 100) if closed > 0 else 0.0
                avg_gor_pts = (net_pts / closed) if closed > 0 else 0.0
                avg_win_pts = (sum(win_pts_list) / len(win_pts_list)) if win_pts_list else 0.0
                avg_loss_pts = (sum(loss_pts_list) / len(loss_pts_list)) if loss_pts_list else 0.0
                
                rec = {
                    'symbol': sym,
                    'timeframe': interval,
                    'tf_name': name,
                    'total_trades': closed,
                    'tp1_hits': tp1_hits,
                    'tp1_pct': round(tp1_pct, 1),
                    'tp2_hits': tp2_hits,
                    'tp2_pct': round(tp2_pct, 1),
                    'sl_hits': sl_hits,
                    'sl_pct': round(sl_pct, 1),
                    'net_pts': round(net_pts, 1),
                    'gor_pts': round(avg_gor_pts, 2),
                    'avg_win_pts': round(avg_win_pts, 2),
                    'avg_loss_pts': round(avg_loss_pts, 2)
                }
                full_stats.append(rec)
                print(f"  [{interval:4}] Trades: {closed:3d} | TP1 Hit: {tp1_pct:5.1f}% | TP2: {tp2_pct:4.1f}% | SL Hit: {sl_pct:5.1f}% | Gor Pt: {avg_gor_pts:+7.2f} pt | Net: {net_pts:+8.1f} pt", flush=True)
            except Exception as e:
                print(f"  [{interval:4}] Notice: {e}", flush=True)
                
    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'full_15stocks_stats.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(full_stats, f, indent=2)
    print(f"\n✅ All stats successfully saved to {out_file}", flush=True)

if __name__ == '__main__':
    asyncio.run(compute_all_stats())
