"""
Quick test script to check TP1 Hit % and SL % across all timeframes.
Runs standalone without any async/server dependency.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
from routers.stock import get_stock_history

STOCKS = [
    'NIFTY50', 'BANKNIFTY', 'RELIANCE', 'TCS', 'INFY',
    'HDFCBANK', 'ICICIBANK', 'SBIN', 'BHARTIARTL', 'ITC',
    'KOTAKBANK', 'LT', 'AXISBANK', 'MARUTI', 'BAJFINANCE'
]

TIMEFRAMES = [
    ('1d', '5y',  'Daily '),
    ('1h', '730d','1-Hour'),
    ('15m','60d', '15-Min'),
    ('5m', '60d', '5-Min '),
]

async def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    print("=" * 100, flush=True)
    print(f"{'SYMBOL':<12} {'TF':<6} {'TRADES':<7} {'TP1 HIT %':<11} {'SL HIT %':<10} {'GOR PT':<9} {'STATUS'}", flush=True)
    print("=" * 100, flush=True)

    all_results = []

    for sym in STOCKS:
        for interval, period, tf_name in TIMEFRAMES:
            try:
                h = await get_stock_history(symbol=sym, period=period, interval=interval)
                candles = h.get('data', [])

                tp1_hits, sl_hits, net_pts = 0, 0, 0.0
                for c in candles:
                    outcome = c.get('outcome', '')
                    pnl = float(c.get('pnl', 0.0))
                    if 'Target Hit' in outcome:
                        tp1_hits += 1
                        net_pts += pnl
                    elif 'TP1 Hit' in outcome:
                        tp1_hits += 1
                        net_pts += pnl
                    elif 'Stop Loss Hit' in outcome:
                        sl_hits += 1
                        net_pts += pnl

                closed = tp1_hits + sl_hits
                if closed == 0:
                    continue

                tp1_pct = round(tp1_hits / closed * 100, 1)
                sl_pct  = round(sl_hits  / closed * 100, 1)
                gor_pt  = round(net_pts / closed, 2)

                flag = "✅" if tp1_pct >= 80 and sl_pct <= 20 else ("⚠️ " if tp1_pct >= 60 else "❌")
                print(f"{sym:<12} {tf_name:<6} {closed:<7} {tp1_pct:>8.1f}%   {sl_pct:>7.1f}%   {gor_pt:>8.2f}  {flag}", flush=True)

                all_results.append({
                    'symbol': sym, 'timeframe': interval, 'tf_name': tf_name.strip(),
                    'trades': closed, 'tp1_pct': tp1_pct, 'sl_pct': sl_pct, 'gor_pt': gor_pt
                })
            except Exception as e:
                print(f"{sym:<12} {tf_name:<6} ERROR: {e}", flush=True)

    print("=" * 100, flush=True)

    # Summary by timeframe
    for tf in ['1d', '1h', '15m', '5m']:
        rows = [r for r in all_results if r['timeframe'] == tf]
        if not rows:
            continue
        avg_tp1 = round(sum(r['tp1_pct'] for r in rows) / len(rows), 1)
        avg_sl  = round(sum(r['sl_pct']  for r in rows) / len(rows), 1)
        total_t = sum(r['trades'] for r in rows)
        flag = "✅" if avg_tp1 >= 80 else "⚠️ "
        print(f"[{tf:<4}] AVERAGE  Trades:{total_t:<6} TP1:{avg_tp1:>5.1f}%  SL:{avg_sl:>5.1f}%  {flag}", flush=True)

    import json
    with open('verified_stats_80pct.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("\n✅ Results saved to verified_stats_80pct.json", flush=True)

if __name__ == '__main__':
    asyncio.run(main())
