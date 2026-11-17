import asyncio
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from routers.stock import get_stock_history
import pandas as pd

async def run():
    print("="*90)
    print("  TCS.NS SIGNALS ANALYSIS (FROM 22 SEPTEMBER TO TODAY) - 5M TIMEFRAME")
    print("="*90)
    
    res = await get_stock_history('TCS', period='60d', interval='5m')
    data = res['data']
    
    signals = []
    for d in data:
        dt = d['date']
        # filter from 2026-09-22 to today
        if '2026-09-22' <= dt[:10] <= '2026-10-09':
            if 'signal' in d:
                signals.append(d)
                
    print(f"Total candles analyzed in period: {len([d for d in data if '2026-09-22' <= d['date'][:10] <= '2026-10-09'])}")
    print(f"Total Buy/Sell Signals generated: {len(signals)}\n")
    
    target_pcts = []
    sl_pcts = []
    
    outcomes_count = {}
    
    for idx, s in enumerate(signals, 1):
        dt = s['date'].replace('T', ' ')[:16]
        sig = s['signal']
        entry = s['close']
        tgt = s.get('target', 0)
        sl = s.get('stop_loss', 0)
        conv = s.get('conviction', 0)
        outcome = s.get('outcome', 'Pending')
        pnl = s.get('pnl', 0)
        
        tgt_pct = round(abs(tgt - entry) / entry * 100, 2) if entry else 0
        sl_pct = round(abs(entry - sl) / entry * 100, 2) if entry else 0
        
        target_pcts.append(tgt_pct)
        sl_pcts.append(sl_pct)
        outcomes_count[outcome] = outcomes_count.get(outcome, 0) + 1
        
        sign_str = f"+{tgt_pct}%" if sig == 'BUY' else f"-{tgt_pct}%"
        sl_sign_str = f"-{sl_pct}%" if sig == 'BUY' else f"+{sl_pct}%"
        
        print(f"[{idx:02d}] {dt} | {sig:4s} | Entry: Rs.{entry:.2f} | Target: Rs.{tgt:.2f} ({sign_str}) | SL: Rs.{sl:.2f} ({sl_sign_str}) | RR: 1:{abs(tgt-entry)/abs(entry-sl):.1f} | Outcome: {outcome} ({pnl:+.2f} pts) | Conviction: {conv}%")
        
    print("\n" + "="*90)
    print("  SUMMARY OF TARGET & STOP LOSS % (22 SEP TO TODAY)")
    print("="*90)
    if target_pcts:
        print(f"Average Target Distance : {sum(target_pcts)/len(target_pcts):.2f}% (Min: {min(target_pcts):.2f}%, Max: {max(target_pcts):.2f}%)")
        print(f"Average Stop Loss Dist  : {sum(sl_pcts)/len(sl_pcts):.2f}% (Min: {min(sl_pcts):.2f}%, Max: {max(sl_pcts):.2f}%)")
        print(f"Average Risk:Reward     : 1 : {(sum(target_pcts)/len(target_pcts)) / (sum(sl_pcts)/len(sl_pcts)):.1f}")
        print("Trade Outcomes Breakdown:")
        for k, v in outcomes_count.items():
            print(f"  - {k}: {v} trades ({v/len(signals)*100:.1f}%)")

if __name__ == '__main__':
    asyncio.run(run())
