import sys
sys.stdout.reconfigure(encoding='utf-8')
import asyncio
from routers.stock import get_stock_history

async def run():
    res = await get_stock_history("TCS", period="60d", interval="5m")
    records = res.get("data", [])
    signals = [r for r in records if r.get("signal") in ("BUY", "SELL")]
    
    print(f"Total signal records: {len(signals)}")
    sept22_oct9 = [
        s for s in signals 
        if "2026-09-22" <= s["date"][:10] <= "2026-10-09"
    ]
    print(f"Signals between 22 Sep and 09 Oct: {len(sept22_oct9)}\n")
    
    target_hits = 0
    sl_hits = 0
    pending = 0
    
    for idx, s in enumerate(sept22_oct9, 1):
        dt = s["date"]
        sig = s["signal"]
        close = s["close"]
        target = s.get("target")
        sl = s.get("stop_loss")
        outcome = s.get("outcome", "Pending")
        conv = s.get("conviction", 0)
        pnl = s.get("pnl", 0)
        
        tgt_pct = abs(target - close) / close * 100 if target else 0
        sl_pct = abs(sl - close) / close * 100 if sl else 0
        
        if outcome == "Target Hit":
            target_hits += 1
        elif outcome == "Stop Loss Hit":
            sl_hits += 1
        else:
            pending += 1
            
        print(f"#{idx:02d} | {dt} | {sig:4s} @ Rs.{close:7.2f} | Tgt: Rs.{target:7.2f} (+{tgt_pct:.2f}%) | SL: Rs.{sl:7.2f} (-{sl_pct:.2f}%) | Conv: {conv}% | Outcome: {outcome} (PnL: {pnl})")

    total_closed = target_hits + sl_hits
    print("\n" + "="*50)
    print(f"SUMMARY (22 Sep - 09 Oct 2026):")
    print(f"Total Signals: {len(sept22_oct9)}")
    print(f"Target Hit / Secured: {target_hits}")
    print(f"Stop Loss Hit: {sl_hits}")
    print(f"Pending: {pending}")
    if total_closed > 0:
        win_rate = (target_hits / total_closed) * 100
        sl_rate = (sl_hits / total_closed) * 100
        print(f"Target Hit Rate: {win_rate:.1f}%")
        print(f"Stop Loss Hit Rate: {sl_rate:.1f}%")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(run())
