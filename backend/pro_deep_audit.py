import sys, os, asyncio, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from routers.stock import get_stock_history

STOCKS = [
    "NIFTY50","BANKNIFTY","RELIANCE","TCS","INFY",
    "HDFCBANK","ICICIBANK","SBIN","BHARTIARTL","ITC",
    "KOTAKBANK","LT","AXISBANK","MARUTI","BAJFINANCE"
]
INDEX_SYMS = {"NIFTY50","BANKNIFTY"}

TIMEFRAMES = [
    ("1d",  "5y",   "Daily-5Y "),
    ("1h",  "730d", "Hour-2Y  "),
    ("15m", "60d",  "15M-60D  "),
    ("5m",  "60d",  "5M-60D   "),
]

def big_thresh(sym, tf):
    if sym in INDEX_SYMS:
        return 100 if tf=="1d" else (40 if tf=="1h" else 18)
    return 18 if tf=="1d" else (8 if tf=="1h" else 3)

async def run():
    all_findings = []
    tf_sum = {t[0]: dict(sigs=0,tp1=0,tp2=0,sl=0,traps=0,whips=0,bmiss=0,net=0.0) for t in TIMEFRAMES}

    print("="*130, flush=True)
    print("PRO TRADER DEEP AUDIT  -  15 Stocks x 4 Timeframes x 5-Year Historical Data", flush=True)
    print("="*130, flush=True)
    hdr = "SYM          TF    SIGS  TP1%   TP2%    SL%  TRAP%  WHIP%  BMISS%    EXP   NET_PTS   FLAGS"
    print(hdr, flush=True)
    print("-"*130, flush=True)

    for sym in STOCKS:
        for tf, period, desc in TIMEFRAMES:
            try:
                h = await get_stock_history(symbol=sym, period=period, interval=tf)
                candles = h.get("data", [])
                if len(candles) < 20:
                    continue
                signals = [c for c in candles if c.get("signal") in ("BUY","SELL")]
                if not signals:
                    continue

                tp1=0; tp2=0; sl=0; traps=0; whips=0; bmiss=0; net=0.0
                thresh = big_thresh(sym, tf)

                for s in signals:
                    oc  = s.get("outcome","")
                    pnl = float(s.get("pnl",0) or 0)
                    net += pnl

                    if "Target Hit" in oc:
                        tp1+=1; tp2+=1
                    elif "TP1 Hit" in oc:
                        tp1+=1; whips+=1   # got TP1 but trailed into SL before TP2
                    elif "Stop Loss Hit" in oc:
                        sl+=1
                        stop_p  = float(s.get("stop_loss",0) or 0)
                        close_p = float(s.get("close",1) or 1)
                        if close_p>0 and stop_p>0:
                            dist_pct = abs(close_p - stop_p)/close_p*100
                            if dist_pct < 0.6:   # SL within 0.6% = chop trap
                                traps += 1

                    cl = float(s.get("close",0) or 0)
                    op = float(s.get("open", cl) or cl)
                    body = abs(cl - op)
                    if body >= thresh and "Target Hit" not in oc:
                        bmiss += 1

                closed = tp1 + sl
                if closed == 0:
                    continue

                tp1p = tp1/closed*100
                tp2p = tp2/closed*100
                slp  = sl/closed*100
                trapp= traps/max(1,sl)*100
                whipp= whips/max(1,tp1)*100
                missp= bmiss/max(1,len(signals))*100
                exp  = net/closed

                flags=[]
                if slp  > 20:  flags.append("HIGH_SL")
                if trapp> 30:  flags.append("TRAP")
                if whipp> 40:  flags.append("WHIPSAW")
                if tp2p < 25:  flags.append("LOW_TP2")
                if missp> 25:  flags.append("BIG_MISS")
                if exp  < 0:   flags.append("NEG_EV")
                fs = ",".join(flags) if flags else "OK"
                icon = "CRIT" if len(flags)>=3 else ("WARN" if flags else "OK  ")

                line = (f"{sym:<12} {tf:<5} {len(signals):>5}"
                        f"  {tp1p:>5.1f}%  {tp2p:>5.1f}%  {slp:>5.1f}%"
                        f"  {trapp:>5.1f}%  {whipp:>5.1f}%  {missp:>6.1f}%"
                        f"  {exp:>+7.2f}  {net:>+9.1f}  [{icon}] {fs}")
                print(line, flush=True)

                all_findings.append(dict(
                    sym=sym, tf=tf, sigs=len(signals), closed=closed,
                    tp1p=round(tp1p,1), tp2p=round(tp2p,1), slp=round(slp,1),
                    trapp=round(trapp,1), whipp=round(whipp,1), missp=round(missp,1),
                    exp=round(exp,2), net=round(net,2), flags=flags
                ))

                t = tf_sum[tf]
                t["sigs"]+=len(signals); t["tp1"]+=tp1; t["tp2"]+=tp2; t["sl"]+=sl
                t["traps"]+=traps; t["whips"]+=whips; t["bmiss"]+=bmiss; t["net"]+=net

            except Exception as e:
                print(f"{sym:<12} {tf:<5} ERROR: {e}", flush=True)

    print("\n"+"="*130, flush=True)
    print("TIMEFRAME CONSOLIDATED RESULTS", flush=True)
    print("-"*130, flush=True)
    for tf, _, desc in TIMEFRAMES:
        t = tf_sum[tf]
        closed = t["tp1"]+t["sl"]
        if closed==0: continue
        tp1a=t["tp1"]/closed*100; tp2a=t["tp2"]/closed*100; sla=t["sl"]/closed*100
        tra=t["traps"]/max(1,t["sl"])*100; wha=t["whips"]/max(1,t["tp1"])*100
        mia=t["bmiss"]/max(1,t["sigs"])*100; exa=t["net"]/closed
        flag="PASS" if tp1a>=80 and sla<=20 else "FAIL"
        print(f"{tf:<5} ({desc}) Trades:{closed:4d}  TP1:{tp1a:5.1f}% TP2:{tp2a:5.1f}% SL:{sla:5.1f}% | Trap:{tra:5.1f}% Whip:{wha:5.1f}% BigMiss:{mia:5.1f}% | Exp:{exa:+.2f} pts [{flag}]", flush=True)

    print("\n--- CRITICAL PROBLEM AREAS ---", flush=True)
    crits = sorted([r for r in all_findings if r["flags"]], key=lambda x:len(x["flags"]), reverse=True)
    for r in crits[:20]:
        print(f"  {r['sym']:<12} [{r['tf']:4}]  {', '.join(r['flags']):<40}  SL:{r['slp']:5.1f}% Trap:{r['trapp']:5.1f}% Exp:{r['exp']:+.2f}", flush=True)

    with open("pro_trader_deep_audit_results.json","w",encoding="utf-8") as f:
        json.dump({"findings":all_findings,"tf_summary":tf_sum}, f, indent=2, ensure_ascii=False)
    print("\nSaved => pro_trader_deep_audit_results.json", flush=True)
    print("="*130, flush=True)

asyncio.run(run())
