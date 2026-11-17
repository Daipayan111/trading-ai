import sys, json
sys.path.insert(0, ".")
import numpy as np
from analytics import fetch_stock_data, add_technical_indicators, generate_signals

STOCKS = [
    ("RELIANCE","Rel"),("TCS","TCS"),("INFY","Inf"),("HDFCBANK","HDFC"),
    ("ICICIBANK","ICICI"),("SBIN","SBI"),("TATASTEEL","TataSt"),("MARUTI","Maruti")
]

def backtest(df):
    sigs = generate_signals(df)
    for c in sigs.columns: df[c] = sigs[c]
    trades = []; i = 60; n = len(df)
    while i < n - 5:
        row = df.iloc[i]; prev = df.iloc[i-1]
        sig = str(row.get("Signal","HOLD"))
        is_buy = "BUY" in sig; is_sell = "SELL" in sig
        if not (is_buy or is_sell): i+=1; continue
        cl = float(row["Close"]); sma = float(row.get("SMA_200",cl)); e50 = float(row.get("EMA_50",cl))
        adx = float(row.get("ADX",25)); bb = float(row.get("BB_Percent",0.5))
        atr = float(row.get("ATR",cl*0.015)); rsi = float(row.get("RSI",50))
        mh = float(row.get("MACD_Hist",0)); pmh = float(prev.get("MACD_Hist",0))
        if is_buy and (cl<sma or cl<e50): i+=1; continue
        if is_sell and (cl>sma or cl>e50): i+=1; continue
        if adx<18: i+=1; continue
        if 0.40<=bb<=0.60 and adx<22: i+=1; continue
        if is_buy and rsi>70: i+=1; continue
        if is_sell and rsi<30: i+=1; continue
        if is_buy and mh<pmh: i+=1; continue
        if is_sell and mh>pmh: i+=1; continue
        ep = cl
        risk = max(atr*1.35, cl*0.006)
        sl = ep-risk if is_buy else ep+risk
        tp1 = ep+max(atr*1.0, cl*0.008) if is_buy else ep-max(atr*1.0, cl*0.008)
        tp2 = ep+max(atr*2.0, cl*0.015) if is_buy else ep-max(atr*2.0, cl*0.015)
        rr2 = round(abs(tp2-ep)/max(0.01,abs(ep-sl)),2)
        h1=h2=hs=False; ex=ep; bh=0; sl_t=sl
        for j in range(i+1, min(i+20, n)):
            f=df.iloc[j]; fh=float(f["High"]); fl=float(f["Low"]); fc=float(f["Close"]); bh=j-i
            if is_buy:
                if (fc<=sl_t or fl<=sl_t-atr*0.35) and not h1: hs=True; ex=sl_t; break
                if fh>=tp1 and not h1: h1=True; ex=tp1; sl_t=ep
                if h1 and fh>=tp2 and not h2: h2=True; ex=tp2; sl_t=tp1; break
                if h1 and fl<=sl_t: ex=sl_t; break
            else:
                if (fc>=sl_t or fh>=sl_t+atr*0.35) and not h1: hs=True; ex=sl_t; break
                if fl<=tp1 and not h1: h1=True; ex=tp1; sl_t=ep
                if h1 and fl<=tp2 and not h2: h2=True; ex=tp2; sl_t=tp1; break
                if h1 and fh>=sl_t: ex=sl_t; break
        pnl = ((ex-ep)/ep*100) if is_buy else ((ep-ex)/ep*100)
        trades.append({"h1":h1,"h2":h2,"hs":hs,"pnl":round(pnl,2),"rr2":rr2})
        i += max(3, bh)
    return trades

print("Symbol      Trades  WR%  TP2%   SL%  RR-TP2   PF   Grade")
all_res = []
for sym, name in STOCKS:
    try:
        df,_,__ = fetch_stock_data(sym, period="5y", interval="1d")
        if df.empty or len(df)<80: print(sym + ": skip"); continue
        df = add_technical_indicators(df)
        trades = backtest(df.copy())
        n = len(trades)
        if n == 0: print(sym + ": 0 trades"); continue
        t1 = sum(1 for t in trades if t["h1"])
        t2 = sum(1 for t in trades if t["h2"])
        sl = sum(1 for t in trades if t["hs"])
        wins = [t["pnl"] for t in trades if t["pnl"]>0]
        loss = [abs(t["pnl"]) for t in trades if t["pnl"]<0]
        pf = round(sum(wins)/max(0.1,sum(loss)),2) if loss else 9.9
        rr2 = round(float(np.mean([t["rr2"] for t in trades])),2)
        wr = round(t1/n*100,1)
        t2p = round(t2/n*100,1)
        slp = round(sl/n*100,1)
        pro = (rr2>=1.8 and wr>=65 and pf>=1.5)
        grade = "PROFESSIONAL" if pro else ("GOOD" if pf>=1.2 else "POOR")
        print("{:<12} {:>4}  {:>5.1f}  {:>5.1f}  {:>5.1f}  1:{:.2f} {:>6.2f}  {}".format(
            sym, n, wr, t2p, slp, rr2, pf, grade))
        all_res.append({"sym":sym,"n":n,"wr":wr,"t2":t2p,"sl":slp,"rr2":rr2,"pf":pf,"pro":pro})
    except Exception as e:
        print(sym + ": ERR " + str(e)[:40])

if all_res:
    avg_wr = round(float(np.mean([r["wr"] for r in all_res])),1)
    avg_rr = round(float(np.mean([r["rr2"] for r in all_res])),2)
    avg_pf = round(float(np.mean([r["pf"] for r in all_res])),2)
    pro_ct = sum(1 for r in all_res if r["pro"])
    print("")
    print("AVG: WR={}% | RR-TP2=1:{} | PF={} | Pro={}/{}".format(
        avg_wr, avg_rr, avg_pf, pro_ct, len(all_res)))
print("DONE")
