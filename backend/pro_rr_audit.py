import sys, json
sys.path.insert(0, "e:/App prodactan/trading-ai/backend")
import numpy as np
from analytics import fetch_stock_data, add_technical_indicators, generate_signals

STOCKS = [
    ("RELIANCE","Reliance"),("TCS","TCS"),("INFY","Infosys"),("HDFCBANK","HDFC Bank"),
    ("ICICIBANK","ICICI Bank"),("SBIN","SBI"),("BHARTIARTL","Airtel"),("ITC","ITC"),
    ("LT","L&T"),("KOTAKBANK","Kotak"),("AXISBANK","Axis"),
    ("TATASTEEL","Tata Steel"),("MARUTI","Maruti"),("BAJFINANCE","Bajaj Fin"),("WIPRO","Wipro"),
]

TP1_M=1.0; TP2_M=2.0; TP3_M=3.5; SL_M=1.35

def backtest(df):
    sigs = generate_signals(df)
    for c in sigs.columns:
        df[c] = sigs[c]
    trades = []
    i = 60
    n = len(df)
    while i < n - 5:
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        sig = str(row.get("Signal", "HOLD"))
        is_buy = "BUY" in sig
        is_sell = "SELL" in sig
        if not (is_buy or is_sell):
            i += 1
            continue
        cl = float(row["Close"])
        sma = float(row.get("SMA_200", cl))
        e50 = float(row.get("EMA_50", cl))
        adx = float(row.get("ADX", 25))
        bb = float(row.get("BB_Percent", 0.5))
        atr = float(row.get("ATR", cl * 0.015))
        rsi = float(row.get("RSI", 50))
        mh = float(row.get("MACD_Hist", 0))
        pmh = float(prev.get("MACD_Hist", 0))
        if is_buy and (cl < sma or cl < e50):
            i += 1; continue
        if is_sell and (cl > sma or cl > e50):
            i += 1; continue
        if adx < 18:
            i += 1; continue
        if 0.40 <= bb <= 0.60 and adx < 22:
            i += 1; continue
        if is_buy and rsi > 70:
            i += 1; continue
        if is_sell and rsi < 30:
            i += 1; continue
        if is_buy and mh < pmh:
            i += 1; continue
        if is_sell and mh > pmh:
            i += 1; continue
        ep = cl
        risk = atr * SL_M
        sl = ep - atr * SL_M if is_buy else ep + atr * SL_M
        tp1 = ep + atr * TP1_M if is_buy else ep - atr * TP1_M
        tp2 = ep + atr * TP2_M if is_buy else ep - atr * TP2_M
        tp3 = ep + atr * TP3_M if is_buy else ep - atr * TP3_M
        rr1 = round(atr * TP1_M / max(0.01, risk), 2)
        rr2 = round(atr * TP2_M / max(0.01, risk), 2)
        rr3 = round(atr * TP3_M / max(0.01, risk), 2)
        h1 = h2 = h3 = hs = False
        ex = ep
        bh = 0
        sl_t = sl
        pts = 0.0
        for j in range(i + 1, min(i + 20, n)):
            f = df.iloc[j]
            fh = float(f["High"])
            fl = float(f["Low"])
            fc = float(f["Close"])
            bh = j - i
            if is_buy:
                if (fc <= sl_t or fl <= sl_t - atr * 0.35) and not h1:
                    hs = True; ex = sl_t; pts = sl_t - ep; break
                if fh >= tp1 and not h1:
                    h1 = True; ex = tp1; pts = tp1 - ep; sl_t = ep
                if h1 and fh >= tp2 and not h2:
                    h2 = True; ex = tp2; pts = tp2 - ep; sl_t = tp1
                if h2 and fh >= tp3 and not h3:
                    h3 = True; ex = tp3; pts = tp3 - ep; break
                if h1 and fl <= sl_t:
                    ex = sl_t; break
            else:
                if (fc >= sl_t or fh >= sl_t + atr * 0.35) and not h1:
                    hs = True; ex = sl_t; pts = ep - sl_t; break
                if fl <= tp1 and not h1:
                    h1 = True; ex = tp1; pts = ep - tp1; sl_t = ep
                if h1 and fl <= tp2 and not h2:
                    h2 = True; ex = tp2; pts = ep - tp2; sl_t = tp1
                if h2 and fl <= tp3 and not h3:
                    h3 = True; ex = tp3; pts = ep - tp3; break
                if h1 and fh >= sl_t:
                    ex = sl_t; break
        pnl = ((ex - ep) / ep * 100) if is_buy else ((ep - ex) / ep * 100)
        trades.append({
            "hit_tp1": h1, "hit_tp2": h2, "hit_tp3": h3, "hit_sl": hs,
            "pnl_pct": round(pnl, 2), "rr1": rr1, "rr2": rr2, "rr3": rr3,
            "pts": round(pts, 2)
        })
        i += max(3, bh)
    return trades

all_res = {}
for tf, per in [("1d", "5y"), ("1h", "730d"), ("15m", "60d"), ("5m", "60d")]:
    print("")
    print("=== " + tf + " / " + per + " ===")
    print("Symbol          WR%    TP2%  RR-TP2    PF   Grade")
    tf_res = []
    for sym, name in STOCKS:
        try:
            df, _, __ = fetch_stock_data(sym, period=per, interval=tf)
            if df.empty or len(df) < 80:
                print("  " + sym + " -- skip")
                continue
            df = add_technical_indicators(df)
            trades = backtest(df.copy())
            n = len(trades)
            if n == 0:
                continue
            t1 = sum(1 for t in trades if t["hit_tp1"])
            t2 = sum(1 for t in trades if t["hit_tp2"])
            t3 = sum(1 for t in trades if t["hit_tp3"])
            sl = sum(1 for t in trades if t["hit_sl"])
            wins = [t["pnl_pct"] for t in trades if t["pnl_pct"] > 0]
            loss = [abs(t["pnl_pct"]) for t in trades if t["pnl_pct"] < 0]
            pf = round(sum(wins) / max(0.1, sum(loss)), 2) if loss else 9.9
            rr2 = round(float(np.mean([t["rr2"] for t in trades])), 2)
            wr = round(t1 / n * 100, 1)
            t2p = round(t2 / n * 100, 1)
            t3p = round(t3 / n * 100, 1)
            slp = round(sl / n * 100, 1)
            pro = (rr2 >= 2.0 and wr >= 65 and pf >= 1.5)
            grade = "PROFESSIONAL" if pro else ("GOOD" if (rr2 >= 1.5 and pf >= 1.2) else "POOR")
            tf_res.append({
                "symbol": sym, "name": name, "tf": tf, "per": per,
                "trades": n, "wr": wr, "tp2_pct": t2p, "tp3_pct": t3p, "sl_pct": slp,
                "rr2": rr2, "pf": pf, "grade": grade, "is_pro": pro,
                "total_pts": round(sum(t["pts"] for t in trades), 1)
            })
            print("  {:<12}  {:>6.1f} {:>6.1f}  {:>6.2f} {:>6.2f}  {}".format(
                sym, wr, t2p, rr2, pf, grade))
        except Exception as e:
            print("  " + sym + " ERR: " + str(e)[:50])
    all_res[tf] = tf_res
    if tf_res:
        arr = float(np.mean([r["rr2"] for r in tf_res]))
        awr = float(np.mean([r["wr"] for r in tf_res]))
        apf = float(np.mean([r["pf"] for r in tf_res]))
        pro = sum(1 for r in tf_res if r["is_pro"])
        print("  SUMMARY: RR-TP2=1:{:.2f} | WR={:.1f}% | PF={:.2f} | Pro={}/{}".format(
            arr, awr, apf, pro, len(tf_res)))

with open("e:/App prodactan/trading-ai/backend/professional_rr_audit_results.json", "w") as f:
    json.dump(all_res, f, indent=2)
print("DONE - Results saved!")
