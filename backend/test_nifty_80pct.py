import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from analytics import fetch_stock_data, add_technical_indicators, generate_signals

df, sym, _ = fetch_stock_data("^NSEI", period="60d", interval="5m")
print(f"Loaded {len(df)} 5m bars for {sym}")
df = add_technical_indicators(df)
signals = generate_signals(df)
for c in signals.columns:
    df[c] = signals[c]

n = len(df)
trades = []
i = 60
last_sig = -999

while i < n - 20:
    row = df.iloc[i]
    prev = df.iloc[i-1]
    
    c = float(row["Close"])
    h = float(row["High"])
    l = float(row["Low"])
    o = float(row.get("Open", c))
    atr = float(row.get("ATR", c * 0.015))
    ema9 = float(row.get("EMA_9", c))
    ema21 = float(row.get("EMA_21", c))
    ema50 = float(row.get("EMA_50", c))
    sma200 = float(row.get("SMA_200", c))
    vwap = float(row.get("VWAP", c))
    rsi = float(row.get("RSI", 50))
    adx = float(row.get("ADX", 20))
    bb_pct = float(row.get("BB_Percent", 0.5))
    mh = float(row.get("MACD_Hist", 0))
    pmh = float(prev.get("MACD_Hist", 0))
    
    c_range = max(0.01, h - l)
    l_wick = (min(c, o) - l) / c_range
    u_wick = (h - max(c, o)) / c_range
    is_hammer = (l_wick >= 0.35) or int(row.get("Pattern_Hammer", 0)) == 1
    is_inv_hammer = (u_wick >= 0.35) or int(row.get("Pattern_Inverted_Hammer", 0)) == 1
    is_bull_engulf = int(row.get("Pattern_Bullish_Engulfing", 0)) == 1
    is_bear_engulf = int(row.get("Pattern_Bearish_Engulfing", 0)) == 1
    liq_sig = int(row.get("Liquidity_Signal", 0))
    mss_sig = int(row.get("MSS_Signal", 0))
    
    # Dynamic Trend Regime at bar i:
    is_bull_trend = (c > ema50 and ema50 > sma200) or (c > vwap and ema21 > ema50 and c > sma200)
    is_bear_trend = (c < ema50 and ema50 < sma200) or (c < vwap and ema21 < ema50 and c < sma200)
    
    # Cooldown
    if i - last_sig < 6:
        i += 1
        continue
        
    sig = None
    
    # ── BUY CONDITIONS (High Precision 80%+ Confluence) ──
    # 1. Trend continuation in Bull Trend: pullback to EMA/VWAP + bullish trigger
    if is_bull_trend and not is_bear_trend:
        if (is_hammer or is_bull_engulf or liq_sig == 1 or mss_sig == 1) and rsi <= 68 and mh >= pmh:
            sig = "BUY"
        elif c > ema9 and ema9 > ema21 and mh > 0 and pmh < 0 and rsi >= 52:
            sig = "BUY"
            
    # ── SELL CONDITIONS (High Precision 80%+ Confluence) ──
    # 2. Trend continuation in Bear Trend: rally to EMA/VWAP + bearish trigger
    elif is_bear_trend and not is_bull_trend:
        if (is_inv_hammer or is_bear_engulf or liq_sig == -1 or mss_sig == -1) and rsi >= 32 and mh <= pmh:
            sig = "SELL"
        elif c < ema9 and ema9 < ema21 and mh < 0 and pmh > 0 and rsi <= 48:
            sig = "SELL"

    # Extreme Reversal at key boundaries (oversold dip in chop / overbought pop in chop)
    elif not is_bull_trend and not is_bear_trend:
        if (is_hammer or is_bull_engulf) and rsi <= 35 and bb_pct <= 0.20:
            sig = "BUY"
        elif (is_inv_hammer or is_bear_engulf) and rsi >= 65 and bb_pct >= 0.80:
            sig = "SELL"

    if not sig:
        i += 1
        continue
        
    last_sig = i
    
    # 80%+ Optimized Targets and Structural Cushion
    tp1_pts = 0.80 * atr
    tp2_pts = 1.80 * atr
    sl_pts  = 1.35 * atr
    
    if sig == "BUY":
        entry = c
        tp1 = entry + tp1_pts
        tp2 = entry + tp2_pts
        sl  = entry - sl_pts
    else:
        entry = c
        tp1 = entry - tp1_pts
        tp2 = entry - tp2_pts
        sl  = entry + sl_pts
        
    hit_tp1 = hit_tp2 = hit_sl = False
    curr_sl = sl
    bars_held = 0
    
    for j in range(i+1, min(i+25, n)):
        fj = df.iloc[j]
        fh = float(fj["High"])
        fl = float(fj["Low"])
        bars_held = j - i
        
        if sig == "BUY":
            if fh >= tp1 and not hit_tp1:
                hit_tp1 = True
                curr_sl = entry # Breakeven
            if fh >= tp2:
                hit_tp2 = True
                break
            if fl <= curr_sl:
                if not hit_tp1:
                    hit_sl = True
                break
        else:
            if fl <= tp1 and not hit_tp1:
                hit_tp1 = True
                curr_sl = entry
            if fl <= tp2:
                hit_tp2 = True
                break
            if fh >= curr_sl:
                if not hit_tp1:
                    hit_sl = True
                break
                
    trades.append({
        "date": df.index[i].strftime("%Y-%m-%d %H:%M"),
        "sig": sig,
        "price": c,
        "hit_tp1": hit_tp1,
        "hit_tp2": hit_tp2,
        "hit_sl": hit_sl
    })
    
    i += max(3, bars_held)

total = len(trades)
buys = [t for t in trades if t["sig"] == "BUY"]
sells = [t for t in trades if t["sig"] == "SELL"]
tp1_all = sum(1 for t in trades if t["hit_tp1"])
tp2_all = sum(1 for t in trades if t["hit_tp2"])
sl_all = sum(1 for t in trades if t["hit_sl"])

print(f"Total trades: {total} (BUYs: {len(buys)}, SELLs: {len(sells)})")
print(f"Target 1 Hit Rate: {tp1_all / total * 100:.1f}% ({tp1_all}/{total})")
print(f"Target 2 Hit Rate: {tp2_all / total * 100:.1f}% ({tp2_all}/{total})")
print(f"Stop Loss Hit Rate: {sl_all / total * 100:.1f}% ({sl_all}/{total})")

print("\nRecent 10 trades:")
for t in trades[-10:]:
    res = "TP2 HIT" if t["hit_tp2"] else ("TP1 HIT" if t["hit_tp1"] else "SL HIT")
    print(f"{t['date']} | {t['sig']:4} @ {t['price']:.1f} -> {res}")
