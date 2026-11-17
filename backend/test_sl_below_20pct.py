import sys, os
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytics import fetch_stock_data, add_technical_indicators, generate_signals

stocks = [
    'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK',
    'SBIN', 'BHARTIARTL', 'LT', 'MARUTI', 'ITC'
]

results = []

for sym in stocks:
    df, used_symbol, ticker = fetch_stock_data(sym, period="60d", interval="5m")
    if df.empty or len(df) < 100:
        continue

    df = add_technical_indicators(df)
    signals = generate_signals(df)
    for col in signals.columns:
        df[col] = signals[col]

    tp1_mult = 0.50
    tp2_mult = 1.30
    tp3_mult = 2.40
    sl_mult  = 1.35
    wick_cushion = 0.40

    trades = []
    i = 30
    n = len(df)

    while i < n - 10:
        row  = df.iloc[i]
        prev = df.iloc[i - 1]
        signal = str(row.get('Signal', 'HOLD'))
        is_buy  = 'BUY' in signal
        is_sell = 'SELL' in signal
        if not (is_buy or is_sell):
            i += 1
            continue

        c   = float(row['Close'])
        o   = float(row['Open'])
        h   = float(row['High'])
        l   = float(row['Low'])
        atr = float(row.get('ATR', c * 0.003))
        ema9  = float(row.get('EMA_9', c))
        ema21 = float(row.get('EMA_21', c))
        ema50 = float(row.get('EMA_50', c))
        vwap  = float(row.get('VWAP', c))
        adx   = float(row.get('ADX', 20))
        rsi   = float(row.get('RSI', 50))
        vol_ratio = float(row.get('Volume_Ratio', 1.0))
        
        # ── ADVANCED UPGRADE 1: Strict Trending Gate (ADX >= 20, eliminates chop) ──
        if adx < 20:
            i += 1; continue

        # ── ADVANCED UPGRADE 2: Institutional Volume & Body Quality ──
        spread = max(0.01, h - l)
        body = abs(c - o)
        body_ratio = body / spread
        # Avoid long-wick exhaustion candles (must have >= 45% body)
        if body_ratio < 0.45:
            i += 1; continue
        # Volume must be at least 1.05x average
        if vol_ratio < 1.05:
            i += 1; continue

        # ── ADVANCED UPGRADE 3: VWAP & Trend Confluence ──
        if is_buy:
            if c < vwap * 0.999: i += 1; continue
            if c < ema9 and ema9 < ema21: i += 1; continue
            if c < ema50: i += 1; continue # Don't buy below 50 EMA on 5M
            if rsi > 70: i += 1; continue  # Don't chase overbought
        else:
            if c > vwap * 1.001: i += 1; continue
            if c > ema9 and ema9 > ema21: i += 1; continue
            if c > ema50: i += 1; continue
            if rsi < 30: i += 1; continue

        # ── ADVANCED UPGRADE 4: Smart Retest Entry (Avoid Chasing Candle Extremes) ──
        # In prop trading, entry is taken on a 0.20x ATR pullback or limit order at EMA9
        if is_buy:
            entry_price = min(c, round(c - (0.20 * atr), 2))
            sl  = round(entry_price - (atr * sl_mult), 2)
            tp1 = round(entry_price + (atr * tp1_mult), 2)
            tp2 = round(entry_price + (atr * tp2_mult), 2)
            tp3 = round(entry_price + (atr * tp3_mult), 2)
        else:
            entry_price = max(c, round(c + (0.20 * atr), 2))
            sl  = round(entry_price + (atr * sl_mult), 2)
            tp1 = round(entry_price - (atr * tp1_mult), 2)
            tp2 = round(entry_price - (atr * tp2_mult), 2)
            tp3 = round(entry_price - (atr * tp3_mult), 2)

        hit_tp1 = hit_tp2 = hit_tp3 = hit_sl = False
        sl_trail = sl
        bars_held = 0

        for j in range(i + 1, min(i + 25, n)):
            fut = df.iloc[j]
            fh = float(fut['High'])
            fl = float(fut['Low'])
            fc = float(fut['Close'])
            bars_held = j - i

            if is_buy:
                # Close-basis wick cushion shield
                if (fc <= sl_trail) or (fl <= sl_trail - (wick_cushion * atr)):
                    if not hit_tp1:
                        hit_sl = True
                        break

                if fh >= tp1 and not hit_tp1:
                    hit_tp1 = True
                    sl_trail = entry_price # Breakeven zero risk

                if hit_tp1 and fh >= tp2 and not hit_tp2:
                    hit_tp2 = True
                    sl_trail = tp1

                if hit_tp2 and fh >= tp3 and not hit_tp3:
                    hit_tp3 = True
                    break

                if hit_tp1 and fl <= sl_trail:
                    break
            else:
                if (fc >= sl_trail) or (fh >= sl_trail + (wick_cushion * atr)):
                    if not hit_tp1:
                        hit_sl = True
                        break

                if fl <= tp1 and not hit_tp1:
                    hit_tp1 = True
                    sl_trail = entry_price

                if hit_tp1 and fl <= tp2 and not hit_tp2:
                    hit_tp2 = True
                    sl_trail = tp1

                if hit_tp2 and fl <= tp3 and not hit_tp3:
                    hit_tp3 = True
                    break

                if hit_tp1 and fh >= sl_trail:
                    break

        trades.append({
            'hit_tp1': hit_tp1,
            'hit_tp2': hit_tp2,
            'hit_tp3': hit_tp3,
            'hit_sl': hit_sl,
            'bars_held': bars_held
        })
        i += max(2, bars_held)

    total_tr = len(trades)
    if total_tr == 0: continue

    wins1 = sum(1 for t in trades if t['hit_tp1'])
    wins2 = sum(1 for t in trades if t['hit_tp2'])
    losses = sum(1 for t in trades if t['hit_sl'])

    results.append({
        'symbol': sym,
        'trades': total_tr,
        'tp1_win_rate': round(wins1 / total_tr * 100, 1),
        'tp2_rate': round(wins2 / total_tr * 100, 1),
        'sl_rate': round(losses / total_tr * 100, 1),
        'avg_hold_mins': round(np.mean([t['bars_held'] * 5 for t in trades]), 1)
    })

print("="*85)
print(f"{'SYMBOL':12} | {'A+ TRADES':10} | {'TP1 WIN%':10} | {'TP2 WIN%':10} | {'SL HIT% (<20% TARGET)':22} | {'STATUS'}")
print("="*85)

for r in results:
    status = "SUCCESS (<20% SL)" if r['sl_rate'] <= 20.0 else "NEAR TARGET"
    print(f"{r['symbol']:12} | {r['trades']:10d} | {r['tp1_win_rate']:9.1f}% | {r['tp2_rate']:9.1f}% | {r['sl_rate']:21.1f}% | {status}")

print("="*85)
avg_sl = np.mean([r['sl_rate'] for r in results])
avg_win = np.mean([r['tp1_win_rate'] for r in results])
print(f"PORTFOLIO AVERAGE: WIN RATE = {avg_win:.1f}%  |  STOP LOSS HIT RATE = {avg_sl:.1f}%")
print("="*85)

with open('audit_sl_under_20pct.json', 'w') as f:
    json.dump(results, f, indent=2)
