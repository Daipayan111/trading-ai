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

    is_index = False
    # 5M calibrated targets:
    # TP1: 0.50x ATR (0.3% - 0.6% fast scalp move)
    # TP2: 1.25x ATR (0.8% - 1.4% trend move)
    # TP3: 2.20x ATR (1.5% - 2.5% runner move)
    # SL : 1.25x ATR (0.7% - 1.2% structure protection)
    tp1_mult = 0.50
    tp2_mult = 1.25
    tp3_mult = 2.20
    sl_mult  = 1.25

    trades = []
    i = 30
    n = len(df)

    while i < n - 10:
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        signal = str(row.get('Signal', 'HOLD'))
        is_buy  = 'BUY' in signal
        is_sell = 'SELL' in signal
        if not (is_buy or is_sell):
            i += 1
            continue

        c = float(row['Close'])
        atr = float(row.get('ATR', c * 0.003))
        ema9 = float(row.get('EMA_9', c))
        ema21 = float(row.get('EMA_21', c))
        vwap = float(row.get('VWAP', c))
        adx = float(row.get('ADX', 20))
        vol_ratio = float(row.get('Volume_Ratio', 1.0))

        # Precision confluence check
        if is_buy:
            if c < ema9 and ema9 < ema21: i += 1; continue
            if c < vwap * 0.998: i += 1; continue
        else:
            if c > ema9 and ema9 > ema21: i += 1; continue
            if c > vwap * 1.002: i += 1; continue

        if vol_ratio < 0.70: i += 1; continue
        if adx < 16: i += 1; continue

        entry_price = c
        if is_buy:
            sl  = entry_price - (atr * sl_mult)
            tp1 = entry_price + (atr * tp1_mult)
            tp2 = entry_price + (atr * tp2_mult)
            tp3 = entry_price + (atr * tp3_mult)
        else:
            sl  = entry_price + (atr * sl_mult)
            tp1 = entry_price - (atr * tp1_mult)
            tp2 = entry_price - (atr * tp2_mult)
            tp3 = entry_price - (atr * tp3_mult)

        tp1_pct = abs(tp1 - entry_price) / entry_price * 100
        tp2_pct = abs(tp2 - entry_price) / entry_price * 100
        tp3_pct = abs(tp3 - entry_price) / entry_price * 100
        sl_pct  = abs(sl - entry_price) / entry_price * 100

        hit_tp1 = hit_tp2 = hit_tp3 = hit_sl = False
        sl_trail = sl
        bars_held = 0

        # Simulate next 24 bars (2 hours)
        for j in range(i + 1, min(i + 25, n)):
            fut = df.iloc[j]
            fh = float(fut['High'])
            fl = float(fut['Low'])
            fc = float(fut['Close'])
            bars_held = j - i

            if is_buy:
                # Stop loss (Liquidity Wick Shield)
                if (fc <= sl_trail) or (fl <= sl_trail - (0.35 * atr)):
                    if not hit_tp1:
                        hit_sl = True
                        break

                if fh >= tp1 and not hit_tp1:
                    hit_tp1 = True
                    sl_trail = entry_price # Breakeven

                if hit_tp1 and fh >= tp2 and not hit_tp2:
                    hit_tp2 = True
                    sl_trail = tp1

                if hit_tp2 and fh >= tp3 and not hit_tp3:
                    hit_tp3 = True
                    break

                if hit_tp1 and fl <= sl_trail:
                    break
            else:
                if (fc >= sl_trail) or (fh >= sl_trail + (0.35 * atr)):
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
            'tp1_pct': tp1_pct,
            'tp2_pct': tp2_pct,
            'tp3_pct': tp3_pct,
            'sl_pct': sl_pct,
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
    wins3 = sum(1 for t in trades if t['hit_tp3'])
    losses = sum(1 for t in trades if t['hit_sl'])

    avg_tp1_pct = np.mean([t['tp1_pct'] for t in trades])
    avg_tp2_pct = np.mean([t['tp2_pct'] for t in trades])
    avg_tp3_pct = np.mean([t['tp3_pct'] for t in trades])
    avg_sl_pct  = np.mean([t['sl_pct'] for t in trades])
    avg_hold_mins = np.mean([t['bars_held'] * 5 for t in trades])

    results.append({
        'symbol': sym,
        'trades': total_tr,
        'avg_tp1_pct': round(avg_tp1_pct, 2),
        'avg_tp2_pct': round(avg_tp2_pct, 2),
        'avg_tp3_pct': round(avg_tp3_pct, 2),
        'avg_sl_pct': round(avg_sl_pct, 2),
        'tp1_win_rate': round(wins1 / total_tr * 100, 1),
        'tp2_rate': round(wins2 / total_tr * 100, 1),
        'tp3_rate': round(wins3 / total_tr * 100, 1),
        'sl_rate': round(losses / total_tr * 100, 1),
        'avg_hold_mins': round(avg_hold_mins, 1),
        'rr_tp1': round(avg_tp1_pct / max(0.01, avg_sl_pct), 2),
        'rr_tp2': round(avg_tp2_pct / max(0.01, avg_sl_pct), 2),
        'rr_tp3': round(avg_tp3_pct / max(0.01, avg_sl_pct), 2),
    })

print("="*115)
print(f"{'SYMBOL':12} | {'TRADES':6} | {'TP1 %':7} | {'TP2 %':7} | {'TP3 %':7} | {'SL %':7} | {'TP1 WIN%':9} | {'TP2 WIN%':9} | {'SL HIT%':8} | {'HOLD TIME'}")
print("="*115)

for r in results:
    print(f"{r['symbol']:12} | {r['trades']:6d} | {r['avg_tp1_pct']:6.2f}% | {r['avg_tp2_pct']:6.2f}% | {r['avg_tp3_pct']:6.2f}% | {r['avg_sl_pct']:6.2f}% | {r['tp1_win_rate']:7.1f}%  | {r['tp2_rate']:7.1f}%  | {r['sl_rate']:6.1f}% | {r['avg_hold_mins']:4.0f} min")

print("="*115)

with open('audit_5m_pct_summary.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Saved to audit_5m_pct_summary.json")
