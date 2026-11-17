import sys, os
import json
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytics import fetch_stock_data, add_technical_indicators, generate_signals

stocks_5m = [
    'NIFTY50', 'BANKNIFTY', 'RELIANCE', 'TCS', 'INFY',
    'ICICIBANK', 'SBIN', 'LT', 'BHARTIARTL', 'MARUTI'
]

def audit_5m_stock(sym):
    df, used_symbol, ticker = fetch_stock_data(sym, period="60d", interval="5m")
    if df.empty or len(df) < 200:
        return {"symbol": sym, "error": f"Insufficient 5M data (bars: {len(df)})"}

    df = add_technical_indicators(df)
    signals = generate_signals(df)
    for col in signals.columns:
        df[col] = signals[col]

    is_index = any(k in sym.upper() for k in ['NIFTY', 'BANK', 'SENSEX'])

    # 5M Intraday targets
    # Intraday scalping targets: TP1 = 0.40x ATR, TP2 = 1.00x ATR, TP3 = 1.80x ATR
    # Intraday SL: 1.20x ATR (Tight structural protection)
    tp1_mult = 0.40 if is_index else 0.45
    tp2_mult = 1.00 if is_index else 1.10
    tp3_mult = 1.80 if is_index else 2.00
    sl_mult  = 1.20 if is_index else 1.25

    trades = []
    i = 30
    n = len(df)
    cooldown = 2 # 2 bars = 10 minutes

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
        atr = float(row.get('ATR', c * 0.003))
        adx = float(row.get('ADX', 22))
        rsi = float(row.get('RSI', 50))
        ema9  = float(row.get('EMA_9', c))
        ema21 = float(row.get('EMA_21', c))

        # Intraday 5M Trend confirmation
        if is_buy and c < ema9:
            i += 1; continue
        if is_sell and c > ema9:
            i += 1; continue

        # Intraday RSI sanity
        if is_buy and rsi > 72: i += 1; continue
        if is_sell and rsi < 28: i += 1; continue

        entry_price = c
        entry_time = df.index[i].strftime("%Y-%m-%d %H:%M")
        
        if is_buy:
            sl  = entry_price - atr * sl_mult
            tp1 = entry_price + atr * tp1_mult
            tp2 = entry_price + atr * tp2_mult
            tp3 = entry_price + atr * tp3_mult
        else:
            sl  = entry_price + atr * sl_mult
            tp1 = entry_price - atr * tp1_mult
            tp2 = entry_price - atr * tp2_mult
            tp3 = entry_price - atr * tp3_mult

        hit_tp1 = hit_tp2 = hit_tp3 = hit_sl = False
        sl_trail = sl
        exit_price = entry_price
        bars_held = 0
        post_tp1_run_pts = 0.0

        # Maximum intraday holding: 24 bars (2 hours)
        max_intraday_bars = 24
        for j in range(i + 1, min(i + max_intraday_bars, n)):
            fut = df.iloc[j]
            fh  = float(fut['High'])
            fl  = float(fut['Low'])
            bars_held = j - i

            # If crossing day boundary, square off at market close (Intraday rule)
            if df.index[j].date() != df.index[i].date():
                exit_price = float(df.iloc[j-1]['Close'])
                break

            if is_buy:
                if fl <= sl_trail and not hit_tp1:
                    hit_sl = True
                    exit_price = sl_trail
                    break

                if fh >= tp1 and not hit_tp1:
                    hit_tp1 = True
                    exit_price = tp1
                    sl_trail = entry_price # Breakeven
                    subseq = df.iloc[j : min(j+10, n)]
                    if len(subseq) > 0:
                        post_tp1_run_pts = max(0.0, float(subseq['High'].max()) - tp1)

                if hit_tp1 and fh >= tp2 and not hit_tp2:
                    hit_tp2 = True
                    exit_price = tp2
                    sl_trail = tp1

                if hit_tp2 and fh >= tp3 and not hit_tp3:
                    hit_tp3 = True
                    exit_price = tp3
                    break

                if hit_tp1 and fl <= sl_trail:
                    exit_price = sl_trail
                    break
            else:
                if fh >= sl_trail and not hit_tp1:
                    hit_sl = True
                    exit_price = sl_trail
                    break

                if fl <= tp1 and not hit_tp1:
                    hit_tp1 = True
                    exit_price = tp1
                    sl_trail = entry_price
                    subseq = df.iloc[j : min(j+10, n)]
                    if len(subseq) > 0:
                        post_tp1_run_pts = max(0.0, tp1 - float(subseq['Low'].min()))

                if hit_tp1 and fl <= tp2 and not hit_tp2:
                    hit_tp2 = True
                    exit_price = tp2
                    sl_trail = tp1

                if hit_tp2 and fl <= tp3 and not hit_tp3:
                    hit_tp3 = True
                    exit_price = tp3
                    break

                if hit_tp1 and fh >= sl_trail:
                    exit_price = sl_trail
                    break

        if is_buy:
            pnl_pts = exit_price - entry_price
        else:
            pnl_pts = entry_price - exit_price

        trades.append({
            "entry_time": entry_time,
            "type": "BUY" if is_buy else "SELL",
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "pnl_pts": round(pnl_pts, 2),
            "hit_tp1": hit_tp1,
            "hit_tp2": hit_tp2,
            "hit_tp3": hit_tp3,
            "hit_sl": hit_sl,
            "bars_held": bars_held,
            "hold_mins": bars_held * 5,
            "post_tp1_run_pts": round(post_tp1_run_pts, 2)
        })

        i += max(cooldown, bars_held)

    n_trades = len(trades)
    if n_trades == 0:
        return {"symbol": sym, "error": "No trades"}

    wins = sum(1 for t in trades if t['hit_tp1'])
    sls  = sum(1 for t in trades if t['hit_sl'])
    tp2s = sum(1 for t in trades if t['hit_tp2'])
    net_pts = sum(t['pnl_pts'] for t in trades)
    hold_times = [t['hold_mins'] for t in trades]
    runs = [t['post_tp1_run_pts'] for t in trades if t['hit_tp1']]

    gross_gain = sum(t['pnl_pts'] for t in trades if t['pnl_pts'] > 0)
    gross_loss = abs(sum(t['pnl_pts'] for t in trades if t['pnl_pts'] < 0))
    pf = round(gross_gain / gross_loss, 2) if gross_loss > 0 else 9.99

    return {
        "symbol": sym,
        "total_bars": len(df),
        "total_intraday_trades": n_trades,
        "trades_per_day": round(n_trades / 42.0, 1), # ~42 trading days in 60d
        "tp1_win_rate": round(wins / n_trades * 100, 1),
        "tp2_rate": round(tp2s / n_trades * 100, 1),
        "sl_rate": round(sls / n_trades * 100, 1),
        "avg_hold_mins": round(np.mean(hold_times), 1),
        "avg_run_past_tp1": round(np.mean(runs), 1) if runs else 0.0,
        "profit_factor": pf,
        "cumulative_intraday_pts": round(net_pts, 1)
    }

print("Running 5-Minute Intraday Audit across 10 Stocks & Indices (60 Days)...")
sys.stdout.flush()

from concurrent.futures import ThreadPoolExecutor, as_completed
results_5m = []

with ThreadPoolExecutor(max_workers=5) as executor:
    fut_map = {executor.submit(audit_5m_stock, s): s for s in stocks_5m}
    for f in as_completed(fut_map):
        sym = fut_map[f]
        try:
            r = f.result()
            if 'error' in r:
                print(f"FAILED {sym}: {r['error']}")
            else:
                results_5m.append(r)
                print(f"5M INTRADAY: {sym:12} | Trades: {r['total_intraday_trades']:3d} ({r['trades_per_day']} /day) | Win: {r['tp1_win_rate']:5.1f}% | SL: {r['sl_rate']:4.1f}% | Avg Hold: {r['avg_hold_mins']:3.0f} min | Net: {r['cumulative_intraday_pts']:+8.1f} pts")
        except Exception as ex:
            print(f"EXCEPTION {sym}: {ex}")
        sys.stdout.flush()

results_5m.sort(key=lambda x: x.get('symbol', ''))
with open('audit_5m_intraday.json', 'w') as f:
    json.dump(results_5m, f, indent=2)

print("\n5M Audit Complete! Saved to audit_5m_intraday.json")
