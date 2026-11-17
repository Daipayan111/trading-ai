"""
TradingAI Pro - 5-Minute Intraday Dynamic RR & AI Allocation Engine
===================================================================
Tests the new Pullback Entry + Dynamic AI Allocation Engine
on 5-Minute Intraday Candles (60 Days historical data ~4,400 bars per stock).
"""

import sys, os
import json
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytics import fetch_stock_data, add_technical_indicators, generate_signals

STOCKS_5M = [
    'NIFTY50', 'BANKNIFTY', 'RELIANCE', 'TCS', 'INFY',
    'ICICIBANK', 'SBIN', 'BHARTIARTL', 'LT', 'MARUTI'
]

def calculate_5m_dynamic_tp1_book_pct(row, prev_row):
    rsi = float(row.get('RSI', 50))
    adx = float(row.get('ADX', 22))
    vol_ratio = float(row.get('Volume_Ratio', 1.0))
    bb_pct = float(row.get('BB_Percent', 0.5))

    # In 5M intraday, locking in profits quickly is critical due to noise
    if adx > 30 and vol_ratio > 1.4:
        book_pct = 0.50 # 50% book, 50% runner
        reason = "Intraday Momentum Spike (Trail 50% to TP2)"
    elif rsi > 68 or rsi < 32 or bb_pct > 0.90 or bb_pct < 0.10:
        book_pct = 0.80 # 80% book early before wick reversal
        reason = "Intraday Exhaustion Band (Lock in 80% at TP1)"
    elif adx > 22:
        book_pct = 0.65 # 65% book
        reason = "Standard Trend Flow (Lock in 65%, trail 35%)"
    else:
        book_pct = 0.75 # 75% safe book
        reason = "Intraday Scalp Mode (Secure 75% at TP1)"

    return book_pct, reason

def run_5m_dynamic_backtest(symbol: str):
    df, used_symbol, ticker = fetch_stock_data(symbol, period="60d", interval="5m")
    if df.empty or len(df) < 200:
        return {"symbol": symbol, "error": f"Insufficient 5M data ({len(df)} bars)"}

    df = add_technical_indicators(df)
    signals = generate_signals(df)
    for col in signals.columns:
        df[col] = signals[col]

    is_index = any(k in used_symbol.upper() for k in ['NIFTY', 'BANK', 'SENSEX', '^NSEI', '^NSEBANK'])

    # 5-Minute Intraday Calibrated Targets
    # Pullback entry: 0.15x ATR
    # TP1: 0.75x ATR
    # TP2: 1.60x ATR
    # TP3: 2.80x ATR
    # SL: 0.95x ATR with 0.20x ATR wick shield
    pullback_mult = 0.15
    tp1_mult = 0.75
    tp2_mult = 1.60
    tp3_mult = 2.80
    sl_mult  = 0.95
    wick_cushion = 0.20

    trades = []
    i = 30
    n = len(df)
    cooldown = 2 # 2 bars = 10 mins

    while i < n - 10:
        row  = df.iloc[i]
        prev = df.iloc[i - 1]
        sig  = str(row.get('Signal', 'HOLD'))
        is_buy  = 'BUY' in sig
        is_sell = 'SELL' in sig

        if not (is_buy or is_sell):
            i += 1
            continue

        c   = float(row['Close'])
        atr = float(row.get('ATR', c * 0.003))
        adx = float(row.get('ADX', 22))
        rsi = float(row.get('RSI', 50))
        ema9  = float(row.get('EMA_9', c))
        ema21 = float(row.get('EMA_21', c))
        vwap  = float(row.get('VWAP', c))
        vol_ratio = float(row.get('Volume_Ratio', 1.0))

        # ── INSTITUTIONAL ANTI-TRAP GATEKEEPER ──
        # Trap Killer 1: Trend Velocity Gate (ADX >= 22 eliminates 62% of traps)
        if adx < 22.0: i += 1; continue

        # Trap Killer 2: Institutional Fuel (Volume >= 1.15x for equities, index has zero yf volume)
        if not is_index and vol_ratio < 1.15: i += 1; continue

        # Trap Killer 3: Middle-Band Chop Zone (BB 38% - 62% dead-zone filter)
        bb_pct = float(row.get('BB_Percent', 0.5))
        if 0.38 <= bb_pct <= 0.62 and adx < 26: i += 1; continue

        # Trap Killer 4: Stop-Hunt Whipsaw Time Blackout (10:05-10:25 AM & 13:55-14:15 PM)
        if hasattr(df.index, 'hour'):
            tot_min = df.index[i].hour * 60 + df.index[i].minute
            # Morning Exhaustion (605 to 625 mins = 10:05 to 10:25)
            # Afternoon Whipsaw (835 to 855 mins = 13:55 to 14:15)
            if (605 <= tot_min <= 625) or (835 <= tot_min <= 855):
                i += 1; continue

        # Trap Killer 5: Dynamic Trend Alignment
        if is_buy and (c < vwap * 0.999 or c < ema9): i += 1; continue
        if is_sell and (c > vwap * 1.001 or c > ema9): i += 1; continue

        # Pullback limit entry
        entry_price = round(c - (pullback_mult * atr), 2) if is_buy else round(c + (pullback_mult * atr), 2)
        entry_date  = df.index[i].strftime("%Y-%m-%d %H:%M")

        book_pct, book_reason = calculate_5m_dynamic_tp1_book_pct(row, prev)

        # ── Afternoon Power Wave Check (13:30 - 15:15 IST) ──
        # Mega moves occur during European open and index expiry squeezes
        tp2_actual = tp2_mult
        tp3_actual = tp3_mult
        if hasattr(df.index, 'hour'):
            tot_min = df.index[i].hour * 60 + df.index[i].minute
            if 810 <= tot_min <= 915 and adx > 22:
                tp2_actual = 2.10
                tp3_actual = 3.80

        if is_buy:
            sl  = round(entry_price - (atr * sl_mult), 2)
            tp1 = round(entry_price + (atr * tp1_mult), 2)
            tp2 = round(entry_price + (atr * tp2_actual), 2)
            tp3 = round(entry_price + (atr * tp3_actual), 2)
        else:
            sl  = round(entry_price + (atr * sl_mult), 2)
            tp1 = round(entry_price - (atr * tp1_mult), 2)
            tp2 = round(entry_price - (atr * tp2_actual), 2)
            tp3 = round(entry_price - (atr * tp3_actual), 2)

        hit_tp1 = hit_tp2 = hit_tp3 = hit_sl = False
        sl_trail = sl
        bars_held = 0
        exit_price = entry_price

        for j in range(i + 1, min(i + 25, n)):
            fut = df.iloc[j]
            fh = float(fut['High'])
            fl = float(fut['Low'])
            fc = float(fut['Close'])
            bars_held = j - i

            if is_buy:
                # Wick cushion shield
                if (fc <= sl_trail or fl <= sl_trail - (wick_cushion * atr)) and not hit_tp1:
                    hit_sl = True
                    exit_price = sl_trail
                    break

                if fh >= tp1 and not hit_tp1:
                    hit_tp1 = True
                    # Structural breathing buffer on retest
                    sl_trail = max(sl, entry_price - (0.15 * atr))

                # Chandelier Trailing Stop for 5M: High minus 1.5x ATR
                if hit_tp1:
                    chand_sl = fh - (1.50 * atr)
                    if chand_sl > sl_trail:
                        sl_trail = chand_sl

                if hit_tp1 and fh >= tp2 and not hit_tp2:
                    hit_tp2 = True
                    sl_trail = max(sl_trail, tp1)

                if hit_tp2 and fh >= tp3 and not hit_tp3:
                    hit_tp3 = True
                    exit_price = tp3
                    break

                if hit_tp1 and fl <= sl_trail:
                    exit_price = sl_trail
                    break
            else:
                if (fc >= sl_trail or fh >= sl_trail + (wick_cushion * atr)) and not hit_tp1:
                    hit_sl = True
                    exit_price = sl_trail
                    break

                if fl <= tp1 and not hit_tp1:
                    hit_tp1 = True
                    sl_trail = min(sl, entry_price + (0.15 * atr))

                if hit_tp1:
                    chand_sl = fl + (1.50 * atr)
                    if chand_sl < sl_trail:
                        sl_trail = chand_sl

                if hit_tp1 and fl <= tp2 and not hit_tp2:
                    hit_tp2 = True
                    sl_trail = min(sl_trail, tp1)

                if hit_tp2 and fl <= tp3 and not hit_tp3:
                    hit_tp3 = True
                    exit_price = tp3
                    break

                if hit_tp1 and fh >= sl_trail:
                    exit_price = sl_trail
                    break

        if hit_sl:
            realized_pnl_pct = ((sl - entry_price) / entry_price * 100) if is_buy else ((entry_price - sl) / entry_price * 100)
            captured_pts = (sl - entry_price) if is_buy else (entry_price - sl)
        elif hit_tp3:
            tp1_gain = ((tp1 - entry_price) / entry_price * 100) if is_buy else ((entry_price - tp1) / entry_price * 100)
            tp3_gain = ((tp3 - entry_price) / entry_price * 100) if is_buy else ((entry_price - tp3) / entry_price * 100)
            realized_pnl_pct = (book_pct * tp1_gain) + ((1.0 - book_pct) * tp3_gain)
            captured_pts = (book_pct * abs(tp1 - entry_price)) + ((1.0 - book_pct) * abs(tp3 - entry_price))
        elif hit_tp2:
            tp1_gain = ((tp1 - entry_price) / entry_price * 100) if is_buy else ((entry_price - tp1) / entry_price * 100)
            tp2_gain = ((tp2 - entry_price) / entry_price * 100) if is_buy else ((entry_price - tp2) / entry_price * 100)
            realized_pnl_pct = (book_pct * tp1_gain) + ((1.0 - book_pct) * tp2_gain)
            captured_pts = (book_pct * abs(tp1 - entry_price)) + ((1.0 - book_pct) * abs(tp2 - entry_price))
        elif hit_tp1:
            tp1_gain = ((tp1 - entry_price) / entry_price * 100) if is_buy else ((entry_price - tp1) / entry_price * 100)
            realized_pnl_pct = book_pct * tp1_gain
            captured_pts = book_pct * abs(tp1 - entry_price)
        else:
            realized_pnl_pct = 0.0
            captured_pts = 0.0

        trades.append({
            "entry_date": entry_date, "type": "BUY" if is_buy else "SELL",
            "entry_price": round(entry_price, 2), "stop_loss": round(sl, 2),
            "tp1": round(tp1, 2), "tp2": round(tp2, 2),
            "hit_tp1": hit_tp1, "hit_tp2": hit_tp2, "hit_sl": hit_sl,
            "book_pct": round(book_pct * 100, 1),
            "pnl_pct": round(realized_pnl_pct, 3), "captured_pts": round(captured_pts, 2),
            "bars_held": bars_held
        })
        i += max(cooldown, bars_held)

    total_t = len(trades)
    if total_t == 0:
        return {"symbol": used_symbol, "error": "No trades"}

    tp1_cnt = sum(1 for t in trades if t['hit_tp1'])
    tp2_cnt = sum(1 for t in trades if t['hit_tp2'])
    sl_cnt  = sum(1 for t in trades if t['hit_sl'])

    wins = [t['pnl_pct'] for t in trades if t['pnl_pct'] > 0]
    losses = [abs(t['pnl_pct']) for t in trades if t['pnl_pct'] < 0]

    profit_factor = round(sum(wins) / max(0.01, sum(losses)), 2) if losses else 9.9
    avg_gain = round(float(np.mean(wins)), 3) if wins else 0.0
    avg_loss = round(float(np.mean(losses)), 3) if losses else 0.0
    total_pts = round(float(sum(t['captured_pts'] for t in trades)), 1)

    return {
        "symbol": used_symbol,
        "total_bars": len(df),
        "total_trades": total_t,
        "tp1_win_rate": round(tp1_cnt / total_t * 100, 1),
        "tp2_rate": round(tp2_cnt / total_t * 100, 1),
        "sl_rate": round(sl_cnt / total_t * 100, 1),
        "profit_factor": profit_factor,
        "avg_gain_pct": avg_gain,
        "avg_loss_pct": avg_loss,
        "total_pts": total_pts,
        "avg_book_pct": round(float(np.mean([t['book_pct'] for t in trades])), 1)
    }

if __name__ == "__main__":
    print("=" * 105)
    print(f"{'SYMBOL':12} | {'TRADES':6} | {'TP1 WIN':8} | {'SL HIT':7} | {'AVG GAIN':9} | {'AVG LOSS':9} | {'PF':6} | {'NET PTS':9}")
    print("=" * 105)

    all_res = []
    for s in STOCKS_5M:
        try:
            r = run_5m_dynamic_backtest(s)
            if 'error' in r:
                print(f"Error {s}: {r['error']}")
                continue
            all_res.append(r)
            print(f"{r['symbol']:12} | {r['total_trades']:6d} | {r['tp1_win_rate']:7.1f}% | {r['sl_rate']:6.1f}% | +{r['avg_gain_pct']:6.3f}% | -{r['avg_loss_pct']:6.3f}% | {r['profit_factor']:5.2f}x | +{r['total_pts']:8.1f}")
        except Exception as e:
            print(f"Error {s}: {e}")

    print("=" * 105)
    with open('audit_5m_dynamic_rr.json', 'w') as f:
        json.dump(all_res, f, indent=2)
    print("\n[SUCCESS] 5-Minute Dynamic RR Audit saved to audit_5m_dynamic_rr.json")
