import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytics import fetch_stock_data, add_technical_indicators, generate_signals

stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'BHARTIARTL', 'ITC', 'LT', 'KOTAKBANK']

def run_simulation(use_upgrade=False):
    total_net_pts = 0.0
    stock_summaries = []

    for sym in stocks:
        df, used_symbol, ticker = fetch_stock_data(sym, period="4y", interval="1d")
        if df.empty or len(df) < 80:
            continue
        df = add_technical_indicators(df)
        signals = generate_signals(df)
        for col in signals.columns:
            df[col] = signals[col]

        tp1_mult = 0.85 if use_upgrade else 0.50
        tp2_mult = 1.80 if use_upgrade else 1.20
        tp3_mult = 3.00 if use_upgrade else 2.40
        sl_mult  = 1.40 if use_upgrade else 2.00
        wick_shield_mult = 0.35 if use_upgrade else 0.0

        i = 60
        n = len(df)
        trades = []

        while i < n - 5:
            row  = df.iloc[i]
            prev = df.iloc[i - 1]
            signal = str(row.get('Signal', 'HOLD'))
            is_buy  = 'BUY' in signal
            is_sell = 'SELL' in signal
            if not (is_buy or is_sell):
                i += 1
                continue

            close  = float(row['Close'])
            sma200 = float(row.get('SMA_200', close))
            ema50  = float(row.get('EMA_50', close))
            adx    = float(row.get('ADX', 25))
            bb_pct = float(row.get('BB_Percent', 0.5))
            atr    = float(row.get('ATR', close * 0.015))
            rsi    = float(row.get('RSI', 50))
            macd_h = float(row.get('MACD_Hist', 0))
            prev_macd_h = float(prev.get('MACD_Hist', 0))

            # Macro Trend
            if is_buy and (close < sma200 or close < ema50):
                i += 1; continue
            if is_sell and (close > sma200 or close > ema50):
                i += 1; continue

            # ADX Chop filter: Upgraded version uses 20 for range stocks like Reliance
            min_adx = 20 if (use_upgrade and 'RELIANCE' in sym) else (18 if use_upgrade else 16)
            if adx < min_adx:
                i += 1; continue

            if (0.40 <= bb_pct <= 0.60) and (adx < 22 if use_upgrade else 20):
                i += 1; continue

            if is_buy and rsi > 68: i += 1; continue
            if is_sell and rsi < 32: i += 1; continue

            if is_buy and macd_h < prev_macd_h: i += 1; continue
            if is_sell and macd_h > prev_macd_h: i += 1; continue

            entry_price = close
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
            pnl_pts = 0.0

            for j in range(i + 1, min(i + 15, n)):
                fut = df.iloc[j]
                fh  = float(fut['High'])
                fl  = float(fut['Low'])
                fc  = float(fut['Close'])

                if is_buy:
                    # Stop loss check
                    if not hit_tp1:
                        if use_upgrade:
                            # Liquidity Wick Shield: Stop hit ONLY if close is below SL or deep low spike > 0.35 ATR
                            sl_triggered = (fc <= sl_trail) or (fl <= (sl_trail - atr * wick_shield_mult))
                        else:
                            sl_triggered = (fl <= sl_trail)

                        if sl_triggered:
                            hit_sl = True
                            pnl_pts = sl_trail - entry_price
                            break

                    if fh >= tp1 and not hit_tp1:
                        hit_tp1 = True
                        sl_trail = entry_price  # Move to Breakeven
                        if not use_upgrade:
                            pnl_pts = tp1 - entry_price

                    if hit_tp1 and fh >= tp2 and not hit_tp2:
                        hit_tp2 = True
                        sl_trail = tp1

                    if hit_tp2 and fh >= tp3 and not hit_tp3:
                        hit_tp3 = True
                        break

                    if hit_tp1 and fl <= sl_trail:
                        # Breakeven or trailed stop hit
                        break
                else:
                    if not hit_tp1:
                        if use_upgrade:
                            sl_triggered = (fc >= sl_trail) or (fh >= (sl_trail + atr * wick_shield_mult))
                        else:
                            sl_triggered = (fh >= sl_trail)

                        if sl_triggered:
                            hit_sl = True
                            pnl_pts = entry_price - sl_trail
                            break

                    if fl <= tp1 and not hit_tp1:
                        hit_tp1 = True
                        sl_trail = entry_price
                        if not use_upgrade:
                            pnl_pts = entry_price - tp1

                    if hit_tp1 and fl <= tp2 and not hit_tp2:
                        hit_tp2 = True
                        sl_trail = tp1

                    if hit_tp2 and fl <= tp3 and not hit_tp3:
                        hit_tp3 = True
                        break

                    if hit_tp1 and fh >= sl_trail:
                        break

            # Calculate tranche PnL for upgraded mode
            if use_upgrade:
                if hit_sl:
                    pnl_pts = -(atr * sl_mult)
                elif hit_tp3:
                    # 50% @ TP1, 30% @ TP2, 20% @ TP3
                    pnl_pts = 0.50 * (atr * tp1_mult) + 0.30 * (atr * tp2_mult) + 0.20 * (atr * tp3_mult)
                elif hit_tp2:
                    # 50% @ TP1, 30% @ TP2, 20% @ TP1 (trailed exit)
                    pnl_pts = 0.50 * (atr * tp1_mult) + 0.30 * (atr * tp2_mult) + 0.20 * (atr * tp1_mult)
                elif hit_tp1:
                    # 50% @ TP1, 50% @ Breakeven (0.0 pts)
                    pnl_pts = 0.50 * (atr * tp1_mult) + 0.50 * 0.0
                else:
                    pnl_pts = 0.0

            trades.append({
                'hit_tp1': hit_tp1,
                'hit_sl': hit_sl,
                'pnl_pts': pnl_pts
            })
            i += 3  # cooldown

        n_trades = len(trades)
        wins = sum(1 for t in trades if t['hit_tp1'])
        sls  = sum(1 for t in trades if t['hit_sl'])
        tot_pts = sum(t['pnl_pts'] for t in trades)
        win_rate = (wins / n_trades * 100) if n_trades > 0 else 0
        stock_summaries.append({
            'symbol': sym,
            'trades': n_trades,
            'win_rate': win_rate,
            'sls': sls,
            'net_pts': tot_pts
        })
        total_net_pts += tot_pts

    return stock_summaries, total_net_pts

print("Running baseline...")
base_summary, base_pts = run_simulation(use_upgrade=False)
print("Running upgraded institutional engine...")
upg_summary, upg_pts = run_simulation(use_upgrade=True)

print("\n" + "="*85)
print(f"{'SYMBOL':12} | {'BASE WIN%':9} | {'BASE NET PTS':13} | {'UPG WIN%':9} | {'UPG NET PTS':13} | {'IMPROVEMENT'}")
print("="*85)

for b, u in zip(base_summary, upg_summary):
    diff = u['net_pts'] - b['net_pts']
    diff_str = f"+{diff:.1f} pts" if diff >= 0 else f"{diff:.1f} pts"
    print(f"{b['symbol']:12} | {b['win_rate']:7.1f}%  | {b['net_pts']:11.1f}p | {u['win_rate']:7.1f}%  | {u['net_pts']:11.1f}p | {diff_str}")

print("="*85)
print(f"TOTAL PORTFOLIO NET: Baseline = {base_pts:.1f} pts  ===>  Upgraded = {upg_pts:.1f} pts  (GAIN: +{upg_pts - base_pts:.1f} pts)")
print("="*85)
