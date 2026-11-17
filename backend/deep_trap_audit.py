"""
Professional Quantitative Trap Audit:
Pinpointing exactly WHERE and WHY the AI triggers BUY/SELL signals
that result in NO movement (Chop), false breakouts, or stop-outs.
"""

import sys, os
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytics import fetch_stock_data, add_technical_indicators, generate_signals

def audit_intraday_traps(symbol="BANKNIFTY"):
    df, used_symbol, _ = fetch_stock_data(symbol, period="60d", interval="5m")
    if df.empty or len(df) < 500:
        return
    df = add_technical_indicators(df)
    signals = generate_signals(df)

    # Let's inspect all BUY/SELL signals
    trapped_trades = []
    good_trades = []

    for i in range(30, len(df) - 20):
        row = df.iloc[i]
        sig = signals.iloc[i]['Signal']
        if sig not in ['BUY', 'STRONG BUY', 'SELL', 'STRONG SELL']:
            continue

        is_buy = 'BUY' in sig
        entry = float(row['Close'])
        atr = float(row.get('ATR', entry * 0.003))
        adx = float(row.get('ADX', 20))
        vol = float(row.get('Volume_Ratio', 1.0))
        bb_w = float(row.get('BB_Width', 0.02))
        bb_p = float(row.get('BB_Percent', 0.5))
        
        # Time of day
        t_str = str(df.index[i])
        hour = df.index[i].hour if hasattr(df.index[i], 'hour') else 12
        minute = df.index[i].minute if hasattr(df.index[i], 'minute') else 0
        tot_min = hour * 60 + minute

        # Look ahead 15 candles (1 hour 15 mins)
        forward = df.iloc[i+1 : i+16]
        if is_buy:
            max_move = (forward['High'].max() - entry) / entry * 100
            max_drop = (entry - forward['Low'].min()) / entry * 100
            net_15 = (forward['Close'].iloc[-1] - entry) / entry * 100
        else:
            max_move = (entry - forward['Low'].min()) / entry * 100
            max_drop = (forward['High'].max() - entry) / entry * 100
            net_15 = (entry - forward['Close'].iloc[-1]) / entry * 100

        # A "TRAPPED" trade is when market barely moved (< 0.12% move) or immediately dropped into loss (> 0.20% drop)
        is_trapped = (max_move < 0.15) or (max_drop > max_move * 1.5)

        info = {
            "time": t_str[:16],
            "sig": sig,
            "hour": hour,
            "tot_min": tot_min,
            "entry": entry,
            "adx": adx,
            "vol": vol,
            "bb_w": bb_w,
            "bb_p": bb_p,
            "max_move_pct": round(max_move, 3),
            "max_drop_pct": round(max_drop, 3),
            "net_15_pct": round(net_15, 3),
            "is_trapped": is_trapped
        }

        if is_trapped:
            trapped_trades.append(info)
        else:
            good_trades.append(info)

    print("=" * 80)
    print(f"DEEP TRAP AUDIT FOR {used_symbol} (5M Intraday - 60 Days)")
    print("=" * 80)
    total = len(trapped_trades) + len(good_trades)
    print(f"Total Signals: {total} | Trapped/Chop Signals: {len(trapped_trades)} ({len(trapped_trades)/max(1,total)*100:.1f}%) | Clean Moves: {len(good_trades)}")
    
    # 1. TIME OF DAY TRAP ANALYSIS
    trap_hours = [t['hour'] for t in trapped_trades]
    print("\n--- [TRAP ZONE 1] TIME OF DAY BREAKDOWN (WHERE TRAPS OCCUR) ---")
    for h in sorted(set(trap_hours)):
        cnt = trap_hours.count(h)
        pct = cnt / len(trapped_trades) * 100
        print(f"Hour {h:02d}:00 - {cnt:2d} traps ({pct:.1f}% of all trapped trades)")

    # 2. ADX & CHOP ANALYSIS
    trap_adx = [t['adx'] for t in trapped_trades]
    good_adx = [t['adx'] for t in good_trades]
    print(f"\n--- [TRAP ZONE 2] ADX MOMENTUM AT SIGNAL ---")
    print(f"Trapped Trades Avg ADX: {np.mean(trap_adx):.2f} (Low momentum / chop)")
    print(f"Clean Trades Avg ADX:   {np.mean(good_adx):.2f}")
    traps_under_22 = sum(1 for a in trap_adx if a < 22)
    print(f"Traps occurring when ADX < 22: {traps_under_22}/{len(trapped_trades)} ({traps_under_22/len(trapped_trades)*100:.1f}%)")

    # 3. BOLLINGER BAND RANGE POSITION
    trap_bb_p = [t['bb_p'] for t in trapped_trades]
    traps_in_middle = sum(1 for p in trap_bb_p if 0.35 <= p <= 0.65)
    print(f"\n--- [TRAP ZONE 3] DEAD-MIDDLE CHOP (BB 35% - 65%) ---")
    print(f"Traps occurring in the dead-middle of Bollinger Bands: {traps_in_middle}/{len(trapped_trades)} ({traps_in_middle/len(trapped_trades)*100:.1f}%)")

    # 4. VOLUME EXHAUSTION
    trap_vol = [t['vol'] for t in trapped_trades]
    low_vol_traps = sum(1 for v in trap_vol if v < 1.10)
    print(f"\n--- [TRAP ZONE 4] VOLUME DROPOFF (FAKE BREAKOUTS) ---")
    print(f"Traps occurring with Volume Ratio < 1.10x: {low_vol_traps}/{len(trapped_trades)} ({low_vol_traps/len(trapped_trades)*100:.1f}%)")

    print("=" * 80)

if __name__ == '__main__':
    audit_intraday_traps('BANKNIFTY')
    audit_intraday_traps('RELIANCE')
