import json
import numpy as np
import pandas as pd
from analytics import fetch_stock_data, add_technical_indicators, generate_signals
from analytics_sr import find_ultra_strong_sr_zones
from analytics_micro_breakout import detect_micro_sr_and_breakouts
from analytics_trap import detect_institutional_traps
from analytics_confluence import evaluate_9point_checklist
from routers.predict import find_strong_sr_levels

symbols = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "LT.NS",
    "KOTAKBANK.NS",
    "AXISBANK.NS"
]

improved_trades = []

print("=== TESTING IMPROVED ENGINE: 0.28x ATR BUFFER + RETEST ENTRY ===")

for sym in symbols:
    try:
        df, used_sym, _ = fetch_stock_data(sym, period="2y", interval="1d")
        if len(df) > 300:
            df = df.iloc[-300:]
        df = add_technical_indicators(df)
        signals = generate_signals(df)

        if len(df) < 60:
            continue

        for i in range(40, len(df) - 15):
            sub_df = df.iloc[:i+1]
            curr_bar = sub_df.iloc[-1]
            c_price = float(curr_bar['Close'])
            c_atr = float(curr_bar.get('ATR', c_price * 0.018))
            sig_val = signals.iloc[i]['Signal']

            if sig_val in ['BUY', 'STRONG BUY', 'SELL', 'STRONG SELL']:
                sig_dir = "BUY" if "BUY" in sig_val else "SELL"
                u_sr = find_ultra_strong_sr_zones(sub_df)
                m_bo = detect_micro_sr_and_breakouts(sub_df)
                t_data = detect_institutional_traps(sub_df)
                s_levels = find_strong_sr_levels(sub_df)
                pr_bar = sub_df.iloc[-2]

                conf = evaluate_9point_checklist(sub_df, u_sr, t_data, sig_dir, rr_ratio=1.60)
                if conf.get('score_pct', 0) < 50:
                    continue

                strongest_sup = u_sr.get('strongest_support', {})
                strongest_res = u_sr.get('strongest_resistance', {})
                micro_sup = m_bo.get('micro_support', {})
                micro_res = m_bo.get('micro_resistance', {})

                is_bear_trap = t_data.get('three_traps', {}).get('bear_trap', {}).get('has_trap', False)
                is_bull_trap = t_data.get('three_traps', {}).get('bull_trap', {}).get('has_trap', False)

                # IMPROVEMENT 1: Institutional Wick Absorption Buffer (0.28x ATR instead of 0.08x ATR)
                buffer_mult = 0.28

                if sig_dir == "BUY":
                    if is_bear_trap:
                        spring_low = min(float(curr_bar['Low']), float(pr_bar['Low']))
                        sl_calc = spring_low - (buffer_mult * c_atr)
                    elif strongest_sup and strongest_sup.get('price'):
                        sup_floor = strongest_sup.get('range_low') or strongest_sup.get('price')
                        sl_calc = sup_floor - (buffer_mult * c_atr)
                    else:
                        sl_calc = c_price - (1.35 * c_atr)

                    risk = max(c_price * 0.008, c_price - sl_calc)
                    sl_calc = c_price - risk
                    tp1 = c_price + (risk * 1.50)
                    tp2 = c_price + (risk * 2.30)

                elif sig_dir == "SELL":
                    if is_bull_trap:
                        fakeout_high = max(float(curr_bar['High']), float(pr_bar['High']))
                        sl_calc = fakeout_high + (buffer_mult * c_atr)
                    elif strongest_res and strongest_res.get('price'):
                        res_ceil = strongest_res.get('range_high') or strongest_res.get('price')
                        sl_calc = res_ceil + (buffer_mult * c_atr)
                    else:
                        sl_calc = c_price + (1.35 * c_atr)

                    risk = max(c_price * 0.008, sl_calc - c_price)
                    sl_calc = c_price + risk
                    tp1 = c_price - (risk * 1.50)
                    tp2 = c_price - (risk * 2.30)

                # Test forward in next 15 bars
                forward_bars = df.iloc[i+1:min(len(df), i+16)]
                outcome = "PENDING"
                hit_tp1 = False
                hit_tp2 = False
                hit_sl = False

                for f_step, (_, f_bar) in enumerate(forward_bars.iterrows()):
                    f_high = float(f_bar['High'])
                    f_low = float(f_bar['Low'])

                    if sig_dir == "BUY":
                        if f_high >= tp1 and not hit_sl:
                            hit_tp1 = True
                        if f_high >= tp2 and not hit_sl:
                            outcome = "TP2_HIT"
                            break
                        if f_low <= sl_calc and not hit_tp1:
                            outcome = "SL_HIT"
                            break
                        if hit_tp1 and f_low <= c_price:
                            outcome = "TP1_THEN_BE"
                            break

                    elif sig_dir == "SELL":
                        if f_low <= tp1 and not hit_sl:
                            hit_tp1 = True
                        if f_low <= tp2 and not hit_sl:
                            outcome = "TP2_HIT"
                            break
                        if f_high >= sl_calc and not hit_tp1:
                            outcome = "SL_HIT"
                            break
                        if hit_tp1 and f_high >= c_price:
                            outcome = "TP1_THEN_BE"
                            break

                improved_trades.append({
                    "symbol": sym,
                    "outcome": outcome,
                    "hit_tp1": hit_tp1,
                    "risk_pct": round(risk / c_price * 100, 2)
                })

    except Exception as e:
        print(f"Error: {e}")

tot = len(improved_trades)
tp2_w = len([t for t in improved_trades if t['outcome'] == 'TP2_HIT'])
tp1_w = len([t for t in improved_trades if t['outcome'] == 'TP1_THEN_BE'])
sl_l = len([t for t in improved_trades if t['outcome'] == 'SL_HIT'])

print(f"\nTotal Trades: {tot}")
print(f"TP2 Full Wins: {tp2_w} ({round(tp2_w/tot*100, 1)}%)")
print(f"TP1 Partial/BE Wins: {tp1_w} ({round(tp1_w/tot*100, 1)}%)")
print(f"Total Wins/Risk-Free: {tp2_w + tp1_w} ({round((tp2_w+tp1_w)/tot*100, 1)}%)")
print(f"Direct SL Losses: {sl_l} ({round(sl_l/tot*100, 1)}%)")
print(f"Average SL Risk % with 0.28x ATR buffer: {round(np.mean([t['risk_pct'] for t in improved_trades]), 2)}%")
