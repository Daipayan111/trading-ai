import json
import numpy as np
import pandas as pd
from analytics import fetch_stock_data, add_technical_indicators, generate_signals
from analytics_sr import find_ultra_strong_sr_zones
from analytics_micro_breakout import detect_micro_sr_and_breakouts
from analytics_trap import detect_institutional_traps
from analytics_confluence import evaluate_9point_checklist
from routers.predict import calc_advanced_trap_sl_target, find_strong_sr_levels

symbols = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "BHARTIARTL.NS"
]

all_trades_confluence = []

print("=== TESTING 9-POINT CONFLUENCE & TRAP-FILTERED HIT RATES ===")

for sym in symbols:
    try:
        df, used_sym, _ = fetch_stock_data(sym, period="1y", interval="1d")
        df = add_technical_indicators(df)
        signals = generate_signals(df)

        if len(df) < 50:
            continue

        for i in range(40, len(df)):
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

                # Filter by 9-Point Confluence & Trap Protection
                conf = evaluate_9point_checklist(sub_df, u_sr, t_data, sig_dir, rr_ratio=1.60)
                
                # ONLY take trades where Confluence is passed (GREEN or score >= 55) and Trap Safe
                is_trap_dangerous = (sig_dir == "BUY" and t_data.get("three_traps", {}).get("bull_trap", {}).get("has_trap")) or \
                                    (sig_dir == "SELL" and t_data.get("three_traps", {}).get("bear_trap", {}).get("has_trap"))
                
                if is_trap_dangerous:
                    continue # Trap Shield avoided this losing trade!

                if conf.get('score_pct', 0) < 50:
                    continue # Filtered out low quality setup!

                sl_calc, tp_calc, rr_calc, tgts_calc, _, _ = calc_advanced_trap_sl_target(
                    current_price=c_price,
                    direction=sig_dir,
                    atr=c_atr,
                    sr_levels=s_levels,
                    ultra_sr=u_sr,
                    micro_breakout=m_bo,
                    trap_data=t_data,
                    conv_score=conf.get('score_pct', 70),
                    interval="1d",
                    latest_candle=curr_bar,
                    prev_candle=pr_bar
                )

                risk_pts = abs(c_price - sl_calc)
                risk_pct = round(risk_pts / c_price * 100, 2)
                tp1_price = tgts_calc['tp1']['price']
                tp2_price = tgts_calc['tp2']['price']
                tp1_rr = tgts_calc['tp1']['risk_reward']
                tp2_rr = tgts_calc['tp2']['risk_reward']

                hit_tp1_first = False
                hit_tp2_first = False
                hit_sl_first = False

                future_bars = df.iloc[i+1:min(len(df), i+25)]
                for f_idx, (_, f_bar) in enumerate(future_bars.iterrows()):
                    f_high = float(f_bar['High'])
                    f_low = float(f_bar['Low'])

                    if sig_dir == "BUY":
                        if f_high >= tp1_price and not hit_sl_first:
                            hit_tp1_first = True
                        if f_high >= tp2_price and not hit_sl_first:
                            hit_tp2_first = True
                        if f_low <= sl_calc and not hit_tp1_first:
                            hit_sl_first = True
                            break
                        if hit_tp2_first:
                            break

                    elif sig_dir == "SELL":
                        if f_low <= tp1_price and not hit_sl_first:
                            hit_tp1_first = True
                        if f_low <= tp2_price and not hit_sl_first:
                            hit_tp2_first = True
                        if f_high >= sl_calc and not hit_tp1_first:
                            hit_sl_first = True
                            break
                        if hit_tp2_first:
                            break

                all_trades_confluence.append({
                    "symbol": sym,
                    "direction": sig_dir,
                    "traffic_light": conf.get('traffic_light'),
                    "score_pct": conf.get('score_pct'),
                    "tp1_rr": tp1_rr,
                    "tp2_rr": tp2_rr,
                    "hit_tp1_first": hit_tp1_first,
                    "hit_tp2_first": hit_tp2_first,
                    "hit_sl_first": hit_sl_first
                })

    except Exception as e:
        print(f"Error on {sym}: {e}")

total = len(all_trades_confluence)
tp1_cnt = len([t for t in all_trades_confluence if t['hit_tp1_first']])
tp2_cnt = len([t for t in all_trades_confluence if t['hit_tp2_first']])
sl_cnt = len([t for t in all_trades_confluence if t['hit_sl_first']])

print(f"\nTotal High-Confluence Filtered Setups: {total}")
print(f"TP1 Hit Rate: {round(tp1_cnt / max(1, total) * 100, 2)}% ({tp1_cnt}/{total})")
print(f"TP2 Hit Rate: {round(tp2_cnt / max(1, total) * 100, 2)}% ({tp2_cnt}/{total})")
print(f"Direct SL Hit Rate: {round(sl_cnt / max(1, total) * 100, 2)}% ({sl_cnt}/{total})")

# By Traffic Light
green_trades = [t for t in all_trades_confluence if t['traffic_light'] == 'GREEN']
if green_trades:
    g_tp1 = len([t for t in green_trades if t['hit_tp1_first']])
    g_tp2 = len([t for t in green_trades if t['hit_tp2_first']])
    g_sl = len([t for t in green_trades if t['hit_sl_first']])
    print(f"\n[🟢 GREEN LIGHT ONLY] Total: {len(green_trades)} | TP1 Hit: {round(g_tp1/len(green_trades)*100,2)}% | TP2 Hit: {round(g_tp2/len(green_trades)*100,2)}% | Direct SL Hit: {round(g_sl/len(green_trades)*100,2)}%")

with open("test_confluence_hit_rates.json", "w") as f:
    json.dump({
        "total": total,
        "tp1_pct": round(tp1_cnt / max(1, total) * 100, 2),
        "tp2_pct": round(tp2_cnt / max(1, total) * 100, 2),
        "sl_pct": round(sl_cnt / max(1, total) * 100, 2),
        "green_count": len(green_trades),
        "green_tp1_pct": round(g_tp1/len(green_trades)*100,2) if green_trades else 0,
        "green_tp2_pct": round(g_tp2/len(green_trades)*100,2) if green_trades else 0,
        "green_sl_pct": round(g_sl/len(green_trades)*100,2) if green_trades else 0,
    }, f, indent=2)
