import json
import numpy as np
import pandas as pd
from analytics import fetch_stock_data, add_technical_indicators, generate_signals
from analytics_sr import find_ultra_strong_sr_zones
from analytics_micro_breakout import detect_micro_sr_and_breakouts
from analytics_trap import detect_institutional_traps
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

all_trades = []

print("=== TESTING EXACT HIT RATES & SUCCESSFUL RISK-REWARD RATIOS ===")

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

                sl_calc, tp_calc, rr_calc, tgts_calc, _, _ = calc_advanced_trap_sl_target(
                    current_price=c_price,
                    direction=sig_dir,
                    atr=c_atr,
                    sr_levels=s_levels,
                    ultra_sr=u_sr,
                    micro_breakout=m_bo,
                    trap_data=t_data,
                    conv_score=75,
                    interval="1d",
                    latest_candle=curr_bar,
                    prev_candle=pr_bar
                )

                risk_pts = abs(c_price - sl_calc)
                risk_pct = round(risk_pts / c_price * 100, 2)
                tp1_price = tgts_calc['tp1']['price']
                tp2_price = tgts_calc['tp2']['price']
                tp3_price = tgts_calc['tp3']['price']
                tp1_rr = tgts_calc['tp1']['risk_reward']
                tp2_rr = tgts_calc['tp2']['risk_reward']

                # Test forward in next 20 bars to see which level hits first
                hit_tp1_first = False
                hit_tp2_first = False
                hit_sl_first = False
                bars_to_outcome = 0

                future_bars = df.iloc[i+1:min(len(df), i+25)]
                for f_idx, (_, f_bar) in enumerate(future_bars.iterrows()):
                    f_high = float(f_bar['High'])
                    f_low = float(f_bar['Low'])

                    # Check for BUY
                    if sig_dir == "BUY":
                        hit_tp1 = f_high >= tp1_price
                        hit_sl = f_low <= sl_calc
                        hit_tp2 = f_high >= tp2_price

                        if hit_tp1 and not hit_sl:
                            hit_tp1_first = True
                        if hit_tp2 and not hit_sl:
                            hit_tp2_first = True
                        if hit_sl and not hit_tp1_first:
                            hit_sl_first = True
                            bars_to_outcome = f_idx + 1
                            break
                        if hit_tp2_first:
                            bars_to_outcome = f_idx + 1
                            break

                    # Check for SELL
                    elif sig_dir == "SELL":
                        hit_tp1 = f_low <= tp1_price
                        hit_sl = f_high >= sl_calc
                        hit_tp2 = f_low <= tp2_price

                        if hit_tp1 and not hit_sl:
                            hit_tp1_first = True
                        if hit_tp2 and not hit_sl:
                            hit_tp2_first = True
                        if hit_sl and not hit_tp1_first:
                            hit_sl_first = True
                            bars_to_outcome = f_idx + 1
                            break
                        if hit_tp2_first:
                            bars_to_outcome = f_idx + 1
                            break

                all_trades.append({
                    "symbol": sym,
                    "date": sub_df.index[-1].strftime("%Y-%m-%d"),
                    "direction": sig_dir,
                    "entry_price": c_price,
                    "sl_risk_pct": risk_pct,
                    "tp1_rr": tp1_rr,
                    "tp2_rr": tp2_rr,
                    "hit_tp1_first": hit_tp1_first,
                    "hit_tp2_first": hit_tp2_first,
                    "hit_sl_first": hit_sl_first,
                    "bars_to_outcome": bars_to_outcome
                })

    except Exception as e:
        print(f"Error on {sym}: {e}")

total_trades = len(all_trades)
tp1_hits = [t for t in all_trades if t['hit_tp1_first']]
tp2_hits = [t for t in all_trades if t['hit_tp2_first']]
sl_first_hits = [t for t in all_trades if t['hit_sl_first']]

tp1_hit_rate = round(len(tp1_hits) / max(1, total_trades) * 100, 2)
tp2_hit_rate = round(len(tp2_hits) / max(1, total_trades) * 100, 2)
sl_first_hit_rate = round(len(sl_first_hits) / max(1, total_trades) * 100, 2)

# Hit rate across RR brackets
rr_brackets = {
    "1:1.50 (TP1 Safe Baseline)": [t for t in all_trades if 1.4 <= t['tp1_rr'] <= 1.8],
    "1:2.00 - 1:2.50 (TP2 Institutional Sweet Spot)": [t for t in all_trades if 1.9 <= t['tp2_rr'] <= 2.6],
    "1:2.50 - 1:3.50 (High Reward)": [t for t in all_trades if 2.6 < t['tp2_rr'] <= 3.6],
    "> 1:3.50 (Mega Squeeze / Trap Runner)": [t for t in all_trades if t['tp2_rr'] > 3.6]
}

bracket_results = {}
for name, b_trades in rr_brackets.items():
    if len(b_trades) > 0:
        b_tp1_wins = len([t for t in b_trades if t['hit_tp1_first']])
        b_tp2_wins = len([t for t in b_trades if t['hit_tp2_first']])
        b_sl_first = len([t for t in b_trades if t['hit_sl_first']])
        bracket_results[name] = {
            "total_trades": len(b_trades),
            "tp1_win_rate_pct": round(b_tp1_wins / len(b_trades) * 100, 2),
            "tp2_win_rate_pct": round(b_tp2_wins / len(b_trades) * 100, 2),
            "sl_loss_rate_pct": round(b_sl_first / len(b_trades) * 100, 2),
            "effective_profitability": "HIGHEST SUCCESS" if b_tp1_wins / len(b_trades) > 0.65 else ("BALANCED HIGH" if b_tp1_wins / len(b_trades) > 0.50 else "RUNNER")
        }

summary = {
    "total_signals_evaluated": total_trades,
    "tp1_hit_rate_pct": tp1_hit_rate,
    "tp2_hit_rate_pct": tp2_hit_rate,
    "sl_hit_first_pct": sl_first_hit_rate,
    "rr_brackets": bracket_results
}

with open("test_hit_rates_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== FINAL TEST RESULTS ===")
print(f"Total Signals Evaluated: {total_trades}")
print(f"TP1 Hit Rate (Target 1 Hits Before SL): {tp1_hit_rate}%")
print(f"TP2 Hit Rate (Full Institutional Target Hits): {tp2_hit_rate}%")
print(f"Stop Loss Hit First (SL hits before reaching TP1): {sl_first_hit_rate}%")
print("\n=== SUCCESS BY RISK-REWARD BRACKET ===")
for b, data in bracket_results.items():
    print(f"[{b}] Trades: {data['total_trades']} | TP1 Hit: {data['tp1_win_rate_pct']}% | TP2 Hit: {data['tp2_win_rate_pct']}% | Direct SL Hit: {data['sl_loss_rate_pct']}%")
