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
    "BHARTIARTL.NS",
    "LT.NS",
    "KOTAKBANK.NS",
    "AXISBANK.NS"
]

all_trades_audit = []
stock_summaries = []

print("=== DEEP 300-DAY 10-STOCK AUDIT: POST-SL & POST-TARGET BEHAVIOR ===")

for sym in symbols:
    try:
        # Fetch 300 days of data
        df, used_sym, _ = fetch_stock_data(sym, period="2y", interval="1d")
        if len(df) > 300:
            df = df.iloc[-300:]
        df = add_technical_indicators(df)
        signals = generate_signals(df)

        total_bars = len(df)
        print(f"\nProcessing {sym}: {total_bars} bars.")

        if total_bars < 60:
            continue

        sym_trades = []

        # Scan historical bars from bar 40 to len(df)-15
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

                # Trap safety filter
                is_counter_trap = (sig_dir == "BUY" and t_data.get("three_traps", {}).get("bull_trap", {}).get("has_trap")) or \
                                  (sig_dir == "SELL" and t_data.get("three_traps", {}).get("bear_trap", {}).get("has_trap"))
                if is_counter_trap:
                    continue

                sl_calc, tp_calc, rr_calc, tgts_calc, sl_rat, tp_rat = calc_advanced_trap_sl_target(
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

                # Test what happened in the next 15 bars
                forward_bars = df.iloc[i+1:min(len(df), i+16)]
                
                outcome = "PENDING"
                exit_idx = 0
                exit_price = 0
                hit_tp1 = False
                hit_tp2 = False
                hit_sl = False

                for f_step, (_, f_bar) in enumerate(forward_bars.iterrows()):
                    f_high = float(f_bar['High'])
                    f_low = float(f_bar['Low'])

                    if sig_dir == "BUY":
                        if f_high >= tp1_price and not hit_sl:
                            hit_tp1 = True
                        if f_high >= tp2_price and not hit_sl:
                            hit_tp2 = True
                            outcome = "TP2_HIT"
                            exit_idx = i + 1 + f_step
                            exit_price = tp2_price
                            break
                        if f_low <= sl_calc and not hit_tp1:
                            hit_sl = True
                            outcome = "SL_HIT"
                            exit_idx = i + 1 + f_step
                            exit_price = sl_calc
                            break
                        if hit_tp1 and f_low <= c_price: # Breakeven exit on remaining
                            outcome = "TP1_THEN_BE"
                            exit_idx = i + 1 + f_step
                            exit_price = c_price
                            break

                    elif sig_dir == "SELL":
                        if f_low <= tp1_price and not hit_sl:
                            hit_tp1 = True
                        if f_low <= tp2_price and not hit_sl:
                            hit_tp2 = True
                            outcome = "TP2_HIT"
                            exit_idx = i + 1 + f_step
                            exit_price = tp2_price
                            break
                        if f_high >= sl_calc and not hit_tp1:
                            hit_sl = True
                            outcome = "SL_HIT"
                            exit_idx = i + 1 + f_step
                            exit_price = sl_calc
                            break
                        if hit_tp1 and f_high >= c_price:
                            outcome = "TP1_THEN_BE"
                            exit_idx = i + 1 + f_step
                            exit_price = c_price
                            break

                if outcome == "PENDING":
                    outcome = "TIME_EXPIRE"
                    exit_idx = min(len(df)-1, i+15)
                    exit_price = float(df.iloc[exit_idx]['Close'])

                # ── POST-EVENT ANALYSIS ──
                # What did the market do AFTER SL or AFTER TP?
                post_event_data = {}
                post_bars = df.iloc[exit_idx+1:min(len(df), exit_idx+11)] # next 10 bars after exit

                if outcome == "SL_HIT" and len(post_bars) > 0:
                    # Check if market reversed and went to original target (Whipsaw!) OR continued crashing (Lifesaver!)
                    max_post_high = float(post_bars['High'].max())
                    min_post_low = float(post_bars['Low'].min())

                    if sig_dir == "BUY":
                        rebounded_to_target = max_post_high >= tp1_price
                        rebounded_to_entry = max_post_high >= c_price
                        plunged_further = min_post_low < (sl_calc - (0.5 * c_atr))
                        max_drawdown_post_sl = round((sl_calc - min_post_low) / c_price * 100, 2)
                    else:
                        rebounded_to_target = min_post_low <= tp1_price
                        rebounded_to_entry = min_post_low <= c_price
                        plunged_further = max_post_high > (sl_calc + (0.5 * c_atr))
                        max_drawdown_post_sl = round((max_post_high - sl_calc) / c_price * 100, 2)

                    if rebounded_to_target:
                        sl_classification = "WHIPSAW_TARGET_REACHED" # SL was too tight!
                    elif rebounded_to_entry:
                        sl_classification = "WHIPSAW_ENTRY_REACHED"  # SL was tagged, but recovered to cost
                    elif plunged_further:
                        sl_classification = "LIFESAVER_SAVED_CAPITAL" # SL saved us from massive drop!
                    else:
                        sl_classification = "CHOP_AFTER_SL"

                    post_event_data = {
                        "sl_classification": sl_classification,
                        "rebounded_to_target": rebounded_to_target,
                        "plunged_further": plunged_further,
                        "extra_drawdown_saved_pct": max_drawdown_post_sl
                    }

                elif outcome in ["TP2_HIT", "TP1_THEN_BE"] and len(post_bars) > 0:
                    # Check if target was the top/bottom (Pinpoint Exit) OR if market ran much further (Left Money on Table)
                    max_p_high = float(post_bars['High'].max())
                    min_p_low = float(post_bars['Low'].min())

                    if sig_dir == "BUY":
                        reversed_from_target = min_p_low < (exit_price - (0.5 * c_atr))
                        ran_further_pct = round((max_p_high - exit_price) / c_price * 100, 2)
                    else:
                        reversed_from_target = max_p_high > (exit_price + (0.5 * c_atr))
                        ran_further_pct = round((exit_price - min_p_low) / c_price * 100, 2)

                    if reversed_from_target and ran_further_pct < 1.0:
                        tp_classification = "PINPOINT_TOP_EXIT" # Exact institutional ceiling!
                    elif ran_further_pct >= 3.0:
                        tp_classification = "BIG_RUNNER_MISSED"  # Could have used TP3 trailing runner!
                    else:
                        tp_classification = "HEALTHY_TARGET"

                    post_event_data = {
                        "tp_classification": tp_classification,
                        "reversed_from_target": reversed_from_target,
                        "extra_move_pct": max(0, ran_further_pct)
                    }

                trade_record = {
                    "symbol": sym,
                    "date": sub_df.index[-1].strftime("%Y-%m-%d"),
                    "direction": sig_dir,
                    "entry_price": c_price,
                    "sl_price": sl_calc,
                    "sl_risk_pct": risk_pct,
                    "tp1_price": tp1_price,
                    "tp2_price": tp2_price,
                    "tp1_rr": tgts_calc['tp1']['risk_reward'],
                    "tp2_rr": tgts_calc['tp2']['risk_reward'],
                    "outcome": outcome,
                    "post_event": post_event_data
                }
                sym_trades.append(trade_record)
                all_trades_audit.append(trade_record)

        # Summarize for this stock
        tot = len(sym_trades)
        tp2_wins = [t for t in sym_trades if t['outcome'] == 'TP2_HIT']
        tp1_be = [t for t in sym_trades if t['outcome'] == 'TP1_THEN_BE']
        sl_hits = [t for t in sym_trades if t['outcome'] == 'SL_HIT']

        # Post-SL breakdown
        sl_whipsaws = [t for t in sl_hits if t.get('post_event', {}).get('sl_classification') in ['WHIPSAW_TARGET_REACHED', 'WHIPSAW_ENTRY_REACHED']]
        sl_lifesavers = [t for t in sl_hits if t.get('post_event', {}).get('sl_classification') == 'LIFESAVER_SAVED_CAPITAL']

        # Post-TP breakdown
        tp_pinpoint = [t for t in tp2_wins if t.get('post_event', {}).get('tp_classification') == 'PINPOINT_TOP_EXIT']
        tp_runners = [t for t in tp2_wins if t.get('post_event', {}).get('tp_classification') == 'BIG_RUNNER_MISSED']

        stock_summaries.append({
            "symbol": sym,
            "total_trades": tot,
            "tp2_hits": len(tp2_wins),
            "tp1_be_hits": len(tp1_be),
            "sl_hits": len(sl_hits),
            "win_rate_pct": round((len(tp2_wins) + len(tp1_be)) / max(1, tot) * 100, 2),
            "sl_whipsaw_count": len(sl_whipsaws),
            "sl_whipsaw_pct": round(len(sl_whipsaws) / max(1, len(sl_hits)) * 100, 2) if sl_hits else 0,
            "sl_lifesaver_count": len(sl_lifesavers),
            "sl_lifesaver_pct": round(len(sl_lifesavers) / max(1, len(sl_hits)) * 100, 2) if sl_hits else 0,
            "tp_pinpoint_count": len(tp_pinpoint),
            "tp_runner_count": len(tp_runners),
            "avg_sl_risk_pct": round(np.mean([t['sl_risk_pct'] for t in sym_trades]), 2) if tot else 0,
            "avg_tp2_rr": round(np.mean([t['tp2_rr'] for t in sym_trades]), 2) if tot else 0,
        })

    except Exception as e:
        print(f"Error on {sym}: {e}")

# Overall Statistics
total_all = len(all_trades_audit)
total_sl = len([t for t in all_trades_audit if t['outcome'] == 'SL_HIT'])
total_tp2 = len([t for t in all_trades_audit if t['outcome'] == 'TP2_HIT'])
total_tp1_be = len([t for t in all_trades_audit if t['outcome'] == 'TP1_THEN_BE'])

all_whipsaws = [t for t in all_trades_audit if t.get('post_event', {}).get('sl_classification') in ['WHIPSAW_TARGET_REACHED', 'WHIPSAW_ENTRY_REACHED']]
all_lifesavers = [t for t in all_trades_audit if t.get('post_event', {}).get('sl_classification') == 'LIFESAVER_SAVED_CAPITAL']
all_pinpoint = [t for t in all_trades_audit if t.get('post_event', {}).get('tp_classification') == 'PINPOINT_TOP_EXIT']
all_runners = [t for t in all_trades_audit if t.get('post_event', {}).get('tp_classification') == 'BIG_RUNNER_MISSED']

final_audit = {
    "total_signals_audited": total_all,
    "tp2_full_wins": total_tp2,
    "tp1_be_partial_wins": total_tp1_be,
    "direct_sl_losses": total_sl,
    "overall_win_rate_pct": round((total_tp2 + total_tp1_be) / max(1, total_all) * 100, 2),
    "post_sl_analysis": {
        "total_sl_events": total_sl,
        "sl_whipsaws_count": len(all_whipsaws),
        "sl_whipsaws_pct": round(len(all_whipsaws) / max(1, total_sl) * 100, 2),
        "sl_lifesavers_count": len(all_lifesavers),
        "sl_lifesavers_pct": round(len(all_lifesavers) / max(1, total_sl) * 100, 2),
    },
    "post_tp_analysis": {
        "total_tp_events": total_tp2,
        "tp_pinpoint_reversal_count": len(all_pinpoint),
        "tp_pinpoint_reversal_pct": round(len(all_pinpoint) / max(1, total_tp2) * 100, 2),
        "tp_runners_left_count": len(all_runners),
        "tp_runners_left_pct": round(len(all_runners) / max(1, total_tp2) * 100, 2),
    },
    "stock_summaries": stock_summaries
}

with open("audit_300days_10stocks.json", "w") as f:
    json.dump(final_audit, f, indent=2)

print("\n=== FINAL AUDIT REPORT ===")
print(f"Total Signals Audited across 10 Stocks (300 Days): {total_all}")
print(f"Full Wins (TP2): {total_tp2} ({round(total_tp2/total_all*100, 1)}%) | Partial/BE Wins (TP1): {total_tp1_be} ({round(total_tp1_be/total_all*100, 1)}%) | SL: {total_sl} ({round(total_sl/total_all*100, 1)}%)")
print(f"Combined Win/Risk-Free Rate: {round((total_tp2+total_tp1_be)/total_all*100, 1)}%")
print(f"\n--- Post-Stop Loss Behavior ---")
print(f"Lifesavers (Market kept crashing after SL): {len(all_lifesavers)} ({round(len(all_lifesavers)/max(1, total_sl)*100, 1)}%)")
print(f"Whipsaws (Market reversed after tagging tight SL): {len(all_whipsaws)} ({round(len(all_whipsaws)/max(1, total_sl)*100, 1)}%)")
print(f"\n--- Post-Target Behavior ---")
print(f"Pinpoint Institutional Tops/Reversals: {len(all_pinpoint)} ({round(len(all_pinpoint)/max(1, total_tp2)*100, 1)}%)")
print(f"Runners that ran further: {len(all_runners)} ({round(len(all_runners)/max(1, total_tp2)*100, 1)}%)")
