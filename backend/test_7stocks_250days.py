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

all_stock_results = []
historical_trade_summaries = []

print("=== STARTING 250-DAY 7-STOCK TEST ===")

for sym in symbols:
    try:
        # Fetch 250 days (approx 1 year daily)
        df, used_sym, _ = fetch_stock_data(sym, period="1y", interval="1d")
        df = add_technical_indicators(df)
        signals = generate_signals(df)

        total_bars = len(df)
        print(f"\nProcessing {sym}: {total_bars} daily bars available.")

        if total_bars < 50:
            print(f"Skipping {sym}, insufficient bars.")
            continue

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        current_price = round(float(latest['Close']), 2)
        atr = float(latest.get('ATR', current_price * 0.018))

        ultra_sr = find_ultra_strong_sr_zones(df)
        micro_bo = detect_micro_sr_and_breakouts(df)
        traps = detect_institutional_traps(df)
        sr_levels = find_strong_sr_levels(df)

        pa_levels = {
            'pdh': latest.get('PDH'),
            'pdl': latest.get('PDL'),
            'pdc': latest.get('PDC'),
            'last_swing_high': latest.get('Last_Swing_High'),
            'last_swing_low': latest.get('Last_Swing_Low'),
        }

        # 1. CURRENT DAILY SETUP (BUY & SELL)
        current_setups = {}
        for direction in ["BUY", "SELL"]:
            sl, tp, rr, targets, sl_rat, tp_rat = calc_advanced_trap_sl_target(
                current_price=current_price,
                direction=direction,
                atr=atr,
                sr_levels=sr_levels,
                ultra_sr=ultra_sr,
                micro_breakout=micro_bo,
                trap_data=traps,
                conv_score=75,
                interval="1d",
                pa_levels=pa_levels,
                latest_candle=latest,
                prev_candle=prev
            )

            risk_pts = round(abs(current_price - sl), 2)
            risk_pct = round((risk_pts / current_price) * 100, 2)
            tp1_pts = round(abs(targets['tp1']['price'] - current_price), 2)
            tp1_pct = round((tp1_pts / current_price) * 100, 2)
            tp2_pts = round(abs(targets['tp2']['price'] - current_price), 2)
            tp2_pct = round((tp2_pts / current_price) * 100, 2)
            tp3_pts = round(abs(targets['tp3']['price'] - current_price), 2)
            tp3_pct = round((tp3_pts / current_price) * 100, 2)

            current_setups[direction] = {
                "price": current_price,
                "sl": sl,
                "sl_risk_pts": risk_pts,
                "sl_risk_pct": risk_pct,
                "sl_rationale": sl_rat,
                "tp1_price": targets['tp1']['price'],
                "tp1_pct": tp1_pct,
                "tp1_rr": targets['tp1']['risk_reward'],
                "tp2_price": targets['tp2']['price'],
                "tp2_pct": tp2_pct,
                "tp2_rr": targets['tp2']['risk_reward'],
                "tp3_price": targets['tp3']['price'],
                "tp3_pct": tp3_pct,
                "tp3_rr": targets['tp3']['risk_reward'],
                "primary_rr": rr,
                "tp_rationale": tp_rat
            }

        # 2. HISTORICAL SIMULATION ACROSS 200-250 DAYS
        # We walk through the bars (from bar 40 to end) to simulate trades taken by this engine
        simulated_trades = []
        in_trade = False
        trade_dir = None
        trade_entry = 0
        trade_sl = 0
        trade_tp1 = 0
        trade_tp2 = 0
        trade_rr1 = 0
        trade_rr2 = 0
        trade_risk_pct = 0
        entry_idx = 0

        # Step through historical bars
        for i in range(40, len(df)):
            sub_df = df.iloc[:i+1]
            curr_bar = sub_df.iloc[-1]
            c_price = float(curr_bar['Close'])
            c_high = float(curr_bar['High'])
            c_low = float(curr_bar['Low'])
            c_atr = float(curr_bar.get('ATR', c_price * 0.018))
            bar_date = sub_df.index[-1].strftime("%Y-%m-%d")

            # If currently in a trade, check exit or target reach
            if in_trade:
                hit_tp2 = (c_high >= trade_tp2) if trade_dir == "BUY" else (c_low <= trade_tp2)
                hit_tp1 = (c_high >= trade_tp1) if trade_dir == "BUY" else (c_low <= trade_tp1)
                hit_sl  = (c_low <= trade_sl)   if trade_dir == "BUY" else (c_high >= trade_sl)

                if hit_tp2:
                    simulated_trades.append({
                        "entry_date": df.index[entry_idx].strftime("%Y-%m-%d"),
                        "exit_date": bar_date,
                        "direction": trade_dir,
                        "entry_price": trade_entry,
                        "outcome": "TP2_HIT",
                        "sl_risk_pct": trade_risk_pct,
                        "target_pct": round(abs(trade_tp2 - trade_entry) / trade_entry * 100, 2),
                        "rr_achieved": trade_rr2,
                        "bars_held": i - entry_idx
                    })
                    in_trade = False
                elif hit_sl:
                    simulated_trades.append({
                        "entry_date": df.index[entry_idx].strftime("%Y-%m-%d"),
                        "exit_date": bar_date,
                        "direction": trade_dir,
                        "entry_price": trade_entry,
                        "outcome": "SL_HIT",
                        "sl_risk_pct": trade_risk_pct,
                        "target_pct": -trade_risk_pct,
                        "rr_achieved": -1.0,
                        "bars_held": i - entry_idx
                    })
                    in_trade = False
                elif (i - entry_idx) >= 20: # Time exit after 20 days
                    ret = ((c_price - trade_entry) / trade_entry * 100) if trade_dir == "BUY" else ((trade_entry - c_price) / trade_entry * 100)
                    simulated_trades.append({
                        "entry_date": df.index[entry_idx].strftime("%Y-%m-%d"),
                        "exit_date": bar_date,
                        "direction": trade_dir,
                        "entry_price": trade_entry,
                        "outcome": "TIME_EXIT",
                        "sl_risk_pct": trade_risk_pct,
                        "target_pct": round(ret, 2),
                        "rr_achieved": round(ret / max(0.01, trade_risk_pct), 2),
                        "bars_held": i - entry_idx
                    })
                    in_trade = False
                continue

            # If not in trade, check for signal
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

                risk_val = abs(c_price - sl_calc)
                r_pct = round(risk_val / c_price * 100, 2)

                # Enter trade
                in_trade = True
                trade_dir = sig_dir
                trade_entry = c_price
                trade_sl = sl_calc
                trade_tp1 = tgts_calc['tp1']['price']
                trade_tp2 = tgts_calc['tp2']['price']
                trade_rr1 = tgts_calc['tp1']['risk_reward']
                trade_rr2 = tgts_calc['tp2']['risk_reward']
                trade_risk_pct = r_pct
                entry_idx = i

        # Calculate historical summary for this stock
        if len(simulated_trades) > 0:
            avg_sl_pct = round(np.mean([t['sl_risk_pct'] for t in simulated_trades]), 2)
            wins = [t for t in simulated_trades if t['outcome'] == 'TP2_HIT' or t['target_pct'] > 0]
            win_rate = round(len(wins) / len(simulated_trades) * 100, 2)
            avg_tgt_pct = round(np.mean([t['target_pct'] for t in wins]), 2) if wins else 0
            avg_rr_win = round(np.mean([t['rr_achieved'] for t in wins]), 2) if wins else 0
            max_rr = round(max([t['rr_achieved'] for t in simulated_trades]), 2)
        else:
            avg_sl_pct, win_rate, avg_tgt_pct, avg_rr_win, max_rr = 0, 0, 0, 0, 0

        stock_summary = {
            "symbol": sym,
            "total_bars": total_bars,
            "current_price": current_price,
            "current_buy": current_setups["BUY"],
            "current_sell": current_setups["SELL"],
            "historical_trades_count": len(simulated_trades),
            "historical_win_rate_pct": win_rate,
            "historical_avg_sl_pct": avg_sl_pct,
            "historical_avg_target_pct": avg_tgt_pct,
            "historical_avg_rr": avg_rr_win,
            "historical_max_rr": max_rr
        }

        all_stock_results.append(stock_summary)

    except Exception as e:
        print(f"Error on {sym}: {e}")

# Save full results to json
with open("test_7stocks_results.json", "w") as f:
    json.dump(all_stock_results, f, indent=2)

print("\n=== COMPLETED TEST FOR ALL 7 STOCKS ===")
for s in all_stock_results:
    cb = s['current_buy']
    print(f"[{s['symbol']}] Price: Rs.{s['current_price']} | SL: Rs.{cb['sl']} (-{cb['sl_risk_pct']}%) | TP1: Rs.{cb['tp1_price']} (+{cb['tp1_pct']}%, RR 1:{cb['tp1_rr']}) | TP2: Rs.{cb['tp2_price']} (+{cb['tp2_pct']}%, RR 1:{cb['tp2_rr']}) | TP3: Rs.{cb['tp3_price']} (+{cb['tp3_pct']}%, RR 1:{cb['tp3_rr']}) | 250d WinRate: {s['historical_win_rate_pct']}% | Avg SL: {s['historical_avg_sl_pct']}% | Avg TP: +{s['historical_avg_target_pct']}% | Avg RR: 1:{s['historical_avg_rr']}")
