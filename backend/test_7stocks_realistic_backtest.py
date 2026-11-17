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

print("=== STARTING 250-DAY REALISTIC MULTI-TIER BACKTEST ===")

for sym in symbols:
    try:
        df, used_sym, _ = fetch_stock_data(sym, period="1y", interval="1d")
        df = add_technical_indicators(df)
        signals = generate_signals(df)

        total_bars = len(df)
        if total_bars < 50:
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

        # 1. CURRENT LIVE DAILY SETUP
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

        # 2. MULTI-TIER REALISTIC BACKTEST ACROSS 200-250 BARS
        simulated_trades = []
        in_trade = False
        trade_dir = None
        trade_entry = 0
        trade_sl = 0
        trade_tp1 = 0
        trade_tp2 = 0
        trade_risk_pct = 0
        tp1_hit = False
        entry_idx = 0

        for i in range(40, len(df)):
            sub_df = df.iloc[:i+1]
            curr_bar = sub_df.iloc[-1]
            c_price = float(curr_bar['Close'])
            c_high = float(curr_bar['High'])
            c_low = float(curr_bar['Low'])
            c_atr = float(curr_bar.get('ATR', c_price * 0.018))
            bar_date = sub_df.index[-1].strftime("%Y-%m-%d")

            if in_trade:
                # Check TP1 first
                if not tp1_hit:
                    if (trade_dir == "BUY" and c_high >= trade_tp1) or (trade_dir == "SELL" and c_low <= trade_tp1):
                        tp1_hit = True
                        # Move SL to Entry Cost! (Risk Free)
                        trade_sl = trade_entry

                # Check TP2
                if (trade_dir == "BUY" and c_high >= trade_tp2) or (trade_dir == "SELL" and c_low <= trade_tp2):
                    tp2_pct = round(abs(trade_tp2 - trade_entry) / trade_entry * 100, 2)
                    tp1_pct = round(abs(trade_tp1 - trade_entry) / trade_entry * 100, 2)
                    avg_realized_pct = round((0.5 * tp1_pct) + (0.5 * tp2_pct), 2)
                    simulated_trades.append({
                        "entry_date": df.index[entry_idx].strftime("%Y-%m-%d"),
                        "exit_date": bar_date,
                        "direction": trade_dir,
                        "entry_price": trade_entry,
                        "outcome": "TP2_FULL_TARGET",
                        "sl_risk_pct": trade_risk_pct,
                        "profit_pct": avg_realized_pct,
                        "rr_achieved": round(avg_realized_pct / max(0.01, trade_risk_pct), 2),
                        "bars_held": i - entry_idx
                    })
                    in_trade = False
                    continue

                # Check SL (either original SL or Breakeven SL if TP1 was already hit)
                if (trade_dir == "BUY" and c_low <= trade_sl) or (trade_dir == "SELL" and c_high >= trade_sl):
                    if tp1_hit:
                        # 50% was booked at TP1, remaining 50% exited at breakeven!
                        tp1_pct = round(abs(trade_tp1 - trade_entry) / trade_entry * 100, 2)
                        net_profit = round(0.5 * tp1_pct, 2)
                        simulated_trades.append({
                            "entry_date": df.index[entry_idx].strftime("%Y-%m-%d"),
                            "exit_date": bar_date,
                            "direction": trade_dir,
                            "entry_price": trade_entry,
                            "outcome": "TP1_BOOKED_BE_EXIT",
                            "sl_risk_pct": trade_risk_pct,
                            "profit_pct": net_profit,
                            "rr_achieved": round(net_profit / max(0.01, trade_risk_pct), 2),
                            "bars_held": i - entry_idx
                        })
                    else:
                        # Pure loss before TP1
                        simulated_trades.append({
                            "entry_date": df.index[entry_idx].strftime("%Y-%m-%d"),
                            "exit_date": bar_date,
                            "direction": trade_dir,
                            "entry_price": trade_entry,
                            "outcome": "SL_HIT",
                            "sl_risk_pct": trade_risk_pct,
                            "profit_pct": -trade_risk_pct,
                            "rr_achieved": -1.0,
                            "bars_held": i - entry_idx
                        })
                    in_trade = False
                    continue

                # Time exit after 25 trading days
                if (i - entry_idx) >= 25:
                    cur_p_pct = ((c_price - trade_entry) / trade_entry * 100) if trade_dir == "BUY" else ((trade_entry - c_price) / trade_entry * 100)
                    if tp1_hit:
                        tp1_pct = round(abs(trade_tp1 - trade_entry) / trade_entry * 100, 2)
                        final_pct = round((0.5 * tp1_pct) + (0.5 * max(0, cur_p_pct)), 2)
                    else:
                        final_pct = round(cur_p_pct, 2)
                    simulated_trades.append({
                        "entry_date": df.index[entry_idx].strftime("%Y-%m-%d"),
                        "exit_date": bar_date,
                        "direction": trade_dir,
                        "entry_price": trade_entry,
                        "outcome": "TIME_EXIT",
                        "sl_risk_pct": trade_risk_pct,
                        "profit_pct": final_pct,
                        "rr_achieved": round(final_pct / max(0.01, trade_risk_pct), 2),
                        "bars_held": i - entry_idx
                    })
                    in_trade = False
                    continue

            # Signal Generation
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

                in_trade = True
                trade_dir = sig_dir
                trade_entry = c_price
                trade_sl = sl_calc
                trade_tp1 = tgts_calc['tp1']['price']
                trade_tp2 = tgts_calc['tp2']['price']
                trade_risk_pct = r_pct
                tp1_hit = False
                entry_idx = i

        total_trades = len(simulated_trades)
        wins = [t for t in simulated_trades if t['profit_pct'] > 0]
        losses = [t for t in simulated_trades if t['profit_pct'] <= 0]
        win_rate = round(len(wins) / max(1, total_trades) * 100, 2)

        avg_sl = round(np.mean([t['sl_risk_pct'] for t in simulated_trades]), 2) if total_trades else 0
        avg_win_profit = round(np.mean([t['profit_pct'] for t in wins]), 2) if wins else 0
        avg_loss = round(np.mean([abs(t['profit_pct']) for t in losses]), 2) if losses else 0
        avg_rr = round(np.mean([t['rr_achieved'] for t in wins]), 2) if wins else 0
        max_rr = round(max([t['rr_achieved'] for t in simulated_trades]), 2) if total_trades else 0

        # Profit Factor
        gross_profit = sum([t['profit_pct'] for t in wins]) if wins else 0
        gross_loss = sum([abs(t['profit_pct']) for t in losses]) if losses else 0.01
        profit_factor = round(gross_profit / max(0.01, gross_loss), 2)

        all_stock_results.append({
            "symbol": sym,
            "total_bars": total_bars,
            "current_price": current_price,
            "current_buy": current_setups["BUY"],
            "current_sell": current_setups["SELL"],
            "total_trades": total_trades,
            "win_rate_pct": win_rate,
            "profit_factor": profit_factor,
            "avg_sl_risk_pct": avg_sl,
            "avg_target_profit_pct": avg_win_profit,
            "avg_rr_ratio": avg_rr,
            "max_rr_ratio": max_rr
        })

    except Exception as e:
        print(f"Error on {sym}: {e}")

with open("test_7stocks_realistic.json", "w") as f:
    json.dump(all_stock_results, f, indent=2)

print("\n=== SUMMARY OF REALISTIC 250-DAY MULTI-TIER BACKTEST ===")
for r in all_stock_results:
    cb = r['current_buy']
    print(f"[{r['symbol']}] Price: Rs.{r['current_price']} | Cur SL: Rs.{cb['sl']} (-{cb['sl_risk_pct']}%) | Cur TP1: Rs.{cb['tp1_price']} (+{cb['tp1_pct']}%, RR 1:{cb['tp1_rr']}) | Cur TP2: Rs.{cb['tp2_price']} (+{cb['tp2_pct']}%, RR 1:{cb['tp2_rr']}) | 250d WinRate: {r['win_rate_pct']}% | Profit Factor: {r['profit_factor']} | 250d Avg SL: -{r['avg_sl_risk_pct']}% | 250d Avg Profit: +{r['avg_target_profit_pct']}% | Avg RR: 1:{r['avg_rr_ratio']}")
