import json
import pandas as pd
from analytics import fetch_stock_data, add_technical_indicators
from analytics_sr import find_ultra_strong_sr_zones
from analytics_micro_breakout import detect_micro_sr_and_breakouts
from analytics_trap import detect_institutional_traps
from routers.predict import calc_advanced_trap_sl_target, find_strong_sr_levels

symbols = ["TCS.NS", "RELIANCE.NS", "TATAMOTORS.NS", "INFY.NS", "HDFCBANK.NS"]
results = []

for sym in symbols:
    try:
        df, used_sym, _ = fetch_stock_data(sym, period="60d", interval="15m")
        df = add_technical_indicators(df)
        if len(df) < 20:
            continue

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        current_price = round(float(latest['Close']), 2)
        atr = float(latest.get('ATR', current_price * 0.015))

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

        # Test both BUY setup and SELL setup metrics
        for direction in ["BUY", "SELL"]:
            sl, tp, rr, targets, sl_rat, tp_rat = calc_advanced_trap_sl_target(
                current_price=current_price,
                direction=direction,
                atr=atr,
                sr_levels=sr_levels,
                ultra_sr=ultra_sr,
                micro_breakout=micro_bo,
                trap_data=traps,
                conv_score=80,
                interval="15m",
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

            results.append({
                "symbol": sym,
                "direction": direction,
                "price": current_price,
                "atr": round(atr, 2),
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
            })
    except Exception as e:
        print(f"Error on {sym}: {e}")

with open("test_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("COMPLETED_METRICS_COUNT:", len(results))
for r in results:
    print(f"[{r['symbol']} - {r['direction']}] Price: Rs.{r['price']} | SL: Rs.{r['sl']} (-{r['sl_risk_pct']}%) | TP1: Rs.{r['tp1_price']} (+{r['tp1_pct']}%, RR 1:{r['tp1_rr']}) | TP2: Rs.{r['tp2_price']} (+{r['tp2_pct']}%, RR 1:{r['tp2_rr']}) | TP3: Rs.{r['tp3_price']} (+{r['tp3_pct']}%, RR 1:{r['tp3_rr']}) | SL Rationale: {r['sl_rationale']}")
