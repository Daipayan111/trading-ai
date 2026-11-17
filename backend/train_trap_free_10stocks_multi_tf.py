"""
TradingAI Pro - High-Speed Multi-Timeframe Trap-Free AI Retraining & Audit Suite
================================================================================
Trained on 10 Instruments across 4 Timeframes (5m, 15m, 1h, 1d) & 4-Year Historical Data.
Vectorized execution for instant results.
"""

import sys, os
import json
import numpy as np
import pandas as pd
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytics import fetch_stock_data, add_technical_indicators
from analytics_ml import (
    train_advanced_ml_model, 
    _add_ratio_features, 
    _get_ensemble_proba,
    FEATURE_COLS
)

INSTRUMENTS = [
    'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK',
    'SBIN', 'BHARTIARTL', 'ITC', 'LT', '^NSEBANK'
]

TIMEFRAMES = [
    {"tf": "5m", "period": "60d", "name": "5-Minute Scalp/Intraday"},
    {"tf": "15m", "period": "60d", "name": "15-Minute Momentum"},
    {"tf": "1h", "period": "730d", "name": "1-Hour Swing"},
    {"tf": "1d", "period": "4y", "name": "Daily Multi-Year Macro"}
]

def train_and_audit_trap_free(symbol: str, tf: str = "5m", period: str = "60d"):
    try:
        df, used_symbol, ticker = fetch_stock_data(symbol, period=period, interval=tf)
        if df.empty or len(df) < 100:
            return {"symbol": symbol, "timeframe": tf, "error": "Insufficient bars"}
        
        # Calculate full technical indicators + new candlestick & price action features
        df = add_technical_indicators(df)
        
        # Train ML model using Triple Barrier labeling
        models, scaler, accuracy, features, wf_result, f1 = train_advanced_ml_model(df)
        
        # Prepare evaluation df
        eval_df = _add_ratio_features(df.copy().dropna())
        avail = [f for f in features if f in eval_df.columns]
        if not avail or len(eval_df) < 50:
            return {"symbol": symbol, "timeframe": tf, "error": "Features unavailable"}

        test_len = min(len(eval_df) - 10, 250)
        eval_slice = eval_df.iloc[-test_len:]
        
        # Vectorized probability generation
        X_s = scaler.transform(eval_slice[avail].values)
        probas = _get_ensemble_proba(models, X_s)
        bull_probs = probas[:, 1] * 100
        
        close_arr = eval_slice['Close'].values
        high_arr  = eval_slice['High'].values
        low_arr   = eval_slice['Low'].values
        atr_arr   = eval_slice['ATR'].fillna(eval_slice['Close'] * 0.01).values if 'ATR' in eval_slice.columns else close_arr * 0.01
        chop_arr  = eval_slice['Choppiness_Index'].values if 'Choppiness_Index' in eval_slice.columns else np.full(test_len, 50.0)
        lunch_arr = eval_slice['Is_Lunch_Chop'].values if 'Is_Lunch_Chop' in eval_slice.columns else np.zeros(test_len)
        vol_arr   = eval_slice['Volume_Ratio'].values if 'Volume_Ratio' in eval_slice.columns else np.ones(test_len)
        uwick_arr = eval_slice['Upper_Wick_Pct'].values if 'Upper_Wick_Pct' in eval_slice.columns else np.zeros(test_len)
        lwick_arr = eval_slice['Lower_Wick_Pct'].values if 'Lower_Wick_Pct' in eval_slice.columns else np.zeros(test_len)
        
        traps_shielded = 0
        total_evals = 0
        signals_generated = 0
        tp_wins = 0
        sl_losses = 0
        total_pts = 0.0
        
        # Audit with lookahead 8
        lookahead = 8
        for i in range(test_len - lookahead):
            total_evals += 1
            bp = bull_probs[i]
            c_chop = chop_arr[i]
            c_lunch = lunch_arr[i]
            c_vol = vol_arr[i]
            c_uwick = uwick_arr[i]
            c_lwick = lwick_arr[i]
            
            # Stage 1 Gatekeeper Trap Shield
            is_trap = False
            if c_chop > 61.8:
                is_trap = True
            elif c_lunch == 1 and c_vol < 1.25:
                is_trap = True
            elif c_uwick >= 0.45 and bp >= 58.0:
                is_trap = True
            elif c_lwick >= 0.45 and bp <= 42.0:
                is_trap = True
                
            if is_trap:
                traps_shielded += 1
                continue
                
            # Signal decision
            sig = None
            if bp >= 60.0:
                sig = "BUY"
            elif bp <= 40.0:
                sig = "SELL"
                
            if not sig:
                continue
                
            signals_generated += 1
            entry = close_arr[i]
            c_atr = atr_arr[i] if atr_arr[i] > 0 else entry * 0.01
            
            # Forward simulation
            f_highs = high_arr[i+1 : i+1+lookahead]
            f_lows  = low_arr[i+1 : i+1+lookahead]
            
            if sig == "BUY":
                tp_price = entry + (1.3 * c_atr)
                sl_price = entry - (0.85 * c_atr)
                win, loss = False, False
                for k in range(len(f_highs)):
                    if f_lows[k] <= sl_price:
                        loss = True
                        break
                    elif f_highs[k] >= tp_price:
                        win = True
                        break
                if win:
                    tp_wins += 1
                    total_pts += (tp_price - entry)
                elif loss:
                    sl_losses += 1
                    total_pts -= (entry - sl_price)
            else:
                tp_price = entry - (1.3 * c_atr)
                sl_price = entry + (0.85 * c_atr)
                win, loss = False, False
                for k in range(len(f_highs)):
                    if f_highs[k] >= sl_price:
                        loss = True
                        break
                    elif f_lows[k] <= tp_price:
                        win = True
                        break
                if win:
                    tp_wins += 1
                    total_pts += (entry - tp_price)
                elif loss:
                    sl_losses += 1
                    total_pts -= (sl_price - entry)
                    
        completed_trades = tp_wins + sl_losses
        win_rate = round((tp_wins / completed_trades * 100), 1) if completed_trades > 0 else 0.0
        trap_shield_rate = round((traps_shielded / total_evals * 100), 1) if total_evals > 0 else 0.0

        return {
            "symbol": used_symbol,
            "timeframe": tf,
            "period": period,
            "model_accuracy": round(accuracy * 100, 1),
            "walk_forward_acc": round(wf_result.get('accuracy', accuracy) * 100, 1),
            "features_used": len(avail),
            "traps_shielded": traps_shielded,
            "trap_shield_rate_pct": trap_shield_rate,
            "active_trades": completed_trades,
            "tp_wins": tp_wins,
            "sl_losses": sl_losses,
            "win_rate_pct": win_rate,
            "total_pts": round(total_pts, 2),
            "status": "SUCCESS"
        }
    except Exception as e:
        return {"symbol": symbol, "timeframe": tf, "error": str(e)}

def run_full_suite():
    print("=" * 80, flush=True)
    print("STARTING TRAP-FREE AI RETRAINING ACROSS 10 INSTRUMENTS & MULTIPLE TIMEFRAMES", flush=True)
    print("=" * 80, flush=True)
    
    results = []
    for tf_info in TIMEFRAMES:
        tf = tf_info["tf"]
        period = tf_info["period"]
        name = tf_info["name"]
        print(f"\nTraining & Auditing Timeframe: {name} ({tf}, Period: {period})...", flush=True)
        
        for sym in INSTRUMENTS:
            res = train_and_audit_trap_free(sym, tf=tf, period=period)
            if res.get("status") == "SUCCESS":
                print(f"  [OK] [{sym:12}] WinRate: {res['win_rate_pct']:5.1f}% | Shielded Traps: {res['traps_shielded']:3d} ({res['trap_shield_rate_pct']:4.1f}%) | ModelAcc: {res['model_accuracy']:4.1f}% | Pts: {res['total_pts']:+7.1f}", flush=True)
            else:
                print(f"  [SKIP] [{sym:12}] Notice: {res.get('error', 'Skipped')}", flush=True)
            results.append(res)
            
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audit_trap_free_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nAudit & Retraining Report successfully cached to: {out_path}", flush=True)
    print("=" * 80, flush=True)

if __name__ == '__main__':
    run_full_suite()
