"""
TradingAI Pro - 15 Stocks Comprehensive Multi-Timeframe Performance Audit
==========================================================================
Evaluates 15 Top Instruments across all Timeframes (5m, 15m, 1h, 5-Year 1d)
using the newly integrated Meta-Labeling Guard, Triple-Barrier, and Wick Shield Engine.
"""

import sys, os
import json
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytics import fetch_stock_data, add_technical_indicators
from analytics_ml import (
    train_advanced_ml_model, 
    _add_ratio_features, 
    _get_ensemble_proba
)

STOCKS_15 = [
    'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK',
    'SBIN', 'BHARTIARTL', 'ITC', 'LT', 'KOTAKBANK',
    'AXISBANK', 'TATAMOTORS', 'MARUTI', 'BAJFINANCE', '^NSEBANK'
]

TIMEFRAMES = [
    {"tf": "1d", "period": "5y", "name": "5-Year Daily Swing"},
    {"tf": "1h", "period": "730d", "name": "1-Hour Swing"},
    {"tf": "15m", "period": "60d", "name": "15-Minute Momentum"},
    {"tf": "5m", "period": "60d", "name": "5-Minute Scalp"}
]

def audit_single_stock_tf(symbol: str, tf: str = "1d", period: str = "5y"):
    try:
        df, used_symbol, ticker = fetch_stock_data(symbol, period=period, interval=tf)
        if df.empty or len(df) < 80:
            return {"symbol": symbol, "timeframe": tf, "error": "Insufficient data"}
            
        df = add_technical_indicators(df)
        models, scaler, accuracy, features, wf_result, f1 = train_advanced_ml_model(df)
        
        eval_df = _add_ratio_features(df.copy().dropna())
        avail = [f for f in features if f in eval_df.columns]
        if not avail or len(eval_df) < 50:
            return {"symbol": symbol, "timeframe": tf, "error": "Features unavailable"}

        test_len = min(len(eval_df) - 10, 350)
        eval_slice = eval_df.iloc[-test_len:]
        
        X_s = scaler.transform(eval_slice[avail].values)
        probas = _get_ensemble_proba(models, X_s)
        bull_probs = probas[:, 1] * 100
        
        close_arr = eval_slice['Close'].values
        high_arr  = eval_slice['High'].values
        low_arr   = eval_slice['Low'].values
        atr_arr   = eval_slice['ATR'].fillna(eval_slice['Close'] * 0.01).values if 'ATR' in eval_slice.columns else close_arr * 0.01
        chop_arr  = eval_slice['Choppiness_Index'].values if 'Choppiness_Index' in eval_slice.columns else np.full(test_len, 50.0)
        rsi_arr   = eval_slice['RSI'].values if 'RSI' in eval_slice.columns else np.full(test_len, 50.0)
        uwick_arr = eval_slice['Upper_Wick_Pct'].values if 'Upper_Wick_Pct' in eval_slice.columns else np.zeros(test_len)
        lwick_arr = eval_slice['Lower_Wick_Pct'].values if 'Lower_Wick_Pct' in eval_slice.columns else np.zeros(test_len)
        
        traps_shielded = 0
        total_evals = 0
        signals_generated = 0
        tp_wins = 0
        sl_losses = 0
        total_pts = 0.0
        
        lookahead = 10
        for i in range(test_len - lookahead):
            total_evals += 1
            bp = bull_probs[i]
            c_chop = chop_arr[i]
            c_rsi = rsi_arr[i]
            c_uwick = uwick_arr[i]
            c_lwick = lwick_arr[i]
            
            # Sharp Gatekeeper Trap Shield (< 15% rejection)
            is_trap = False
            if c_chop > 72.0:
                is_trap = True
            elif c_uwick >= 0.58 and bp >= 60.0:
                is_trap = True
            elif c_lwick >= 0.58 and bp <= 40.0:
                is_trap = True
                
            if is_trap:
                traps_shielded += 1
                continue
                
            # Dual-Mode Decision: Trend Momentum vs Range Mean Reversion
            sig = None
            if c_chop <= 58.0:
                if bp >= 54.0: sig = "BUY"
                elif bp <= 46.0: sig = "SELL"
            else:
                if c_rsi <= 38.0 or bp >= 56.0: sig = "BUY"
                elif c_rsi >= 62.0 or bp <= 44.0: sig = "SELL"
                
            if not sig:
                continue
                
            signals_generated += 1
            entry = close_arr[i]
            c_atr = atr_arr[i] if atr_arr[i] > 0 else entry * 0.01
            
            f_highs = high_arr[i+1 : i+1+lookahead]
            f_lows  = low_arr[i+1 : i+1+lookahead]
            
            if sig == "BUY":
                effective_entry = entry - (0.15 * c_atr)
                tp_price = effective_entry + (0.75 * c_atr)
                sl_price = effective_entry - (1.25 * c_atr)
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
                    total_pts += (tp_price - effective_entry)
                elif loss:
                    sl_losses += 1
                    total_pts -= (effective_entry - sl_price)
            else:
                effective_entry = entry + (0.15 * c_atr)
                tp_price = effective_entry - (0.75 * c_atr)
                sl_price = effective_entry + (1.25 * c_atr)
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
                    total_pts += (effective_entry - tp_price)
                elif loss:
                    sl_losses += 1
                    total_pts -= (sl_price - effective_entry)
                    
        completed_trades = tp_wins + sl_losses
        win_rate = round((tp_wins / completed_trades * 100), 1) if completed_trades > 0 else 0.0
        trap_shield_rate = round((traps_shielded / total_evals * 100), 1) if total_evals > 0 else 0.0

        return {
            "symbol": used_symbol,
            "timeframe": tf,
            "period": period,
            "total_bars": len(df),
            "model_accuracy": round(accuracy * 100, 1),
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

def run_15_audit():
    print("=" * 90, flush=True)
    print("🚀 RUNNING 15 STOCKS COMPREHENSIVE MULTI-TIMEFRAME PERFORMANCE AUDIT", flush=True)
    print("=" * 90, flush=True)
    
    full_data = {}
    for tf_info in TIMEFRAMES:
        tf = tf_info["tf"]
        period = tf_info["period"]
        name = tf_info["name"]
        print(f"\n⚡ Auditing Timeframe: {name} ({tf}, Period: {period})...", flush=True)
        tf_results = []
        
        for sym in STOCKS_15:
            res = audit_single_stock_tf(sym, tf=tf, period=period)
            if res.get("status") == "SUCCESS":
                print(f"  ✓ [{sym:12}] WinRate: {res['win_rate_pct']:5.1f}% | Shield: {res['trap_shield_rate_pct']:4.1f}% | Trades: {res['active_trades']:3d} | Pts: {res['total_pts']:+8.1f} | Acc: {res['model_accuracy']:4.1f}%", flush=True)
            else:
                print(f"  ✗ [{sym:12}] Notice: {res.get('error', 'Skipped')}", flush=True)
            tf_results.append(res)
            
        full_data[tf] = tf_results
        
    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audit_15stocks_comprehensive.json')
    with open(out_file, 'w') as f:
        json.dump(full_data, f, indent=2)
    print(f"\n✅ 15-Stock Complete Audit Successfully Saved To: {out_file}", flush=True)
    print("=" * 90, flush=True)

if __name__ == '__main__':
    run_15_audit()
