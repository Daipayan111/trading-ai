"""
Ultra-Precision Multi-Timeframe AI Engine (Target 1 Hit >= 80%, SL <= 20%)
Tests across 15 Stocks x 4 Timeframes (1D, 1H, 15M, 5M)
"""

import sys, os
import numpy as np
import pandas as pd
import yfinance as yf

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytics import add_technical_indicators, detect_macro_chart_patterns
from analytics_ml import _add_ratio_features, _build_target, _get_available_features, _train_rf, _train_xgb, _get_ensemble_proba, HAS_XGB

STOCKS_15 = [
    'NIFTY50', 'BANKNIFTY', 'RELIANCE', 'TCS', 'INFY',
    'HDFCBANK', 'ICICIBANK', 'SBIN', 'BHARTIARTL', 'ITC',
    'KOTAKBANK', 'LT', 'AXISBANK', 'MARUTI', 'BAJFINANCE'
]

SYMBOL_MAP = {
    'NIFTY50': '^NSEI',
    'BANKNIFTY': '^NSEBANK',
    'RELIANCE': 'RELIANCE.NS',
    'TCS': 'TCS.NS',
    'INFY': 'INFY.NS',
    'HDFCBANK': 'HDFCBANK.NS',
    'ICICIBANK': 'ICICIBANK.NS',
    'SBIN': 'SBIN.NS',
    'BHARTIARTL': 'BHARTIARTL.NS',
    'ITC': 'ITC.NS',
    'KOTAKBANK': 'KOTAKBANK.NS',
    'LT': 'LT.NS',
    'AXISBANK': 'AXISBANK.NS',
    'MARUTI': 'MARUTI.NS',
    'BAJFINANCE': 'BAJFINANCE.NS',
}

TIMEFRAMES = [
    ('1d', '5y', 'Daily (5Y)'),
    ('1h', '730d', '1-Hour (2Y)'),
    ('15m', '60d', '15-Min (60D)'),
    ('5m', '60d', '5-Min (60D)')
]

def evaluate_symbol_tf(sym, interval, period):
    ticker = SYMBOL_MAP.get(sym, sym)
    # Fetch 5 years or appropriate period
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty or len(df) < 80:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(subset=['Close', 'High', 'Low'], inplace=True)

    df = add_technical_indicators(df)
    df = detect_macro_chart_patterns(df)
    df = _add_ratio_features(df)
    
    # Train ML Model on this dataset
    df['Target'] = _build_target(df)
    df_clean = df.dropna()
    features = _get_available_features(df_clean)
    if len(df_clean) < 80 or len(features) < 5:
        return None
        
    X = df_clean[features].astype(float).fillna(0).values
    y = df_clean['Target'].astype(int).values
    
    # Train / Test Split
    train_size = int(len(X) * 0.70)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    if len(np.unique(y_train)) < 2:
        return None
        
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    models = {'rf': _train_rf(X_train_s, y_train)}
    if HAS_XGB:
        xgb_m = _train_xgb(X_train_s, y_train)
        if xgb_m: models['xgb'] = xgb_m
        
    test_df = df_clean.iloc[train_size:].copy()
    probs = _get_ensemble_proba(models, X_test_s)
    bull_probs = probs[:, 1] * 100
    
    close_arr = test_df['Close'].values
    high_arr  = test_df['High'].values
    low_arr   = test_df['Low'].values
    open_arr  = test_df['Open'].values
    atr_arr   = test_df['ATR'].values if 'ATR' in test_df.columns else close_arr * 0.015
    chop_arr  = test_df['Choppiness_Index'].values if 'Choppiness_Index' in test_df.columns else np.zeros(len(test_df))
    rsi_arr   = test_df['RSI'].values if 'RSI' in test_df.columns else np.full(len(test_df), 50.0)
    
    # Price Action wicks & candles
    candle_ranges = np.maximum(0.01, high_arr - low_arr)
    lower_wicks   = (np.minimum(close_arr, open_arr) - low_arr) / candle_ranges
    upper_wicks   = (high_arr - np.maximum(close_arr, open_arr)) / candle_ranges
    
    hammer_arr = test_df['Pattern_Hammer'].values if 'Pattern_Hammer' in test_df.columns else np.zeros(len(test_df))
    engulf_bull = test_df['Pattern_Bullish_Engulfing'].values if 'Pattern_Bullish_Engulfing' in test_df.columns else np.zeros(len(test_df))
    engulf_bear = test_df['Pattern_Bearish_Engulfing'].values if 'Pattern_Bearish_Engulfing' in test_df.columns else np.zeros(len(test_df))
    liq_arr = test_df['Liquidity_Signal'].values if 'Liquidity_Signal' in test_df.columns else np.zeros(len(test_df))
    mss_arr = test_df['MSS_Signal'].values if 'MSS_Signal' in test_df.columns else np.zeros(len(test_df))
    
    tp1_hits = 0
    tp2_hits = 0
    sl_hits = 0
    net_pts = 0.0
    
    lookahead = 20 if interval in ['1m', '5m'] else 15
    last_trade_idx = -999
    min_spacing = 4 if interval == '5m' else (3 if interval == '15m' else 2)
    
    # Target calibration:
    # TP1 (Safe Harvest): 0.75x ATR
    # TP2 (Runner): 1.60x ATR
    # SL (Structural cushion): 1.35x ATR
    # Limit entry discount: 0.15x ATR
    tp1_mult = 0.75
    tp2_mult = 1.60
    sl_mult  = 1.35
    discount_mult = 0.15
    
    for i in range(len(test_df) - lookahead):
        if i - last_trade_idx < min_spacing:
            continue
            
        bp = bull_probs[i]
        c_chop = chop_arr[i]
        c_rsi = rsi_arr[i]
        c_lw = lower_wicks[i]
        c_uw = upper_wicks[i]
        
        # Extreme chop filter
        if c_chop > 70.0:
            continue
            
        sig = None
        # Confluence of ML Probability + Candlestick / Indicator trigger
        has_bull_candle = (hammer_arr[i] == 1) or (engulf_bull[i] == 1) or (c_lw >= 0.35) or (liq_arr[i] == 1) or (mss_arr[i] == 1)
        has_bear_candle = (engulf_bear[i] == 1) or (c_uw >= 0.35) or (liq_arr[i] == -1) or (mss_arr[i] == -1)
        
        if bp >= 53.0 and (has_bull_candle or bp >= 58.0 or c_rsi <= 36.0):
            sig = 'BUY'
        elif bp <= 47.0 and (has_bear_candle or bp <= 42.0 or c_rsi >= 64.0):
            sig = 'SELL'
            
        if not sig:
            continue
            
        last_trade_idx = i
        c = close_arr[i]
        c_atr = atr_arr[i] if atr_arr[i] > 0 else c * 0.015
        
        f_highs = high_arr[i+1 : i+1+lookahead]
        f_lows  = low_arr[i+1 : i+1+lookahead]
        
        if sig == 'BUY':
            entry = c - (discount_mult * c_atr)
            tp1 = entry + (tp1_mult * c_atr)
            tp2 = entry + (tp2_mult * c_atr)
            sl  = entry - (sl_mult * c_atr)
            
            hit_tp1, hit_tp2, hit_sl = False, False, False
            curr_sl = sl
            
            for k in range(len(f_highs)):
                if f_highs[k] >= tp1 and not hit_tp1:
                    hit_tp1 = True
                    curr_sl = entry - (0.10 * c_atr) # Trailing Breakeven cushion
                    
                if f_highs[k] >= tp2:
                    hit_tp2 = True
                    break
                if f_lows[k] <= curr_sl:
                    if hit_tp1:
                        # TP1 was booked before SL
                        break
                    else:
                        hit_sl = True
                        break
                        
            if hit_tp1:
                tp1_hits += 1
                if hit_tp2:
                    tp2_hits += 1
                    net_pts += (tp2 - entry)
                else:
                    net_pts += (tp1 - entry)
            elif hit_sl:
                sl_hits += 1
                net_pts -= (entry - sl)
                
        else: # SELL
            entry = c + (discount_mult * c_atr)
            tp1 = entry - (tp1_mult * c_atr)
            tp2 = entry - (tp2_mult * c_atr)
            sl  = entry + (sl_mult * c_atr)
            
            hit_tp1, hit_tp2, hit_sl = False, False, False
            curr_sl = sl
            
            for k in range(len(f_highs)):
                if f_lows[k] <= tp1 and not hit_tp1:
                    hit_tp1 = True
                    curr_sl = entry + (0.10 * c_atr)
                    
                if f_lows[k] <= tp2:
                    hit_tp2 = True
                    break
                if f_highs[k] >= curr_sl:
                    if hit_tp1:
                        break
                    else:
                        hit_sl = True
                        break
                        
            if hit_tp1:
                tp1_hits += 1
                if hit_tp2:
                    tp2_hits += 1
                    net_pts += (entry - tp2)
                else:
                    net_pts += (entry - tp1)
            elif hit_sl:
                sl_hits += 1
                net_pts -= (sl - entry)

    closed = tp1_hits + sl_hits
    tp1_pct = round((tp1_hits / closed * 100), 1) if closed > 0 else 0.0
    sl_pct  = round((sl_hits / closed * 100), 1) if closed > 0 else 0.0
    tp2_pct = round((tp2_hits / closed * 100), 1) if closed > 0 else 0.0
    gor_pts = round((net_pts / closed), 2) if closed > 0 else 0.0
    
    return {
        'symbol': sym,
        'interval': interval,
        'period': period,
        'trades': closed,
        'tp1_pct': tp1_pct,
        'tp2_pct': tp2_pct,
        'sl_pct': sl_pct,
        'gor_pts': gor_pts,
        'net_pts': round(net_pts, 1)
    }

if __name__ == '__main__':
    test_symbols = ['NIFTY50', 'BANKNIFTY', 'RELIANCE', 'TCS', 'SBIN']
    print(f"{'SYM':<10} | {'TF':<5} | {'TRADES':<6} | {'TP1 HIT %':<10} | {'SL HIT %':<10} | {'GOR PTS':<10} | {'NET PTS':<10}")
    print("-" * 80)
    for sym in test_symbols:
        for tf, per, name in TIMEFRAMES:
            res = evaluate_symbol_tf(sym, tf, per)
            if res:
                print(f"{res['symbol']:<10} | {res['interval']:<5} | {res['trades']:<6} | {res['tp1_pct']:>8.1f}% | {res['sl_pct']:>8.1f}% | {res['gor_pts']:>9.2f} | {res['net_pts']:>9.1f}")
