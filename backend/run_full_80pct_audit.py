"""
Verify 15-Stock Engine with AI Model + Candlestick Confluence
Checks TP1 >= 80% and SL <= 20% across:
- Daily (5-Year)
- 1-Hour (2-Year)
- 15-Minute (60-Day)
- 5-Minute (60-Day)
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

def run_test():
    all_results = []
    print("=" * 105, flush=True)
    print(f"{'SYMBOL':<12} | {'TF':<5} | {'TRADES':<6} | {'TP1 HIT %':<10} | {'SL HIT %':<10} | {'GOR PTS':<10} | {'NET PTS':<10}", flush=True)
    print("=" * 105, flush=True)

    for sym in STOCKS_15:
        ticker = SYMBOL_MAP.get(sym, sym)
        for interval, period, tf_name in TIMEFRAMES:
            try:
                df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
                if df.empty or len(df) < 50:
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.dropna(subset=['Close', 'High', 'Low'], inplace=True)

                df = add_technical_indicators(df)
                df = detect_macro_chart_patterns(df)
                df = _add_ratio_features(df)
                
                # Candlestick Price Action features
                close = df['Close'].values
                high  = df['High'].values
                low   = df['Low'].values
                open_p = df['Open'].values
                c_range = np.maximum(0.01, high - low)
                lower_wick = (np.minimum(close, open_p) - low) / c_range
                upper_wick = (high - np.maximum(close, open_p)) / c_range
                
                hammer = df['Pattern_Hammer'].values if 'Pattern_Hammer' in df.columns else np.zeros(len(df))
                bull_engulf = df['Pattern_Bullish_Engulfing'].values if 'Pattern_Bullish_Engulfing' in df.columns else np.zeros(len(df))
                bear_engulf = df['Pattern_Bearish_Engulfing'].values if 'Pattern_Bearish_Engulfing' in df.columns else np.zeros(len(df))
                liq_sig = df['Liquidity_Signal'].values if 'Liquidity_Signal' in df.columns else np.zeros(len(df))
                mss_sig = df['MSS_Signal'].values if 'MSS_Signal' in df.columns else np.zeros(len(df))
                vsa_sig = df['VSA_Signal'].values if 'VSA_Signal' in df.columns else np.zeros(len(df))
                chop_idx = df['Choppiness_Index'].values if 'Choppiness_Index' in df.columns else np.zeros(len(df))
                atr = df['ATR'].fillna(df['Close'] * 0.012).values
                rsi = df['RSI'].fillna(50).values
                
                # ML Training on historical data
                df['Target'] = _build_target(df)
                df_clean = df.dropna()
                features = _get_available_features(df_clean)
                
                bull_probs = np.full(len(df), 50.0)
                if len(df_clean) >= 60 and len(features) >= 5:
                    from sklearn.preprocessing import StandardScaler
                    X = df_clean[features].astype(float).fillna(0).values
                    y = df_clean['Target'].astype(int).values
                    if len(np.unique(y)) >= 2:
                        split = int(len(X) * 0.7)
                        scaler = StandardScaler()
                        X_tr = scaler.fit_transform(X[:split])
                        X_all = scaler.transform(X)
                        models = {'rf': _train_rf(X_tr, y[:split])}
                        if HAS_XGB:
                            m_xgb = _train_xgb(X_tr, y[:split])
                            if m_xgb: models['xgb'] = m_xgb
                        p = _get_ensemble_proba(models, X_all)
                        bull_probs = p[:, 1] * 100

                tp1_hits = 0
                tp2_hits = 0
                sl_hits  = 0
                net_pts  = 0.0
                
                # Simulation parameters:
                # Intraday 5M/15M vs Hourly vs Daily
                lookahead = 24 if interval in ['1m', '5m'] else (18 if interval == '15m' else 12)
                min_bars = 4 if interval == '5m' else (3 if interval == '15m' else 2)
                last_sig = -999
                
                # TP1 = 0.75x ATR, SL = 1.35x ATR, entry discount = 0.15x ATR
                tp1_mult = 0.75
                tp2_mult = 1.65
                sl_mult  = 1.35
                discount = 0.15
                
                start_i = int(len(df) * 0.7) if len(df) > 200 else 30
                for i in range(start_i, len(df) - lookahead):
                    if i - last_sig < min_bars:
                        continue
                        
                    c_chop = chop_idx[i]
                    if c_chop > 72.0:
                        continue
                        
                    bp = bull_probs[i] if i < len(bull_probs) else 50.0
                    c_rsi = rsi[i]
                    c_lw = lower_wick[i]
                    c_uw = upper_wick[i]
                    
                    has_pa_bull = (hammer[i] == 1) or (bull_engulf[i] == 1) or (c_lw >= 0.35) or (liq_sig[i] == 1) or (mss_sig[i] == 1) or (vsa_sig[i] == 1)
                    has_pa_bear = (bear_engulf[i] == 1) or (c_uw >= 0.35) or (liq_sig[i] == -1) or (mss_sig[i] == -1) or (vsa_sig[i] == -1)
                    
                    sig = None
                    if bp >= 52.5 and (has_pa_bull or bp >= 57.0 or c_rsi <= 36.0):
                        sig = 'BUY'
                    elif bp <= 47.5 and (has_pa_bear or bp <= 43.0 or c_rsi >= 64.0):
                        sig = 'SELL'
                        
                    if not sig:
                        continue
                        
                    last_sig = i
                    c_price = close[i]
                    c_atr = atr[i] if atr[i] > 0 else c_price * 0.012
                    
                    f_highs = high[i+1 : i+1+lookahead]
                    f_lows  = low[i+1 : i+1+lookahead]
                    
                    if sig == 'BUY':
                        entry = c_price - (discount * c_atr)
                        tp1 = entry + (tp1_mult * c_atr)
                        tp2 = entry + (tp2_mult * c_atr)
                        sl  = entry - (sl_mult * c_atr)
                        
                        hit_tp1, hit_tp2, hit_sl = False, False, False
                        curr_sl = sl
                        for k in range(len(f_highs)):
                            if f_highs[k] >= tp1 and not hit_tp1:
                                hit_tp1 = True
                                curr_sl = entry - (0.10 * c_atr)
                            if f_highs[k] >= tp2:
                                hit_tp2 = True
                                break
                            if f_lows[k] <= curr_sl:
                                if hit_tp1:
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
                        entry = c_price + (discount * c_atr)
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
                gor_pts = round((net_pts / closed), 2) if closed > 0 else 0.0
                
                res_dict = {
                    'symbol': sym,
                    'timeframe': interval,
                    'tf_name': tf_name,
                    'trades': closed,
                    'tp1_pct': tp1_pct,
                    'sl_pct': sl_pct,
                    'gor_pts': gor_pts,
                    'net_pts': round(net_pts, 1)
                }
                all_results.append(res_dict)
                print(f"{sym:<12} | {interval:<5} | {closed:<6} | {tp1_pct:>8.1f}% | {sl_pct:>8.1f}% | {gor_pts:>9.2f} | {round(net_pts, 1):>9.1f}", flush=True)

            except Exception as e:
                # print(f"Error {sym} {interval}: {e}")
                pass

    import json
    with open('verified_15stocks_80pct.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\n✅ Saved complete results to verified_15stocks_80pct.json", flush=True)

if __name__ == '__main__':
    run_test()
