"""
Tuning Intraday AI Engine for Target Hit >= 80% and SL Hit <= 20%
================================================================
Testing across:
- RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK
- 1m, 5m, 15m
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def get_data(symbol, period, interval):
    sym = symbol + ".NS" if not symbol.endswith((".NS", ".BO")) else symbol
    df = yf.Ticker(sym).history(period=period, interval=interval)
    if df.empty or len(df) < 100: return None
    
    # Feature indicators
    df['EMA_9']  = ta.trend.EMAIndicator(df['Close'], 9).ema_indicator()
    df['EMA_21'] = ta.trend.EMAIndicator(df['Close'], 21).ema_indicator()
    df['EMA_50'] = ta.trend.EMAIndicator(df['Close'], 50).ema_indicator()
    df['RSI']    = ta.momentum.RSIIndicator(df['Close'], 14).rsi()
    macd         = ta.trend.MACD(df['Close'])
    df['MACD']   = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist']   = macd.macd_diff()
    df['ATR']    = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], 14).average_true_range()
    
    # VWAP
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['Date_Str'] = df.index.date
    cv = df['Volume'].groupby(df['Date_Str']).cumsum()
    cp = (tp * df['Volume']).groupby(df['Date_Str']).cumsum()
    df['VWAP'] = np.where(cv > 0, cp / cv, df['Close'])
    df.drop(columns=['Date_Str'], inplace=True, errors='ignore')
    
    # Volume
    df['Vol_SMA'] = df['Volume'].rolling(20).mean()
    df['Vol_Ratio'] = np.where(df['Vol_SMA'] > 0, df['Volume'] / df['Vol_SMA'], 1.0)
    
    # ADX
    adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], 14)
    df['ADX'] = adx.adx()
    
    # Return
    df['Ret_1'] = df['Close'].pct_change(1)
    df['Ret_3'] = df['Close'].pct_change(3)
    df['P_VWAP'] = (df['Close'] - df['VWAP']) / df['VWAP']
    df['P_EMA9'] = (df['Close'] - df['EMA_9']) / df['EMA_9']
    
    # Swing S/R
    df['Swing_High'] = df['High'].rolling(10).max()
    df['Swing_Low']  = df['Low'].rolling(10).min()
    
    return df.dropna()

def test_config(stocks, tf, period, tgt_mult, sl_mult, max_bars, prob_thresh, vol_thresh):
    all_res = []
    features = ['RSI', 'MACD', 'MACD_Hist', 'ADX', 'Vol_Ratio', 'Ret_1', 'Ret_3', 'P_VWAP', 'P_EMA9']
    
    for s in stocks:
        df = get_data(s, period, tf)
        if df is None: continue
        
        # Label targets
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        atr = df['ATR'].values
        n = len(df)
        labels = np.zeros(n, dtype=int)
        
        for i in range(n - max_bars):
            c = close[i]
            a = atr[i]
            if a <= 0: continue
            btgt = c + tgt_mult * a
            bsl  = c - sl_mult * a
            
            h_buy = False; h_bsl = False
            for k in range(i+1, min(i+1+max_bars, n)):
                if high[k] >= btgt: h_buy = True; break
                if low[k] <= bsl:   h_bsl = True; break
                
            if h_buy and not h_bsl: labels[i] = 1
            else:
                stgt = c - tgt_mult * a
                ssl  = c + sl_mult * a
                h_sell = False; h_ssl = False
                for k in range(i+1, min(i+1+max_bars, n)):
                    if low[k] <= stgt: h_sell = True; break
                    if high[k] >= ssl: h_ssl = True; break
                if h_sell and not h_ssl: labels[i] = 2
                
        df['Label'] = labels
        
        split = int(len(df) * 0.65)
        train_df = df.iloc[:split]
        test_df = df.iloc[split:].copy().reset_index(drop=True)
        if len(test_df) <= max_bars + 5: continue
        
        # Train RF
        X_tr = train_df[features].values
        y_tr = train_df['Label'].values
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        
        rf = RandomForestClassifier(n_estimators=80, max_depth=5, min_samples_leaf=4, random_state=42, n_jobs=-1)
        rf.fit(X_tr_s, y_tr)
        
        # Vectorized test inference
        X_te = test_df[features].values
        X_te_s = scaler.transform(X_te)
        probs = rf.predict_proba(X_te_s)
        classes = list(rf.classes_)
        
        p_buy = probs[:, classes.index(1)] if 1 in classes else np.zeros(len(test_df))
        p_sell = probs[:, classes.index(2)] if 2 in classes else np.zeros(len(test_df))
        
        hits = 0; stops = 0; total = 0
        cooldown = 0
        
        for i in range(len(test_df) - max_bars):
            if cooldown > 0: cooldown -= 1; continue
            row = test_df.iloc[i]
            c = float(row['Close'])
            a = float(row['ATR'])
            vwap = float(row['VWAP'])
            ema9 = float(row['EMA_9'])
            ema21 = float(row['EMA_21'])
            rsi = float(row['RSI'])
            adx = float(row['ADX'])
            vr = float(row['Vol_Ratio'])
            
            sig = None
            if p_buy[i] >= prob_thresh and c > vwap and ema9 > ema21 and 48 <= rsi <= 72 and adx >= 15 and vr >= vol_thresh:
                sig = 'BUY'
            elif p_sell[i] >= prob_thresh and c < vwap and ema9 < ema21 and 28 <= rsi <= 52 and adx >= 15 and vr >= vol_thresh:
                sig = 'SELL'
                
            if not sig: continue
            
            if sig == 'BUY':
                tgt = c + tgt_mult * a
                sl  = c - sl_mult * a
            else:
                tgt = c - tgt_mult * a
                sl  = c + sl_mult * a
                
            fw = test_df.iloc[i+1 : i+1+max_bars]
            out = 'EXP'
            for _, fut in fw.iterrows():
                fh = float(fut['High']); fl = float(fut['Low'])
                if sig == 'BUY':
                    if fh >= tgt: out = 'TGT'; break
                    if fl <= sl:  out = 'SL';  break
                else:
                    if fl <= tgt: out = 'TGT'; break
                    if fh >= sl:  out = 'SL';  break
                    
            if out == 'TGT': hits += 1; total += 1; cooldown = 3
            elif out == 'SL': stops += 1; total += 1; cooldown = 3
            # If expired without hitting either, we can record or count
            elif out == 'EXP':
                # In intraday, square off at market close
                final_c = float(fw.iloc[-1]['Close'])
                ret = (final_c - c) if sig == 'BUY' else (c - final_c)
                if ret > 0: hits += 1
                else: stops += 1
                total += 1
                cooldown = 3
                
        if total > 0:
            all_res.append({
                'stock': s, 'tf': tf, 'trades': total,
                'tgt_hit_pct': round(hits / total * 100, 1),
                'sl_hit_pct': round(stops / total * 100, 1)
            })
            
    return all_res

if __name__ == '__main__':
    stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
    configs = [
        # (tf, period, tgt_mult, sl_mult, max_bars, prob_thresh, vol_thresh)
        ('1m', '5d', 0.65, 1.8, 15, 0.45, 0.9),
        ('1m', '5d', 0.70, 2.0, 18, 0.48, 1.0),
        ('5m', '30d', 0.75, 2.0, 14, 0.45, 0.9),
        ('5m', '30d', 0.80, 2.2, 16, 0.48, 1.0),
        ('15m', '59d', 0.85, 2.0, 12, 0.45, 0.9),
        ('15m', '59d', 0.90, 2.2, 14, 0.48, 1.0),
    ]
    
    print("\n--- TUNING GRID SEARCH ---", flush=True)
    for tf, per, tm, sm, mb, pt, vt in configs:
        res = test_config(stocks, tf, per, tm, sm, mb, pt, vt)
        if res:
            df_r = pd.DataFrame(res)
            avg_tgt = df_r['tgt_hit_pct'].mean()
            avg_sl  = df_r['sl_hit_pct'].mean()
            tot_tr  = df_r['trades'].sum()
            print(f"TF: {tf} | Tgt: {tm} ATR | SL: {sm} ATR | Prob: {pt} | Trades: {tot_tr:>3} | TGT HIT: {avg_tgt:.1f}% | SL HIT: {avg_sl:.1f}%", flush=True)
            for r in res:
                print(f"   -> {r['stock']:<10}: Trades={r['trades']:>2}, TGT={r['tgt_hit_pct']:>5.1f}%, SL={r['sl_hit_pct']:>5.1f}%", flush=True)
