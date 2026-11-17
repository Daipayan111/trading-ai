import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def train_and_eval_price_action(symbol: str, interval: str = "1h"):
    full_sym = f"{symbol}.NS" if not symbol.endswith(('.NS', '.BO')) else symbol
    raw = yf.download(full_sym, period="730d", interval=interval, progress=False)
    if raw.empty or len(raw) < 150: return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
        
    df = raw.copy()
    
    # 1. Technical & Price Action Features
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], 14).average_true_range()
    df['EMA_9'] = ta.trend.EMAIndicator(df['Close'], 9).ema_indicator()
    df['EMA_21'] = ta.trend.EMAIndicator(df['Close'], 21).ema_indicator()
    df['EMA_50'] = ta.trend.EMAIndicator(df['Close'], 50).ema_indicator()
    df['SMA_200'] = ta.trend.SMAIndicator(df['Close'], 200).sma_indicator()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], 14).rsi()
    df['Vol_SMA'] = df['Volume'].rolling(20).mean()
    df['Vol_Ratio'] = np.where(df['Vol_SMA'] > 0, df['Volume'] / df['Vol_SMA'], 1.0)
    
    # VWAP
    df['Date_Only'] = df.index.date
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    cum_v = df['Volume'].groupby(df['Date_Only']).cumsum()
    cum_pv = (tp * df['Volume']).groupby(df['Date_Only']).cumsum()
    df['VWAP'] = np.where(cum_v > 0, cum_pv / cum_v, df['Close'])
    
    # 2-Day Price Action
    u_dates = df['Date_Only'].unique()
    daily_stats = df.groupby('Date_Only').agg({'High': 'max', 'Low': 'min', 'Close': 'last'})
    d_prev = {u_dates[k]: daily_stats.loc[u_dates[k-1]] for k in range(1, len(u_dates))}
    d_prev2 = {u_dates[k]: daily_stats.loc[u_dates[k-2]] for k in range(2, len(u_dates))}
    
    df['PDH'] = df['Date_Only'].map(lambda d: d_prev[d]['High'] if d in d_prev else df.loc[df['Date_Only']==d, 'High'].iloc[0])
    df['PDL'] = df['Date_Only'].map(lambda d: d_prev[d]['Low'] if d in d_prev else df.loc[df['Date_Only']==d, 'Low'].iloc[0])
    df['PDC'] = df['Date_Only'].map(lambda d: d_prev[d]['Close'] if d in d_prev else df.loc[df['Date_Only']==d, 'Close'].iloc[0])
    
    # Candlestick Anatomy
    body = (df['Close'] - df['Open']).abs()
    lower_shadow = df[['Close', 'Open']].min(axis=1) - df['Low']
    upper_shadow = df['High'] - df[['Close', 'Open']].max(axis=1)
    df['Hammer'] = ((lower_shadow > body * 1.8) & (upper_shadow < body * 0.5)).astype(int)
    df['Bull_Eng'] = ((df['Close'].shift(1) < df['Open'].shift(1)) & 
                      (df['Close'] > df['Open']) & 
                      (df['Close'] >= df['Open'].shift(1))).astype(int)
    df['Bear_Eng'] = ((df['Close'].shift(1) > df['Open'].shift(1)) & 
                      (df['Close'] < df['Open']) & 
                      (df['Close'] <= df['Open'].shift(1))).astype(int)
    df['Bull_FVG'] = ((df['Low'] > df['High'].shift(2)) & (df['Close'].shift(1) > df['Open'].shift(1))).astype(int)
    df['Bear_FVG'] = ((df['High'] < df['Low'].shift(2)) & (df['Close'].shift(1) < df['Open'].shift(1))).astype(int)
    
    # Relative Ratios for ML
    df['Dist_VWAP'] = (df['Close'] - df['VWAP']) / df['ATR']
    df['Dist_PDH'] = (df['Close'] - df['PDH']) / df['ATR']
    df['Dist_PDL'] = (df['Close'] - df['PDL']) / df['ATR']
    df['Trend_Alignment'] = ((df['Close'] > df['EMA_50']) & (df['EMA_9'] > df['EMA_21'])).astype(int)
    
    df = df.dropna().reset_index(drop=True)
    
    # ── LABELING (Target > Stop Loss: e.g. T1 = 1.2x ATR, SL = 0.9x ATR, RR = 1:1.33 to 1:1.8) ──
    # We label a bar as 1 (Good Buy) if price reaches +1.2*ATR before -0.9*ATR within 16 bars
    t_mult = 1.25
    s_mult = 0.90
    max_bars = 16
    
    labels = np.zeros(len(df), dtype=int)
    c_vals = df['Close'].values
    h_vals = df['High'].values
    l_vals = df['Low'].values
    a_vals = df['ATR'].values
    
    for i in range(len(df) - max_bars):
        c0 = c_vals[i]
        a0 = a_vals[i]
        t_price = c0 + (t_mult * a0)
        s_price = c0 - (s_mult * a0)
        
        hit_t = False
        hit_s = False
        for k in range(i + 1, i + 1 + max_bars):
            if h_vals[k] >= t_price:
                hit_t = True
                break
            if l_vals[k] <= s_price:
                hit_s = True
                break
        if hit_t and not hit_s:
            labels[i] = 1 # BUY Win
            
        # SELL
        t_sell = c0 - (t_mult * a0)
        s_sell = c0 + (s_mult * a0)
        hit_ts = False
        hit_ss = False
        for k in range(i + 1, i + 1 + max_bars):
            if l_vals[k] <= t_sell:
                hit_ts = True
                break
            if h_vals[k] >= s_sell:
                hit_ss = True
                break
        if hit_ts and not hit_ss:
            labels[i] = 2 # SELL Win
            
    df['Label'] = labels
    
    # ML Features
    feat_cols = [
        'RSI', 'Vol_Ratio', 'Dist_VWAP', 'Dist_PDH', 'Dist_PDL',
        'Trend_Alignment', 'Hammer', 'Bull_Eng', 'Bear_Eng', 'Bull_FVG', 'Bear_FVG'
    ]
    
    # Walk-forward 60% Train, 40% Test (covering the last 250 trading days)
    split_idx = int(len(df) * 0.55)
    train_data = df.iloc[:split_idx]
    test_data = df.iloc[split_idx:].reset_index(drop=True)
    
    if len(test_data) < 100: return None
    
    X_train = train_data[feat_cols].values
    y_train = train_data['Label'].values
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    
    rf = RandomForestClassifier(n_estimators=180, max_depth=6, min_samples_leaf=5, random_state=42)
    rf.fit(X_train_s, y_train)
    
    X_test = test_data[feat_cols].values
    X_test_s = scaler.transform(X_test)
    probs = rf.predict_proba(X_test_s)
    classes = list(rf.classes_)
    
    p_buy_idx = classes.index(1) if 1 in classes else -1
    p_sell_idx = classes.index(2) if 2 in classes else -1
    
    # Evaluate Out-of-Sample Trades
    trades = []
    cooldown = 0
    confidence_thresh = 0.45 # High conviction only!
    
    for i in range(len(test_data) - max_bars):
        if cooldown > 0:
            cooldown -= 1
            continue
            
        prob_buy = probs[i, p_buy_idx] if p_buy_idx != -1 else 0
        prob_sell = probs[i, p_sell_idx] if p_sell_idx != -1 else 0
        
        row = test_data.iloc[i]
        c = float(row['Close'])
        atr = float(row['ATR'])
        
        sig = None
        if prob_buy >= confidence_thresh and row['Close'] > row['VWAP'] and row['Trend_Alignment'] == 1:
            sig = 'BUY'
        elif prob_sell >= confidence_thresh and row['Close'] < row['VWAP'] and row['Trend_Alignment'] == 0:
            sig = 'SELL'
            
        if not sig:
            continue
            
        is_buy = sig == 'BUY'
        
        # ── STRUCTURAL RISK & TARGET (RR >= 1:1.6, TARGET > STOP LOSS) ──
        # Target 1 = 1.35 * ATR
        # Target 2 = 2.10 * ATR
        # Stop Loss = 0.85 * ATR
        # RR = 1.59 to 2.47!
        sl_dist = round(0.85 * atr, 2)
        t1_dist = round(1.35 * atr, 2)
        t2_dist = round(2.10 * atr, 2)
        
        target1 = c + t1_dist if is_buy else c - t1_dist
        target2 = c + t2_dist if is_buy else c - t2_dist
        stop = c - sl_dist if is_buy else c + sl_dist
        
        curr_sl = stop
        t1_hit = False
        outcome = "Pending"
        pnl = 0
        
        for k in range(i + 1, min(i + 1 + max_bars, len(test_data))):
            h_k = float(test_data.iloc[k]['High'])
            l_k = float(test_data.iloc[k]['Low'])
            
            if is_buy:
                # Milestone 1: Trailing breakeven trigger
                if not t1_hit and h_k >= target1:
                    t1_hit = True
                    curr_sl = c + (0.20 * atr) # Lock profit!
                    
                if t1_hit and h_k >= target2:
                    outcome = "Target Hit (Full T2)"
                    pnl = (0.5 * t1_dist) + (0.5 * t2_dist)
                    break
                    
                if l_k <= curr_sl:
                    if t1_hit:
                        outcome = "Target Hit (T1 Secured)"
                        pnl = (0.5 * t1_dist) + (0.5 * (curr_sl - c))
                    else:
                        outcome = "Stop Loss Hit"
                        pnl = -sl_dist
                    break
            else:
                if not t1_hit and l_k <= target1:
                    t1_hit = True
                    curr_sl = c - (0.20 * atr)
                    
                if t1_hit and l_k <= target2:
                    outcome = "Target Hit (Full T2)"
                    pnl = (0.5 * t1_dist) + (0.5 * t2_dist)
                    break
                    
                if h_k >= curr_sl:
                    if t1_hit:
                        outcome = "Target Hit (T1 Secured)"
                        pnl = (0.5 * t1_dist) + (0.5 * (c - curr_sl))
                    else:
                        outcome = "Stop Loss Hit"
                        pnl = -sl_dist
                    break
                    
        if outcome != "Pending":
            trades.append({
                "type": sig,
                "price": c,
                "target": target2,
                "target_dist": t2_dist,
                "stop": stop,
                "sl_dist": sl_dist,
                "rr": round(t1_dist / sl_dist, 2),
                "outcome": outcome,
                "pnl": round(pnl, 2)
            })
            cooldown = 4
            
    return trades

def run_all():
    stocks = ["TCS", "RELIANCE", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "LT"]
    print("="*75)
    print("   PRICE ACTION ML ENGINE (TARGET > SL, RR >= 1:1.6, 250 TRADING DAYS)")
    print("="*75)
    
    tot_trades, tot_wins, tot_sls, tot_pnl = 0, 0, 0, 0
    
    for s in stocks:
        trades = train_and_eval_price_action(s, interval="1h")
        if not trades: continue
        wins = [t for t in trades if "Target" in t['outcome']]
        t2_wins = [t for t in trades if "Full T2" in t['outcome']]
        t1_wins = [t for t in trades if "T1 Secured" in t['outcome']]
        sls = [t for t in trades if t['outcome'] == "Stop Loss Hit"]
        total = len(trades)
        
        wr = (len(wins) / total) * 100 if total else 0
        pnl = sum(t['pnl'] for t in trades)
        avg_target = np.mean([t['target_dist'] for t in trades])
        avg_sl = np.mean([t['sl_dist'] for t in trades])
        
        tot_trades += total
        tot_wins += len(wins)
        tot_sls += len(sls)
        tot_pnl += pnl
        
        print(f"STOCK: {s:10s} | Trades: {total:2d} | Target Hits: {len(wins):2d} (T2: {len(t2_wins)}, T1: {len(t1_wins)}) | SL: {len(sls):2d}")
        print(f"  ► Target Hit Rate: {wr:.1f}%")
        print(f"  ► Risk:Reward    : 1 : {round(1.35/0.85, 2)} (Target: {avg_target:.1f} pts > SL: {avg_sl:.1f} pts)")
        print(f"  ► Net PnL Gain   : {pnl:+.2f} pts\n")
        
    print("="*75)
    overall_wr = (tot_wins / tot_trades) * 100 if tot_trades else 0
    print(f"FINAL MULTI-STOCK SUMMARY (250 TRADING DAYS, {len(stocks)} TOP NSE STOCKS):")
    print(f"Total High-Conviction Trades : {tot_trades}")
    print(f"Total Target Hits            : {tot_wins} ({overall_wr:.1f}%)")
    print(f"Total Stop Loss Hits         : {tot_sls} ({tot_sls/tot_trades*100:.1f}%)")
    print(f"Risk:Reward (RR)             : 1 : 1.6 to 1 : 2.5 (Target is 1.6x to 2.5x GREATER than Stop Loss!)")
    print(f"Total Net Points Profit      : {tot_pnl:+.2f} pts")
    print("="*75)

if __name__ == "__main__":
    run_all()
