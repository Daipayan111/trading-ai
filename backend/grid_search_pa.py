import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd
import numpy as np
import ta

def grid_search_price_action():
    symbols = ['TCS.NS', 'RELIANCE.NS', 'INFY.NS', 'HDFCBANK.NS']
    dfs = {}
    for s in symbols:
        raw = yf.download(s, period='730d', interval='1h', progress=False)
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
        df = raw.copy()
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], 14).average_true_range()
        df['EMA9'] = ta.trend.EMAIndicator(df['Close'], 9).ema_indicator()
        df['EMA21'] = ta.trend.EMAIndicator(df['Close'], 21).ema_indicator()
        df['EMA50'] = ta.trend.EMAIndicator(df['Close'], 50).ema_indicator()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], 14).rsi()
        df['Date'] = df.index.date
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (tp * df['Volume']).groupby(df['Date']).cumsum() / df['Volume'].groupby(df['Date']).cumsum()
        
        # 2-Day Levels
        u_dates = df['Date'].unique()
        d_stats = df.groupby('Date').agg({'High': 'max', 'Low': 'min', 'Close': 'last'})
        d_prev = {u_dates[k]: d_stats.loc[u_dates[k-1]] for k in range(1, len(u_dates))}
        df['PDH'] = df['Date'].map(lambda d: d_prev[d]['High'] if d in d_prev else df['High'])
        df['PDL'] = df['Date'].map(lambda d: d_prev[d]['Low'] if d in d_prev else df['Low'])
        
        dfs[s] = df.dropna().reset_index(drop=True)
        
    print("Testing Parameter Configurations across 250+ days (15m candles)...")
    
    # We want:
    # 1. Target >= Stop Loss (i.e. tgt_mult >= sl_mult, RR >= 1:1.0 to 1:1.5)
    # 2. Win Rate >= 80%
    # 3. High Net Profit Points
    
    candidates = [
        # (tgt_mult, sl_mult, trail_mult, trail_lock)
        (1.20, 1.00, 0.50, 0.15),
        (1.30, 1.10, 0.55, 0.20),
        (1.40, 1.10, 0.60, 0.20),
        (1.50, 1.20, 0.65, 0.25),
        (1.20, 1.10, 0.45, 0.15),
        (1.35, 1.15, 0.50, 0.20),
        (1.50, 1.25, 0.55, 0.25),
        (1.60, 1.30, 0.60, 0.30),
    ]
    
    for tgt_m, sl_m, tr_trig, tr_lock in candidates:
        rr = tgt_m / sl_m
        tot_trades, tot_wins, tot_pnl = 0, 0, 0
        
        for s, df in dfs.items():
            for i in range(50, len(df) - 30):
                row = df.iloc[i]
                c = float(row['Close'])
                atr = float(row['ATR'])
                vwap = float(row['VWAP'])
                ema9 = float(row['EMA9'])
                ema21 = float(row['EMA21'])
                ema50 = float(row['EMA50'])
                rsi = float(row['RSI'])
                pdh = float(row['PDH'])
                pdl = float(row['PDL'])
                
                # Strict Price Action Confluence:
                # 1. 2-Day Breakout or Bounce:
                #    BUY: c > vwap and ema9 > ema21 and ema21 > ema50 and (c > pdh or (c > ema21 and 52 <= rsi <= 68))
                is_buy = False
                is_sell = False
                if c > vwap and ema9 > ema21 and ema21 > ema50 and (c > pdh or (c > ema21 and 52 <= rsi <= 68)):
                    is_buy = True
                elif c < vwap and ema9 < ema21 and ema21 < ema50 and (c < pdl or (c < ema21 and 32 <= rsi <= 48)):
                    is_sell = True
                    
                if not is_buy and not is_sell: continue
                
                t_dist = tgt_m * atr
                s_dist = sl_m * atr
                trig_dist = tr_trig * atr
                lock_dist = tr_lock * atr
                
                target = c + t_dist if is_buy else c - t_dist
                stop = c - s_dist if is_buy else c + s_dist
                
                curr_sl = stop
                trailed = False
                outcome = 'Pending'
                pnl = 0
                
                for j in range(i + 1, min(i + 35, len(df))):
                    h = float(df.iloc[j]['High'])
                    l = float(df.iloc[j]['Low'])
                    if is_buy:
                        if h >= (c + trig_dist) and not trailed:
                            curr_sl = c + lock_dist
                            trailed = True
                        if h >= target:
                            outcome = 'Target Hit'
                            pnl = t_dist
                            break
                        if l <= curr_sl:
                            outcome = 'Target Hit' if trailed else 'SL Hit'
                            pnl = (curr_sl - c) if trailed else -s_dist
                            break
                    else:
                        if l <= (c - trig_dist) and not trailed:
                            curr_sl = c - lock_dist
                            trailed = True
                        if l <= target:
                            outcome = 'Target Hit'
                            pnl = t_dist
                            break
                        if h >= curr_sl:
                            outcome = 'Target Hit' if trailed else 'SL Hit'
                            pnl = (c - curr_sl) if trailed else -s_dist
                            break
                            
                if outcome != 'Pending':
                    tot_trades += 1
                    if outcome == 'Target Hit': tot_wins += 1
                    tot_pnl += pnl
                    
        wr = tot_wins / tot_trades * 100 if tot_trades else 0
        avg_pts = tot_pnl / tot_trades if tot_trades else 0
        print(f"Tgt: {tgt_m:.2f}*ATR | SL: {sl_m:.2f}*ATR | RR: 1 : {rr:.2f} | Trail: {tr_trig:.2f} -> Trades: {tot_trades:4d} | Win Rate: {wr:.1f}% | Net PnL: {tot_pnl:+8.1f} pts (Avg: {avg_pts:+.2f} pts/trade)")

if __name__ == "__main__":
    grid_search_price_action()
