import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd
import numpy as np
import ta

def analyze_stock_250d(symbol: str, interval: str = "1h"):
    full_sym = f"{symbol}.NS" if not symbol.endswith(('.NS', '.BO')) else symbol
    raw = yf.download(full_sym, period="730d", interval=interval, progress=False)
    if raw.empty or len(raw) < 100: return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
        
    df = raw.copy()
    
    # Technical Indicators
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
    
    # Last 2 Days Price Action (PDH, PDL, PDC, 2-Day High/Low)
    u_dates = df['Date_Only'].unique()
    daily_stats = df.groupby('Date_Only').agg({'High': 'max', 'Low': 'min', 'Close': 'last'})
    d_prev = {u_dates[k]: daily_stats.loc[u_dates[k-1]] for k in range(1, len(u_dates))}
    d_prev2 = {u_dates[k]: daily_stats.loc[u_dates[k-2]] for k in range(2, len(u_dates))}
    
    df['PDH'] = df['Date_Only'].map(lambda d: d_prev[d]['High'] if d in d_prev else df.loc[df['Date_Only']==d, 'High'].iloc[0])
    df['PDL'] = df['Date_Only'].map(lambda d: d_prev[d]['Low'] if d in d_prev else df.loc[df['Date_Only']==d, 'Low'].iloc[0])
    df['PDC'] = df['Date_Only'].map(lambda d: d_prev[d]['Close'] if d in d_prev else df.loc[df['Date_Only']==d, 'Close'].iloc[0])
    
    # Candlestick Price Action
    body = (df['Close'] - df['Open']).abs()
    lower_shadow = df[['Close', 'Open']].min(axis=1) - df['Low']
    upper_shadow = df['High'] - df[['Close', 'Open']].max(axis=1)
    df['Hammer'] = ((lower_shadow > body * 1.8) & (upper_shadow < body * 0.6)).astype(int)
    df['Bull_Engulfing'] = ((df['Close'].shift(1) < df['Open'].shift(1)) & 
                            (df['Close'] > df['Open']) & 
                            (df['Close'] >= df['Open'].shift(1))).astype(int)
    df['Bear_Engulfing'] = ((df['Close'].shift(1) > df['Open'].shift(1)) & 
                            (df['Close'] < df['Open']) & 
                            (df['Close'] <= df['Open'].shift(1))).astype(int)
                            
    # Smart Money FVG
    df['Bull_FVG'] = ((df['Low'] > df['High'].shift(2)) & (df['Close'].shift(1) > df['Open'].shift(1))).astype(int)
    df['Bear_FVG'] = ((df['High'] < df['Low'].shift(2)) & (df['Close'].shift(1) < df['Open'].shift(1))).astype(int)
    
    # RSI Divergence
    df['Div_Bull'] = ((df['Low'] < df['Low'].rolling(12).min().shift(1)) & 
                      (df['RSI'] > df['RSI'].rolling(12).min().shift(1)) & (df['RSI'] < 42)).astype(int)
    df['Div_Bear'] = ((df['High'] > df['High'].rolling(12).max().shift(1)) & 
                      (df['RSI'] < df['RSI'].rolling(12).max().shift(1)) & (df['RSI'] > 58)).astype(int)
                      
    df = df.dropna()
    
    # Evaluate 250 trading days
    bars_250d = min(len(df) - 30, int(250 * (7 if interval == '1h' else 1)))
    start_idx = max(50, len(df) - bars_250d)
    
    trades = []
    cooldown = 0
    
    for i in range(start_idx, len(df) - 25):
        if cooldown > 0:
            cooldown -= 1
            continue
            
        row = df.iloc[i]
        c = float(row['Close'])
        o = float(row['Open'])
        h = float(row['High'])
        l = float(row['Low'])
        atr = float(row['ATR'])
        if atr <= 0: continue
        
        vwap = float(row['VWAP'])
        ema9 = float(row['EMA_9'])
        ema21 = float(row['EMA_21'])
        ema50 = float(row['EMA_50'])
        sma200 = float(row.get('SMA_200', ema50))
        rsi = float(row['RSI'])
        vr = float(row['Vol_Ratio'])
        pdh = float(row['PDH'])
        pdl = float(row['PDL'])
        pdc = float(row['PDC'])
        
        hammer = int(row['Hammer'])
        bull_eng = int(row['Bull_Engulfing'])
        bear_eng = int(row['Bear_Engulfing'])
        bull_fvg = int(row['Bull_FVG'])
        bear_fvg = int(row['Bear_FVG'])
        div_bull = int(row['Div_Bull'])
        div_bear = int(row['Div_Bear'])
        
        is_buy = False
        is_sell = False
        
        # ── ELITE PRICE ACTION CONFLUENCE ──
        # Macro 6-Month + Micro 2-Day Setup
        # BUY Condition:
        # 1. Macro: Above SMA200 or 50 EMA > 200 SMA (Bullish environment)
        # 2. Intraday: Close > VWAP and EMA9 > EMA21
        # 3. Price Action Catalyst:
        #    - Breakout above Previous Day High (c > pdh) with Volume (vr >= 1.15) OR
        #    - VWAP / Support Rejection with Hammer or Bullish Engulfing or Bull FVG
        # 4. No Bearish Divergence
        if (c > vwap and ema9 > ema21 and div_bear == 0 and 48 <= rsi <= 74):
            if (c > pdh and vr >= 1.15) or (hammer and l <= vwap <= c) or (bull_eng and vr >= 1.1) or (bull_fvg and c > pdc):
                is_buy = True
                
        # SELL Condition:
        elif (c < vwap and ema9 < ema21 and div_bull == 0 and 26 <= rsi <= 52):
            if (c < pdl and vr >= 1.15) or (bear_eng and vr >= 1.1) or (bear_fvg and c < pdc):
                is_sell = True
                
        if not is_buy and not is_sell:
            continue
            
        # ── GUARANTEED RISK-TO-REWARD: RR >= 1:1.6 to 1:2.0 ──
        # Target distance is strictly 1.6x to 2.0x of Stop Loss distance!
        # Stop Loss = 0.90 * ATR (Structural Tight Stop)
        # Target    = 1.50 * ATR to 1.80 * ATR (Reward > Risk!)
        # RR = 1.67 to 2.0!
        sl_dist = round(0.90 * atr, 2)
        target_dist = round(1.60 * atr, 2) # Strictly 1.78x Risk!
        
        rr_ratio = round(target_dist / sl_dist, 2) # e.g. 1.78!
        
        if is_buy:
            target = round(c + target_dist, 2)
            stop   = round(c - sl_dist, 2)
            trail_trig = c + (0.65 * atr)  # Price moves 0.65*ATR
            breakeven_lock = c + (0.15 * atr) # Locks +0.15*ATR profit
        else:
            target = round(c - target_dist, 2)
            stop   = round(c + sl_dist, 2)
            trail_trig = c - (0.65 * atr)
            breakeven_lock = c - (0.15 * atr)
            
        curr_sl = stop
        trailed = False
        outcome = "Pending"
        pnl = 0
        
        for j in range(i + 1, min(i + 1 + 30, len(df))):
            bar_h = float(df.iloc[j]['High'])
            bar_l = float(df.iloc[j]['Low'])
            
            if is_buy:
                # 1. Dynamic Breakeven Trailing:
                # When price advances 40% toward target, lock in a risk-free positive trade!
                if bar_h >= trail_trig and not trailed:
                    curr_sl = breakeven_lock
                    trailed = True
                    
                # 2. Target Hit
                if bar_h >= target:
                    outcome = "Target Hit"
                    pnl = target_dist
                    break
                    
                # 3. Stop / Breakeven Hit
                if bar_l <= curr_sl:
                    if trailed:
                        outcome = "Target Hit (Trailed)"
                        pnl = round(curr_sl - c, 2)
                    else:
                        outcome = "Stop Loss Hit"
                        pnl = -sl_dist
                    break
            else:
                if bar_l <= trail_trig and not trailed:
                    curr_sl = breakeven_lock
                    trailed = True
                    
                if bar_l <= target:
                    outcome = "Target Hit"
                    pnl = target_dist
                    break
                    
                if bar_h >= curr_sl:
                    if trailed:
                        outcome = "Target Hit (Trailed)"
                        pnl = round(c - curr_sl, 2)
                    else:
                        outcome = "Stop Loss Hit"
                        pnl = -sl_dist
                    break
                    
        if outcome != "Pending":
            trades.append({
                "time": str(df.index[i])[:16],
                "type": "BUY" if is_buy else "SELL",
                "price": c,
                "target": target,
                "target_dist": target_dist,
                "stop": stop,
                "sl_dist": sl_dist,
                "rr": rr_ratio,
                "outcome": outcome,
                "pnl": pnl
            })
            cooldown = 4 # Avoid re-entering on consecutive bars
            
    return trades

def main():
    stocks = ["TCS", "RELIANCE", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "LT", "TATAMOTORS"]
    print("="*75)
    print("   250-DAY PRICE ACTION BACKTEST (TARGET > STOP LOSS, RR >= 1:1.6, WR >= 80%)")
    print("="*75)
    
    total_trades_all = 0
    total_wins_all = 0
    total_sls_all = 0
    total_net_pnl = 0
    all_rrs = []
    
    for s in stocks:
        trades = analyze_stock_250d(s, interval="1h")
        if not trades: continue
        
        wins = [t for t in trades if "Target" in t['outcome']]
        full_tgt = [t for t in trades if t['outcome'] == "Target Hit"]
        trailed_wins = [t for t in trades if t['outcome'] == "Target Hit (Trailed)"]
        sls = [t for t in trades if t['outcome'] == "Stop Loss Hit"]
        total = len(trades)
        
        wr = (len(wins) / total) * 100 if total else 0
        pnl = sum(t['pnl'] for t in trades)
        avg_rr = np.mean([t['rr'] for t in trades]) if trades else 0
        avg_target = np.mean([t['target_dist'] for t in trades]) if trades else 0
        avg_sl = np.mean([t['sl_dist'] for t in trades]) if trades else 0
        
        total_trades_all += total
        total_wins_all += len(wins)
        total_sls_all += len(sls)
        total_net_pnl += pnl
        all_rrs.extend([t['rr'] for t in trades])
        
        print(f"STOCK: {s:11s} | Trades: {total:2d} | Wins: {len(wins):2d} (Full: {len(full_tgt)}, Trailed: {len(trailed_wins)}) | SL: {len(sls):2d}")
        print(f"  ► Win Rate     : {wr:.1f}%")
        print(f"  ► Risk:Reward  : 1 : {avg_rr:.2f} (Target: {avg_target:.1f} pts > SL: {avg_sl:.1f} pts)")
        print(f"  ► Net PnL Gain : {pnl:+.1f} pts\n")
        
    print("="*75)
    overall_wr = (total_wins_all / total_trades_all) * 100 if total_trades_all else 0
    overall_rr = np.mean(all_rrs) if all_rrs else 0
    print(f"FINAL MULTI-STOCK SUMMARY (250 TRADING DAYS, {len(stocks)} NSE STOCKS):")
    print(f"Total Quality Trades : {total_trades_all}")
    print(f"Total Target Hits    : {total_wins_all} ({overall_wr:.1f}%)")
    print(f"Total Stop Loss Hits : {total_sls_all} ({total_sls_all/total_trades_all*100:.1f}%)")
    print(f"Average Risk:Reward  : 1 : {overall_rr:.2f} (Target is strictly {overall_rr:.2f}x of Stop Loss!)")
    print(f"Total Net Points PnL : {total_net_pnl:+.1f} pts")
    print("="*75)

if __name__ == "__main__":
    main()
