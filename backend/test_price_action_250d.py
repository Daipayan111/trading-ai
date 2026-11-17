import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd
import numpy as np
from analytics import add_technical_indicators, generate_signals

def test_ml_price_action_250d():
    stocks = ["TCS", "RELIANCE", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "LT"]
    print("="*75)
    print("   TESTING INSTITUTIONAL PRICE ACTION (RR >= 1:1.5, TARGET > SL, WIN RATE >= 80%)")
    print("="*75)
    
    results_summary = []
    
    for symbol in stocks:
        raw = yf.download(f"{symbol}.NS", period="730d", interval="1h", progress=False)
        if raw.empty or len(raw) < 100: continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
            
        df = add_technical_indicators(raw)
        
        # Price Action features:
        # 1. 20-period Donchian Channels (Breakout of 2-3 days)
        df['Upper_Chan'] = df['High'].rolling(20).max()
        df['Lower_Chan'] = df['Low'].rolling(20).min()
        
        # 2. Previous Day High (PDH) and Low (PDL)
        df['Date_Only'] = df.index.date
        d_summary = df.groupby('Date_Only').agg({'High': 'max', 'Low': 'min', 'Close': 'last'})
        u_dates = df['Date_Only'].unique()
        d_prev = {u_dates[k]: d_summary.loc[u_dates[k-1]] for k in range(1, len(u_dates))}
        df['PDH'] = df['Date_Only'].map(lambda d: d_prev[d]['High'] if d in d_prev else df.loc[df['Date_Only']==d, 'High'].iloc[0])
        df['PDL'] = df['Date_Only'].map(lambda d: d_prev[d]['Low'] if d in d_prev else df.loc[df['Date_Only']==d, 'Low'].iloc[0])
        df['PDC'] = df['Date_Only'].map(lambda d: d_prev[d]['Close'] if d in d_prev else df.loc[df['Date_Only']==d, 'Close'].iloc[0])
        
        # 3. Volatility & Volume
        df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Filter for the last 250 trading days (~1750 1-hour candles)
        start_idx = max(50, len(df) - 1750)
        
        trades = []
        for i in range(start_idx, len(df) - 35):
            row = df.iloc[i]
            c = float(row['Close'])
            atr = float(row['ATR'])
            if pd.isna(atr) or atr <= 0: atr = c * 0.01
            
            vwap = float(row.get('VWAP', c))
            ema9 = float(row.get('EMA_9', c))
            ema21 = float(row.get('EMA_21', c))
            ema50 = float(row.get('EMA_50', c))
            rsi = float(row.get('RSI', 50))
            div = int(row.get('Divergence_Signal', 0))
            smc = int(row.get('SMC_Signal', 0))
            hammer = int(row.get('Pattern_Hammer', 0))
            bull_eng = int(row.get('Pattern_Bullish_Engulfing', 0))
            bear_eng = int(row.get('Pattern_Bearish_Engulfing', 0))
            vol_ratio = float(row.get('Vol_Ratio', 1.0))
            pdh = float(row['PDH'])
            pdl = float(row['PDL'])
            
            # ── STRICT PRICE ACTION ENTRY RULES ──
            # BUY: Price Action confirmation
            # 1. Bullish Candle (Hammer / Engulfing / SMC FVG) OR Breakout above PDH/Upper Channel
            # 2. Strong Volume (> 1.1x)
            # 3. Confirmed by VWAP & EMA Alignment
            # 4. No Bearish Divergence
            is_buy = False
            is_sell = False
            
            buy_score = 0
            if c > vwap: buy_score += 1
            if ema9 > ema21 and ema21 > ema50: buy_score += 1
            if bull_eng or hammer or smc == 1: buy_score += 1.5
            if c > pdh: buy_score += 1.5 # Previous day high breakout
            if vol_ratio > 1.15: buy_score += 1
            if 50 <= rsi <= 75: buy_score += 1
            if div == -1: buy_score = 0 # Trap!
            
            sell_score = 0
            if c < vwap: sell_score += 1
            if ema9 < ema21 and ema21 < ema50: sell_score += 1
            if bear_eng or smc == -1: sell_score += 1.5
            if c < pdl: sell_score += 1.5 # Previous day low breakdown
            if vol_ratio > 1.15: sell_score += 1
            if 25 <= rsi <= 50: sell_score += 1
            if div == 1: sell_score = 0 # Trap!
            
            if buy_score >= 4.5:
                is_buy = True
            elif sell_score >= 4.5:
                is_sell = True
            else:
                continue
                
            # ── STRUCTURAL RISK & TARGET (RR >= 1:1.6, TARGET > STOP LOSS) ──
            # Stop Loss is tight: 0.85 * ATR
            # Target 1: 1.35 * ATR -> RR = 1 : 1.59
            # Target 2: 2.10 * ATR -> RR = 1 : 2.47
            # Blended RR = 1 : 1.75!
            risk_dist = round(0.85 * atr, 2)
            t1_dist = round(1.35 * atr, 2)
            t2_dist = round(2.10 * atr, 2)
            
            if is_buy:
                stop = round(c - risk_dist, 2)
                t1 = round(c + t1_dist, 2)
                t2 = round(c + t2_dist, 2)
            else:
                stop = round(c + risk_dist, 2)
                t1 = round(c - t1_dist, 2)
                t2 = round(c - t2_dist, 2)
                
            # Trailing Breakeven once T1 is hit:
            # At T1: 50% position booked at +1.35*ATR, Stop Loss moved to Entry + 0.2*ATR (risk-free)
            # Remaining 50% rides to T2 or exits at Entry + 0.2*ATR
            t1_hit = False
            curr_sl = stop
            outcome = "Pending"
            realized_pnl = 0
            
            for j in range(i + 1, min(i + 1 + 35, len(df))):
                h = float(df.iloc[j]['High'])
                l = float(df.iloc[j]['Low'])
                
                if is_buy:
                    # Check T1
                    if not t1_hit and h >= t1:
                        t1_hit = True
                        curr_sl = c + (0.20 * atr) # Risk-free breakeven locked
                        
                    # Check T2
                    if t1_hit and h >= t2:
                        outcome = "Target Hit (Full T2)"
                        realized_pnl = (0.5 * t1_dist) + (0.5 * t2_dist) # ~1.725 * ATR
                        break
                        
                    # Check Stop
                    if l <= curr_sl:
                        if t1_hit:
                            outcome = "Target Hit (T1 + Breakeven)"
                            realized_pnl = (0.5 * t1_dist) + (0.5 * (curr_sl - c)) # ~0.775 * ATR
                        else:
                            outcome = "Stop Loss Hit"
                            realized_pnl = -risk_dist
                        break
                else:
                    if not t1_hit and l <= t1:
                        t1_hit = True
                        curr_sl = c - (0.20 * atr)
                        
                    if t1_hit and l <= t2:
                        outcome = "Target Hit (Full T2)"
                        realized_pnl = (0.5 * t1_dist) + (0.5 * t2_dist)
                        break
                        
                    if h >= curr_sl:
                        if t1_hit:
                            outcome = "Target Hit (T1 + Breakeven)"
                            realized_pnl = (0.5 * t1_dist) + (0.5 * (c - curr_sl))
                        else:
                            outcome = "Stop Loss Hit"
                            realized_pnl = -risk_dist
                        break
                        
            if outcome != "Pending":
                trades.append({
                    "time": str(df.index[i])[:16],
                    "type": "BUY" if is_buy else "SELL",
                    "price": c,
                    "target": t2,
                    "target_dist": t2_dist,
                    "stop": stop,
                    "risk_dist": risk_dist,
                    "rr": round(t1_dist / risk_dist, 2),
                    "outcome": outcome,
                    "pnl": round(realized_pnl, 2)
                })
                
        wins = [t for t in trades if "Target" in t['outcome']]
        t2_wins = [t for t in trades if "Full T2" in t['outcome']]
        t1_wins = [t for t in trades if "T1 + Breakeven" in t['outcome']]
        sls = [t for t in trades if t['outcome'] == "Stop Loss Hit"]
        total = len(trades)
        
        if total > 0:
            wr = (len(wins) / total) * 100
            total_pts = sum(t['pnl'] for t in trades)
            avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
            avg_loss = np.mean([abs(t['pnl']) for t in sls]) if sls else 0
            
            print(f"STOCK: {symbol:10s} | Trades: {total:2d} | Target Wins: {len(wins):2d} (T2: {len(t2_wins)}, T1: {len(t1_wins)}) | SL: {len(sls):2d}")
            print(f"  ► Win Rate: {wr:.1f}% | SL Rate: {len(sls)/total*100:.1f}%")
            print(f"  ► RR Displayed: 1 : {round(1.35/0.85, 2)} (Target 1 > Stop Loss, Target 2 >> Stop Loss!)")
            print(f"  ► Avg Win: +{avg_win:.2f} pts | Avg Loss: -{avg_loss:.2f} pts | Net PnL: {total_pts:+.2f} pts\n")
            results_summary.append({"symbol": symbol, "total": total, "wins": len(wins), "sls": len(sls), "wr": wr, "pnl": total_pts})

    print("="*75)
    tot_trades = sum(r['total'] for r in results_summary)
    tot_wins = sum(r['wins'] for r in results_summary)
    tot_sls = sum(r['sls'] for r in results_summary)
    tot_pnl = sum(r['pnl'] for r in results_summary)
    print(f"GRAND SUMMARY (250 TRADING DAYS, 7 TOP NSE STOCKS):")
    print(f"Total High-Conviction Trades: {tot_trades}")
    print(f"Total Target Hits           : {tot_wins} ({tot_wins/tot_trades*100:.1f}%)")
    print(f"Total Stop Loss Hits        : {tot_sls} ({tot_sls/tot_trades*100:.1f}%)")
    print(f"Risk:Reward (RR)            : 1 : 1.6 to 1 : 2.5 (Target is 1.6x to 2.5x GREATER than Stop Loss!)")
    print(f"Total Net Points Profit     : {tot_pnl:+.2f} pts")
    print("="*75)

if __name__ == "__main__":
    test_ml_price_action_250d()
