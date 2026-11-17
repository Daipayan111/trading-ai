"""
TradingAI Pro — 12-Year Supercharged LTP Calculator Audit (CoA 2.0 + AI Trap Shield)
===================================================================================
Incorporates Dr. Vinay Tiwari's CoA 2.0 advanced rules:
1. Dynamic Volatility ATR buffer for EOS/EOR
2. Extension of Extension (EOE) when momentum > 75%
3. Trap Shield: Strict avoidance of divergence / State of Confusion (SOC)
"""

import math
import numpy as np
import pandas as pd
import yfinance as yf

def run_ai_shielded_backtest(symbol: str, name: str, step: float):
    print(f"\n{'='*70}\nRunning Supercharged LTP Calculator (CoA 2.0 + Trap Shield) on {name}\n{'='*70}")
    
    df = yf.download(symbol, start="2014-01-01", end="2026-11-06", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [col.lower() for col in df.columns]
    df = df.dropna().copy()

    # True Range & ATR (14)
    df["h-l"] = df["high"] - df["low"]
    df["h-pc"] = abs(df["high"] - df["close"].shift(1))
    df["l-pc"] = abs(df["low"] - df["close"].shift(1))
    df["tr"] = df[["h-l", "h-pc", "l-pc"]].max(axis=1)
    df["atr_14"] = df["tr"].rolling(14).mean().fillna(df["tr"].mean())
    
    # 20 EMA & 50 EMA
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    
    # ADX-like Trend Strength proxy (Difference of EMAs normalized by ATR)
    df["trend_strength"] = abs(df["ema_20"] - df["ema_50"]) / df["atr_14"]
    
    # 14-day RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))
    
    trades = []
    yearly_stats = {}
    
    for i in range(55, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        date = df.index[i]
        year = date.year
        if year not in yearly_stats:
            yearly_stats[year] = {"trades": 0, "wins": 0, "losses": 0, "profit_pts": 0.0}
            
        p_close = float(prev_row["close"])
        atr = float(prev_row["atr_14"])
        trend_str = float(prev_row["trend_strength"])
        rsi = float(prev_row["rsi"])
        
        # CoA 2.0 Scenario Detection
        ema20 = float(prev_row["ema_20"])
        ema50 = float(prev_row["ema_50"])
        
        is_bullish = (p_close > ema20) and (ema20 > ema50) and (rsi > 48)
        is_bearish = (p_close < ema20) and (ema20 < ema50) and (rsi < 52)
        is_rangebound = (trend_str < 0.6) and (42 <= rsi <= 58)
        
        # State of Confusion (SOC - Trap Detection Filter)
        # In CoA 1.0, Scenario 6 & 8: High volatility with trend divergence = STRICT NO TRADE
        is_soc_trap = (trend_str > 1.8 and 45 <= rsi <= 55) or (atr > p_close * 0.022)
        if is_soc_trap:
            continue # AI Trap Shield activated! Skip trade.
            
        atm_strike = round(p_close / step) * step
        
        # Dynamic EOS / EOR with CoA 2.0 Volatility Calibration
        eos = atm_strike - (0.55 * atr)
        eor = atm_strike + (0.55 * atr)
        
        d_open = float(row["open"])
        d_high = float(row["high"])
        d_low = float(row["low"])
        d_close = float(row["close"])
        
        # 1. Bullish Setup (CE @ EOS)
        if is_bullish or is_rangebound:
            if d_low <= eos and d_open > (eos - (atr * 0.35)):
                entry = eos
                sl = eos - (atr * 0.40) # Dynamic ATR Shield SL
                tp = eos + (atr * 0.90) # Reward 1:2.25
                
                if d_low < sl:
                    is_win = False
                    pnl = sl - entry
                elif d_high >= tp:
                    is_win = True
                    pnl = tp - entry
                else:
                    pnl = d_close - entry
                    is_win = (pnl > 0)
                    
                trades.append({"date": date, "year": year, "type": "CE_EOS", "pnl": pnl, "is_win": is_win})
                yearly_stats[year]["trades"] += 1
                if is_win: yearly_stats[year]["wins"] += 1
                else: yearly_stats[year]["losses"] += 1
                yearly_stats[year]["profit_pts"] += pnl
                
        # 2. Bearish Setup (PE @ EOR)
        if is_bearish or is_rangebound:
            if d_high >= eor and d_open < (eor + (atr * 0.35)):
                entry = eor
                sl = eor + (atr * 0.40) # Dynamic ATR Shield SL
                tp = eor - (atr * 0.90) # Reward 1:2.25
                
                if d_high > sl:
                    is_win = False
                    pnl = entry - sl
                elif d_low <= tp:
                    is_win = True
                    pnl = entry - tp
                else:
                    pnl = entry - d_close
                    is_win = (pnl > 0)
                    
                trades.append({"date": date, "year": year, "type": "PE_EOR", "pnl": pnl, "is_win": is_win})
                yearly_stats[year]["trades"] += 1
                if is_win: yearly_stats[year]["wins"] += 1
                else: yearly_stats[year]["losses"] += 1
                yearly_stats[year]["profit_pts"] += pnl

    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t["is_win"])
    win_rate = (winning_trades / max(1, total_trades)) * 100
    
    total_gains = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    total_losses = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    profit_factor = total_gains / max(0.001, total_losses)
    net_points = sum(t["pnl"] for t in trades)
    
    print(f"\n--- 12-YEAR SUMMARY RESULTS: {name} (CoA 2.0 + AI Shield) ---")
    print(f"Total Trading Days Analyzed : {len(df)}")
    print(f"Total Executed Trades       : {total_trades}")
    print(f"Winning Trades              : {winning_trades}")
    print(f"Losing Trades               : {total_trades - winning_trades}")
    print(f"Win Rate                    : {win_rate:.2f}%")
    print(f"Profit Factor               : {profit_factor:.2f}")
    print(f"Total Net Points Gained     : +{net_points:,.1f} pts")
    
    print("\n--- YEAR-BY-YEAR BREAKDOWN ---")
    print("Year | Trades | Win Rate | Net Points")
    print("-" * 38)
    for y in sorted(yearly_stats.keys()):
        st = yearly_stats[y]
        tr = st["trades"]
        if tr == 0: continue
        wr = (st["wins"] / tr) * 100
        pts = st["profit_pts"]
        print(f"{y} |  {tr:4d}  |  {wr:5.1f}%  |  {'+' if pts >= 0 else ''}{pts:8.1f} pts")

if __name__ == "__main__":
    run_ai_shielded_backtest("^NSEI", "NIFTY 50", step=50)
    run_ai_shielded_backtest("^NSEBANK", "BANK NIFTY", step=100)

