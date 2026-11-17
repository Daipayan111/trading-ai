"""
TradingAI Pro — 12-Year Institutional Backtest Audit for LTP Calculator (CoA 1.0)
================================================================================
Comprehensive 12-Year Backtest (2014–2026) of Dr. Vinay Tiwari's LTP Calculator
mechanics on NIFTY 50 and BANK NIFTY.
"""

import math
import numpy as np
import pandas as pd
import yfinance as yf

def run_backtest_for_symbol(symbol: str, name: str, step: float):
    print(f"\n{'='*70}\nStarting 12-Year LTP Calculator Audit for {name} ({symbol})\n{'='*70}")
    
    # Download 12-year daily OHLCV
    df = yf.download(symbol, start="2014-01-01", end="2026-11-06", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [col.lower() for col in df.columns]

    df = df.dropna().copy()
    
    # Rolling 20-day historical volatility (annualized)
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["vol_20"] = df["log_ret"].rolling(20).std() * math.sqrt(252)
    df["vol_20"] = df["vol_20"].fillna(0.16)
    
    # EMAs for Chart of Accuracy Trend Biasing (CoA 1.0)
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    
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
        vol = float(prev_row["vol_20"])
        
        # Expected daily 1-sigma move & option premium estimation
        daily_vol_pts = p_close * (vol / math.sqrt(252))
        atm_strike = round(p_close / step) * step
        expected_premium = max(step * 0.8, daily_vol_pts * 0.45)
        
        # Determine Market State according to CoA 1.0 (Chart of Accuracy)
        ema20 = float(prev_row["ema_20"])
        ema50 = float(prev_row["ema_50"])
        
        is_bullish = (p_close > ema20) and (ema20 > ema50)
        is_bearish = (p_close < ema20) and (ema20 < ema50)
        is_rangebound = not is_bullish and not is_bearish
        
        # Calculate Support & Resistance Strikes
        if is_bullish:
            # Resistance is WTT, Support is Strong
            res_strike = atm_strike + step
            sup_strike = atm_strike
            allowed_trades = ["CE"] # Buy CE only at EOS
        elif is_bearish:
            # Resistance is WTB / Strong, Support is WTB
            res_strike = atm_strike
            sup_strike = atm_strike - step
            allowed_trades = ["PE"] # Buy PE only at EOR
        else:
            # Rangebound: Scenario 1 (Both Strong)
            res_strike = atm_strike + step
            sup_strike = atm_strike - step
            allowed_trades = ["CE", "PE"] # Buy both at reversals
            
        # Reversal levels (EOS & EOR)
        eos = sup_strike - (expected_premium * 0.65)
        eor = res_strike + (expected_premium * 0.65)
        
        d_open = float(row["open"])
        d_high = float(row["high"])
        d_low = float(row["low"])
        d_close = float(row["close"])
        
        # Check EOS (Buy CE Trade)
        if "CE" in allowed_trades:
            if d_low <= eos and d_open > (eos - (step * 0.5)):
                # Entry triggered at EOS!
                entry = eos
                sl = eos - (step * 0.6) # Risk: ~0.6 step
                tp = eos + (step * 1.4) # Reward: ~1.4 step (RR 1:2.3)
                
                # Check outcome:
                # If day low violated SL deeply
                if d_low < sl:
                    is_win = False
                    pnl = sl - entry
                elif d_high >= tp:
                    is_win = True
                    pnl = tp - entry
                else:
                    # Intraday close exit
                    pnl = d_close - entry
                    is_win = (pnl > 0)
                    
                trades.append({
                    "date": date,
                    "year": year,
                    "type": "BUY_CE_EOS",
                    "entry": entry,
                    "exit": entry + pnl,
                    "pnl": pnl,
                    "is_win": is_win
                })
                yearly_stats[year]["trades"] += 1
                if is_win:
                    yearly_stats[year]["wins"] += 1
                else:
                    yearly_stats[year]["losses"] += 1
                yearly_stats[year]["profit_pts"] += pnl

        # Check EOR (Buy PE Trade)
        if "PE" in allowed_trades:
            if d_high >= eor and d_open < (eor + (step * 0.5)):
                # Entry triggered at EOR!
                entry = eor
                sl = eor + (step * 0.6) # Risk: ~0.6 step
                tp = eor - (step * 1.4) # Reward: ~1.4 step (RR 1:2.3)
                
                if d_high > sl:
                    is_win = False
                    pnl = entry - sl
                elif d_low <= tp:
                    is_win = True
                    pnl = entry - tp
                else:
                    # Intraday close exit
                    pnl = entry - d_close
                    is_win = (pnl > 0)
                    
                trades.append({
                    "date": date,
                    "year": year,
                    "type": "BUY_PE_EOR",
                    "entry": entry,
                    "exit": entry - pnl,
                    "pnl": pnl,
                    "is_win": is_win
                })
                yearly_stats[year]["trades"] += 1
                if is_win:
                    yearly_stats[year]["wins"] += 1
                else:
                    yearly_stats[year]["losses"] += 1
                yearly_stats[year]["profit_pts"] += pnl
                
    # Compile Statistics
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t["is_win"])
    win_rate = (winning_trades / max(1, total_trades)) * 100
    
    total_gains = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    total_losses = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    profit_factor = total_gains / max(0.001, total_losses)
    net_points = sum(t["pnl"] for t in trades)
    
    # Compute Drawdown
    equity_curve = np.cumsum([t["pnl"] for t in trades])
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = peak - equity_curve
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0
    
    print(f"\n--- 12-YEAR SUMMARY RESULTS: {name} ---")
    print(f"Total Trading Days Analyzed : {len(df)}")
    print(f"Total Executed Reversal Trades: {total_trades}")
    print(f"Winning Trades              : {winning_trades}")
    print(f"Losing Trades               : {total_trades - winning_trades}")
    print(f"Win Rate                    : {win_rate:.2f}%")
    print(f"Profit Factor               : {profit_factor:.2f}")
    print(f"Total Net Points Gained     : +{net_points:,.1f} pts")
    print(f"Max Peak-to-Trough Drawdown : -{max_dd:,.1f} pts")
    
    print("\n--- YEAR-BY-YEAR BREAKDOWN ---")
    print("Year | Trades | Win Rate | Net Points")
    print("-" * 38)
    for y in sorted(yearly_stats.keys()):
        st = yearly_stats[y]
        tr = st["trades"]
        wr = (st["wins"] / max(1, tr)) * 100
        pts = st["profit_pts"]
        print(f"{y} |  {tr:4d}  |  {wr:5.1f}%  |  {'+' if pts >= 0 else ''}{pts:8.1f} pts")
        
    return {
        "symbol": symbol,
        "name": name,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "net_points": net_points,
        "max_dd": max_dd,
        "yearly": yearly_stats
    }

if __name__ == "__main__":
    nifty_res = run_backtest_for_symbol("^NSEI", "NIFTY 50", step=50)
    bank_res = run_backtest_for_symbol("^NSEBANK", "BANK NIFTY", step=100)

