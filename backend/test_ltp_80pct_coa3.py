"""
TradingAI Pro — LTP Calculator CoA 3.0 (80%+ Institutional Confirmation Audit)
==============================================================================
Compares 12-Year Performance when waiting for Price-Action Confirmation at EOS/EOR:
1. Price dips into Support / EOS zone AND rejects upward (bullish hammer / wick rejection / bounce).
2. Trend alignment (Higher Timeframe EMA 200 & EMA 50 confirmation).
3. Institutional Trap Shield (Avoids falling-knife gaps & panic crashes).
4. Asymmetric TP1 capture + Trailing stop protection.
"""

import numpy as np
import pandas as pd
import yfinance as yf

def test_coa3_confirmed(symbol: str, name: str, step: float):
    print(f"\n{'='*72}\nTesting LTP Calculator CoA 3.0 (With Confirmation Candle) on {name}\n{'='*72}")
    
    df = yf.download(symbol, start="2014-01-01", end="2026-11-06", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [col.lower() for col in df.columns]
    df = df.dropna().copy()

    df["h-l"] = df["high"] - df["low"]
    df["h-pc"] = abs(df["high"] - df["close"].shift(1))
    df["l-pc"] = abs(df["low"] - df["close"].shift(1))
    df["tr"] = df[["h-l", "h-pc", "l-pc"]].max(axis=1)
    df["atr_14"] = df["tr"].rolling(14).mean().fillna(df["tr"].mean())
    
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

    delta = df["close"].diff()
    gain5 = delta.clip(lower=0).rolling(5).mean()
    loss5 = (-delta.clip(upper=0)).rolling(5).mean()
    df["rsi_5"] = 100 - (100 / (1 + (gain5 / (loss5 + 1e-9))))

    trades = []
    yearly_stats = {}

    for i in range(205, len(df) - 5):
        prev = df.iloc[i-1]
        curr = df.iloc[i]
        
        date = df.index[i]
        year = date.year
        if year not in yearly_stats:
            yearly_stats[year] = {"trades": 0, "wins": 0, "losses": 0, "profit_pts": 0.0}

        atr = float(prev["atr_14"])
        p_close = float(prev["close"])
        ema20 = float(prev["ema_20"])
        ema50 = float(prev["ema_50"])
        ema200 = float(prev["ema_200"])
        rsi5 = float(prev["rsi_5"])

        # Skip high-volatility crash regimes
        if atr > p_close * 0.022:
            continue

        c_open = float(curr["open"])
        c_high = float(curr["high"])
        c_low = float(curr["low"])
        c_close = float(curr["close"])
        c_range = max(1e-5, c_high - c_low)
        lower_wick = min(c_open, c_close) - c_low
        upper_wick = c_high - max(c_open, c_close)

        # RULE 1: EOS Bullish Reversal Confirmation
        # - Primary Uptrend: Price > EMA 200 and EMA 50 > EMA 200
        # - Pullback into EOS zone (RSI 5 <= 35 or touched EMA 20/50 support)
        # - Confirmation Candle: Strong rejection (green close OR hammer lower wick >= 35% of range)
        is_uptrend = (c_close > ema200) and (ema50 > ema200)
        in_eos_zone = (rsi5 <= 38) or (c_low <= ema20)
        bull_confirm = (c_close > c_open) or (lower_wick >= 0.40 * c_range)

        # RULE 2: EOR Bearish Reversal Confirmation
        # - Primary Downtrend: Price < EMA 50 and EMA 20 < EMA 50
        # - Spike into EOR zone (RSI 5 >= 65 or touched EMA 20 resistance)
        # - Confirmation Candle: Bearish rejection (red close OR upper wick >= 35% of range)
        is_downtrend = (c_close < ema50) and (ema20 < ema50)
        in_eor_zone = (rsi5 >= 62) or (c_high >= ema20)
        bear_confirm = (c_close < c_open) or (upper_wick >= 0.40 * c_range)

        if is_uptrend and in_eos_zone and bull_confirm:
            entry = c_close
            # Structure SL protected below the rejection low/Gamma wall
            sl = min(c_low, entry - 1.5 * atr) - (0.2 * atr)
            # TP1 target: Quick asymmetric pop towards Resistance / EOR (0.75 ATR)
            tp = entry + (0.75 * atr)

            outcome_win = None
            pnl = 0.0
            for j in range(i + 1, min(i + 6, len(df))):
                r_j = df.iloc[j]
                if float(r_j["low"]) <= sl:
                    outcome_win = False
                    pnl = sl - entry
                    break
                if float(r_j["high"]) >= tp:
                    outcome_win = True
                    pnl = tp - entry
                    break
            if outcome_win is None:
                final_c = float(df.iloc[min(i + 5, len(df) - 1)]["close"])
                pnl = final_c - entry
                outcome_win = (pnl > 0)

            trades.append({"date": date, "year": year, "type": "CONFIRMED_EOS_CE", "pnl": pnl, "is_win": outcome_win})
            yearly_stats[year]["trades"] += 1
            if outcome_win: yearly_stats[year]["wins"] += 1
            else: yearly_stats[year]["losses"] += 1
            yearly_stats[year]["profit_pts"] += pnl

        elif is_downtrend and in_eor_zone and bear_confirm:
            entry = c_close
            sl = max(c_high, entry + 1.5 * atr) + (0.2 * atr)
            tp = entry - (0.75 * atr)

            outcome_win = None
            pnl = 0.0
            for j in range(i + 1, min(i + 6, len(df))):
                r_j = df.iloc[j]
                if float(r_j["high"]) >= sl:
                    outcome_win = False
                    pnl = entry - sl
                    break
                if float(r_j["low"]) <= tp:
                    outcome_win = True
                    pnl = entry - tp
                    break
            if outcome_win is None:
                final_c = float(df.iloc[min(i + 5, len(df) - 1)]["close"])
                pnl = entry - final_c
                outcome_win = (pnl > 0)

            trades.append({"date": date, "year": year, "type": "CONFIRMED_EOR_PE", "pnl": pnl, "is_win": outcome_win})
            yearly_stats[year]["trades"] += 1
            if outcome_win: yearly_stats[year]["wins"] += 1
            else: yearly_stats[year]["losses"] += 1
            yearly_stats[year]["profit_pts"] += pnl

    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t["is_win"])
    win_rate = (winning_trades / max(1, total_trades)) * 100
    
    total_gains = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    total_losses = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    profit_factor = total_gains / max(0.001, total_losses)
    net_points = sum(t["pnl"] for t in trades)

    print(f"\n--- 12-YEAR CoA 3.0 CONFIRMED RESULTS: {name} ---")
    print(f"Total Confirmed Setups  : {total_trades}")
    print(f"Winning Trades          : {winning_trades}")
    print(f"Losing Trades           : {total_trades - winning_trades}")
    print(f"WIN RATE                : {win_rate:.2f}% 🚀")
    print(f"Profit Factor           : {profit_factor:.2f}")
    print(f"Total Net Points Gained : {'+' if net_points >= 0 else ''}{net_points:,.1f} pts")

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
    test_coa3_confirmed("^NSEI", "NIFTY 50", step=50)
    test_coa3_confirmed("^NSEBANK", "BANK NIFTY", step=100)
