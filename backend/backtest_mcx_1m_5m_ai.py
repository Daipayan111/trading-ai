"""
MCX Commodities 1-Minute & 5-Minute High-Frequency AI Backtesting & Optimization Engine
========================================================================================
Tests CRUDEOIL, GOLD, SILVER, NATURALGAS, COPPER on:
- 5-Minute (60-day intraday horizon)
- 1-Minute (7-day micro scalping horizon)

Compares:
1. Standard Static Model (Traditional EMA + RSI + 0.9x ATR SL)
2. Upgraded AI Scalp Engine (Displacement + FVG + Anti-Hunt Buffer + Dynamic Runner)
"""

import sys, os
import math
import json
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytics import fetch_stock_data, add_technical_indicators

MCX_SPECS = {
    "CRUDEOIL":   {"lot_size": 100,  "step": 50,  "tick": 1.0,  "unit": "bbl", "name": "Crude Oil"},
    "GOLD":       {"lot_size": 1,    "step": 100, "tick": 1.0,  "unit": "10g", "name": "Gold"},
    "SILVER":     {"lot_size": 30,   "step": 500, "tick": 1.0,  "unit": "1kg", "name": "Silver"},
    "NATURALGAS": {"lot_size": 1250, "step": 5,   "tick": 0.1,  "unit": "mmBtu", "name": "Natural Gas"},
    "COPPER":     {"lot_size": 2500, "step": 5,   "tick": 0.05, "unit": "1kg", "name": "Copper"},
}

def run_micro_backtest(symbol: str, interval: str = "5m", use_ai_upgrade: bool = True):
    spec = MCX_SPECS.get(symbol, {"lot_size": 100, "step": 50, "tick": 1.0, "unit": "unit", "name": symbol})
    lot_size = spec["lot_size"]
    period = "7d" if interval == "1m" else "60d"

    df, _, _ = fetch_stock_data(symbol, period=period, interval=interval)
    if df.empty or len(df) < 60:
        return {"error": f"Insufficient {interval} data for {symbol}"}

    df = add_technical_indicators(df)
    
    # ATR & Micro Trend indicators
    high = df['High']
    low = df['Low']
    close = df['Close']
    open_ = df['Open']
    
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    df['ATR'] = tr.rolling(14).mean().bfill()
    df['EMA9'] = close.ewm(span=9).mean()
    df['EMA21'] = close.ewm(span=21).mean()
    df['EMA50'] = close.ewm(span=50).mean()
    
    # AI Feature Engineering: Body Ratio & Displacement
    spread = (high - low).replace(0, 1e-6)
    df['Body_Ratio'] = abs(close - open_) / spread
    vol = df['Volume'] if 'Volume' in df.columns else pd.Series(1, index=df.index)
    vol_mean = vol.rolling(20, min_periods=5).mean().replace(0, 1)
    df['Vol_Surge'] = vol / vol_mean

    trades = []
    in_trade = False
    trade = {}
    lookahead_bars = 8

    # Multipliers
    if not use_ai_upgrade:
        # Standard retail model: tight static SL (0.9x ATR)
        sl_mult = 0.90
        tp1_mult = 1.25
        tp2_mult = 2.00
    else:
        # Upgraded AI model: Anti-hunt safe buffer + dynamic runner
        sl_mult = 1.45   # Beyond micro-liquidity sweeps
        tp1_mult = 1.35  # High-conviction milestone (50% book)
        tp2_mult = 2.80  # Runner target (50% runner)

    start_idx = 50
    end_idx = len(df) - lookahead_bars

    for i in range(start_idx, end_idx):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        curr_p = float(row['Close'])
        atr = float(row['ATR'])
        rsi = float(row.get('RSI', 50))
        ema9 = float(row['EMA9'])
        ema21 = float(row['EMA21'])
        ema50 = float(row['EMA50'])
        body_ratio = float(row['Body_Ratio'])
        vol_surge = float(row['Vol_Surge'])

        if not in_trade:
            if not use_ai_upgrade:
                # Naive setup: simple EMA cross
                buy_cond = (ema9 > ema21) and (prev['Close'] <= prev['EMA9']) and (curr_p > ema9) and (rsi > 50)
                sell_cond = (ema9 < ema21) and (prev['Close'] >= prev['EMA9']) and (curr_p < ema9) and (rsi < 50)
            else:
                # Upgraded AI setup:
                # 1. Trend alignment (EMA9 > EMA21 > EMA50 for Buy)
                # 2. Institutional displacement (body > 60% of candle)
                # 3. Volume surge > 1.15x
                # 4. RSI not exhausted (45 < RSI < 68)
                buy_cond = (ema9 > ema21 and curr_p > ema21) and (body_ratio >= 0.58) and (vol_surge >= 1.12) and (46 <= rsi <= 68)
                sell_cond = (ema9 < ema21 and curr_p < ema21) and (body_ratio >= 0.58) and (vol_surge >= 1.12) and (32 <= rsi <= 54)

            if buy_cond:
                in_trade = True
                trade = {
                    "side": "BUY",
                    "entry_idx": i,
                    "entry_time": df.index[i],
                    "entry_price": curr_p,
                    "atr": atr,
                    "sl": curr_p - (sl_mult * atr),
                    "tp1": curr_p + (tp1_mult * atr),
                    "tp2": curr_p + (tp2_mult * atr),
                    "tp1_hit": False,
                }
            elif sell_cond:
                in_trade = True
                trade = {
                    "side": "SELL",
                    "entry_idx": i,
                    "entry_time": df.index[i],
                    "entry_price": curr_p,
                    "atr": atr,
                    "sl": curr_p + (sl_mult * atr),
                    "tp1": curr_p - (tp1_mult * atr),
                    "tp2": curr_p - (tp2_mult * atr),
                    "tp1_hit": False,
                }
        else:
            bar_h = float(row['High'])
            bar_l = float(row['Low'])
            is_buy = trade["side"] == "BUY"
            bars_held = i - trade["entry_idx"]

            # Check SL
            hit_sl = (bar_l <= trade["sl"]) if is_buy else (bar_h >= trade["sl"])
            hit_tp1 = (bar_h >= trade["tp1"]) if is_buy else (bar_l <= trade["tp1"])
            hit_tp2 = (bar_h >= trade["tp2"]) if is_buy else (bar_l <= trade["tp2"])

            # In upgraded model, when TP1 is hit, move SL to Breakeven
            if use_ai_upgrade and hit_tp1 and not trade["tp1_hit"]:
                trade["tp1_hit"] = True
                trade["sl"] = trade["entry_price"] # Breakeven stop

            exit_reason = None
            exit_price = None

            if hit_sl:
                exit_reason = "STOP_LOSS" if not trade.get("tp1_hit") else "BREAKEVEN_PROTECT"
                exit_price = trade["sl"]
            elif hit_tp2:
                exit_reason = "TARGET_2_RUNNER"
                exit_price = trade["tp2"]
            elif hit_tp1 and not use_ai_upgrade:
                exit_reason = "TARGET_1"
                exit_price = trade["tp1"]
            elif bars_held >= (15 if interval == "1m" else 24):
                exit_reason = "TIME_EXIT"
                exit_price = curr_p

            if exit_reason:
                # Post-exit analysis over next 8 bars
                subsequent = df.iloc[i+1 : i+1+lookahead_bars]
                if is_buy:
                    post_max = float(subsequent['High'].max()) if not subsequent.empty else exit_price
                    post_min = float(subsequent['Low'].min()) if not subsequent.empty else exit_price
                    pts_pnl = exit_price - trade["entry_price"]
                    post_run = max(0.0, post_max - exit_price)
                else:
                    post_max = float(subsequent['High'].max()) if not subsequent.empty else exit_price
                    post_min = float(subsequent['Low'].min()) if not subsequent.empty else exit_price
                    pts_pnl = trade["entry_price"] - exit_price
                    post_run = max(0.0, exit_price - post_min)

                # SL Trap check: did market reverse back into profit?
                sl_was_trap = False
                if exit_reason == "STOP_LOSS":
                    if is_buy and post_max > trade["entry_price"]:
                        sl_was_trap = True
                    elif not is_buy and post_min < trade["entry_price"]:
                        sl_was_trap = True

                # Partial allocation accounting for upgraded model
                if use_ai_upgrade and trade.get("tp1_hit"):
                    # 50% booked at TP1, 50% exited at exit_price
                    tp1_pts = (trade["tp1"] - trade["entry_price"]) if is_buy else (trade["entry_price"] - trade["tp1"])
                    effective_pts = (0.50 * tp1_pts) + (0.50 * pts_pnl)
                else:
                    effective_pts = pts_pnl

                trades.append({
                    "symbol": symbol,
                    "side": trade["side"],
                    "pts_pnl": round(effective_pts, 2),
                    "inr_pnl": round(effective_pts * lot_size, 2),
                    "exit_reason": exit_reason,
                    "sl_was_trap": sl_was_trap,
                    "post_run_atr": round(post_run / (trade["atr"] + 1e-6), 2),
                })
                in_trade = False
                trade = {}

    if not trades:
        return {"error": "No trades"}

    tdf = pd.DataFrame(trades)
    tot = len(tdf)
    win_df = tdf[tdf['pts_pnl'] > 0]
    loss_df = tdf[tdf['pts_pnl'] <= 0]
    win_rate = round((len(win_df) / tot) * 100, 1)

    tot_inr = round(float(tdf['inr_pnl'].sum()), 2)
    win_inr = float(win_df['inr_pnl'].sum()) if not win_df.empty else 0.0
    loss_inr = abs(float(loss_df['inr_pnl'].sum())) if not loss_df.empty else 1.0
    pf = round(win_inr / max(1.0, loss_inr), 2)

    sl_trades = tdf[tdf['exit_reason'] == "STOP_LOSS"]
    sl_trap_cnt = int(sl_trades['sl_was_trap'].sum()) if not sl_trades.empty else 0
    sl_trap_pct = round((sl_trap_cnt / max(1, len(sl_trades))) * 100, 1) if not sl_trades.empty else 0.0

    return {
        "symbol": symbol,
        "name": spec["name"],
        "interval": interval,
        "model": "AI_UPGRADED" if use_ai_upgrade else "BASELINE_STATIC",
        "total_trades": tot,
        "win_rate_pct": win_rate,
        "total_inr": tot_inr,
        "profit_factor": pf,
        "sl_trap_pct": sl_trap_pct,
        "avg_post_tp_run_atr": round(float(tdf['post_run_atr'].mean()), 2),
    }

if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING 1-MINUTE & 5-MINUTE MCX COMMODITIES AI TEST & OPTIMIZATION")
    print("=" * 80)

    symbols = ["CRUDEOIL", "GOLD", "SILVER", "NATURALGAS", "COPPER"]
    intervals = ["5m", "1m"]

    comparison = []

    for sym in symbols:
        print(f"\n--- {sym} ---")
        for intv in intervals:
            base = run_micro_backtest(sym, interval=intv, use_ai_upgrade=False)
            upg = run_micro_backtest(sym, interval=intv, use_ai_upgrade=True)

            if "error" not in base and "error" not in upg:
                print(f"[{intv.upper()}] Baseline Static:  Trades={base['total_trades']}, WinRate={base['win_rate_pct']}%, PnL=₹{base['total_inr']:,.2f}, PF={base['profit_factor']}, SL-Trap={base['sl_trap_pct']}%")
                print(f"[{intv.upper()}] AI Upgraded:      Trades={upg['total_trades']}, WinRate={upg['win_rate_pct']}%, PnL=₹{upg['total_inr']:,.2f}, PF={upg['profit_factor']}, SL-Trap={upg['sl_trap_pct']}%")
                comparison.append({"symbol": sym, "interval": intv, "baseline": base, "upgraded": upg})
            else:
                print(f"[{intv.upper()}] Error: {base.get('error') or upg.get('error')}")

    out_file = os.path.join(os.path.dirname(__file__), "mcx_micro_backtest_results.json")
    with open(out_file, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[DONE] Saved micro backtest comparison to: {out_file}")

