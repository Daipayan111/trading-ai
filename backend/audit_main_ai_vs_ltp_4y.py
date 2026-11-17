"""
4-Year Historical Data A/B Audit:
  Engine A (BEFORE): Previous Main AI (Analytics + Technical Confluence)
  Engine B (AFTER):  Main AI + LTP Calculator Fusion (EOS/EOR Gamma Walls + CoA 1.0 Gate + EOS/EOR Precision Dip/Reject)

Zero hindsight, raw forward OHLCV candle simulation (no calibration JSON overrides).
Note: For daily (1d) stocks, applies the 0.8% percentage floor; for intraday (1h/15m/5m),
uses pure ATR multipliers (0.80x ATR TP1, 1.80x ATR TP2, 3.20x ATR TP3, 1.35x ATR SL)
for both Engine A and Engine B identically (100% apples-to-apples comparison).
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytics import fetch_stock_data, add_technical_indicators, generate_signals
from analytics_ltp_calculator import estimate_bar_ltp_context

TP1_MULT = 0.80
TP2_MULT = 1.80
TP3_MULT = 3.20
SL_MULT = 1.35
MAX_HOLD = 20
COOLDOWN = 3


def simulate_engine(df: pd.DataFrame, symbol: str, interval: str, mode: str = "OLD") -> dict:
    """
    mode:
      'OLD'     -> Previous Main AI
      'LTP_GATE'-> Main AI + LTP CoA & EOS/EOR Anti-Trap Gate (no extra bypass entries)
      'LTP_FULL'-> Main AI + LTP Gate + Strict-Vetted EOS/EOR Pullback Triggers (all 6 Main AI filters enforced)
    """
    use_ltp_fusion = mode in ("LTP_GATE", "LTP_FULL")
    allow_ltp_pullback = mode == "LTP_FULL"
    trades = []
    filtered_by_ltp = 0
    ltp_added = 0
    i = 60
    n = len(df)
    is_daily = interval == "1d"

    while i < n - 5:
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        signal = str(row.get("Signal", "HOLD"))
        is_buy = "BUY" in signal
        is_sell = "SELL" in signal

        close = float(row["Close"])
        open_p = float(row["Open"])
        high = float(row["High"])
        low = float(row["Low"])
        sma200 = float(row.get("SMA_200", close))
        ema50 = float(row.get("EMA_50", close))
        ema21 = float(row.get("EMA_21", close))
        adx = float(row.get("ADX", 25))
        bb_pct = float(row.get("BB_Percent", 0.5))
        atr = float(row.get("ATR", close * 0.015))
        if pd.isna(atr) or atr <= 0:
            atr = close * 0.015
        rsi = float(row.get("RSI", 50))
        macd_h = float(row.get("MACD_Hist", 0))
        prev_macd_h = float(prev.get("MACD_Hist", 0))
        vol_ratio = float(row.get("Volume_Ratio", 1.0))

        ltp_ctx = None
        if use_ltp_fusion:
            ltp_ctx = estimate_bar_ltp_context(
                close=close,
                ema21=ema21,
                ema50=ema50,
                atr=atr,
                rsi=rsi,
                vol_ratio=vol_ratio,
                high=high,
                low=low,
                macd_hist=macd_h,
                prev_macd_hist=prev_macd_h,
                adx=adx,
                symbol=symbol,
            )

        if allow_ltp_pullback and not (is_buy or is_sell) and ltp_ctx and not ltp_ctx["is_confusion"]:
            if (
                close > open_p
                and 0.15 <= ltp_ctx["dist_to_eos_atr"] <= 0.65
                and ltp_ctx["market_trend"] in ("BULLISH", "EXTREME_BULLISH", "RANGEBOUND")
            ):
                is_buy = True
                ltp_added += 1
            elif (
                close < open_p
                and 0.15 <= ltp_ctx["dist_to_eor_atr"] <= 0.65
                and ltp_ctx["market_trend"] in ("BEARISH", "BEARISH_TOP", "BLOODBATH_BEARISH")
            ):
                is_sell = True
                ltp_added += 1

        if not (is_buy or is_sell):
            i += 1
            continue

        # ── Baseline Main AI 6 Confluence Filters (ALWAYS enforced — never bypassed!) ──
        if is_buy and (close < sma200 or close < ema50):
            i += 1
            continue
        if is_sell and (close > sma200 or close > ema50):
            i += 1
            continue
        if adx < 18:
            i += 1
            continue
        if (0.40 <= bb_pct <= 0.60) and (adx < 22):
            i += 1
            continue
        if is_buy and rsi > 70:
            i += 1
            continue
        if is_sell and rsi < 30:
            i += 1
            continue
        if is_buy and macd_h < prev_macd_h:
            i += 1
            continue
        if is_sell and macd_h > prev_macd_h:
            i += 1
            continue
        if vol_ratio < 0.95:
            i += 1
            continue

        # ── Engine B Anti-Trap LTP CoA & Wall Gate ──
        if use_ltp_fusion and ltp_ctx:
            if ltp_ctx["is_confusion"]:
                filtered_by_ltp += 1
                i += 1
                continue
            if is_buy and (ltp_ctx["near_eor_resistance"] or ltp_ctx["market_trend"] in ("BEARISH", "BLOODBATH_BEARISH")):
                filtered_by_ltp += 1
                i += 1
                continue
            if is_sell and (ltp_ctx["near_eos_support"] or ltp_ctx["market_trend"] in ("BULLISH", "EXTREME_BULLISH")):
                filtered_by_ltp += 1
                i += 1
                continue

        # Identical Target/SL framing for apples-to-apples comparison
        if is_daily:
            pct_floor_tp1 = max(atr * TP1_MULT, close * 0.009 if close < 300 else close * 0.008)
            pct_floor_tp2 = max(atr * TP2_MULT, close * 0.022 if close < 300 else close * 0.015)
            pct_floor_tp3 = max(atr * TP3_MULT, close * 0.038 if close < 300 else close * 0.025)
            sl_dist = max(atr * SL_MULT, close * 0.012 if close < 300 else close * 0.006)
        else:
            pct_floor_tp1 = atr * TP1_MULT
            pct_floor_tp2 = atr * TP2_MULT
            pct_floor_tp3 = atr * TP3_MULT
            sl_dist = atr * SL_MULT

        if is_buy:
            sl = close - sl_dist
            tp1 = close + pct_floor_tp1
            tp2 = close + pct_floor_tp2
            tp3 = close + pct_floor_tp3
        else:
            sl = close + sl_dist
            tp1 = close - pct_floor_tp1
            tp2 = close - pct_floor_tp2
            tp3 = close - pct_floor_tp3

        hit_tp1 = hit_tp2 = hit_tp3 = hit_sl = False
        bars_held = 0
        sl_trail = sl
        t1_exit = t2_exit = t3_exit = None
        fc = close

        for j in range(i + 1, min(i + MAX_HOLD, n)):
            fut = df.iloc[j]
            fh = float(fut["High"])
            fl = float(fut["Low"])
            fc = float(fut["Close"])
            bars_held = j - i

            if is_buy:
                sl_triggered = (fc <= sl_trail) or (fl <= (sl_trail - atr * 0.35))
                if sl_triggered and not hit_tp1:
                    hit_sl = True
                    t1_exit = t2_exit = t3_exit = sl_trail
                    break
                if fh >= tp1 and not hit_tp1:
                    hit_tp1 = True
                    t1_exit = tp1
                    sl_trail = close
                if hit_tp1 and fh >= tp2 and not hit_tp2:
                    hit_tp2 = True
                    t2_exit = tp2
                    sl_trail = tp1
                if hit_tp2 and fh >= tp3 and not hit_tp3:
                    hit_tp3 = True
                    t3_exit = tp3
                    break
                if hit_tp1 and fl <= sl_trail:
                    if t2_exit is None:
                        t2_exit = sl_trail
                    if t3_exit is None:
                        t3_exit = sl_trail
                    break
            else:
                sl_triggered = (fc >= sl_trail) or (fh >= (sl_trail + atr * 0.35))
                if sl_triggered and not hit_tp1:
                    hit_sl = True
                    t1_exit = t2_exit = t3_exit = sl_trail
                    break
                if fl <= tp1 and not hit_tp1:
                    hit_tp1 = True
                    t1_exit = tp1
                    sl_trail = close
                if hit_tp1 and fl <= tp2 and not hit_tp2:
                    hit_tp2 = True
                    t2_exit = tp2
                    sl_trail = tp1
                if hit_tp2 and fl <= tp3 and not hit_tp3:
                    hit_tp3 = True
                    t3_exit = tp3
                    break
                if hit_tp1 and fh >= sl_trail:
                    if t2_exit is None:
                        t2_exit = sl_trail
                    if t3_exit is None:
                        t3_exit = sl_trail
                    break

        if t1_exit is None:
            t1_exit = fc
        if t2_exit is None:
            t2_exit = fc
        if t3_exit is None:
            t3_exit = fc

        if is_buy:
            captured_pts = 0.50 * (t1_exit - close) + 0.30 * (t2_exit - close) + 0.20 * (t3_exit - close)
        else:
            captured_pts = 0.50 * (close - t1_exit) + 0.30 * (close - t2_exit) + 0.20 * (close - t3_exit)

        trades.append({
            "hit_tp1": hit_tp1,
            "hit_tp2": hit_tp2,
            "hit_sl": hit_sl,
            "captured_pts": captured_pts,
        })
        i += max(COOLDOWN, bars_held)

    total = len(trades)
    if total == 0:
        return {"trades": 0, "tp1_pct": 0.0, "sl_pct": 0.0, "win_pct": 0.0, "pf": 0.0, "net_pts": 0.0, "filtered": filtered_by_ltp, "added": ltp_added}

    tp1_hits = sum(1 for t in trades if t["hit_tp1"])
    sl_hits = sum(1 for t in trades if t["hit_sl"])
    wins = sum(1 for t in trades if t["captured_pts"] > 0)
    gross_win = sum(t["captured_pts"] for t in trades if t["captured_pts"] > 0)
    gross_loss = abs(sum(t["captured_pts"] for t in trades if t["captured_pts"] < 0))
    pf = round(gross_win / max(0.01, gross_loss), 2)
    net_pts = round(sum(t["captured_pts"] for t in trades), 2)

    return {
        "trades": total,
        "tp1_pct": round(tp1_hits / total * 100, 2),
        "sl_pct": round(sl_hits / total * 100, 2),
        "win_pct": round(wins / total * 100, 2),
        "pf": pf,
        "net_pts": net_pts,
        "filtered": filtered_by_ltp,
        "added": ltp_added,
    }


def main():
    tests = [
        ("NIFTY50", "4y", "1d"),
        ("BANKNIFTY", "4y", "1d"),
        ("RELIANCE", "4y", "1d"),
        ("HDFCBANK", "4y", "1d"),
        ("SBIN", "4y", "1d"),
        ("TATASTEEL", "4y", "1d"),
        ("NIFTY50", "2y", "1h"),
        ("BANKNIFTY", "2y", "1h"),
        ("NIFTY50", "60d", "15m"),
        ("BANKNIFTY", "60d", "15m"),
        ("NIFTY50", "60d", "5m"),
        ("BANKNIFTY", "60d", "5m"),
    ]

    print("=" * 112)
    print("4-YEAR HISTORICAL A/B BACKTEST: PREVIOUS MAIN AI (BEFORE) vs. MAIN AI + LTP CALCULATOR FUSION (AFTER)")
    print("=" * 112)
    print(f"{'SYMBOL':10} {'TF':4} {'PER':4} | {'OLD TR':6} {'OLD TP1%':8} {'OLD SL%':7} {'OLD PF':6} {'OLD PTS':9} | {'NEW TR':6} {'NEW TP1%':8} {'NEW SL%':7} {'NEW PF':6} {'NEW PTS':9} | {'TP1 DIFF':8}")
    print("-" * 112)

    old_total_trades = old_tp1_sum = old_sl_sum = old_win_sum = 0
    new_total_trades = new_tp1_sum = new_sl_sum = new_win_sum = 0
    old_total_pts = 0.0
    new_total_pts = 0.0

    for sym, per, tf in tests:
        df = None
        used_sym = sym
        for attempt in range(2):
            try:
                df, used_sym, _ = fetch_stock_data(sym, period=per, interval=tf)
                if df is not None and not df.empty:
                    break
            except Exception as err:
                if attempt == 0:
                    import time
                    time.sleep(2)
                else:
                    print(f"{sym:10} {tf:4} {per:4} | FETCH ERROR: {err}")
                    df = None
        if df is None or df.empty or len(df) < 80:
            print(f"{sym:10} {tf:4} {per:4} | INSUFFICIENT DATA")
            continue
        df = add_technical_indicators(df)
        sigs = generate_signals(df)
        for col in sigs.columns:
            df[col] = sigs[col]

        old_res = simulate_engine(df, used_sym, tf, mode="OLD")
        gate_res = simulate_engine(df, used_sym, tf, mode="LTP_GATE")
        new_res = simulate_engine(df, used_sym, tf, mode="LTP_FULL")

        tp1_diff = round(new_res["tp1_pct"] - old_res["tp1_pct"], 2)
        gate_diff = round(gate_res["tp1_pct"] - old_res["tp1_pct"], 2)
        diff_str = f"{'+' if tp1_diff >= 0 else ''}{tp1_diff}% (G:{'+' if gate_diff >= 0 else ''}{gate_diff}%)"
        print(
            f"{sym:10} {tf:4} {per:4} | "
            f"{old_res['trades']:6d} {old_res['tp1_pct']:7.2f}% {old_res['sl_pct']:6.2f}% {old_res['pf']:6.2f} {old_res['net_pts']:9.1f} | "
            f"{new_res['trades']:6d} {new_res['tp1_pct']:7.2f}% {new_res['sl_pct']:6.2f}% {new_res['pf']:6.2f} {new_res['net_pts']:9.1f} | "
            f"{diff_str:8}"
        )

        old_total_trades += old_res["trades"]
        old_tp1_sum += old_res["tp1_pct"] * old_res["trades"] / 100.0
        old_sl_sum += old_res["sl_pct"] * old_res["trades"] / 100.0
        old_win_sum += old_res["win_pct"] * old_res["trades"] / 100.0
        old_total_pts += old_res["net_pts"]

        new_total_trades += new_res["trades"]
        new_tp1_sum += new_res["tp1_pct"] * new_res["trades"] / 100.0
        new_sl_sum += new_res["sl_pct"] * new_res["trades"] / 100.0
        new_win_sum += new_res["win_pct"] * new_res["trades"] / 100.0
        new_total_pts += new_res["net_pts"]

    print("=" * 112)
    old_agg_tp1 = round(old_tp1_sum / max(1, old_total_trades) * 100, 2)
    old_agg_sl = round(old_sl_sum / max(1, old_total_trades) * 100, 2)
    old_agg_win = round(old_win_sum / max(1, old_total_trades) * 100, 2)
    new_agg_tp1 = round(new_tp1_sum / max(1, new_total_trades) * 100, 2)
    new_agg_sl = round(new_sl_sum / max(1, new_total_trades) * 100, 2)
    new_agg_win = round(new_win_sum / max(1, new_total_trades) * 100, 2)
    print(f"AGGREGATE OLD MAIN AI : Trades={old_total_trades}, TP1 Hit={old_agg_tp1}%, WinRate={old_agg_win}%, SL Hit={old_agg_sl}%, NetPts={round(old_total_pts, 1)}")
    print(f"AGGREGATE NEW + LTP AI: Trades={new_total_trades}, TP1 Hit={new_agg_tp1}%, WinRate={new_agg_win}%, SL Hit={new_agg_sl}%, NetPts={round(new_total_pts, 1)}")
    print(f"NET IMPROVEMENT       : TP1 {'+' if new_agg_tp1 >= old_agg_tp1 else ''}{round(new_agg_tp1 - old_agg_tp1, 2)}% | WinRate {'+' if new_agg_win >= old_agg_win else ''}{round(new_agg_win - old_agg_win, 2)}% | SL {'+' if new_agg_sl >= old_agg_sl else ''}{round(new_agg_sl - old_agg_sl, 2)}% | NetPts {'+' if new_total_pts >= old_total_pts else ''}{round(new_total_pts - old_total_pts, 1)}")


if __name__ == "__main__":
    main()
