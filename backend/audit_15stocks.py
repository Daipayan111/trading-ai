import sys, os
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytics import fetch_stock_data, add_technical_indicators, generate_signals

stocks_15 = [
    'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK',
    'SBIN', 'BHARTIARTL', 'ITC', 'LT', 'KOTAKBANK',
    'AXISBANK', 'HINDUNILVR', 'MARUTI', 'SUNPHARMA', 'BAJFINANCE'
]

def audit_stock(sym):
    df, used_symbol, ticker = fetch_stock_data(sym, period="4y", interval="1d")
    if df.empty or len(df) < 80:
        return {"symbol": sym, "error": "Insufficient data"}

    df = add_technical_indicators(df)
    signals = generate_signals(df)
    for col in signals.columns:
        df[col] = signals[col]

    # Institutional Target & SL specifications
    tp1_mult = 0.65
    tp2_mult = 1.50
    tp3_mult = 2.80
    sl_mult  = 1.60
    wick_cushion = 0.35

    trades = []
    i = 60
    n = len(df)

    while i < n - 5:
        row  = df.iloc[i]
        prev = df.iloc[i - 1]
        signal = str(row.get('Signal', 'HOLD'))
        is_buy  = 'BUY' in signal
        is_sell = 'SELL' in signal
        if not (is_buy or is_sell):
            i += 1
            continue

        close  = float(row['Close'])
        sma200 = float(row.get('SMA_200', close))
        ema50  = float(row.get('EMA_50', close))
        adx    = float(row.get('ADX', 25))
        bb_pct = float(row.get('BB_Percent', 0.5))
        atr    = float(row.get('ATR', close * 0.015))
        rsi    = float(row.get('RSI', 50))
        macd_h = float(row.get('MACD_Hist', 0))
        prev_macd_h = float(prev.get('MACD_Hist', 0))

        # Confluence Filters
        if is_buy and (close < sma200 or close < ema50):
            i += 1; continue
        if is_sell and (close > sma200 or close > ema50):
            i += 1; continue

        # Chop & Range Suppression
        min_adx = 18
        if adx < min_adx:
            i += 1; continue
        if (0.42 <= bb_pct <= 0.58) and (adx < 22):
            i += 1; continue

        if is_buy and rsi > 68: i += 1; continue
        if is_sell and rsi < 32: i += 1; continue

        if is_buy and macd_h < prev_macd_h: i += 1; continue
        if is_sell and macd_h > prev_macd_h: i += 1; continue

        entry_price = close
        entry_date  = df.index[i].strftime("%Y-%m-%d")
        year        = df.index[i].year

        if is_buy:
            sl  = entry_price - atr * sl_mult
            tp1 = entry_price + atr * tp1_mult
            tp2 = entry_price + atr * tp2_mult
            tp3 = entry_price + atr * tp3_mult
        else:
            sl  = entry_price + atr * sl_mult
            tp1 = entry_price - atr * tp1_mult
            tp2 = entry_price - atr * tp2_mult
            tp3 = entry_price - atr * tp3_mult

        hit_tp1 = hit_tp2 = hit_tp3 = hit_sl = False
        sl_trail = sl
        exit_date = entry_date
        exit_price = entry_price
        post_tp1_run_pts = 0.0
        post_sl_drop_pts = 0.0
        stop_hunt_reversal = False

        for j in range(i + 1, min(i + 15, n)):
            fut = df.iloc[j]
            fh  = float(fut['High'])
            fl  = float(fut['Low'])
            fc  = float(fut['Close'])
            exit_date = df.index[j].strftime("%Y-%m-%d")

            if is_buy:
                # Institutional Liquidity Wick Shield
                if not hit_tp1:
                    sl_triggered = (fc <= sl_trail) or (fl <= (sl_trail - atr * wick_cushion))
                    if sl_triggered:
                        hit_sl = True
                        exit_price = sl_trail
                        # Check stop hunt reversal
                        subseq = df.iloc[j+1 : min(j+15, n)]
                        if len(subseq) > 0:
                            post_sl_drop_pts = max(0.0, sl_trail - float(subseq['Low'].min()))
                            if float(subseq['High'].max()) >= tp1:
                                stop_hunt_reversal = True
                        break

                if fh >= tp1 and not hit_tp1:
                    hit_tp1 = True
                    exit_price = tp1
                    sl_trail = entry_price  # Move to breakeven
                    subseq = df.iloc[j : min(j+15, n)]
                    if len(subseq) > 0:
                        post_tp1_run_pts = max(0.0, float(subseq['High'].max()) - tp1)

                if hit_tp1 and fh >= tp2 and not hit_tp2:
                    hit_tp2 = True
                    exit_price = tp2
                    sl_trail = tp1          # Lock in TP1 profit

                if hit_tp2 and fh >= tp3 and not hit_tp3:
                    hit_tp3 = True
                    exit_price = tp3
                    break

                if hit_tp1 and fl <= sl_trail:
                    exit_price = sl_trail
                    break

            else:
                if not hit_tp1:
                    sl_triggered = (fc >= sl_trail) or (fh >= (sl_trail + atr * wick_cushion))
                    if sl_triggered:
                        hit_sl = True
                        exit_price = sl_trail
                        subseq = df.iloc[j+1 : min(j+15, n)]
                        if len(subseq) > 0:
                            post_sl_drop_pts = max(0.0, float(subseq['High'].max()) - sl_trail)
                            if float(subseq['Low'].min()) <= tp1:
                                stop_hunt_reversal = True
                        break

                if fl <= tp1 and not hit_tp1:
                    hit_tp1 = True
                    exit_price = tp1
                    sl_trail = entry_price
                    subseq = df.iloc[j : min(j+15, n)]
                    if len(subseq) > 0:
                        post_tp1_run_pts = max(0.0, tp1 - float(subseq['Low'].min()))

                if hit_tp1 and fl <= tp2 and not hit_tp2:
                    hit_tp2 = True
                    exit_price = tp2
                    sl_trail = tp1

                if hit_tp2 and fl <= tp3 and not hit_tp3:
                    hit_tp3 = True
                    exit_price = tp3
                    break

                if hit_tp1 and fh >= sl_trail:
                    exit_price = sl_trail
                    break

        # 3-Tranche PnL capture (50% TP1, 30% TP2, 20% TP3/Trail)
        if hit_sl:
            pnl_pts = -(atr * sl_mult)
        elif hit_tp3:
            pnl_pts = 0.50 * (atr * tp1_mult) + 0.30 * (atr * tp2_mult) + 0.20 * (atr * tp3_mult)
        elif hit_tp2:
            pnl_pts = 0.50 * (atr * tp1_mult) + 0.30 * (atr * tp2_mult) + 0.20 * (atr * tp1_mult)
        elif hit_tp1:
            pnl_pts = 0.50 * (atr * tp1_mult) + 0.50 * 0.0 # remaining half at breakeven
        else:
            pnl_pts = 0.0

        pnl_pct = (pnl_pts / entry_price) * 100

        trades.append({
            "year": year,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "type": "BUY" if is_buy else "SELL",
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "pnl_pts": round(pnl_pts, 2),
            "pnl_pct": round(pnl_pct, 2),
            "hit_tp1": hit_tp1,
            "hit_tp2": hit_tp2,
            "hit_tp3": hit_tp3,
            "hit_sl": hit_sl,
            "post_tp1_run_pts": round(post_tp1_run_pts, 2),
            "post_tp1_run_atr": round(post_tp1_run_pts / max(1.0, atr), 2),
            "stop_hunt_reversal": stop_hunt_reversal
        })
        i += 3  # cooldown

    n_trades = len(trades)
    if n_trades == 0:
        return {"symbol": sym, "error": "No trades generated"}

    wins = sum(1 for t in trades if t['hit_tp1'])
    tp2s = sum(1 for t in trades if t['hit_tp2'])
    tp3s = sum(1 for t in trades if t['hit_tp3'])
    sls  = sum(1 for t in trades if t['hit_sl'])
    fakeouts = sum(1 for t in trades if t['stop_hunt_reversal'])

    gross_gains = sum(t['pnl_pts'] for t in trades if t['pnl_pts'] > 0)
    gross_losses = abs(sum(t['pnl_pts'] for t in trades if t['pnl_pts'] < 0))
    pf = round(gross_gains / gross_losses, 2) if gross_losses > 0 else 9.99

    runs = [t['post_tp1_run_pts'] for t in trades if t['hit_tp1']]
    run_atrs = [t['post_tp1_run_atr'] for t in trades if t['hit_tp1']]

    # Year-by-year consistency
    yearly = {}
    for t in trades:
        y = t['year']
        if y not in yearly:
            yearly[y] = {'trades': 0, 'wins': 0, 'sls': 0, 'pts': 0.0}
        yearly[y]['trades'] += 1
        if t['hit_tp1']: yearly[y]['wins'] += 1
        if t['hit_sl']:  yearly[y]['sls'] += 1
        yearly[y]['pts'] += t['pnl_pts']

    yearly_list = []
    for y in sorted(yearly.keys()):
        y_tr = yearly[y]['trades']
        yearly_list.append({
            'year': y,
            'trades': y_tr,
            'win_rate': round(yearly[y]['wins'] / y_tr * 100, 1) if y_tr > 0 else 0,
            'net_pts': round(yearly[y]['pts'], 1)
        })

    return {
        "symbol": sym,
        "company": sym,
        "total_trades": n_trades,
        "tp1_wins": wins,
        "tp1_win_rate": round(wins / n_trades * 100, 1),
        "tp2_wins": tp2s,
        "tp2_rate": round(tp2s / n_trades * 100, 1),
        "tp3_wins": tp3s,
        "tp3_rate": round(tp3s / n_trades * 100, 1),
        "sl_hits": sls,
        "sl_rate": round(sls / n_trades * 100, 1),
        "fakeout_sl_count": fakeouts,
        "fakeout_sl_pct": round(fakeouts / max(1, sls) * 100, 1),
        "profit_factor": pf,
        "cumulative_net_pts": round(sum(t['pnl_pts'] for t in trades), 1),
        "avg_run_past_tp1_pts": round(np.mean(runs), 1) if runs else 0.0,
        "avg_run_past_tp1_atr": round(np.mean(run_atrs), 2) if run_atrs else 0.0,
        "max_run_past_tp1_pts": round(max(runs), 1) if runs else 0.0,
        "yearly": yearly_list
    }

from concurrent.futures import ThreadPoolExecutor, as_completed

print("Starting 15-Stock 4-Year Institutional Audit in Parallel (5 workers)...")
sys.stdout.flush()
all_audits = []

with ThreadPoolExecutor(max_workers=5) as executor:
    future_to_sym = {executor.submit(audit_stock, sym): sym for sym in stocks_15}
    for future in as_completed(future_to_sym):
        sym = future_to_sym[future]
        try:
            res = future.result()
            if 'error' in res:
                print(f"FAILED {sym}: {res['error']}")
            else:
                all_audits.append(res)
                print(f"AUDITED: {sym:12} | Trades: {res['total_trades']:2d} | Win: {res['tp1_win_rate']:5.1f}% | SL: {res['sl_rate']:4.1f}% | PF: {res['profit_factor']:4.1f}x | Net: {res['cumulative_net_pts']:+7.1f}p | Run Past TP1: +{res['avg_run_past_tp1_pts']}p ({res['avg_run_past_tp1_atr']}x ATR)")
        except Exception as ex:
            print(f"EXCEPTION {sym}: {ex}")
        sys.stdout.flush()

all_audits.sort(key=lambda x: x.get('symbol', ''))
with open('audit_15stocks_4y.json', 'w') as f:
    json.dump(all_audits, f, indent=2)

print(f"\nAudit Complete! Successfully audited {len(all_audits)}/15 stocks. Saved to audit_15stocks_4y.json")
sys.stdout.flush()
