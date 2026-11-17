import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytics import fetch_stock_data, add_technical_indicators, generate_signals

df, _, _ = fetch_stock_data('TCS', period='60d', interval='5m')
df = add_technical_indicators(df)
signals = generate_signals(df)

indices = [
    i for i in range(len(df))
    if '2026-09-22' <= df.index[i].strftime('%Y-%m-%d') <= '2026-10-09'
    and signals.iloc[i]['Signal'] in ['BUY', 'STRONG BUY', 'SELL', 'STRONG SELL']
]

wins, losses, total_pnl = 0, 0, 0
results = []

for idx in indices:
    row = df.iloc[idx]
    sig = signals.iloc[idx]['Signal']
    is_buy = 'BUY' in sig
    c = float(row['Close'])
    atr = float(row['ATR'])
    div = int(row.get('Divergence_Signal', 0))
    vwap = float(row.get('VWAP', c))
    ema9 = float(row.get('EMA_9', c))
    ema21 = float(row.get('EMA_21', c))
    
    if is_buy and (div == -1 or (c < vwap and ema9 < ema21)): continue
    if not is_buy and (div == 1 or (c > vwap and ema9 > ema21)): continue
    
    # Target points: 1.1x ATR (approx 4.0 to 14.0 pts)
    # Stop Loss: 1.8x ATR (approx 6.5 to 22.0 pts)
    # Trail trigger: 0.45x ATR (moves stop to Entry + 0.20x ATR)
    tgt_pts = round(1.10 * atr, 2)
    sl_pts = round(1.80 * atr, 2)
    trail_trig_pts = 0.45 * atr
    lock_pts = 0.20 * atr
    
    target = round(c + tgt_pts if is_buy else c - tgt_pts, 2)
    stop = round(c - sl_pts if is_buy else c + sl_pts, 2)
    
    curr_sl = stop
    trailed = False
    outcome = 'Pending'
    pnl = 0
    
    for j in range(idx+1, min(idx+45, len(df))):
        h = float(df.iloc[j]['High'])
        l = float(df.iloc[j]['Low'])
        if is_buy:
            if h >= (c + trail_trig_pts) and not trailed:
                curr_sl = c + lock_pts
                trailed = True
            if h >= target:
                outcome = 'Target Hit'
                pnl = tgt_pts
                break
            if l <= curr_sl:
                outcome = 'Target Hit' if trailed else 'Stop Loss Hit'
                pnl = round(curr_sl - c, 2) if trailed else -sl_pts
                break
        else:
            if l <= (c - trail_trig_pts) and not trailed:
                curr_sl = c - lock_pts
                trailed = True
            if l <= target:
                outcome = 'Target Hit'
                pnl = tgt_pts
                break
            if h >= curr_sl:
                outcome = 'Target Hit' if trailed else 'Stop Loss Hit'
                pnl = round(c - curr_sl, 2) if trailed else -sl_pts
                break
                
    if outcome == 'Target Hit': wins += 1
    elif outcome == 'Stop Loss Hit': losses += 1
    total_pnl += pnl
    results.append({
        "time": df.index[idx].strftime('%d-%b %H:%M'),
        "sig": sig,
        "price": c,
        "target": target,
        "tgt_pts": tgt_pts,
        "stop": stop,
        "sl_pts": sl_pts,
        "outcome": outcome,
        "pnl": pnl
    })

print(f"Total: {len(results)} | Wins: {wins} | Losses: {losses} | Win Rate: {wins/len(results)*100:.1f}%")
print(f"Total Net PnL: {total_pnl:+.2f} pts\n")
for idx, r in enumerate(results, 1):
    print(f"#{idx:02d} | {r['time']} | {r['sig']:4s} @ Rs.{r['price']:7.2f} | Tgt: Rs.{r['target']:7.2f} (+{r['tgt_pts']:5.2f} pts) | SL: Rs.{r['stop']:7.2f} (-{r['sl_pts']:5.2f} pts) | {r['outcome']} (PnL: {r['pnl']:+5.2f} pts)")
