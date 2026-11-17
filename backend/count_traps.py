import json

def report_counts():
    print("=" * 75)
    print("1. 5-MINUTE INTRADAY DATASET (60 Days - 10 Instruments)")
    print("=" * 75)
    with open('audit_5m_dynamic_rr.json') as f:
        d5m = json.load(f)

    tot_trades_5m = sum(s['total_trades'] for s in d5m)
    tot_sl_5m = 0
    tot_chop_5m = 0

    print(f"{'SYMBOL':12} | {'TOTAL':5} | {'SL HIT (Loss)':14} | {'TP1 (Low Win)':14} | {'NET PTS'}")
    print("-" * 75)
    for s in d5m:
        n = s['total_trades']
        sl = round(n * s['sl_rate'] / 100)
        tp1_low = n - sl
        tot_sl_5m += sl
        tot_chop_5m += tp1_low
        print(f"{s['symbol']:12} | {n:5d} | {sl:4d} ({s['sl_rate']:4.1f}%)   | {tp1_low:4d} ({s['tp1_win_rate']:4.1f}%)   | {s['total_pts']:+8.1f}")
    
    print("-" * 75)
    print(f"5M TOTALS: Total Trades = {tot_trades_5m} | Full SL Hit = {tot_sl_5m} ({tot_sl_5m/tot_trades_5m*100:.1f}%) | Small TP1 Wins = {tot_chop_5m}")

    print("\n" + "=" * 75)
    print("2. 5-YEAR DAILY SWING DATASET (2021 - 2026 - 10 Stocks)")
    print("=" * 75)
    with open('audit_5y_dynamic_rr.json') as f:
        d5y = json.load(f)

    tot_trades_5y = sum(s['total_trades'] for s in d5y)
    tot_sl_5y = 0
    tot_chop_5y = 0

    print(f"{'SYMBOL':12} | {'TOTAL':5} | {'SL HIT (Loss)':14} | {'TP1 (Win/Chop)':14} | {'NET PTS'}")
    print("-" * 75)
    for s in d5y:
        n = s['total_trades']
        sl = round(n * s['sl_rate'] / 100)
        tp1_low = n - sl
        tot_sl_5y += sl
        tot_chop_5y += tp1_low
        print(f"{s['symbol']:12} | {n:5d} | {sl:4d} ({s['sl_rate']:4.1f}%)   | {tp1_low:4d} ({s['tp1_win_rate']:4.1f}%)   | {s['total_pts']:+8.1f}")
    
    print("-" * 75)
    print(f"5Y TOTALS: Total Trades = {tot_trades_5y} | Full SL Hit = {tot_sl_5y} ({tot_sl_5y/tot_trades_5y*100:.1f}%) | TP1 Captures = {tot_chop_5y}")
    print("=" * 75)

if __name__ == '__main__':
    report_counts()
