import sys, os
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytics_target_sl_backtest import calculate_4y_target_sl_stats

stocks = [
    'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK',
    'SBIN', 'BHARTIARTL', 'ITC', 'LT', 'TATAMOTORS'
]

results = []

print("="*105)
print(f"{'SYMBOL':12} | {'TRADES':6} | {'TP1 WIN':8} | {'TP2 HIT':8} | {'SL HIT':8} | {'PF':5} | {'RUN PAST TP1':18} | {'SL REVERSAL':11} | {'NET PTS'}")
print("="*105)

for sym in stocks:
    try:
        r = calculate_4y_target_sl_stats(sym, period='4y', interval='1d')
        if 'error' in r:
            print(f"Error for {sym}: {r['error']}")
            continue
            
        summary = {
            'symbol': sym,
            'total_trades': r.get('total_trades', 0),
            'tp1_rate': r.get('tp1_hit_rate_pct', 0),
            'tp2_rate': r.get('tp2_hit_rate_pct', 0),
            'tp3_rate': r.get('tp3_hit_rate_pct', 0),
            'sl_rate': r.get('sl_hit_rate_pct', 0),
            'profit_factor': r.get('profit_factor', 0),
            'avg_gain_pct': r.get('avg_gain_pct', 0),
            'avg_loss_pct': r.get('avg_loss_pct', 0),
            'post_tp1_run_pts': r.get('post_tp1_run_avg_pts', 0),
            'post_tp1_run_atr': r.get('post_tp1_run_avg_atr', 0),
            'max_post_tp1_run_pts': r.get('post_tp1_run_max_pts', 0),
            'post_sl_drop_pts': r.get('post_sl_drop_avg_pts', 0),
            'stop_hunt_rev_rate': r.get('stop_hunt_reversal_rate_pct', 0),
            'cumulative_pts': r.get('cumulative_profit_pts', 0),
            'yearly': r.get('yearly_consistency', [])
        }
        results.append(summary)
        print(f"{sym:12} | {summary['total_trades']:6d} | {summary['tp1_rate']:7.1f}% | {summary['tp2_rate']:7.1f}% | {summary['sl_rate']:7.1f}% | {summary['profit_factor']:4.1f}x | +{summary['post_tp1_run_pts']}p ({summary['post_tp1_run_atr']}x ATR) | {summary['stop_hunt_rev_rate']:9.1f}% | +{summary['cumulative_pts']:8.1f} pts")
    except Exception as e:
        print(f"Error for {sym}: {e}")

print("="*105)

with open('audit_10stocks_4y.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nAudit saved to audit_10stocks_4y.json successfully.")
