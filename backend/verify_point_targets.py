import sys, os
import requests
import json

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from routers.predict import predict_stock

TEST_CASES = [
    # (Symbol, Category, Expected Min TP1 Points)
    ('SBIN', 'Ordinary Stock (< Rs 700)', 10.0),
    ('ITC', 'Ordinary Stock (< Rs 700)', 10.0),
    ('RELIANCE', 'Mid/Large Stock (>= Rs 700)', 18.0),
    ('TCS', 'Mid/Large Stock (>= Rs 700)', 18.0),
    ('MARUTI', 'High-Value Large Cap (>= Rs 700)', 20.0),
    ('^NSEBANK', 'Major Index (^NSEBANK)', 100.0),
    ('^NSEI', 'Major Index (NIFTY 50)', 100.0)
]

def verify_live_targets():
    print("=" * 105)
    print("LIVE MARKET TARGET ENGINE VERIFICATION: ENFORCED MINIMUM POINT FLOORS")
    print("=" * 105)
    
    for sym, category, min_floor in TEST_CASES:
        try:
            res = predict_stock(sym, interval="1d")
            curr_p = res.get('current_price', 0)
            pred = res.get('prediction', 'HOLD')
            sl = res.get('stop_loss', 0)
            tp1_data = res.get('multi_targets', {}).get('tp1', {})
            tp2_data = res.get('multi_targets', {}).get('tp2', {})
            tp3_data = res.get('multi_targets', {}).get('tp3', {})
            
            tp1_p = tp1_data.get('price', 0)
            tp1_pts = tp1_data.get('distance_pts', 0)
            tp2_pts = tp2_data.get('distance_pts', 0)
            tp3_pts = tp3_data.get('distance_pts', 0)
            rr1 = tp1_data.get('risk_reward', 0)
            rr2 = tp2_data.get('risk_reward', 0)
            
            passed = tp1_pts >= min_floor
            status = "PASSED [OK]" if passed else "FAILED [TOO LOW]"
            
            print(f"[{status:15}] {sym:10} | Price: {curr_p:8.2f} | Category: {category}")
            print(f"                Signal: {pred:10} | SL: {sl:8.2f} | TP1: {tp1_p:8.2f} (+{tp1_pts:6.1f} pts) [Min Req: {min_floor} pts]")
            print(f"                TP2: +{tp2_pts:6.1f} pts (RR 1:{rr2}) | TP3: +{tp3_pts:6.1f} pts | Dynamic Alloc: {tp1_data.get('dynamic_book_pct')}%")
            print("-" * 105)
        except Exception as e:
            print(f"[ERROR] {sym:10}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    verify_live_targets()
