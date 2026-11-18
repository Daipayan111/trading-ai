from fastapi import APIRouter, HTTPException
from analytics import fetch_stock_data, add_technical_indicators, generate_signals
from analytics_sr import find_ultra_strong_sr_zones
from analytics_trap import detect_institutional_traps
from analytics_confluence import evaluate_9point_checklist
from analytics_micro_breakout import detect_micro_sr_and_breakouts
from routers.predict import calc_advanced_trap_sl_target, find_strong_sr_levels
import pandas as pd
import numpy as np

import os
import json
import time
import threading
import concurrent.futures

router = APIRouter()

CACHE_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "elite_signals_cache.json")
SYMBOLS_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nifty500_symbols.json")

# In-memory cache for elite-scan to provide ultra-fast instant responses
_elite_cache = {"data": None, "timestamp": 0, "is_scanning": False}

def _load_broad_market_universe():
    symbols = []
    if os.path.exists(SYMBOLS_FILE_PATH):
        try:
            with open(SYMBOLS_FILE_PATH, "r") as f:
                symbols = json.load(f)
        except Exception:
            pass

    # Core high-liquidity stocks to ensure all leaders are present
    core_leaders = [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL",
        "AXISBANK", "KOTAKBANK", "LT", "SUNPHARMA", "HINDUNILVR", "BAJFINANCE", "MARUTI",
        "NTPC", "POWERGRID", "ONGC", "COALINDIA", "NESTLEIND", "TITAN", "ULTRACEMCO",
        "ASIANPAINT", "DRREDDY", "CIPLA", "APOLLOHOSP", "TATACONSUM", "INDUSINDBK",
        "BAJAJFINSV", "ADANIENT", "ADANIPORTS", "GRASIM", "HEROMOTOCO", "EICHERMOT",
        "BPCL", "BRITANNIA", "ITC", "M&M", "TECHM", "HCLTECH", "SHREECEM", "DIVISLAB",
        "JSWSTEEL", "HINDALCO", "TATASTEEL", "WIPRO", "SBILIFE", "HDFCLIFE", "BAJAJ-AUTO",
        "TRENT", "BEL", "HAL", "ZOMATO", "JIOFIN", "VEDL", "DLF", "PIDILITIND", "SIEMENS",
        "ABB", "TVSMOTOR", "POLYCAB", "CHOLAFIN", "GAIL", "IOC", "AMBUJACEM", "BANKBARODA",
        "CANBK", "PNB", "PFC", "RECLTD", "SHRIRAMFIN", "HAVELLS", "DABUR", "GODREJCP",
        "MARICO", "BERGEPAINT", "ICICIPRULI", "ICICIGI", "MOTHERSON", "BOSCHLTD",
        "CUMMINSIND", "COLPAL", "INDIGO", "NAUKRI", "PERSISTENT", "MUTHOOTFIN",
        "OBEROIRLTY", "LODHA", "PHOENIXLTD", "TATACOMM", "TATAPOWER", "SRF",
        "ASTRAL", "PIIND", "AUROPHARMA", "LUPIN", "ALKEM", "TORNTPOWER", "UNITDSPR",
        "JUBLFOOD", "VOLTAS", "PAGEIND", "ASHOKLEY", "BHEL", "COFORGE", "MPHASIS",
        "LTTS", "DIXON", "KALYANKJIL", "FEDERALBNK", "IDFCFIRSTB", "BANDHANBNK",
        "AUBANK", "LICHSGFIN", "MANAPPURAM", "BALKRISIND", "MRF", "APOLLOTYRE",
        "BHARATFORG", "ESCORTS", "EXIDEIND", "CONCOR", "IRCTC", "INDHOTEL", "MAXHEALTH",
        "FORTIS", "SYNGENE", "BIOCON", "GLENMARK", "IPCALAB", "TATACHEM", "DEEPAKNTR",
        "ATUL", "NAVINFLUOR", "GUJGASLTD", "IGL", "MGL", "PETRONET", "HINDPETRO",
        "SAIL", "JINDALSTEL", "NMDC", "NATIONALUM", "ACC", "DALBHARAT", "RAMCOCEM",
        "PRESTIGE", "BRIGADE", "GODREJPROP", "SUNTV", "ZEEL", "COROMANDEL", "FACT",
        "GNFC", "METROPOLIS", "LALPATHLAB", "TATAELXSI", "KPITTECH", "CYIENT", "SONACOMS",
        "TIINDIA", "KEI", "SUPREMEIND", "AIAENG", "THERMAX", "CARBORUNIV", "SCHAEFFLER",
        "TIMKEN", "SKFINDIA", "ENDURANCE", "SUNDRMFAST", "BATAINDIA", "RELAXO"
    ]
    combined = list(dict.fromkeys(core_leaders + symbols))
    return combined

# Pre-load saved cache from disk on startup
if os.path.exists(CACHE_FILE_PATH) and os.path.getsize(CACHE_FILE_PATH) > 10:
    try:
        with open(CACHE_FILE_PATH, "r") as f:
            _elite_cache["data"] = json.load(f)
            _elite_cache["timestamp"] = time.time()
    except Exception:
        pass

# High-Liquidity Stocks specifically suited for Fast 5-Minute Intraday Scalping
INTRADAY_5M_UNIVERSE = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL", "TCS", "INFY", "LT",
    "AXISBANK", "KOTAKBANK", "MARUTI", "BAJFINANCE", "TITAN", "TATASTEEL", "ADANIENT",
    "ADANIPORTS", "JSWSTEEL", "POWERGRID", "NTPC", "COALINDIA", "SUNPHARMA", "DRREDDY",
    "CIPLA", "APOLLOHOSP", "DIVISLAB", "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA",
    "ASIANPAINT", "GRASIM", "ULTRACEMCO", "SHRIRAMFIN", "M&M", "EICHERMOT", "HEROMOTOCO",
    "BAJAJ-AUTO", "HDFCLIFE", "SBILIFE", "BPCL", "IOC", "ONGC", "WIPRO", "HCLTECH",
    "TECHM", "LTIM", "BEL", "HAL", "VEDL", "HINDALCO", "TRENT", "ZOMATO", "DLF",
    "INDIGO", "CANBK", "PNB", "PFC", "RECLTD", "CHOLAFIN", "FEDERALBNK", "IDFCFIRSTB",
    "BANKBARODA", "MUTHOOTFIN", "TATAPOWER", "POLYCAB", "DIXON", "KALYANKJIL", "CUMMINSIND",
    "ASHOKLEY", "BHEL", "PERSISTENT", "COFORGE", "MPHASIS", "AUBANK", "SAIL", "JINDALSTEL"
]

def _evaluate_stock_setup(symbol: str, interval: str = "1d"):
    """
    Evaluates a single stock against the full AI engine for a specific timeframe:
    - '5m': 5-Minute Intraday Scalp / Day Trading (period='5d')
    - '1d': 1-Day Daily Setup / Swing Trading (period='6mo')
    Strictly enforces Price >= 150.
    """
    try:
        period = "5d" if interval == "5m" else "6mo"
        df, used_symbol, ticker = fetch_stock_data(symbol, period=period, interval=interval)
        if df is None or len(df) < 20:
            return None

        latest = df.iloc[-1]
        curr_price = round(float(latest['Close']), 2)

        # ─── Strict User Rule: Price >= 150 (Exclude penny stocks / low price shares) ───
        if curr_price < 150.0:
            return None

        df = add_technical_indicators(df)
        signals = generate_signals(df)

        prev = df.iloc[-2] if len(df) > 1 else latest
        latest_sig = signals.iloc[-1]

        # For 5M Intraday Scalping: if the very last candle is HOLD, check recent 3 candles or strong EMA trend
        sig_dir = "BUY" if "BUY" in latest_sig['Signal'] else ("SELL" if "SELL" in latest_sig['Signal'] else "HOLD")
        if interval == "5m" and sig_dir == "HOLD":
            for offset in [-2, -3]:
                if len(signals) >= abs(offset):
                    past_sig = signals.iloc[offset]['Signal']
                    if "BUY" in past_sig:
                        sig_dir = "BUY"
                        latest_sig = signals.iloc[offset]
                        break
                    elif "SELL" in past_sig:
                        sig_dir = "SELL"
                        latest_sig = signals.iloc[offset]
                        break
            if sig_dir == "HOLD" and 'EMA_9' in df.columns and 'EMA_21' in df.columns:
                rsi_val = float(latest.get('RSI', 50))
                if latest['Close'] > latest['EMA_9'] > latest['EMA_21'] and rsi_val > 50:
                    sig_dir = "BUY"
                    latest_sig = pd.Series({'Signal': 'BUY (5M Intraday Momentum)'})
                elif latest['Close'] < latest['EMA_9'] < latest['EMA_21'] and rsi_val < 50:
                    sig_dir = "SELL"
                    latest_sig = pd.Series({'Signal': 'SELL (5M Intraday Breakdown)'})

        trap_data = detect_institutional_traps(df)
        ultra_sr = find_ultra_strong_sr_zones(df)
        micro_bo = detect_micro_sr_and_breakouts(df)
        sr_levels = find_strong_sr_levels(df)

        atr_val = float(latest.get('ATR', curr_price * (0.005 if interval == '5m' else 0.015)))

        sl_val, tp_val, rr_val, targets_val, sl_rat_val, tp_rat_val = calc_advanced_trap_sl_target(
            current_price=curr_price,
            direction=sig_dir,
            atr=atr_val,
            sr_levels=sr_levels,
            ultra_sr=ultra_sr,
            micro_breakout=micro_bo,
            trap_data=trap_data,
            conv_score=75,
            interval=interval,
            latest_candle=latest,
            prev_candle=prev
        )

        confluence = evaluate_9point_checklist(df, ultra_sr, trap_data, sig_dir, rr_ratio=rr_val)
        conf_score = confluence.get('score_pct', 0)
        is_trap_safe = confluence.get('trap_safe', True)
        has_trap = trap_data.get('has_trap', False)
        traffic_light = confluence.get('traffic_light', 'YELLOW')

        # ─── Elite Filter: High quality only ───
        if sig_dir == "HOLD":
            return None
        min_conf = 35 if interval == "5m" else 50
        min_rr = 1.10 if interval == "5m" else 1.20
        if conf_score < min_conf:
            return None
        if rr_val < min_rr:
            return None
        if has_trap and not is_trap_safe:
            return None

        # ─── Volume Surge Score ───
        vol_ratio = float(latest.get('Volume_Ratio', 1))
        volume_surge_score = min(100, (vol_ratio - 1.0) * 50) if vol_ratio > 1.0 else 0

        # ─── Trend Strength (ADX) ───
        adx_val = float(latest.get('ADX', 20))
        trend_score = min(100, adx_val * 2.5)

        # ─── Trap-Safe Bonus ───
        trap_safe_bonus = 100 if is_trap_safe else 0

        # ─── RR Score (capped at 4:1) ───
        rr_score = min(100, (rr_val / 4.0) * 100)

        # ─── Composite Elite Score ───
        elite_score = round(
            (conf_score * 0.40) +
            (rr_score * 0.25) +
            (volume_surge_score * 0.20) +
            (trend_score * 0.10) +
            (trap_safe_bonus * 0.05),
            2
        )

        # ─── Win Probability (targeting 85-99%) ───
        base_win = conf_score * 0.60
        rr_bonus = min(20, rr_val * 5)
        vol_bonus = min(10, vol_ratio * 3)
        trap_bonus = 10 if is_trap_safe else -10
        strong_signal_bonus = 5 if "STRONG" in latest_sig['Signal'] else 0
        win_probability = round(min(99.0, max(60.0, base_win + rr_bonus + vol_bonus + trap_bonus + strong_signal_bonus)), 1)

        # ─── Setup Quality Label ───
        if elite_score >= 70 or win_probability >= 90:
            setup_quality = "💎 DIAMOND"
        elif elite_score >= 55 or win_probability >= 80:
            setup_quality = "🥇 GOLD"
        elif elite_score >= 45:
            setup_quality = "🥈 SILVER"
        else:
            setup_quality = "🥉 BRONZE"

        effective_signal = latest_sig['Signal']
        if has_trap and not is_trap_safe:
            if "BUY" in effective_signal and trap_data.get('trap_type') == "BULL_TRAP":
                effective_signal = "HOLD"
            elif "SELL" in effective_signal and trap_data.get('trap_type') == "BEAR_TRAP":
                effective_signal = "HOLD"

        timeframe_label = "5M Intraday Scalp" if interval == "5m" else "1D Daily Setup"

        return {
            "symbol": used_symbol,
            "name": symbol,
            "price": curr_price,
            "change_pct": round(float(latest.get('Price_Change', 0)) * 100, 2),
            "signal": effective_signal,
            "direction": sig_dir,
            "timeframe": interval,
            "timeframe_label": timeframe_label,
            "rsi": round(float(latest.get('RSI', 50)), 2),
            "adx": round(adx_val, 1),
            "volume_ratio": round(vol_ratio, 2),
            "traffic_light": traffic_light,
            "confluence_score": conf_score,
            "passed_count": confluence.get('passed_count', 0),
            "has_trap": has_trap,
            "trap_type": trap_data.get('trap_type', 'NONE'),
            "is_trap_safe": is_trap_safe,
            "target": tp_val,
            "stop_loss": sl_val,
            "risk_reward": round(rr_val, 2),
            "multi_targets": targets_val,
            "sl_rationale": sl_rat_val,
            "tp_rationale": tp_rat_val,
            "elite_score": elite_score,
            "win_probability": win_probability,
            "setup_quality": setup_quality,
            "volume_surge_score": round(volume_surge_score, 1),
            "trend_score": round(trend_score, 1),
        }
    except Exception:
        return None

def _run_elite_scan_computation():
    global _elite_cache
    _elite_cache["is_scanning"] = True

    universe = _load_broad_market_universe()
    all_5m_results = []
    all_1d_results = []

    # Parallel scan with low worker count to prevent CPU/memory spikes on cloud dynos
    def _scan_5m(sym):
        return _evaluate_stock_setup(sym, interval="5m")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        evaluated_5m = list(executor.map(_scan_5m, INTRADAY_5M_UNIVERSE))

    for item in evaluated_5m:
        if item is not None and item.get('direction') != 'HOLD':
            all_5m_results.append(item)

    def _scan_1d(sym):
        return _evaluate_stock_setup(sym, interval="1d")

    # Limit broad market scan to top 50 active leaders to avoid overwhelming container
    scan_universe = universe[:50] if len(universe) > 50 else universe
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        evaluated_1d = list(executor.map(_scan_1d, scan_universe))

    for item in evaluated_1d:
        if item is not None and item.get('direction') != 'HOLD':
            all_1d_results.append(item)

    # Sort each list by win_probability & elite_score descending
    all_5m_results.sort(key=lambda x: (x['win_probability'], x['elite_score']), reverse=True)
    all_1d_results.sort(key=lambda x: (x['win_probability'], x['elite_score']), reverse=True)

    for rank, item in enumerate(all_5m_results, start=1):
        item['elite_rank'] = rank
    for rank, item in enumerate(all_1d_results, start=1):
        item['elite_rank'] = rank

    # Curate balanced combined list: Interleave top 5M intraday setups with top 1D daily setups
    top_5m = all_5m_results[:10]
    top_1d = all_1d_results[:15]
    combined_all = []
    max_len = max(len(top_5m), len(top_1d))
    for i in range(max_len):
        if i < len(top_5m):
            combined_all.append(top_5m[i])
        if i < len(top_1d):
            combined_all.append(top_1d[i])

    for rank, item in enumerate(combined_all, start=1):
        item['elite_rank'] = rank

    top_results = combined_all[:20]

    response_payload = {
        "results": top_results,
        "intraday_5m": all_5m_results[:20],
        "daily_1d": all_1d_results[:20],
        "intraday_5m_count": len(all_5m_results),
        "daily_1d_count": len(all_1d_results),
        "total_scanned": len(universe) + len(INTRADAY_5M_UNIVERSE),
        "total_qualified": len(all_5m_results) + len(all_1d_results),
        "top_count": len(top_results),
        "scan_note": f"Scanned 500+ Stocks (₹150+) — {len(all_5m_results)} Intraday (5M) & {len(all_1d_results)} Daily (1D) setups",
        "cached_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    _elite_cache["data"] = response_payload
    _elite_cache["timestamp"] = time.time()
    _elite_cache["is_scanning"] = False

    # Persist to disk with default=str for numpy/int64/float compatibility
    try:
        os.makedirs(os.path.dirname(CACHE_FILE_PATH), exist_ok=True)
        with open(CACHE_FILE_PATH, "w") as f:
            json.dump(response_payload, f, indent=2, default=str)
    except Exception as e:
        print("[CACHE SAVE ERROR]:", e)

    return response_payload

def _background_scan_worker():
    """Background loop that rescans active universe every 15 minutes (with delayed start to ensure fast boot)."""
    time.sleep(180)  # Wait 3 minutes after boot before running first background scan
    while True:
        try:
            _run_elite_scan_computation()
        except Exception as e:
            print("[BACKGROUND SCANNER] Error:", e)
        time.sleep(900)  # Every 15 minutes

# Start background scanner thread once
_bg_thread = threading.Thread(target=_background_scan_worker, daemon=True)
_bg_thread.start()


@router.get("/elite-scan")
def elite_scan(interval: str = "all"):
    """
    Elite AI Scanner: Scans 500+ NSE Market Stocks (price >= 150),
    providing both 5M Intraday Scalp and 1D Daily Setups.
    Query parameters:
    - interval='5m': Returns exclusively 5M Intraday Setups
    - interval='1d': Returns exclusively 1D Daily Setups
    - interval='all': Returns combined top setups (5M + 1D)
    """
    global _elite_cache

    if _elite_cache["data"] is not None:
        data = _elite_cache["data"]
    else:
        data = _run_elite_scan_computation()

    filter_mode = (interval or "all").lower()
    if filter_mode == "5m":
        res_list = data.get("intraday_5m", [])
        return {**data, "results": res_list, "active_filter": "5m"}
    elif filter_mode == "1d":
        res_list = data.get("daily_1d", [])
        return {**data, "results": res_list, "active_filter": "1d"}
    return data


@router.get("/{symbol}")
async def get_trading_signals(symbol: str, period: str = "6mo"):
    """Get comprehensive trading signals for a stock"""
    try:
        df, used_symbol, ticker = fetch_stock_data(symbol, period=period, interval="1d")
        df = add_technical_indicators(df)
        signals = generate_signals(df)

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        latest_signals = signals.iloc[-1]

        # Determine overall recommendation
        current_signal = latest_signals['Signal']
        combined_score = latest_signals['Combined']

        # Stop Loss, Target and 90%+ S/R Confluence
        current_price = float(latest['Close'])
        atr = float(latest.get('ATR', current_price * 0.02))

        ultra_sr = find_ultra_strong_sr_zones(df)
        trap_data = detect_institutional_traps(df)
        micro_bo = detect_micro_sr_and_breakouts(df)
        sr_levels = find_strong_sr_levels(df)

        support = ultra_sr.get('strongest_support', {}).get('price') or round(float(latest.get('S1', latest['Low'])), 2)
        resistance = ultra_sr.get('strongest_resistance', {}).get('price') or round(float(latest.get('R1', latest['High'])), 2)

        sl_buy, tp_buy, rr_buy, targets_buy, sl_rat_buy, tp_rat_buy = calc_advanced_trap_sl_target(
            current_price=current_price,
            direction="BUY",
            atr=atr,
            sr_levels=sr_levels,
            ultra_sr=ultra_sr,
            micro_breakout=micro_bo,
            trap_data=trap_data,
            conv_score=75,
            interval="1d",
            latest_candle=latest,
            prev_candle=prev
        )

        sl_sell, tp_sell, rr_sell, targets_sell, sl_rat_sell, tp_rat_sell = calc_advanced_trap_sl_target(
            current_price=current_price,
            direction="SELL",
            atr=atr,
            sr_levels=sr_levels,
            ultra_sr=ultra_sr,
            micro_breakout=micro_bo,
            trap_data=trap_data,
            conv_score=75,
            interval="1d",
            latest_candle=latest,
            prev_candle=prev
        )

        direction = "BUY" if "BUY" in current_signal else ("SELL" if "SELL" in current_signal else "HOLD")
        active_rr = rr_buy if direction == "BUY" else (rr_sell if direction == "SELL" else rr_buy)
        confluence = evaluate_9point_checklist(df, ultra_sr, trap_data, direction, rr_ratio=active_rr)

        # Trap Override: Suppress false breakouts
        if not confluence.get("trap_safe", True):
            if "BUY" in current_signal and "BULL_TRAP" in str(trap_data.get("trap_type", "")):
                current_signal = "HOLD"
            elif "SELL" in current_signal and "BEAR_TRAP" in str(trap_data.get("trap_type", "")):
                current_signal = "HOLD"

        # Recent signal history (last 20 days)
        signal_history = []
        for i in range(min(30, len(signals))):
            idx = -(i + 1)
            sig_row = signals.iloc[idx]
            df_row = df.iloc[idx]
            signal_history.append({
                "date": df.index[idx].strftime("%Y-%m-%d"),
                "price": round(float(df_row['Close']), 2),
                "signal": sig_row['Signal'],
                "combined_score": round(float(sig_row['Combined']), 2),
                "rsi": round(float(df_row['RSI']), 2) if not pd.isna(df_row.get('RSI', float('nan'))) else None,
            })

        signal_history.reverse()

        # Individual indicator statuses
        rsi_val = float(latest.get('RSI', 50))
        macd_val = float(latest.get('MACD', 0))
        macd_sig = float(latest.get('MACD_Signal', 0))
        adx_val = float(latest.get('ADX', 0))
        bb_pct = float(latest.get('BB_Percent', 0.5))

        indicators = {
            "RSI": {
                "value": round(rsi_val, 2),
                "status": "OVERSOLD" if rsi_val < 30 else "OVERBOUGHT" if rsi_val > 70 else "NEUTRAL",
                "signal": "BUY" if rsi_val < 30 else "SELL" if rsi_val > 70 else "HOLD"
            },
            "MACD": {
                "value": round(macd_val, 4),
                "signal_line": round(macd_sig, 4),
                "status": "BULLISH" if macd_val > macd_sig else "BEARISH",
                "signal": "BUY" if macd_val > macd_sig else "SELL"
            },
            "ADX": {
                "value": round(adx_val, 2),
                "status": "STRONG TREND" if adx_val > 25 else "WEAK TREND",
                "signal": "TREND CONFIRMED" if adx_val > 25 else "NO CLEAR TREND"
            },
            "Bollinger Bands": {
                "percent": round(bb_pct, 4),
                "status": "NEAR LOWER BAND" if bb_pct < 0.2 else "NEAR UPPER BAND" if bb_pct > 0.8 else "MIDDLE",
                "signal": "BUY" if bb_pct < 0.2 else "SELL" if bb_pct > 0.8 else "HOLD"
            },
            "Volume": {
                "ratio": round(float(latest.get('Volume_Ratio', 1)), 2),
                "status": "HIGH VOLUME" if float(latest.get('Volume_Ratio', 1)) > 1.5 else "NORMAL",
                "signal": "CONFIRMING" if float(latest.get('Volume_Ratio', 1)) > 1.5 else "NEUTRAL"
            }
        }

        return {
            "symbol": used_symbol,
            "current_price": round(current_price, 2),
            "main_signal": current_signal,
            "combined_score": round(float(combined_score), 2),
            "traffic_light": confluence.get("traffic_light", "YELLOW"),
            "confluence_checklist": confluence,
            "trap_detector": trap_data,
            "ultra_sr_zones": ultra_sr,
            "support": support,
            "resistance": resistance,
            "stop_loss_buy": sl_buy,
            "target_buy": tp_buy,
            "multi_targets_buy": targets_buy,
            "sl_rationale_buy": sl_rat_buy,
            "tp_rationale_buy": tp_rat_buy,
            "stop_loss_sell": sl_sell,
            "target_sell": tp_sell,
            "multi_targets_sell": targets_sell,
            "sl_rationale_sell": sl_rat_sell,
            "tp_rationale_sell": tp_rat_sell,
            "risk_reward_ratio": active_rr,
            "multi_targets": targets_buy if direction == "BUY" else (targets_sell if direction == "SELL" else targets_buy),
            "sl_rationale": sl_rat_buy if direction == "BUY" else sl_rat_sell,
            "tp_rationale": tp_rat_buy if direction == "BUY" else tp_rat_sell,
            "indicators": indicators,
            "signal_history": signal_history
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/multi/scan")
def scan_multiple_stocks(symbols: str = "RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK,SBIN,WIPRO,TATAMOTORS"):
    """Scan multiple stocks and return signals with Traffic Light and Trap Filter"""
    symbol_list = [s.strip() for s in symbols.split(",")]
    results = []

    for symbol in symbol_list[:15]:  # Max 15 at a time
        try:
            df, used_symbol, ticker = fetch_stock_data(symbol, period="3mo", interval="1d")
            df = add_technical_indicators(df)
            signals = generate_signals(df)

            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            latest_sig = signals.iloc[-1]

            trap_data = detect_institutional_traps(df)
            ultra_sr = find_ultra_strong_sr_zones(df)
            micro_bo = detect_micro_sr_and_breakouts(df)
            sr_levels = find_strong_sr_levels(df)

            sig_dir = "BUY" if "BUY" in latest_sig['Signal'] else ("SELL" if "SELL" in latest_sig['Signal'] else "HOLD")
            curr_price = round(float(latest['Close']), 2)
            atr_val = float(latest.get('ATR', curr_price * 0.015))

            sl_val, tp_val, rr_val, targets_val, sl_rat_val, tp_rat_val = calc_advanced_trap_sl_target(
                current_price=curr_price,
                direction=sig_dir,
                atr=atr_val,
                sr_levels=sr_levels,
                ultra_sr=ultra_sr,
                micro_breakout=micro_bo,
                trap_data=trap_data,
                conv_score=75,
                interval="1d",
                latest_candle=latest,
                prev_candle=prev
            )

            confluence = evaluate_9point_checklist(df, ultra_sr, trap_data, sig_dir, rr_ratio=rr_val)

            effective_signal = latest_sig['Signal']
            if trap_data.get('has_trap') and not confluence.get('trap_safe', True):
                if "BUY" in effective_signal and trap_data.get('trap_type') == "BULL_TRAP":
                    effective_signal = "HOLD"
                elif "SELL" in effective_signal and trap_data.get('trap_type') == "BEAR_TRAP":
                    effective_signal = "HOLD"

            results.append({
                "symbol": used_symbol,
                "name": symbol,
                "price": curr_price,
                "change_pct": round(float(latest.get('Price_Change', 0)) * 100, 2),
                "signal": effective_signal,
                "rsi": round(float(latest.get('RSI', 50)), 2),
                "volume_ratio": round(float(latest.get('Volume_Ratio', 1)), 2),
                "traffic_light": confluence.get('traffic_light', 'YELLOW'),
                "confluence_score": confluence.get('score_pct', 50),
                "passed_count": confluence.get('passed_count', 4),
                "has_trap": trap_data.get('has_trap', False),
                "trap_type": trap_data.get('trap_type', 'NONE'),
                "target": tp_val,
                "stop_loss": sl_val,
                "risk_reward": rr_val,
                "multi_targets": targets_val,
                "sl_rationale": sl_rat_val,
                "tp_rationale": tp_rat_val,
            })
        except Exception:
            continue

    # Sort by traffic light (GREEN first) and then strongest signal
    signal_order = {'STRONG BUY': 5, 'BUY': 4, 'HOLD': 3, 'SELL': 2, 'STRONG SELL': 1}
    light_order = {'GREEN': 3, 'YELLOW': 2, 'RED': 1}
    results.sort(key=lambda x: (light_order.get(x.get('traffic_light'), 2), signal_order.get(x['signal'], 3)), reverse=True)

    return {"results": results, "total": len(results)}
