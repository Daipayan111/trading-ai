"""
Advanced Multi-Timeframe (MTF) Alignment Matrix & Institutional Regime Engine
=============================================================================
Analyzes real market structure, EMAs, RSI, and momentum across 5 key timeframes:
- 1M: Micro Scalp Momentum
- 5M: Intraday Execution
- 15M: Short-Term Structure / BOS
- 1H: Institutional Trend / Order Blocks
- 1D: Macro Trend / Daily Higher Highs
"""
import time
import pandas as pd
import numpy as np
import yfinance as yf

# In-memory cache with 45-second TTL to keep API ultra-fast
_MTF_CACHE = {}
_CACHE_TTL = 45  # seconds

def calc_tf_trend(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 15:
        return {
            "trend": "NEUTRAL",
            "label": "🟡 FLAT",
            "score": 50,
            "rsi": 50.0,
            "ema_status": "Neutral"
        }
    
    close = df['Close']
    curr = float(close.iloc[-1])
    
    # EMAs
    ema20 = float(close.ewm(span=min(20, len(close))).mean().iloc[-1])
    ema50 = float(close.ewm(span=min(50, len(close))).mean().iloc[-1])
    
    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(min(14, len(close))).mean()
    loss = (-delta.clip(upper=0)).rolling(min(14, len(close))).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
    
    score = 50
    # Price vs EMA 20
    if curr > ema20: score += 15
    else: score -= 15
    
    # EMA 20 vs EMA 50 (Trend Direction)
    if ema20 > ema50: score += 15
    else: score -= 15
    
    # RSI Momentum
    if rsi >= 60: score += 15
    elif rsi > 52: score += 8
    elif rsi <= 40: score -= 15
    elif rsi < 48: score -= 8
    
    # Candlestick Momentum (Last 3 candles higher closes)
    if len(close) >= 3:
        if close.iloc[-1] > close.iloc[-2] > close.iloc[-3]:
            score += 5
        elif close.iloc[-1] < close.iloc[-2] < close.iloc[-3]:
            score -= 5

    score = max(5, min(95, score))
    
    if score >= 75:
        trend = "STRONG_BULL"
        label = "🚀 STRONG"
        ema_status = "Price well above EMA 20 & 50"
    elif score >= 58:
        trend = "BULLISH"
        label = "🟢 BULL"
        ema_status = "Price above EMA 20"
    elif score <= 25:
        trend = "STRONG_BEAR"
        label = "🔻 HEAVY"
        ema_status = "Price dumping below EMA 20 & 50"
    elif score <= 42:
        trend = "BEARISH"
        label = "🔴 BEAR"
        ema_status = "Price below EMA 20"
    else:
        trend = "NEUTRAL"
        label = "🟡 FLAT"
        ema_status = "Consolidating near EMAs"
        
    return {
        "trend": trend,
        "label": label,
        "score": int(score),
        "rsi": round(rsi, 1),
        "ema_status": ema_status,
        "price": round(curr, 2)
    }


def analyze_multi_timeframe(symbol: str) -> dict:
    """
    Returns an institutional-grade Multi-Timeframe alignment matrix for a stock symbol.
    """
    clean_symbol = symbol.replace('.NS', '').replace('.BO', '').strip().upper()

    INDEX_MAPPING = {
        "NIFTY": "^NSEI",
        "NIFTY50": "^NSEI",
        "^NSEI": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "NIFTYBANK": "^NSEBANK",
        "^NSEBANK": "^NSEBANK",
        "SENSEX": "^BSESN",
        "^BSESN": "^BSESN",
        "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
        "NIFTYIT": "^CNXIT",
        "^CNXIT": "^CNXIT",
        "NIFTYMIDCAP": "^NSEMDCP50",
        "^NSEMDCP50": "^NSEMDCP50",
    }

    COMMODITY_MAPPING = {
        "CRUDEOIL": "CL=F",
        "CRUDEOILM": "CL=F",
        "GOLD": "GC=F",
        "GOLDM": "GC=F",
        "SILVER": "SI=F",
        "SILVERM": "SI=F",
        "NATURALGAS": "NG=F",
        "COPPER": "HG=F",
    }

    if clean_symbol in COMMODITY_MAPPING:
        symbol_ns = COMMODITY_MAPPING[clean_symbol]
    elif clean_symbol in INDEX_MAPPING:
        symbol_ns = INDEX_MAPPING[clean_symbol]
    elif clean_symbol.startswith("^"):
        symbol_ns = clean_symbol
    else:
        symbol_ns = f"{clean_symbol}.NS"

    now = time.time()
    
    # Check cache
    if clean_symbol in _MTF_CACHE:
        entry = _MTF_CACHE[clean_symbol]
        if now - entry['timestamp'] < _CACHE_TTL:
            return entry['data']
            
    ticker = yf.Ticker(symbol_ns)
    
    try:
        # Fetch 1D and 5M data
        df_1d = ticker.history(period="6mo", interval="1d")
        df_5m = ticker.history(period="5d", interval="5m")
        
        if df_5m.empty:
            # Fallback to daily
            df_5m = df_1d
    except Exception as e:
        print("MTF Download Error:", e)
        df_1d = pd.DataFrame()
        df_5m = pd.DataFrame()
        
    # Micro 1M from 5M tail or short download
    df_1m = df_5m.tail(30).copy() if not df_5m.empty else pd.DataFrame()
    
    # Resample 5M to 15M and 1H
    if not df_5m.empty and len(df_5m) >= 10:
        df_15m = df_5m.resample('15min').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
        df_1h = df_5m.resample('1h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
    else:
        df_15m = pd.DataFrame()
        df_1h = pd.DataFrame()
        
    tf_1m = calc_tf_trend(df_1m)
    tf_5m = calc_tf_trend(df_5m)
    tf_15m = calc_tf_trend(df_15m)
    tf_1h = calc_tf_trend(df_1h)
    tf_1d = calc_tf_trend(df_1d)
    
    all_tfs = [tf_1m, tf_5m, tf_15m, tf_1h, tf_1d]
    bull_count = sum(1 for t in all_tfs if 'BULL' in t['trend'])
    bear_count = sum(1 for t in all_tfs if 'BEAR' in t['trend'])
    neutral_count = 5 - bull_count - bear_count
    
    avg_score = int(np.mean([t['score'] for t in all_tfs]))
    
    # Institutional Regime & Actionable Verdict
    is_macro_bull = ('BULL' in tf_1d['trend']) or ('BULL' in tf_1h['trend'])
    is_macro_bear = ('BEAR' in tf_1d['trend']) or ('BEAR' in tf_1h['trend'])
    is_intra_bear = ('BEAR' in tf_5m['trend']) or ('BEAR' in tf_1m['trend'])
    is_intra_bull = ('BULL' in tf_5m['trend']) or ('BULL' in tf_1m['trend'])
    
    if bull_count >= 4:
        regime = "FULL_BULL_ALIGNMENT"
        verdict_badge = "🔥 ULTRA BULLISH (HIGH CONVICTION)"
        badge_color = "#10B981"
        action_advice = "All timeframes aligned in institutional sync. Safe for aggressive runner targets."
    elif bear_count >= 4:
        regime = "FULL_BEAR_ALIGNMENT"
        verdict_badge = "🔻 ULTRA BEARISH (HEAVY SELLOFF)"
        badge_color = "#EF4444"
        action_advice = "Heavy distribution across all timeframes. Avoid longs, trail shorts."
    elif is_macro_bull and is_intra_bear:
        regime = "BULLISH_PULLBACK"
        verdict_badge = "⚡ BULLISH PULLBACK (DIP BUY ZONE)"
        badge_color = "#3B82F6"
        action_advice = "1D & 1H macro trend is Bullish. 5M drop is a discount pullback. Look for support bounce."
    elif is_macro_bear and is_intra_bull:
        regime = "BEARISH_RALLY"
        verdict_badge = "⚠️ BEARISH PULLBACK (FADE THE RALLY)"
        badge_color = "#F59E0B"
        action_advice = "1D & 1H macro trend is Bearish. 5M rally is a dead-cat bounce. Do not chase highs."
    elif bull_count > bear_count:
        regime = "MODERATE_BULL"
        verdict_badge = "🟢 MODERATE BULLISH ALIGNMENT"
        badge_color = "#059669"
        action_advice = "Bullish momentum leading. Wait for local 5M confirmation before entry."
    elif bear_count > bull_count:
        regime = "MODERATE_BEAR"
        verdict_badge = "🔴 MODERATE BEARISH ALIGNMENT"
        badge_color = "#DC2626"
        action_advice = "Sellers have local control. Confirm micro support break before shorting."
    else:
        regime = "CHOPPY_RANGE"
        verdict_badge = "⏸ TIMEFRAME DIVERGENCE (CHOPPY)"
        badge_color = "#64748B"
        action_advice = "Timeframes in conflict. Trade strictly with key S/R boundaries and tight SL."
        
    result = {
        "symbol": clean_symbol,
        "timeframes": {
            "1m": tf_1m,
            "5m": tf_5m,
            "15m": tf_15m,
            "1h": tf_1h,
            "1d": tf_1d
        },
        "bullish_count": bull_count,
        "bearish_count": bear_count,
        "neutral_count": neutral_count,
        "avg_power_score": avg_score,
        "regime": regime,
        "verdict_badge": verdict_badge,
        "badge_color": badge_color,
        "action_advice": action_advice,
        "timestamp": now
    }
    
    # Save to cache
    _MTF_CACHE[clean_symbol] = {"timestamp": now, "data": result}
    return result
