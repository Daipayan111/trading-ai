"""
Sector Rotation Analysis Router
=================================
Tracks NSE Sector Indices to identify:
  - Which sectors have strongest momentum
  - Sector-level BUY/SELL signals
  - Hot sector alert for stock selection
"""
from fastapi import APIRouter, HTTPException
import yfinance as yf
import pandas as pd
import ta
import numpy as np

router = APIRouter()

# NSE Sector ETFs / Indices (Yahoo Finance symbols)
NSE_SECTORS = {
    "IT":        "^CNXIT",
    "BANK":      "^NSEBANK",
    "PHARMA":    "^CNXPHARMA",
    "AUTO":      "^CNXAUTO",
    "FMCG":      "^CNXFMCG",
    "METAL":     "^CNXMETAL",
    "REALTY":    "^CNXREALTY",
    "ENERGY":    "^CNXENERGY",
    "INFRA":     "^CNXINFRA",
    "MEDIA":     "^CNXMEDIA",
}

# Top stocks per sector for drill-down
SECTOR_STOCKS = {
    "IT":     ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"],
    "BANK":   ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK"],
    "PHARMA": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "BIOCON"],
    "AUTO":   ["TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "MARUTI"],
    "FMCG":   ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR"],
    "METAL":  ["TATASTEEL", "HINDALCO", "JSWSTEEL", "SAIL", "VEDL"],
    "REALTY": ["DLF", "OBEROIRLTY", "GODREJPROP", "PRESTIGE", "PHOENIXLTD"],
    "ENERGY": ["RELIANCE", "ONGC", "BPCL", "IOC", "NTPC"],
}


def _analyze_sector_momentum(symbol: str, period: str = "3mo") -> dict:
    """Analyze momentum indicators for a sector index."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval="1d")

        if df.empty or len(df) < 20:
            return None

        close = df['Close']

        # RSI
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]

        # EMA Cross
        ema9  = ta.trend.EMAIndicator(close, window=9).ema_indicator().iloc[-1]
        ema21 = ta.trend.EMAIndicator(close, window=21).ema_indicator().iloc[-1]
        ema50 = ta.trend.EMAIndicator(close, window=50).ema_indicator().iloc[-1]

        # MACD
        macd_obj  = ta.trend.MACD(close)
        macd_val  = macd_obj.macd().iloc[-1]
        macd_sig  = macd_obj.macd_signal().iloc[-1]
        macd_hist = macd_obj.macd_diff().iloc[-1]

        # Price performance
        change_1w  = (close.iloc[-1] / close.iloc[-5]  - 1) * 100 if len(close) >= 5  else 0
        change_1mo = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0
        change_3mo = (close.iloc[-1] / close.iloc[0]   - 1) * 100

        # ADX
        adx_obj = ta.trend.ADXIndicator(df['High'], df['Low'], close)
        adx = adx_obj.adx().iloc[-1]

        # Momentum Score (0-100)
        score = 50  # Base

        # RSI contribution (±15)
        if rsi < 30:   score += 10   # Oversold = potential reversal
        elif rsi > 70: score -= 10   # Overbought = caution
        else:          score += (rsi - 50) * 0.3

        # EMA alignment (±15)
        if ema9 > ema21 > ema50:   score += 15   # Fully bullish
        elif ema9 < ema21 < ema50: score -= 15   # Fully bearish
        elif ema9 > ema21:         score += 7
        elif ema9 < ema21:         score -= 7

        # MACD (±10)
        if macd_val > macd_sig:  score += 10
        else:                    score -= 10

        # Momentum trend (1w change, ±10)
        score += min(10, max(-10, change_1w * 1.5))

        score = max(0, min(100, score))

        # Signal
        if score >= 65:   sector_signal = "BUY"
        elif score <= 35: sector_signal = "SELL"
        else:             sector_signal = "HOLD"

        strength = "STRONG" if (score >= 75 or score <= 25) else "MODERATE"

        return {
            "symbol": symbol,
            "momentum_score": round(float(score), 1),
            "signal": sector_signal,
            "strength": strength,
            "rsi": round(float(rsi), 2),
            "adx": round(float(adx), 2),
            "macd_bullish": bool(macd_val > macd_sig),
            "ema_aligned_bullish": bool(ema9 > ema21 > ema50),
            "performance": {
                "1w_pct":  round(float(change_1w), 2),
                "1mo_pct": round(float(change_1mo), 2),
                "3mo_pct": round(float(change_3mo), 2),
            },
            "current_price": round(float(close.iloc[-1]), 2),
        }
    except Exception as e:
        return None


@router.get("/rotation")
def get_sector_rotation():
    """
    Analyze all NSE sectors and rank by momentum strength.
    Returns hot sectors, cold sectors, and rotation recommendations.
    """
    try:
        results = []

        for sector_name, symbol in NSE_SECTORS.items():
            data = _analyze_sector_momentum(symbol)
            if data:
                data['sector'] = sector_name
                data['top_stocks'] = SECTOR_STOCKS.get(sector_name, [])
                results.append(data)

        if not results:
            raise ValueError("Could not fetch any sector data")

        # Sort by momentum score
        results.sort(key=lambda x: x['momentum_score'], reverse=True)

        # Classify
        hot_sectors  = [r for r in results if r['signal'] == 'BUY']
        cold_sectors = [r for r in results if r['signal'] == 'SELL']
        neutral      = [r for r in results if r['signal'] == 'HOLD']

        # Rotation insight
        top_sector = results[0] if results else None
        worst_sector = results[-1] if results else None

        rotation_insight = ""
        if top_sector and worst_sector:
            rotation_insight = (
                f"Money is flowing INTO {top_sector['sector']} "
                f"(Score: {top_sector['momentum_score']}) and OUT OF "
                f"{worst_sector['sector']} (Score: {worst_sector['momentum_score']})"
            )

        return {
            "sectors": results,
            "hot_sectors": [r['sector'] for r in hot_sectors],
            "cold_sectors": [r['sector'] for r in cold_sectors],
            "neutral_sectors": [r['sector'] for r in neutral],
            "top_sector": top_sector,
            "worst_sector": worst_sector,
            "rotation_insight": rotation_insight,
            "total_sectors_analyzed": len(results),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{sector_name}")
def get_sector_detail(sector_name: str):
    """Get detailed analysis for a specific sector and its top stocks."""
    sector_upper = sector_name.upper()
    symbol = NSE_SECTORS.get(sector_upper)

    if not symbol:
        available = list(NSE_SECTORS.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Sector '{sector_name}' not found. Available: {available}"
        )

    sector_data = _analyze_sector_momentum(symbol, period="6mo")
    if not sector_data:
        raise HTTPException(status_code=400, detail=f"Could not fetch data for sector {sector_upper}")

    sector_data['sector'] = sector_upper
    sector_data['top_stocks'] = SECTOR_STOCKS.get(sector_upper, [])

    return {
        "sector": sector_upper,
        "analysis": sector_data,
        "stocks_to_watch": sector_data['top_stocks'],
        "recommendation": (
            f"Focus on {', '.join(sector_data['top_stocks'][:3])} — "
            f"sector signal is {sector_data['signal']}"
        ),
    }
