"""
MCX Commodity Trading Router
==============================
Endpoints:
  GET  /api/commodity/list              - All commodities with live LTP
  GET  /api/commodity/price/{symbol}    - Single commodity price
  GET  /api/commodity/chart/{symbol}    - OHLCV chart data
  GET  /api/commodity/signals           - All commodity AI signals
  GET  /api/commodity/signal/{symbol}   - Single commodity AI signal
  GET  /api/commodity/status            - MCX market status
  POST /api/commodity/test-signal       - Test push notification
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
import datetime
import logging
import pytz

logger = logging.getLogger("CommodityRouter")
router = APIRouter()
IST = pytz.timezone("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Market Status
# ---------------------------------------------------------------------------

@router.get("/status")
async def get_market_status():
    """MCX market open/closed status with timing info."""
    from commodity_market_data import is_mcx_market_open, get_market_status, MCX_COMMODITIES
    is_open = is_mcx_market_open()
    status = get_market_status()
    now = datetime.datetime.now(IST)
    return {
        "is_open": is_open,
        "status": status,
        "exchange": "MCX",
        "trading_hours": "09:00 AM – 11:30 PM IST (Mon–Fri)",
        "current_time": now.strftime("%H:%M:%S IST"),
        "current_date": now.strftime("%A, %d %b %Y"),
        "commodities_count": len(MCX_COMMODITIES),
    }


# ---------------------------------------------------------------------------
# Commodity List (all with live prices)
# ---------------------------------------------------------------------------

@router.get("/list")
async def get_commodity_list():
    """Returns all MCX commodities with live LTP, change%, metadata."""
    from commodity_market_data import get_commodity_prices, MCX_COMMODITIES, get_market_status

    prices = get_commodity_prices()
    market_status = get_market_status()

    result = []
    for sym, meta in MCX_COMMODITIES.items():
        price_data = prices.get(sym, {})
        ltp        = price_data.get("ltp", 0.0)
        change     = price_data.get("change", 0.0)
        change_pct = price_data.get("change_pct", 0.0)
        volume     = price_data.get("volume", 0)
        high       = price_data.get("high", ltp)
        low        = price_data.get("low", ltp)
        open_      = price_data.get("open", ltp)
        source     = price_data.get("source", "none")

        result.append({
            "symbol":      sym,
            "name":        meta["name"],
            "exchange":    "MCX",
            "emoji":       meta["emoji"],
            "color":       meta["color"],
            "lot_size":    meta["lot_size"],
            "unit":        meta["unit"],
            "ltp":         ltp,
            "open":        open_,
            "high":        high,
            "low":         low,
            "change":      change,
            "change_pct":  change_pct,
            "is_up":       change >= 0,
            "volume":      volume,
            "data_source": source,
            "token":       meta.get("token"),
        })

    return {
        "market_status": market_status,
        "count":         len(result),
        "commodities":   result,
        "updated_at":    datetime.datetime.now(IST).isoformat(),
    }


# ---------------------------------------------------------------------------
# Single Commodity Price
# ---------------------------------------------------------------------------

@router.get("/price/{symbol}")
async def get_commodity_price(symbol: str):
    """Live price for a single MCX commodity."""
    from commodity_market_data import get_commodity_price, MCX_COMMODITIES
    sym = symbol.upper()
    if sym not in MCX_COMMODITIES:
        raise HTTPException(status_code=404, detail=f"Commodity '{symbol}' not found. Available: {list(MCX_COMMODITIES.keys())}")
    data = get_commodity_price(sym)
    if not data:
        raise HTTPException(status_code=503, detail="Price data unavailable — backend fetching")
    return data


# ---------------------------------------------------------------------------
# Chart Data (OHLCV)
# ---------------------------------------------------------------------------

@router.get("/chart/{symbol}")
async def get_commodity_chart(
    symbol: str,
    period: str   = Query("30d", description="Period: 5d/30d/90d/1y"),
    interval: str = Query("1d", description="Interval: 1h/1d/1wk"),
):
    """OHLCV chart data for a commodity. Uses yfinance global futures as proxy."""
    from commodity_market_data import MCX_COMMODITIES
    import yfinance as yf

    sym = symbol.upper()
    meta = MCX_COMMODITIES.get(sym)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Unknown commodity: {symbol}")

    # Map periods
    period_map  = {"5d": "5d", "30d": "1mo", "90d": "3mo", "6mo": "6mo", "1y": "1y", "1mo": "1mo", "3mo": "3mo"}
    yf_period   = period_map.get(period, "1mo")
    yf_interval = interval if interval in ["1m","5m","15m","1h","1d","1wk"] else "1d"

    yf_sym_map = {
        "CRUDEOIL": "CL=F", "CRUDEOILM": "CL=F",
        "GOLD": "GC=F", "GOLDM": "GC=F",
        "SILVER": "SI=F", "SILVERM": "SI=F",
        "NATURALGAS": "NG=F", "COPPER": "HG=F",
    }
    yf_symbol = meta.get("yf_symbol") or meta.get("yf") or yf_sym_map.get(sym, "CL=F")

    try:
        ticker = yf.Ticker(yf_symbol)
        hist = ticker.history(period=yf_period, interval=yf_interval)
        if hist.empty:
            raise HTTPException(status_code=503, detail="Chart data unavailable")

        # Approximate USD → INR for MCX
        try:
            usdinr = yf.Ticker("INR=X").fast_info.get("last_price", 84.0) or 84.0
        except Exception:
            usdinr = 84.0

        multipliers = {
            "CRUDEOIL":   usdinr,
            "CRUDEOILM":  usdinr,
            "GOLD":       (usdinr / 31.1035) * 10,
            "GOLDM":      (usdinr / 31.1035) * 10,
            "SILVER":     (usdinr / 31.1035) * 1000,
            "SILVERM":    (usdinr / 31.1035) * 1000,
            "NATURALGAS": usdinr,
            "COPPER":     (usdinr / 0.453592),
        }
        mult = multipliers.get(sym, usdinr)

        candles = []
        for idx, row in hist.iterrows():
            candles.append({
                "timestamp": idx.isoformat(),
                "open":      round(float(row["Open"])  * mult, 2),
                "high":      round(float(row["High"])  * mult, 2),
                "low":       round(float(row["Low"])   * mult, 2),
                "close":     round(float(row["Close"]) * mult, 2),
                "volume":    int(row["Volume"]) if row["Volume"] else 0,
            })

        # Summary stats
        closes = [c["close"] for c in candles]
        last   = closes[-1]  if closes else 0
        first  = closes[0]   if closes else 0
        period_change     = round(last - first, 2)
        period_change_pct = round((period_change / first) * 100, 2) if first else 0

        return {
            "symbol":             sym,
            "name":               meta["name"],
            "interval":           yf_interval,
            "period":             period,
            "candles":            candles,
            "candles_count":      len(candles),
            "period_change":      period_change,
            "period_change_pct":  period_change_pct,
            "source":             "yfinance_global_futures",
            "note":               f"Global {yf_symbol} futures converted to INR proxy",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Commodity Chart] {sym}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# AI Signals
# ---------------------------------------------------------------------------

@router.get("/signals")
async def get_all_commodity_signals():
    """AI buy/sell signals for all MCX commodities."""
    from commodity_market_data import compute_commodity_signal, MCX_COMMODITIES, get_market_status
    import asyncio, concurrent.futures

    market_status = get_market_status()
    results = []

    # Compute in parallel using thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {executor.submit(compute_commodity_signal, sym): sym for sym in MCX_COMMODITIES}
        for future in concurrent.futures.as_completed(future_map):
            try:
                results.append(future.result())
            except Exception as e:
                sym = future_map[future]
                results.append({"symbol": sym, "error": str(e), "signal": "ERROR"})

    # Sort: STRONG first, then BUY/SELL, then NEUTRAL
    priority = {"STRONG BUY": 0, "STRONG SELL": 1, "BUY": 2, "SELL": 3, "NEUTRAL": 4, "WAIT": 5}
    results.sort(key=lambda x: priority.get(x.get("signal", "WAIT"), 6))

    return {
        "market_status": market_status,
        "signals":       results,
        "count":         len(results),
        "generated_at":  datetime.datetime.now(IST).isoformat(),
    }


@router.get("/signal/{symbol}")
async def get_commodity_signal(symbol: str):
    """AI signal for a single MCX commodity."""
    from commodity_market_data import compute_commodity_signal, MCX_COMMODITIES, get_market_status
    sym = symbol.upper()
    if sym not in MCX_COMMODITIES:
        raise HTTPException(status_code=404, detail=f"Unknown commodity: {symbol}")
    signal = compute_commodity_signal(sym)
    signal["market_status"] = get_market_status()
    return signal


# ---------------------------------------------------------------------------
# Test push notification
# ---------------------------------------------------------------------------

@router.post("/test-signal")
async def test_commodity_signal_notification(symbol: str = "CRUDEOIL"):
    """
    Trigger a test push notification for a commodity signal.
    Used for development verification.
    """
    from commodity_market_data import compute_commodity_signal, _send_commodity_signal_notification, MCX_COMMODITIES
    sym = symbol.upper()
    if sym not in MCX_COMMODITIES:
        raise HTTPException(status_code=404, detail=f"Unknown commodity: {symbol}")
    signal = compute_commodity_signal(sym)
    signal["signal"] = "STRONG BUY"   # Force for test
    signal["direction"] = "BUY"
    _send_commodity_signal_notification(signal)
    return {"message": f"Test push notification triggered for {sym}", "signal": signal}


# ---------------------------------------------------------------------------
# Commodity Options Chain (Crude Oil, Gold — if MCX options available)
# ---------------------------------------------------------------------------

@router.get("/options/{symbol}")
async def get_commodity_options(symbol: str):
    """
    MCX Options chain for commodities (Crude Oil, Gold have MCX options).
    Returns strikes via Angel One NFO/MCX options search.
    """
    from commodity_market_data import get_commodity_price, MCX_COMMODITIES
    sym = symbol.upper()
    if sym not in MCX_COMMODITIES:
        raise HTTPException(status_code=404, detail=f"Unknown commodity: {symbol}")

    price_data = get_commodity_price(sym)
    if not price_data or not price_data.get("ltp"):
        return {"symbol": sym, "chain": [], "note": "Live price unavailable — options chain requires live LTP"}

    ltp = price_data["ltp"]
    # Build theoretical chain (Black-Scholes based)
    # MCX Crude Oil options: strikes spaced 50 INR, Gold: 100 INR
    strike_step = {"CRUDEOIL": 50, "CRUDEOILM": 50, "GOLD": 100, "GOLDM": 100, "SILVER": 500, "SILVERM": 500, "NATURALGAS": 5, "COPPER": 5}.get(sym, 50)
    atm_strike = round(ltp / strike_step) * strike_step
    strikes = [atm_strike + i * strike_step for i in range(-5, 6)]

    import math
    import datetime

    def bs_price(S, K, T, r, sigma, opt_type):
        if T <= 0 or sigma <= 0:
            intrinsic = max(S - K, 0) if opt_type == "CE" else max(K - S, 0)
            return round(intrinsic, 2)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        from scipy.stats import norm
        try:
            if opt_type == "CE":
                return round(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2), 2)
            else:
                return round(K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1), 2)
        except Exception:
            intrinsic = max(S - K, 0) if opt_type == "CE" else max(K - S, 0)
            return round(intrinsic, 2)

    # Approximate: T = 20 trading days / 252, vol = 25%, r = 6.5%
    T = 20 / 252
    sigma = 0.25
    r = 0.065

    chain = []
    for strike in strikes:
        ce = bs_price(ltp, strike, T, r, sigma, "CE")
        pe = bs_price(ltp, strike, T, r, sigma, "PE")
        otm_distance = abs(ltp - strike)
        chain.append({
            "strike":    strike,
            "is_atm":   strike == atm_strike,
            "ce_ltp":   ce,
            "pe_ltp":   pe,
            "ce_iv":    round(sigma * 100 + (otm_distance / ltp) * 20, 2),
            "pe_iv":    round(sigma * 100 + (otm_distance / ltp) * 20, 2),
            "ce_oi":    0,
            "pe_oi":    0,
        })

    return {
        "symbol":    sym,
        "name":      MCX_COMMODITIES[sym]["name"],
        "spot":      ltp,
        "atm":       atm_strike,
        "chain":     chain,
        "note":      "Theoretical BS prices — live MCX option tokens require Angel One MCXFO subscription",
        "generated_at": datetime.datetime.now(IST).isoformat(),
    }


# ---------------------------------------------------------------------------
# MCX Commodity LTP Calculator (Investing Daddy / CoA 1.0 & 2.0 Engine)
# ---------------------------------------------------------------------------

@router.get("/ltp-calculator/{symbol}")
async def get_commodity_ltp_calculator(symbol: str):
    """
    Supercharged Institutional LTP Calculator for MCX Commodities.
    Inspired by Dr. Vinay Prakash Tiwari's Investing Daddy LTP Calculator:
    - Exact Support & Resistance Strikes
    - Extension of Support (EOS) & Extension of Resistance (EOR)
    - WTT (Weak Towards Top) & WTB (Weak Towards Bottom) State Analysis
    - Chart of Accuracy (CoA 1.0 & 2.0) 9-Scenario Recommendation
    - Upper Reversal (UR) & Lower Reversal (LR) for every strike
    """
    from commodity_market_data import get_commodity_price, MCX_COMMODITIES
    from analytics_ltp_calculator import build_commodity_ltp_calculator

    sym = symbol.upper().replace(".NS", "").replace(".BO", "")
    if sym not in MCX_COMMODITIES:
        raise HTTPException(status_code=404, detail=f"Commodity '{symbol}' not found. Available: {list(MCX_COMMODITIES.keys())}")

    price_data = get_commodity_price(sym)
    meta = MCX_COMMODITIES[sym]
    ltp = float(price_data.get("ltp") or meta["baseline_ltp"])

    # Compute Supercharged LTP Calculator metrics
    ltp_calc = build_commodity_ltp_calculator(sym, ltp)

    # Attach commodity metadata
    ltp_calc["commodity_meta"] = {
        "symbol": sym,
        "name": meta["name"],
        "emoji": meta["emoji"],
        "lot_size": meta["lot_size"],
        "unit": meta["unit"],
        "strike_step": ltp_calc.get("step", 50),
        "change": price_data.get("change", 0.0),
        "change_pct": price_data.get("change_pct", 0.0),
        "is_up": price_data.get("is_up", True),
    }

    return ltp_calc

