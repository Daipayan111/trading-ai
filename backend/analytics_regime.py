"""
Market Regime Detection Module
================================
Uses statistical methods to classify the market into:
  - BULL  (Trending Up)
  - BEAR  (Trending Down)
  - SIDEWAYS (Ranging / Mean-Reverting)

This module adjusts signal weights dynamically based on the detected regime.
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def detect_market_regime(df: pd.DataFrame) -> dict:
    """
    Detect market regime using multiple methods:
    1. ADX-based trend strength
    2. 200 SMA slope
    3. Volatility percentile (VIX-proxy)
    4. Returns a regime label and signal weight multipliers
    """
    if len(df) < 50:
        return _default_regime()

    latest = df.iloc[-1]
    close = df['Close']

    # --- Method 1: ADX-based ---
    adx = float(latest.get('ADX', 20))
    adx_pos = float(latest.get('ADX_Pos', 0))
    adx_neg = float(latest.get('ADX_Neg', 0))

    # --- Method 2: 200 SMA Slope ---
    sma200_slope = 0.0
    if 'SMA_200' in df.columns and len(df) >= 210:
        sma_series = df['SMA_200'].dropna()
        if len(sma_series) >= 10:
            recent_sma = sma_series.iloc[-1]
            old_sma = sma_series.iloc[-10]
            sma200_slope = (recent_sma - old_sma) / old_sma * 100  # % change over 10 days

    # --- Method 3: Price vs SMA200 ---
    price_vs_sma200 = 0
    if 'SMA_200' in df.columns:
        sma200_val = float(latest.get('SMA_200', close.iloc[-1]))
        current_price = float(latest['Close'])
        price_vs_sma200 = (current_price - sma200_val) / sma200_val * 100

    # --- Method 4: Volatility Regime ---
    if len(df) >= 30:
        recent_vol = close.pct_change().rolling(10).std().iloc[-1]
        hist_vol = close.pct_change().rolling(252).std().iloc[-1] if len(df) >= 252 else close.pct_change().std()
        vol_percentile = (recent_vol / hist_vol * 100) if hist_vol > 0 else 50
    else:
        vol_percentile = 50

    # --- Method 5: EMA Alignment Score ---
    ema_alignment = 0
    if all(k in df.columns for k in ['EMA_9', 'EMA_21', 'EMA_50']):
        e9 = float(latest.get('EMA_9', 0))
        e21 = float(latest.get('EMA_21', 0))
        e50 = float(latest.get('EMA_50', 0))
        if e9 > e21 > e50:
            ema_alignment = 2   # Fully bullish aligned
        elif e9 < e21 < e50:
            ema_alignment = -2  # Fully bearish aligned
        elif e9 > e21:
            ema_alignment = 1
        elif e9 < e21:
            ema_alignment = -1

    # --- Composite Regime Score ---
    # Each component gives -2 to +2, total range: -10 to +10
    regime_score = 0.0

    # ADX component: strong trend?
    if adx > 30:
        if adx_pos > adx_neg:
            regime_score += 2   # Strong Bull trend
        else:
            regime_score -= 2   # Strong Bear trend
    elif adx > 20:
        if adx_pos > adx_neg:
            regime_score += 1
        else:
            regime_score -= 1
    else:
        # Weak trend — sideways
        regime_score += 0

    # SMA slope component
    if sma200_slope > 0.5:
        regime_score += 2
    elif sma200_slope > 0:
        regime_score += 1
    elif sma200_slope < -0.5:
        regime_score -= 2
    elif sma200_slope < 0:
        regime_score -= 1

    # Price vs SMA200
    if price_vs_sma200 > 5:
        regime_score += 2
    elif price_vs_sma200 > 0:
        regime_score += 1
    elif price_vs_sma200 < -5:
        regime_score -= 2
    elif price_vs_sma200 < 0:
        regime_score -= 1

    # EMA alignment
    regime_score += ema_alignment

    # Volatility (high vol = uncertainty / crisis)
    if vol_percentile > 150:
        regime_score -= 1  # High stress / bearish bias

    # --- Classify Regime ---
    if regime_score >= 4:
        regime = "BULL"
        regime_emoji = "🟢"
        description = "Strong Uptrend — বড় BUY signals কাজ করবে"
        buy_weight_multiplier = 1.5
        sell_weight_multiplier = 0.7
        strategy_hint = "Trend Following — Breakouts buy করো, dips hold করো"
    elif regime_score <= -4:
        regime = "BEAR"
        regime_emoji = "🔴"
        description = "Strong Downtrend — SELL/SHORT signals বিশ্বাসযোগ্য"
        buy_weight_multiplier = 0.7
        sell_weight_multiplier = 1.5
        strategy_hint = "Trend Following (Short) — Rallies sell করো, cash রাখো"
    else:
        regime = "SIDEWAYS"
        regime_emoji = "🟡"
        description = "Ranging Market — Mean Reversion কাজ করবে"
        buy_weight_multiplier = 1.0
        sell_weight_multiplier = 1.0
        strategy_hint = "Mean Reversion — Support buy, Resistance sell, tight SL"

    return {
        "regime": regime,
        "regime_emoji": regime_emoji,
        "regime_score": round(regime_score, 2),
        "description": description,
        "strategy_hint": strategy_hint,
        "buy_weight_multiplier": buy_weight_multiplier,
        "sell_weight_multiplier": sell_weight_multiplier,
        "components": {
            "adx_strength": round(adx, 2),
            "sma200_slope_pct": round(sma200_slope, 3),
            "price_vs_sma200_pct": round(price_vs_sma200, 2),
            "ema_alignment_score": ema_alignment,
            "volatility_percentile": round(float(vol_percentile), 1)
        }
    }


def apply_regime_adjusted_score(combined_score: float, regime_info: dict, signal_direction: str) -> float:
    """
    Adjust the combined signal score based on the market regime.
    Bull market: BUY signals amplified, SELL signals dampened.
    Bear market: SELL signals amplified, BUY signals dampened.
    """
    if signal_direction == "BUY":
        return combined_score * regime_info.get("buy_weight_multiplier", 1.0)
    elif signal_direction == "SELL":
        return combined_score * regime_info.get("sell_weight_multiplier", 1.0)
    return combined_score


def _default_regime() -> dict:
    return {
        "regime": "SIDEWAYS",
        "regime_emoji": "🟡",
        "regime_score": 0,
        "description": "Not enough data to determine regime",
        "strategy_hint": "Use caution — insufficient data",
        "buy_weight_multiplier": 1.0,
        "sell_weight_multiplier": 1.0,
        "components": {}
    }
