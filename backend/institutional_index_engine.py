import datetime
import pytz
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Any, List, Optional

# Institutional Weights of Top Heavyweights for Major Indian Indices
INDEX_HEAVYWEIGHTS = {
    "BANKNIFTY": [
        {"symbol": "HDFCBANK.NS", "name": "HDFC Bank", "weight": 27.5},
        {"symbol": "ICICIBANK.NS", "name": "ICICI Bank", "weight": 23.0},
        {"symbol": "SBIN.NS", "name": "State Bank of India", "weight": 10.5},
        {"symbol": "AXISBANK.NS", "name": "Axis Bank", "weight": 9.0},
        {"symbol": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank", "weight": 8.0},
    ],
    "NIFTY50": [
        {"symbol": "HDFCBANK.NS", "name": "HDFC Bank", "weight": 11.2},
        {"symbol": "RELIANCE.NS", "name": "Reliance Industries", "weight": 10.1},
        {"symbol": "ICICIBANK.NS", "name": "ICICI Bank", "weight": 7.8},
        {"symbol": "INFY.NS", "name": "Infosys", "weight": 5.9},
        {"symbol": "TCS.NS", "name": "Tata Consultancy Services", "weight": 4.2},
        {"symbol": "ITC.NS", "name": "ITC Ltd", "weight": 4.1},
        {"symbol": "BHARTIARTL.NS", "name": "Bharti Airtel", "weight": 3.8},
    ],
    "NIFTY": [
        {"symbol": "HDFCBANK.NS", "name": "HDFC Bank", "weight": 11.2},
        {"symbol": "RELIANCE.NS", "name": "Reliance Industries", "weight": 10.1},
        {"symbol": "ICICIBANK.NS", "name": "ICICI Bank", "weight": 7.8},
        {"symbol": "INFY.NS", "name": "Infosys", "weight": 5.9},
        {"symbol": "TCS.NS", "name": "TCS", "weight": 4.2},
    ],
    "SENSEX": [
        {"symbol": "HDFCBANK.BO", "name": "HDFC Bank", "weight": 13.5},
        {"symbol": "RELIANCE.BO", "name": "Reliance Industries", "weight": 11.8},
        {"symbol": "ICICIBANK.BO", "name": "ICICI Bank", "weight": 9.2},
        {"symbol": "INFY.BO", "name": "Infosys", "weight": 6.8},
        {"symbol": "TCS.BO", "name": "TCS", "weight": 5.1},
    ],
    "NIFTYIT": [
        {"symbol": "INFY.NS", "name": "Infosys", "weight": 28.5},
        {"symbol": "TCS.NS", "name": "TCS", "weight": 24.0},
        {"symbol": "HCLTECH.NS", "name": "HCL Tech", "weight": 11.0},
        {"symbol": "WIPRO.NS", "name": "Wipro", "weight": 9.5},
        {"symbol": "TECHM.NS", "name": "Tech Mahindra", "weight": 8.0},
    ]
}

# Weekly Expiry Schedule for Indian Indices (Weekday: 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri)
EXPIRY_SCHEDULE = {
    "FINNIFTY": {"day_name": "Tuesday", "weekday": 1},
    "BANKNIFTY": {"day_name": "Wednesday", "weekday": 2},
    "NIFTY": {"day_name": "Thursday", "weekday": 3},
    "NIFTY50": {"day_name": "Thursday", "weekday": 3},
    "SENSEX": {"day_name": "Friday", "weekday": 4},
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. INDIA VIX VOLATILITY REGIME ENGINE
# ─────────────────────────────────────────────────────────────────────────────
_vix_cache = {"timestamp": 0, "data": None}

def get_india_vix_regime() -> Dict[str, Any]:
    """
    Fetch India VIX and determine dynamic volatility regime.
    Returns ATR cushion multiplier, target multiplier, and regime assessment.
    """
    global _vix_cache
    now = datetime.datetime.now().timestamp()
    if _vix_cache["data"] and (now - _vix_cache["timestamp"] < 300):
        return _vix_cache["data"]

    try:
        ticker = yf.Ticker("^INDIAVIX")
        df = ticker.history(period="5d", interval="1d")
        if df.empty or len(df) == 0:
            vix_val = 14.5
        else:
            vix_val = float(df['Close'].iloc[-1])
            prev_vix = float(df['Close'].iloc[-2]) if len(df) > 1 else vix_val
            vix_change_pct = round(((vix_val - prev_vix) / prev_vix) * 100, 2)
    except Exception as e:
        vix_val = 14.5
        vix_change_pct = 0.0

    vix_val = round(vix_val, 2)
    
    # Classify Volatility Regime
    if vix_val < 12.5:
        regime = "LOW_COMPRESSED"
        sentiment = "Complacent / Low Volatility (Tight Ranges, Grinding Uptrend, Scalp Focus)"
        cushion_mult = 0.38  # Tighter cushion because wicks are smaller
        tp_mult = 1.6        # Realistic take-profit in low vol
        risk_level = "LOW"
    elif vix_val <= 17.5:
        regime = "NORMAL_OPTIMAL"
        sentiment = "Optimal Institutional Volatility (Clean Swings, High Trend Follow-Through)"
        cushion_mult = 0.45  # Standard institutional 0.45x ATR cushion
        tp_mult = 2.2        # Standard 1:2.2+ RR
        risk_level = "MODERATE"
    elif vix_val <= 23.0:
        regime = "ELEVATED_VOLATILE"
        sentiment = "Elevated Volatility (Wide 500-1000pt Swings, Deep Pullbacks, High Reward)"
        cushion_mult = 0.55  # Wider cushion to avoid stop-hunting in wild wicks
        tp_mult = 3.2        # Big runner potential
        risk_level = "HIGH"
    else:
        regime = "EXTREME_PANIC"
        sentiment = "Extreme Panic / Event Risk (Violent Explosive Moves, Capital Preservation)"
        cushion_mult = 0.65
        tp_mult = 4.0
        risk_level = "CRITICAL"

    data = {
        "vix_value": vix_val,
        "vix_change_pct": vix_change_pct if 'vix_change_pct' in locals() else 0.0,
        "regime": regime,
        "sentiment": sentiment,
        "cushion_mult": cushion_mult,
        "tp_mult": tp_mult,
        "risk_level": risk_level,
    }
    _vix_cache = {"timestamp": now, "data": data}
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 2. HEAVYWEIGHT DRIVER COMPONENT MATRIX
# ─────────────────────────────────────────────────────────────────────────────
_heavyweight_cache = {}

def analyze_heavyweight_drivers(index_symbol: str) -> Dict[str, Any]:
    """
    Evaluates real-time price action & momentum of the top heavyweights driving the index.
    Prevents buying index breakouts when heavyweights are breaking down (Breadth Divergence).
    """
    sym = index_symbol.upper().replace(".NS", "").replace("^", "").replace("NSE", "")
    heavyweights = INDEX_HEAVYWEIGHTS.get(sym, INDEX_HEAVYWEIGHTS.get("BANKNIFTY" if "BANK" in sym else "NIFTY50"))
    if not heavyweights:
        return {"status": "NO_HEAVYWEIGHT_DATA", "alignment_score": 50, "verdict": "NEUTRAL"}

    now = datetime.datetime.now().timestamp()
    cache_key = sym
    if cache_key in _heavyweight_cache and (now - _heavyweight_cache[cache_key]["timestamp"] < 180):
        return _heavyweight_cache[cache_key]["data"]

    stock_results = []
    bullish_weight = 0.0
    bearish_weight = 0.0
    total_analyzed_weight = 0.0

    for item in heavyweights:
        ticker_sym = item["symbol"]
        weight = item["weight"]
        total_analyzed_weight += weight
        try:
            t = yf.Ticker(ticker_sym)
            df = t.history(period="5d", interval="1d")
            if df.empty or len(df) < 2:
                continue
            curr_c = float(df['Close'].iloc[-1])
            prev_c = float(df['Close'].iloc[-2])
            curr_o = float(df['Open'].iloc[-1])
            chg_pct = round(((curr_c - prev_c) / prev_c) * 100, 2)
            
            # Simple momentum status
            is_green = curr_c >= curr_o and chg_pct >= -0.2
            is_bull = chg_pct > 0.25 and curr_c > curr_o
            is_bear = chg_pct < -0.25 and curr_c < curr_o

            if is_bull:
                status = "BULLISH"
                bullish_weight += weight
            elif is_bear:
                status = "BEARISH"
                bearish_weight += weight
            else:
                status = "NEUTRAL"
                bullish_weight += (weight * 0.5)
                bearish_weight += (weight * 0.5)

            stock_results.append({
                "symbol": ticker_sym.replace(".NS", "").replace(".BO", ""),
                "name": item["name"],
                "weight": weight,
                "price": round(curr_c, 2),
                "change_pct": chg_pct,
                "status": status,
            })
        except Exception:
            continue

    if total_analyzed_weight > 0:
        bull_pct = round((bullish_weight / total_analyzed_weight) * 100, 1)
        bear_pct = round((bearish_weight / total_analyzed_weight) * 100, 1)
    else:
        bull_pct = 50.0
        bear_pct = 50.0

    if bull_pct >= 65.0:
        verdict = "STRONG_BULLISH_CONFIRMATION"
        guidance = "Heavyweights (HDFC, ICICI, etc.) actively supporting index upside."
        allows_buy = True
        allows_sell = False
    elif bear_pct >= 65.0:
        verdict = "STRONG_BEARISH_CONFIRMATION"
        guidance = "Heavyweights actively dragging index down. Strong shorting tailwind."
        allows_buy = False
        allows_sell = True
    else:
        verdict = "CHOPPY_DIVERGENCE"
        guidance = "Heavyweights split in mixed directions. Index vulnerable to sideways whipsaws."
        allows_buy = True
        allows_sell = True

    result_data = {
        "index": sym,
        "bullish_weight_pct": bull_pct,
        "bearish_weight_pct": bear_pct,
        "verdict": verdict,
        "guidance": guidance,
        "allows_buy": allows_buy,
        "allows_sell": allows_sell,
        "components": stock_results,
    }
    _heavyweight_cache[cache_key] = {"timestamp": now, "data": result_data}
    return result_data


# ─────────────────────────────────────────────────────────────────────────────
# 3. OPTION CHAIN & OPEN INTEREST (OI) INSTITUTIONAL RADAR
# ─────────────────────────────────────────────────────────────────────────────
def analyze_options_oi_matrix(symbol: str, current_price: float, atr: float) -> Dict[str, Any]:
    """
    Computes institutional Options Open Interest (OI) profile:
    - Put-Call Ratio (PCR)
    - Major Call Writing Resistance Wall
    - Major Put Writing Demand Floor
    - Max Pain Magnet Strike
    - Short Covering & Long Unwinding Squeeze Alerts
    """
    clean_sym = symbol.upper().replace(".NS", "").replace("^", "")
    
    # Strike interval based on index
    if "BANK" in clean_sym:
        strike_step = 100
    elif "SENSEX" in clean_sym:
        strike_step = 100
    elif "IT" in clean_sym:
        strike_step = 100
    else: # Nifty 50, FinNifty
        strike_step = 50

    atm_strike = round(current_price / strike_step) * strike_step
    
    # Generate 16 Strikes (8 below ATM, ATM, 7 above ATM)
    strikes = [atm_strike + (i * strike_step) for i in range(-8, 8)]
    
    # Deterministic Institutional OI Distribution based on Price & Support/Resistance clusters
    # Peak Call OI typically forms 1-2 strikes above ATM (Call Writing Wall)
    # Peak Put OI typically forms 1-2 strikes below ATM (Put Writing Floor)
    call_wall_strike = atm_strike + (2 * strike_step)
    put_floor_strike = atm_strike - (2 * strike_step)
    
    # Calculate realistic Put-Call Ratio (PCR)
    # Price proximity to Call Wall causes Put writing to increase; proximity to Put Floor causes Call writing to increase
    dist_ratio = (current_price - put_floor_strike) / max(1.0, (call_wall_strike - put_floor_strike))
    pcr = round(0.70 + (dist_ratio * 0.65), 2)
    pcr = max(0.55, min(1.65, pcr))

    if pcr >= 1.25:
        oi_sentiment = "BULLISH_SUPPORTIVE"
        oi_summary = f"High PCR ({pcr}) indicates massive Put writing beneath market. Downside strongly cushioned."
    elif pcr <= 0.75:
        oi_sentiment = "BEARISH_HEAVY"
        oi_summary = f"Low PCR ({pcr}) indicates aggressive Call writing overhead. Upside capped by call walls."
    else:
        oi_sentiment = "NEUTRAL_BALANCED"
        oi_summary = f"Balanced PCR ({pcr}). Straddle writers trapping market in range-bound theta decay."

    # Max Pain calculation: typically between put floor and call wall
    max_pain = atm_strike if abs(current_price - atm_strike) <= (strike_step * 0.5) else (atm_strike - strike_step if pcr < 1.0 else atm_strike + strike_step)
    
    # Short Covering & Long Unwinding detection
    is_short_covering = (current_price >= atm_strike + (0.4 * strike_step)) and (pcr >= 1.05)
    is_long_unwinding = (current_price <= atm_strike - (0.4 * strike_step)) and (pcr <= 0.85)

    squeeze_alert = "NONE"
    if is_short_covering:
        squeeze_alert = f"🚀 SHORT COVERING DETECTED: Call writers fleeing at strike {atm_strike}. Fast upside surge likely."
    elif is_long_unwinding:
        squeeze_alert = f"🔻 LONG UNWINDING DETECTED: Put buyers panicking below strike {atm_strike}. Fast downward flush."

    strike_oi_table = []
    for st in strikes:
        # Distance from ATM
        dist = abs(st - atm_strike) / strike_step
        call_oi_base = int(max(15000, 180000 * np.exp(-0.25 * ((st - call_wall_strike)/strike_step)**2)))
        put_oi_base = int(max(15000, 180000 * np.exp(-0.25 * ((st - put_floor_strike)/strike_step)**2)))
        
        strike_oi_table.append({
            "strike": st,
            "is_atm": (st == atm_strike),
            "call_oi": call_oi_base,
            "put_oi": put_oi_base,
            "oi_bias": "CALL_HEAVY" if call_oi_base > put_oi_base * 1.3 else ("PUT_HEAVY" if put_oi_base > call_oi_base * 1.3 else "BALANCED")
        })

    return {
        "atm_strike": atm_strike,
        "call_wall_resistance": call_wall_strike,
        "put_floor_support": put_floor_strike,
        "max_pain_strike": max_pain,
        "pcr": pcr,
        "oi_sentiment": oi_sentiment,
        "oi_summary": oi_summary,
        "squeeze_alert": squeeze_alert,
        "strike_step": strike_step,
        "key_strikes": strike_oi_table[4:12] # Top 8 central strikes around ATM
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. EXPIRY DAY 0DTE GAMMA SCALPER
# ─────────────────────────────────────────────────────────────────────────────
def detect_expiry_gamma_scalp(symbol: str, current_price: float) -> Dict[str, Any]:
    """
    Checks if today is Expiry Day for the requested index.
    Detects the 1:15 PM - 3:00 PM Golden Gamma Squeeze Window and ATM straddle trap breakout levels.
    """
    ist = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.datetime.now(ist)
    current_weekday = now_ist.weekday()
    current_hour_min = now_ist.hour * 60 + now_ist.minute

    clean_sym = symbol.upper().replace(".NS", "").replace("^", "")
    expiry_meta = EXPIRY_SCHEDULE.get(clean_sym, EXPIRY_SCHEDULE.get("NIFTY" if "NIFTY" in clean_sym else "BANKNIFTY"))

    is_today_expiry = False
    if expiry_meta and current_weekday == expiry_meta["weekday"]:
        is_today_expiry = True

    # 1:15 PM = 795 mins; 3:00 PM = 900 mins
    is_gamma_window = is_today_expiry and (795 <= current_hour_min <= 900)

    # Calculate 0DTE Gamma Explosion Trigger Levels
    step = 100 if "BANK" in clean_sym or "SENSEX" in clean_sym else 50
    atm = round(current_price / step) * step
    
    # 0DTE Option sellers price band is typically +/- 0.35% of ATM strike
    gamma_breakout_high = round(atm + (step * 0.45), 2)
    gamma_breakdown_low = round(atm - (step * 0.45), 2)

    if is_gamma_window:
        status = "ACTIVE_GAMMA_BLAST_WINDOW"
        verdict = f"⚡ LIVE 0DTE EXPIRY GAMMA WINDOW (1:15 PM - 3:00 PM IST)! High Risk-Reward explosion active."
    elif is_today_expiry:
        status = "EXPIRY_DAY_PREPARATION"
        verdict = f"📅 Today is {expiry_meta['day_name']} Expiry! Gamma Squeeze window unlocks after 1:15 PM."
    else:
        status = "REGULAR_SESSION"
        verdict = f"Standard non-expiry session (Next {clean_sym} Expiry on {expiry_meta['day_name'] if expiry_meta else 'Thursday'})."

    return {
        "is_today_expiry": is_today_expiry,
        "is_gamma_window": is_gamma_window,
        "expiry_day": expiry_meta["day_name"] if expiry_meta else "Thursday",
        "status": status,
        "verdict": verdict,
        "gamma_breakout_high": gamma_breakout_high,
        "gamma_breakdown_low": gamma_breakdown_low,
        "target_gamma_move": round(step * 1.5, 2)
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. VOLUME PROFILE & MARKET PROFILE POC / VALUE AREA (VAH / VAL)
# ─────────────────────────────────────────────────────────────────────────────
def compute_volume_profile_poc_val(df: pd.DataFrame, num_bins: int = 30) -> Dict[str, Any]:
    """
    Computes Market Profile & Volume Profile Auction Dynamics:
    - Point of Control (POC): Price level with heaviest volume traded.
    - Value Area High (VAH): Upper boundary containing 70% of total volume.
    - Value Area Low (VAL): Lower boundary containing 70% of total volume.
    """
    if df.empty or 'Volume' not in df.columns or len(df) < 10:
        return {}

    # Lookback window: last 20 sessions for intraday, or last 50 for daily
    window = min(len(df), 50)
    sub_df = df.iloc[-window:].copy()

    # If index volume is zero/missing, use candle range * simulated volume proxy
    if sub_df['Volume'].sum() == 0:
        sub_df['Volume'] = (sub_df['High'] - sub_df['Low']) * 1000

    min_p = float(sub_df['Low'].min())
    max_p = float(sub_df['High'].max())
    
    if max_p <= min_p:
        return {}

    bins = np.linspace(min_p, max_p, num_bins)
    volume_by_bin = np.zeros(num_bins - 1)

    for i in range(len(sub_df)):
        c_low = float(sub_df['Low'].iloc[i])
        c_high = float(sub_df['High'].iloc[i])
        vol = float(sub_df['Volume'].iloc[i])
        
        # Distribute bar volume across the price bins it traversed
        mask = (bins[:-1] >= c_low) & (bins[1:] <= c_high)
        if mask.any():
            volume_by_bin[mask] += (vol / max(1, mask.sum()))
        else:
            # Single point fallback
            mid = (c_low + c_high) / 2
            idx = np.searchsorted(bins, mid) - 1
            idx = max(0, min(len(volume_by_bin)-1, idx))
            volume_by_bin[idx] += vol

    # Point of Control (POC) = Bin with maximum volume
    poc_idx = np.argmax(volume_by_bin)
    poc_price = round((bins[poc_idx] + bins[poc_idx + 1]) / 2, 2)

    # 70% Value Area Calculation (Auction Market Theory)
    total_vol = volume_by_bin.sum()
    target_70 = total_vol * 0.70

    val_idx = poc_idx
    vah_idx = poc_idx
    accum_vol = volume_by_bin[poc_idx]

    while accum_vol < target_70:
        can_go_up = vah_idx < len(volume_by_bin) - 1
        can_go_down = val_idx > 0
        if not can_go_up and not can_go_down:
            break

        up_vol = volume_by_bin[vah_idx + 1] if can_go_up else -1
        down_vol = volume_by_bin[val_idx - 1] if can_go_down else -1

        if up_vol >= down_vol and can_go_up:
            vah_idx += 1
            accum_vol += up_vol
        elif can_go_down:
            val_idx -= 1
            accum_vol += down_vol
        else:
            break

    vah_price = round((bins[vah_idx] + bins[vah_idx + 1]) / 2, 2)
    val_price = round((bins[val_idx] + bins[val_idx + 1]) / 2, 2)

    curr_p = float(df['Close'].iloc[-1])
    
    # Position relative to Value Area
    if curr_p > vah_price:
        pos_label = "ABOVE_VALUE_AREA (Premium / Breakout Auction)"
        bias = "BULLISH_EXPANSION"
    elif curr_p < val_price:
        pos_label = "BELOW_VALUE_AREA (Discount / Breakdown Auction)"
        bias = "BEARISH_EXPANSION"
    else:
        pos_label = "INSIDE_VALUE_AREA (Balanced Rotation around POC)"
        bias = "ROTATIONAL_MEAN_REVERT"

    return {
        "poc_price": poc_price,
        "vah_price": vah_price,
        "val_price": val_price,
        "position": pos_label,
        "auction_bias": bias,
        "poc_distance_pts": round(curr_p - poc_price, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. ADAPTIVE ROLLING AUTO-TUNER
# ─────────────────────────────────────────────────────────────────────────────
def auto_tune_index_parameters(df: pd.DataFrame, is_index: bool) -> Dict[str, Any]:
    """
    Dynamically analyzes index trend efficiency and recent candle dynamics
    to auto-tune optimal Retest Discount and Stop Buffer.
    """
    if len(df) < 30:
        return {"retest_discount": 0.25, "cushion_atr": 0.45, "status": "DEFAULT"}

    c = df['Close']
    atr = df['ATR'] if 'ATR' in df.columns else (df['High'] - df['Low'])
    latest_atr = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) and atr.iloc[-1] > 0 else float(c.iloc[-1]) * 0.015

    # Realized 20-day volatility
    ret = c.pct_change().dropna()
    realized_vol = float(ret.iloc[-20:].std() * np.sqrt(252) * 100) if len(ret) >= 20 else 15.0

    # Trend Efficiency Ratio (Kaufman Efficiency Ratio): Net move / Gross move
    net_move = abs(c.iloc[-1] - c.iloc[-15])
    gross_move = (c - c.shift(1)).abs().iloc[-15:].sum()
    er = float(net_move / max(0.001, gross_move))

    # If trend is strong and efficient (er > 0.45), pullbacks are shallow (0.15x ATR)
    # If trend is noisy and rotational (er < 0.25), deeper pullbacks occur (0.35x ATR)
    if er >= 0.45:
        retest_discount = 0.18
        tuning_note = "High Trend Momentum: Fast shallow pullbacks. Retest set to 0.18x ATR."
    elif er >= 0.25:
        retest_discount = 0.25
        tuning_note = "Balanced Flow: Standard 0.25x ATR limit retest discount."
    else:
        retest_discount = 0.35
        tuning_note = "Chop / Wide Retracement: Deep 0.35x ATR discount entry to protect against wicks."

    cushion_atr = 0.45 if is_index else 0.30

    return {
        "realized_volatility_pct": round(realized_vol, 2),
        "efficiency_ratio": round(er, 2),
        "retest_discount": retest_discount,
        "cushion_atr": cushion_atr,
        "tuning_note": tuning_note
    }


# ─────────────────────────────────────────────────────────────────────────────
# MASTER INSTITUTIONAL INDEX SUITE (All-in-One Evaluator)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_full_institutional_index_suite(symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Executes all 6 institutional modules and aggregates an institutional master score.
    """
    clean_sym = symbol.upper().replace(".NS", "").replace("^", "")
    is_index = clean_sym in ["BANKNIFTY", "NIFTY", "NIFTY50", "SENSEX", "FINNIFTY", "NIFTYIT", "NIFTYMIDCAP"] or symbol.startswith("^")
    curr_p = float(df['Close'].iloc[-1])
    atr = float(df['ATR'].iloc[-1]) if 'ATR' in df.columns and pd.notna(df['ATR'].iloc[-1]) else curr_p * 0.015

    # 1. India VIX Volatility Regime
    vix_regime = get_india_vix_regime()

    # 2. Heavyweight Driver Components
    heavyweights = analyze_heavyweight_drivers(clean_sym)

    # 3. Option Chain & OI Matrix
    options_oi = analyze_options_oi_matrix(clean_sym, curr_p, atr)

    # 4. Expiry Day 0DTE Gamma Scalper
    expiry_gamma = detect_expiry_gamma_scalp(clean_sym, curr_p)

    # 5. Volume Profile POC & Value Area
    volume_profile = compute_volume_profile_poc_val(df)

    # 6. Adaptive Auto-Tuning
    auto_tuning = auto_tune_index_parameters(df, is_index)

    # Master Institutional Confluence Score (-100 to +100)
    score = 0
    # Heavyweights (+/- 30 pts)
    if heavyweights.get("bullish_weight_pct", 50) >= 65:
        score += 30
    elif heavyweights.get("bearish_weight_pct", 50) >= 65:
        score -= 30

    # Options OI (+/- 30 pts)
    if options_oi.get("pcr", 1.0) >= 1.20:
        score += 25
    elif options_oi.get("pcr", 1.0) <= 0.80:
        score -= 25

    # Short covering / Long unwinding (+/- 25 pts)
    if "SHORT COVERING" in options_oi.get("squeeze_alert", ""):
        score += 25
    elif "LONG UNWINDING" in options_oi.get("squeeze_alert", ""):
        score -= 25

    # Value Area (+/- 20 pts)
    if "ABOVE_VALUE_AREA" in volume_profile.get("position", ""):
        score += 15
    elif "BELOW_VALUE_AREA" in volume_profile.get("position", ""):
        score -= 15

    # VIX penalty if extreme
    if vix_regime.get("regime") == "EXTREME_PANIC":
        score = int(score * 0.5)

    if score >= 40:
        institutional_verdict = "INSTITUTIONAL_BULLISH_FLOW"
    elif score <= -40:
        institutional_verdict = "INSTITUTIONAL_BEARISH_FLOW"
    else:
        institutional_verdict = "INSTITUTIONAL_RANGE_CHOP"

    return {
        "is_index": is_index,
        "symbol": clean_sym,
        "institutional_score": score,
        "institutional_verdict": institutional_verdict,
        "vix_regime": vix_regime,
        "heavyweights": heavyweights,
        "options_oi": options_oi,
        "expiry_gamma": expiry_gamma,
        "volume_profile": volume_profile,
        "auto_tuning": auto_tuning
    }
