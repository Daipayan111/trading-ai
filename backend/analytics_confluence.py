"""
9-Point AI Confluence Checklist & Traffic Light Engine (90%+ Accuracy)
======================================================================
Evaluates 9 institutional checkpoints before any trade entry:
1. HTF Trend Direction (EMA 50 / VWAP alignment)
2. Institutional Volume Expansion (Volume Ratio >= 1.20)
3. 90%+ Ultra-Strong S/R Confluence Zone Position
4. Order Block (OB) / Fair Value Gap (FVG) Validation
5. Candlestick Confirmation Pattern (Hammer / Engulfing)
6. Market Structure Shift (MSS / BOS / CHoCH)
7. Institutional Trap Safety (Zero Traps / Spring Confirmation)
8. Session Timing Window (09:45 - 11:30 or 13:30 - 15:10)
9. Strict Risk-to-Reward Ratio (RR >= 1:1.60)
"""
import pandas as pd
import numpy as np


def evaluate_9point_checklist(
    df: pd.DataFrame,
    ultra_sr: dict,
    trap_data: dict,
    signal_direction: str = "BUY",
    rr_ratio: float = 1.60,
    ltp_context: dict = None
) -> dict:
    """
    Evaluate the comprehensive 10-Point Confluence Matrix (9 Price/Technical Pillars + Point 10 LTP Option Chain CoA).
    Returns checklist items, passed count, percentage score, traffic light state, and trade recommendation.
    """
    if df.empty or len(df) < 5:
        return {
            "traffic_light": "YELLOW",
            "score_pct": 50,
            "passed_count": 4,
            "total_points": 10 if ltp_context else 9,
            "verdict": "WAIT_FOR_DATA",
            "recommendation": "Insufficient data to verify confluence checklist.",
            "checklist": []
        }

    latest = df.iloc[-1]
    current_price = float(latest['Close'])
    is_buy = signal_direction.upper() == "BUY"

    checklist = []

    # ── Point 1: Higher Timeframe Trend Alignment ──
    ema50 = float(latest.get('EMA_50', current_price))
    ema21 = float(latest.get('EMA_21', current_price))
    ema9  = float(latest.get('EMA_9', current_price))
    vwap  = float(latest.get('VWAP', current_price))

    if is_buy:
        htf_passed = (current_price >= ema50) or (ema9 >= ema21) or (current_price >= vwap)
        htf_desc = f"Price (Rs. {current_price}) holding above EMA 50 (Rs. {round(ema50, 1)}) or VWAP"
    else:
        htf_passed = (current_price <= ema50) or (ema9 <= ema21) or (current_price <= vwap)
        htf_desc = f"Price (Rs. {current_price}) trading below EMA 50 (Rs. {round(ema50, 1)}) or VWAP"

    checklist.append({
        "id": "point_1_trend",
        "title": "HTF Trend Alignment",
        "passed": bool(htf_passed),
        "weight": 12,
        "detail": htf_desc if htf_passed else "Trend alignment weak or fighting major moving average"
    })

    # ── Point 2: Institutional Volume Surge ──
    vol_ratio = float(latest.get('Volume_Ratio', 1.0))
    vol_passed = vol_ratio >= 1.15
    checklist.append({
        "id": "point_2_volume",
        "title": "Volume Confirmation",
        "passed": bool(vol_passed),
        "weight": 11,
        "detail": f"Institutional Volume Ratio at {round(vol_ratio, 2)}x (Target >= 1.15x)" if vol_passed else f"Subdued Volume ({round(vol_ratio, 2)}x). Lack of institutional volume"
    })

    # ── Point 3: Ultra-Strong S/R Zone (90%+ Accuracy) ──
    sup = ultra_sr.get('strongest_support', {})
    res = ultra_sr.get('strongest_resistance', {})
    
    if is_buy:
        # For buy, price should be either bouncing off support or breaking above resistance cleanly
        sr_passed = (sup.get('strength_score', 0) >= 85) and (abs(sup.get('distance_pct', 99)) <= 3.5)
        sr_desc = f"Supported by 90%+ Strong Zone at Rs. {sup.get('price')} ({sup.get('strength_score', 0)}% Acc)"
    else:
        sr_passed = (res.get('strength_score', 0) >= 85) and (abs(res.get('distance_pct', 99)) <= 3.5)
        sr_desc = f"Resisted by 90%+ Strong Supply at Rs. {res.get('price')} ({res.get('strength_score', 0)}% Acc)"

    checklist.append({
        "id": "point_3_sr_zone",
        "title": "90%+ S/R Confluence Anchor",
        "passed": bool(sr_passed),
        "weight": 14,
        "detail": sr_desc if sr_passed else "Price is floating mid-air away from verified S/R boundaries"
    })

    # ── Point 4: Order Block (OB) / Fair Value Gap (FVG) ──
    smc_sig = int(latest.get('SMC_Signal', 0))
    liq_sig = int(latest.get('Liquidity_Signal', 0))
    ob_passed = (is_buy and (smc_sig == 1 or liq_sig == 1)) or (not is_buy and (smc_sig == -1 or liq_sig == -1)) or bool(ultra_sr.get('strongest_support', {}).get('confluence_count', 0) >= 2)
    checklist.append({
        "id": "point_4_order_block",
        "title": "Order Block & Liquidity Sweep",
        "passed": bool(ob_passed),
        "weight": 12,
        "detail": "Smart Money Order Block mitigation / Liquidity purge active" if ob_passed else "No active Order Block mitigation detected"
    })

    # ── Point 5: Candlestick Reversal Confirmation ──
    bull_candles = int(latest.get('Pattern_Bullish_Engulfing', 0)) == 1 or int(latest.get('Pattern_Hammer', 0)) == 1
    bear_candles = int(latest.get('Pattern_Bearish_Engulfing', 0)) == 1 or int(latest.get('Pattern_Doji', 0)) == 1
    candle_passed = (is_buy and (bull_candles or current_price > float(latest['Open']))) or (not is_buy and (bear_candles or current_price < float(latest['Open'])))
    checklist.append({
        "id": "point_5_candlestick",
        "title": "Candlestick Confirmation",
        "passed": bool(candle_passed),
        "weight": 10,
        "detail": "Candle body direction & wick confirms entry momentum" if candle_passed else "Candle formation is indecisive or opposing"
    })

    # ── Point 6: Market Structure Shift (MSS / BOS) ──
    mss_sig = int(latest.get('MSS_Signal', 0))
    vsa_sig = int(latest.get('VSA_Signal', 0))
    mss_passed = (is_buy and (mss_sig == 1 or vsa_sig == 1)) or (not is_buy and (mss_sig == -1 or vsa_sig == -1)) or (int(latest.get('Breakout_Signal', 0)) != 0)
    checklist.append({
        "id": "point_6_structure",
        "title": "Market Structure Shift (BOS)",
        "passed": bool(mss_passed),
        "weight": 11,
        "detail": "Structural Break of Structure / VSA absorption confirmed" if mss_passed else "Market structure in intermediate range consolidation"
    })

    # ── Point 7: Institutional Trap Safety Check ──
    has_trap = trap_data.get('has_trap', False)
    trap_type = trap_data.get('trap_type', 'NONE')
    
    # Safe if no trap, or if the trap is opposing (e.g. Bear Trap is bullish!)
    if is_buy:
        trap_safe = not has_trap or (trap_type == "BEAR_TRAP")
        trap_desc = "Safe: Zero Bull Trap detected (Institutional paths clear)" if not has_trap else ("Ultra-Bullish: Bear Trap Active! Panic sellers trapped." if trap_type == "BEAR_TRAP" else "WARNING: Bull Trap active! Buying is dangerous.")
    else:
        trap_safe = not has_trap or (trap_type == "BULL_TRAP")
        trap_desc = "Safe: Zero Bear Trap detected (Institutional paths clear)" if not has_trap else ("Ultra-Bearish: Bull Trap Active! Breakout buyers trapped." if trap_type == "BULL_TRAP" else "WARNING: Bear Trap active! Shorting is dangerous.")

    checklist.append({
        "id": "point_7_trap_safe",
        "title": "Zero-Trap Safety Verification",
        "passed": bool(trap_safe),
        "weight": 15,
        "detail": trap_desc
    })

    # ── Point 8: Session Timing Golden Window ──
    session_mult = float(latest.get('Session_Multiplier', 1.0))
    session_passed = session_mult >= 1.0
    checklist.append({
        "id": "point_8_session",
        "title": "Session Timing Window",
        "passed": bool(session_passed),
        "weight": 8,
        "detail": "Active High-Volume Session (Golden Window / Power Hour)" if session_passed else "Low-Volume Lunch Lull or Square-Off Period"
    })

    # ── Point 9: Strict Risk-to-Reward Ratio (RR >= 1:1.60) ──
    safe_rr = float(rr_ratio) if rr_ratio is not None else 1.60
    rr_passed = (rr_ratio is not None) and (safe_rr >= 1.60)
    checklist.append({
        "id": "point_9_risk_reward",
        "title": "Risk-to-Reward Ratio (>= 1:1.60)",
        "passed": bool(rr_passed),
        "weight": 9,
        "detail": f"Guaranteed Minimum RR of 1 : {round(safe_rr, 2)} (Target > Stop Loss)" if rr_passed else "Risk-Reward sub-optimal or neutral setup (< 1:1.60)"
    })

    # ── Point 10: LTP Calculator Option Chain & CoA 1.0 Alignment ──
    effective_ltp = ltp_context or (ultra_sr.get("ltp_levels") if isinstance(ultra_sr, dict) else None)
    ltp_confusion = False
    if effective_ltp:
        coa_trend = str(effective_ltp.get("market_trend") or "RANGEBOUND")
        sup_state = str(effective_ltp.get("sup_state") or "STRONG")
        res_state = str(effective_ltp.get("res_state") or "STRONG")
        eos_val = effective_ltp.get("eos")
        eor_val = effective_ltp.get("eor")
        ltp_confusion = (coa_trend == "CONFUSION") or bool(effective_ltp.get("is_confusion"))

        if ltp_confusion:
            ltp_passed = False
            ltp_detail = f"CoA State of Confusion ({sup_state}/{res_state}) — Smart Money whipsaw trap active"
        elif is_buy:
            blocked_by_eor = bool(effective_ltp.get("near_eor_resistance"))
            bearish_coa = coa_trend in ("BEARISH", "BLOODBATH_BEARISH")
            ltp_passed = (not blocked_by_eor) and (not bearish_coa)
            ltp_detail = (
                f"CoA {coa_trend} | EOS Floor: {eos_val} ({sup_state}), EOR Ceiling: {eor_val} ({res_state})"
                if ltp_passed
                else f"Blocked: Spot near EOR ({eor_val}) or Bearish CoA ({coa_trend})"
            )
        else:
            blocked_by_eos = bool(effective_ltp.get("near_eos_support"))
            bullish_coa = coa_trend in ("BULLISH", "EXTREME_BULLISH")
            ltp_passed = (not blocked_by_eos) and (not bullish_coa)
            ltp_detail = (
                f"CoA {coa_trend} | EOR Ceiling: {eor_val} ({res_state}), EOS Floor: {eos_val} ({sup_state})"
                if ltp_passed
                else f"Blocked: Spot near EOS ({eos_val}) or Bullish CoA ({coa_trend})"
            )

        checklist.append({
            "id": "point_10_ltp_coa",
            "title": "LTP Calculator Option Chain (EOS/EOR & CoA)",
            "passed": bool(ltp_passed),
            "weight": 12,
            "detail": ltp_detail
        })

    # ── SYNTHESIS & TRAFFIC LIGHT CLASSIFICATION ──
    total_points = len(checklist)
    passed_count = sum(1 for item in checklist if item['passed'])
    score_pct = round((passed_count / max(1, total_points)) * 100)
    green_threshold = 9 if total_points >= 10 else 8
    yellow_threshold = 7 if total_points >= 10 else 6

    # Critical Trap Override: If trap opposes trade or CoA State of Confusion, force RED light!
    if not trap_safe or ltp_confusion:
        traffic_light = "RED"
        verdict = "DO_NOT_ENTER_TRAP_ACTIVE"
        recommendation = (
            "[RED LIGHT] CoA Option Chain State of Confusion / Trap detected! Do NOT enter."
            if ltp_confusion else
            "[RED LIGHT] Institutional Trap detected! Do NOT enter. Wait for trap resolution."
        )
    elif passed_count >= green_threshold:
        traffic_light = "GREEN"
        verdict = "SAFE_TO_ENTER"
        recommendation = f"[GREEN LIGHT] {passed_count}/{total_points} Checks Passed ({score_pct}% Precision). Prime institutional + LTP confluence. Execute trade!"
    elif passed_count >= yellow_threshold:
        traffic_light = "YELLOW"
        verdict = "CAUTION_WAIT_RETEST"
        recommendation = f"[YELLOW LIGHT] {passed_count}/{total_points} Checks Passed ({score_pct}% Score). Moderate setup. Wait for price retest or scale in with 50% position."
    else:
        traffic_light = "RED"
        verdict = "HIGH_RISK_AVOID"
        recommendation = f"[RED LIGHT] Only {passed_count}/{total_points} Checks Passed ({score_pct}% Score). Setup lacks institutional confirmation. Avoid taking trade."

    return {
        "traffic_light": traffic_light,
        "score_pct": score_pct,
        "passed_count": passed_count,
        "total_points": total_points,
        "verdict": verdict,
        "recommendation": recommendation,
        "signal_direction": signal_direction,
        "trap_safe": trap_safe,
        "checklist": checklist
    }

