"""
Options Flow Intelligence Module — PCR, Max Pain & OI Analysis
===============================================================
Computes from live yfinance options chain:
  1. Put-Call Ratio (PCR) by OI — Institutional Positioning Signal
  2. Max Pain Price — Where most options expire worthless (Market Maker magnet)
  3. OI Distribution — Net Call/Put OI imbalance near current price (+-5%)
  4. Net Options Flow Direction — Final actionable directional signal

Signal Logic:
  - PCR < 0.70  => Bearish (too many calls = complacency)
  - PCR > 1.30  => Bullish (too many puts = fear / smart money reversal)
  - Max Pain > price => price should rise (bullish pull)
  - Max Pain < price => price should fall (bearish push)
"""
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

_OI_CACHE = {}
_OI_CACHE_TTL = 120  # 2 minutes


def _empty_options_flow(reason: str = "No data") -> dict:
    return {
        "available": False,
        "reason": reason,
        "pcr": None,
        "max_pain": None,
        "oi_signal": "NEUTRAL",
        "flow_direction": "NEUTRAL",
        "flow_score": 0,
        "confidence": 0,
        "signals": [],
        "details": {}
    }


def compute_max_pain(calls_df: pd.DataFrame, puts_df: pd.DataFrame):
    """Max Pain = strike where total option buyer loss is maximized."""
    try:
        all_strikes = sorted(set(
            list(calls_df['strike'].values) + list(puts_df['strike'].values)
        ))
        if not all_strikes:
            return None

        min_pain = float('inf')
        max_pain_strike = all_strikes[0]

        for test_strike in all_strikes:
            call_pain = 0.0
            for _, row in calls_df.iterrows():
                if row['strike'] > test_strike:
                    call_pain += (row['strike'] - test_strike) * float(row.get('openInterest', 0) or 0)

            put_pain = 0.0
            for _, row in puts_df.iterrows():
                if row['strike'] < test_strike:
                    put_pain += (test_strike - row['strike']) * float(row.get('openInterest', 0) or 0)

            total_pain = call_pain + put_pain
            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = test_strike

        return float(max_pain_strike)
    except Exception:
        return None


def analyze_options_flow(symbol: str, current_price: float) -> dict:
    """
    Main Options Flow Intelligence — returns PCR, Max Pain, OI imbalance, net direction.
    """
    if not HAS_YF:
        return _empty_options_flow("yfinance not available")

    now = time.time()
    if symbol in _OI_CACHE and (now - _OI_CACHE[symbol]['ts']) < _OI_CACHE_TTL:
        return _OI_CACHE[symbol]['data']

    try:
        INDEX_MAP = {
            "NIFTY": "^NSEI", "NIFTY50": "^NSEI", "^NSEI": "^NSEI",
            "BANKNIFTY": "^NSEBANK", "^NSEBANK": "^NSEBANK",
            "SENSEX": "^BSESN", "^BSESN": "^BSESN",
        }
        clean = symbol.replace('.NS', '').replace('.BO', '').upper()
        yf_sym = INDEX_MAP.get(clean, clean if clean.startswith('^') else f"{clean}.NS")

        ticker = yf.Ticker(yf_sym)
        expirations = ticker.options

        if not expirations:
            result = _empty_options_flow("No options chain available")
            _OI_CACHE[symbol] = {'ts': now, 'data': result}
            return result

        target_expiries = expirations[:min(2, len(expirations))]
        all_calls, all_puts = [], []

        for expiry in target_expiries:
            try:
                chain = ticker.option_chain(expiry)
                if chain.calls is not None and not chain.calls.empty:
                    all_calls.append(chain.calls)
                if chain.puts is not None and not chain.puts.empty:
                    all_puts.append(chain.puts)
            except Exception:
                continue

        if not all_calls or not all_puts:
            result = _empty_options_flow("Options chain empty")
            _OI_CACHE[symbol] = {'ts': now, 'data': result}
            return result

        calls_df = pd.concat(all_calls, ignore_index=True)
        puts_df = pd.concat(all_puts, ignore_index=True)

        calls_df['openInterest'] = pd.to_numeric(calls_df.get('openInterest', 0), errors='coerce').fillna(0)
        puts_df['openInterest'] = pd.to_numeric(puts_df.get('openInterest', 0), errors='coerce').fillna(0)

        total_call_oi = float(calls_df['openInterest'].sum())
        total_put_oi = float(puts_df['openInterest'].sum())

        if total_call_oi <= 0:
            result = _empty_options_flow("Zero call OI")
            _OI_CACHE[symbol] = {'ts': now, 'data': result}
            return result

        pcr = round(total_put_oi / total_call_oi, 3)

        max_pain = compute_max_pain(
            calls_df[['strike', 'openInterest']],
            puts_df[['strike', 'openInterest']]
        )

        # ATM band OI (+-5%)
        atm_low = current_price * 0.95
        atm_high = current_price * 1.05
        atm_calls = calls_df[(calls_df['strike'] >= atm_low) & (calls_df['strike'] <= atm_high)]
        atm_puts = puts_df[(puts_df['strike'] >= atm_low) & (puts_df['strike'] <= atm_high)]
        atm_call_oi = float(atm_calls['openInterest'].sum())
        atm_put_oi = float(atm_puts['openInterest'].sum())
        atm_pcr = round(atm_put_oi / atm_call_oi, 3) if atm_call_oi > 0 else 1.0

        # Score derivation
        signals = []
        score = 0

        if pcr > 1.30:
            signals.append(f"PCR={pcr} >1.30 => BULLISH (excessive puts = fear/reversal zone)")
            score += 2
        elif pcr > 1.10:
            signals.append(f"PCR={pcr} >1.10 => Mildly Bullish")
            score += 1
        elif pcr < 0.70:
            signals.append(f"PCR={pcr} <0.70 => BEARISH (excessive calls = complacency)")
            score -= 2
        elif pcr < 0.90:
            signals.append(f"PCR={pcr} <0.90 => Mildly Bearish")
            score -= 1
        else:
            signals.append(f"PCR={pcr} => NEUTRAL range (0.90-1.10)")

        if max_pain:
            pain_pct = (max_pain - current_price) / current_price * 100
            if pain_pct > 1.5:
                signals.append(f"Max Pain Rs.{max_pain:.0f} is {pain_pct:.1f}% above price => BULLISH pull")
                score += 1
            elif pain_pct < -1.5:
                signals.append(f"Max Pain Rs.{max_pain:.0f} is {abs(pain_pct):.1f}% below price => BEARISH push")
                score -= 1
            else:
                signals.append(f"Max Pain Rs.{max_pain:.0f} near price => NEUTRAL")

        if atm_pcr > 1.40:
            signals.append(f"ATM PCR={atm_pcr:.2f} => BULLISH (heavy put wall = strong support)")
            score += 1
        elif atm_pcr < 0.70:
            signals.append(f"ATM PCR={atm_pcr:.2f} => BEARISH (heavy call wall = resistance)")
            score -= 1

        if score >= 2:
            flow_direction = "BULLISH"
            oi_signal = "STRONG_BUY"
            confidence = min(90, 60 + score * 10)
        elif score == 1:
            flow_direction = "MILDLY_BULLISH"
            oi_signal = "BUY"
            confidence = 58
        elif score <= -2:
            flow_direction = "BEARISH"
            oi_signal = "STRONG_SELL"
            confidence = min(90, 60 + abs(score) * 10)
        elif score == -1:
            flow_direction = "MILDLY_BEARISH"
            oi_signal = "SELL"
            confidence = 58
        else:
            flow_direction = "NEUTRAL"
            oi_signal = "NEUTRAL"
            confidence = 42

        result = {
            "available": True,
            "reason": "Live options chain data",
            "pcr": pcr,
            "atm_pcr": atm_pcr,
            "max_pain": round(max_pain, 2) if max_pain else None,
            "max_pain_distance_pct": round((max_pain - current_price) / current_price * 100, 2) if max_pain else None,
            "total_call_oi": int(total_call_oi),
            "total_put_oi": int(total_put_oi),
            "atm_call_oi": int(atm_call_oi),
            "atm_put_oi": int(atm_put_oi),
            "oi_signal": oi_signal,
            "flow_direction": flow_direction,
            "flow_score": score,
            "confidence": confidence,
            "expiries_analyzed": list(target_expiries),
            "signals": signals,
            "details": {
                "pcr_interpretation": (
                    "Extremely Bullish" if pcr > 1.50 else
                    "Bullish (Fear/Hedging)" if pcr > 1.30 else
                    "Neutral-Bullish" if pcr > 1.10 else
                    "Neutral" if pcr >= 0.90 else
                    "Neutral-Bearish" if pcr >= 0.70 else
                    "Bearish (Complacency)"
                ),
                "atm_oi_band": f"+-5% band ({round(atm_low)}-{round(atm_high)})"
            }
        }

        _OI_CACHE[symbol] = {'ts': now, 'data': result}
        return result

    except Exception as e:
        result = _empty_options_flow(f"Error: {str(e)[:100]}")
        _OI_CACHE[symbol] = {'ts': now, 'data': result}
        return result
