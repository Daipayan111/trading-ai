from fastapi import APIRouter, HTTPException
import yfinance as yf
import pandas as pd

router = APIRouter()

# Yahoo Finance symbols for Indian market indices
INDICES = {
    "NIFTY50": {
        "symbol": "^NSEI",
        "name": "Nifty 50",
        "exchange": "NSE"
    },
    "BANKNIFTY": {
        "symbol": "^NSEBANK",
        "name": "Bank Nifty",
        "exchange": "NSE"
    },
    "SENSEX": {
        "symbol": "^BSESN",
        "name": "BSE Sensex",
        "exchange": "BSE"
    },
    "NIFTYIT": {
        "symbol": "^CNXIT",
        "name": "Nifty IT",
        "exchange": "NSE"
    },
    "NIFTYMIDCAP": {
        "symbol": "^NSEMDCP50",
        "name": "Nifty Midcap 50",
        "exchange": "NSE"
    },
}


def get_index_data(yf_symbol: str, period: str = "5d"):
    """Fetch index data from Yahoo Finance"""
    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(period=period, interval="1d")
    if df.empty:
        raise ValueError(f"No data found for index: {yf_symbol}")
    df.index = pd.to_datetime(df.index)
    return df


@router.get("/live")
def get_all_indices_live():
    """Get live data for all major Indian indices"""
    results = []
    for key, meta in INDICES.items():
        try:
            ticker = yf.Ticker(meta["symbol"])
            fi = ticker.fast_info
            price = getattr(fi, 'last_price', None)
            prev_close = getattr(fi, 'previous_close', None)
            open_p = getattr(fi, 'open', None)
            high_p = getattr(fi, 'day_high', None)
            low_p = getattr(fi, 'day_low', None)
            
            if price is not None and not pd.isna(price) and price > 0:
                pc = prev_close if (prev_close and not pd.isna(prev_close)) else price
                change = float(price) - float(pc)
                change_pct = (change / float(pc)) * 100 if pc else 0.0
                results.append({
                    "id": key,
                    "name": meta["name"],
                    "symbol": meta["symbol"],
                    "exchange": meta["exchange"],
                    "price": round(float(price), 2),
                    "open": round(float(open_p), 2) if open_p else round(float(price), 2),
                    "high": round(float(high_p), 2) if high_p else round(float(price), 2),
                    "low": round(float(low_p), 2) if low_p else round(float(price), 2),
                    "change": round(change, 2),
                    "change_pct": round(change_pct, 2),
                    "is_up": change >= 0,
                })
                continue
        except Exception:
            pass

        try:
            df = get_index_data(meta["symbol"], period="5d")
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest

            change = float(latest["Close"]) - float(prev["Close"])
            change_pct = (change / float(prev["Close"])) * 100

            results.append({
                "id": key,
                "name": meta["name"],
                "symbol": meta["symbol"],
                "exchange": meta["exchange"],
                "price": round(float(latest["Close"]), 2),
                "open": round(float(latest["Open"]), 2),
                "high": round(float(latest["High"]), 2),
                "low": round(float(latest["Low"]), 2),
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
                "is_up": change >= 0,
            })
        except Exception as e:
            results.append({
                "id": key,
                "name": meta["name"],
                "symbol": meta["symbol"],
                "exchange": meta["exchange"],
                "error": str(e),
            })

    return {"indices": results, "total": len(results)}


@router.get("/{index_key}")
async def get_index_detail(index_key: str, period: str = "6mo"):
    """Get detailed data + chart history for a specific index"""
    index_key = index_key.upper()
    if index_key not in INDICES:
        raise HTTPException(
            status_code=404,
            detail=f"Index '{index_key}' not found. Available: {list(INDICES.keys())}"
        )

    meta = INDICES[index_key]
    try:
        df = get_index_data(meta["symbol"], period=period)

        # Latest quote
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        change = float(latest["Close"]) - float(prev["Close"])
        change_pct = (change / float(prev["Close"])) * 100

        # Chart history
        chart_data = []
        for idx, row in df.iterrows():
            chart_data.append({
                "date": idx.strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]) if row["Volume"] else 0,
            })

        # 52-week high/low
        df_52w = get_index_data(meta["symbol"], period="1y")
        week52_high = round(float(df_52w["High"].max()), 2)
        week52_low = round(float(df_52w["Low"].min()), 2)

        return {
            "id": index_key,
            "name": meta["name"],
            "symbol": meta["symbol"],
            "exchange": meta["exchange"],
            "price": round(float(latest["Close"]), 2),
            "open": round(float(latest["Open"]), 2),
            "high": round(float(latest["High"]), 2),
            "low": round(float(latest["Low"]), 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "is_up": change >= 0,
            "week52_high": week52_high,
            "week52_low": week52_low,
            "period": period,
            "chart_data": chart_data,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
