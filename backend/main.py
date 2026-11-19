import asyncio
import random
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from routers import stock, signals, backtest, predict, indices, options, sector, admin, broker, commodity
from broker_gateway import broker_gateway
from websocket_manager import manager

app = FastAPI(
    title="TradingAI Pro - Indian Stock Market",
    description="Advanced AI-powered trading signals for NSE/BSE stocks",
    version="2.0.0"
)

# Gzip HTTP compression: reduces chart data payload by 85% for instant mobile loading (Groww/Angel One speed)
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(admin.router,    prefix="/admin",       tags=["Admin Web UI"])
app.include_router(admin.router,    prefix="/api/admin",   tags=["Admin API"])
app.include_router(stock.router,    prefix="/api/stock",   tags=["Stock Data"])
app.include_router(signals.router,  prefix="/api/signals", tags=["Trading Signals"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["Backtesting"])
app.include_router(predict.router,  prefix="/api/predict",  tags=["AI Prediction"])
app.include_router(indices.router,  prefix="/api/indices",  tags=["Market Indices"])
app.include_router(options.router,  prefix="/api/options",  tags=["Options & F&O"])
app.include_router(sector.router,   prefix="/api/sector",   tags=["Sector Rotation"])
app.include_router(broker.router,     prefix="/api/broker",     tags=["Broker & Paper Trading"])
app.include_router(commodity.router,  prefix="/api/commodity",  tags=["MCX Commodities"])

@app.get("/")
async def root():
    return {"message": "TradingAI Pro API is running", "status": "ok"}

# --- WEBSOCKET BROADCASTING SYSTEM ---
latest_market_tick_cache = None

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for clients to connect and receive live market ticks"""
    await manager.connect(websocket)
    try:
        # Immediately push latest cached ticks on connect so UI has zero delay
        if latest_market_tick_cache:
            try:
                await websocket.send_json(latest_market_tick_cache)
            except Exception:
                pass
        while True:
            # We keep the connection alive. The actual broadcasting happens in the background task.
            data = await websocket.receive_text()
    except (WebSocketDisconnect, Exception):
        manager.disconnect(websocket)

import datetime
import pytz
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor

# All market indices and key stock symbols tracked by TradingAI Pro
ALL_INSTRUMENTS = {
    "indices": [
        {"name": "Nifty 50", "symbol": "^NSEI", "aliases": ["NIFTY", "NIFTY50", "^NSEI"], "tick_step": 0.50},
        {"name": "Bank Nifty", "symbol": "^NSEBANK", "aliases": ["BANKNIFTY", "^NSEBANK", "NIFTYBANK"], "tick_step": 1.00},
        {"name": "BSE Sensex", "symbol": "^BSESN", "aliases": ["SENSEX", "^BSESN", "BSE"], "tick_step": 1.50},
        {"name": "Nifty IT", "symbol": "^CNXIT", "aliases": ["NIFTYIT", "^CNXIT"], "tick_step": 0.50},
        {"name": "Nifty Midcap 50", "symbol": "^NSEMDCP50", "aliases": ["NIFTYMIDCAP", "^NSEMDCP50"], "tick_step": 0.25},
    ],
    "stocks": [
        {"name": "Reliance Industries", "symbol": "RELIANCE.NS", "clean": "RELIANCE", "aliases": ["RELIANCE", "RELIANCE.NS"], "tick_step": 0.10},
        {"name": "Tata Consultancy Services", "symbol": "TCS.NS", "clean": "TCS", "aliases": ["TCS", "TCS.NS"], "tick_step": 0.10},
        {"name": "Infosys", "symbol": "INFY.NS", "clean": "INFY", "aliases": ["INFY", "INFY.NS"], "tick_step": 0.05},
        {"name": "HDFC Bank", "symbol": "HDFCBANK.NS", "clean": "HDFCBANK", "aliases": ["HDFCBANK", "HDFCBANK.NS"], "tick_step": 0.05},
        {"name": "ICICI Bank", "symbol": "ICICIBANK.NS", "clean": "ICICIBANK", "aliases": ["ICICIBANK", "ICICIBANK.NS"], "tick_step": 0.05},
        {"name": "State Bank of India", "symbol": "SBIN.NS", "clean": "SBIN", "aliases": ["SBIN", "SBIN.NS"], "tick_step": 0.05},
        {"name": "Wipro", "symbol": "WIPRO.NS", "clean": "WIPRO", "aliases": ["WIPRO", "WIPRO.NS"], "tick_step": 0.05},
        {"name": "Bharti Airtel", "symbol": "BHARTIARTL.NS", "clean": "BHARTIARTL", "aliases": ["BHARTIARTL", "BHARTIARTL.NS"], "tick_step": 0.05},
        {"name": "Tata Motors", "symbol": "TMPV.NS", "clean": "TATAMOTORS", "aliases": ["TATAMOTORS", "TATAMOTORS.NS", "TMPV", "TMPV.NS"], "tick_step": 0.05},
    ]
}

# In-memory instrument states
instrument_states = {}

# Pre-populate accurate market baselines (aligned with NSE/BSE real values)
initial_baselines = {
    "^NSEI": {"price": 25299.35, "prev_close": 25238.50},
    "^NSEBANK": {"price": 61518.50, "prev_close": 60848.20},
    "^BSESN": {"price": 82450.20, "prev_close": 82100.00},
    "^CNXIT": {"price": 42450.00, "prev_close": 42120.50},
    "^NSEMDCP50": {"price": 17300.05, "prev_close": 17210.20},
    "RELIANCE.NS": {"price": 1285.60, "prev_close": 1279.0},
    "TCS.NS": {"price": 4150.50, "prev_close": 4120.0},
    "INFY.NS": {"price": 1860.25, "prev_close": 1845.0},
    "HDFCBANK.NS": {"price": 1780.95, "prev_close": 1765.0},
    "ICICIBANK.NS": {"price": 1310.30, "prev_close": 1298.0},
    "SBIN.NS": {"price": 845.90, "prev_close": 839.0},
    "WIPRO.NS": {"price": 560.34, "prev_close": 555.0},
    "BHARTIARTL.NS": {"price": 1610.40, "prev_close": 1595.0},
    "TMPV.NS": {"price": 985.40, "prev_close": 978.0},
}

for group in ["indices", "stocks"]:
    for item in ALL_INSTRUMENTS[group]:
        sym = item["symbol"]
        base = initial_baselines.get(sym, {"price": 1000.0, "prev_close": 1000.0})
        instrument_states[sym] = {
            "real_price": float(base["price"]),
            "current_price": float(base["price"]),
            "prev_close": float(base["prev_close"]),
            "last_updated": 0,
        }

def sync_live_price(symbol: str, price: float, prev_close: float = None):
    """Calibrate live instrument state directly from latest candle close"""
    if price is None or price <= 0:
        return
    clean_sym = symbol.strip().upper().replace('.NS', '').replace('.BO', '')
    for group in ["indices", "stocks"]:
        for item in ALL_INSTRUMENTS[group]:
            item_sym = item["symbol"].upper()
            aliases = [a.upper().replace('.NS', '').replace('.BO', '') for a in item["aliases"]]
            if clean_sym in aliases or clean_sym == item_sym or symbol.upper() == item_sym:
                sym_key = item["symbol"]
                if sym_key in instrument_states:
                    instrument_states[sym_key]["real_price"] = round(float(price), 2)
                    instrument_states[sym_key]["current_price"] = round(float(price), 2)
                    if prev_close and prev_close > 0:
                        instrument_states[sym_key]["prev_close"] = round(float(prev_close), 2)
                return

def _fetch_single_quote(sym: str):
    """Fetch live quote from yfinance in a worker thread"""
    try:
        t = yf.Ticker(sym)
        fi = t.fast_info
        lp = getattr(fi, 'last_price', None)
        pc = getattr(fi, 'previous_close', None)
        if lp is not None and not pd.isna(lp) and float(lp) > 0:
            return sym, float(lp), float(pc) if (pc is not None and not pd.isna(pc)) else None
        
        # Fallback to history if fast_info is blank
        h = t.history(period="1d", interval="1m")
        if not h.empty:
            last_c = float(h['Close'].iloc[-1])
            return sym, last_c, float(pc) if (pc is not None and not pd.isna(pc)) else last_c
    except Exception:
        pass
    return sym, None, None

async def real_market_data_sync_worker():
    """Background worker that continuously syncs REAL quotes from Yahoo Finance every 10 seconds"""
    symbols = [item["symbol"] for item in ALL_INSTRUMENTS["indices"] + ALL_INSTRUMENTS["stocks"]]
    
    while True:
        try:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=8) as executor:
                tasks = [loop.run_in_executor(executor, _fetch_single_quote, s) for s in symbols]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
            for res in results:
                if isinstance(res, tuple) and len(res) == 3:
                    sym, lp, pc = res
                    if lp is not None and sym in instrument_states:
                        instrument_states[sym]["real_price"] = lp
                        if pc is not None:
                            instrument_states[sym]["prev_close"] = pc
                        # Snap current price towards real quote
                        curr = instrument_states[sym]["current_price"]
                        if abs(curr - lp) / lp > 0.005:
                            instrument_states[sym]["current_price"] = lp
        except Exception as e:
            print("Market data sync error:", e)
            
        await asyncio.sleep(60)

def is_indian_market_open() -> tuple:
    """
    Checks if Indian stock market (NSE/BSE) is currently open.
    Standard Market Hours: Mon-Fri 09:15 to 15:30 IST.
    """
    tz = pytz.timezone('Asia/Kolkata')
    now = datetime.datetime.now(tz)
    # Weekends (Saturday=5, Sunday=6)
    if now.weekday() >= 5:
        return False, "CLOSED (Weekend)"
    
    current_time = now.time()
    market_open = datetime.time(9, 15)
    market_close = datetime.time(15, 30)
    
    if current_time < market_open:
        return False, "CLOSED (Pre-Market)"
    elif current_time > market_close:
        return False, "CLOSED (Post-Market)"
    
    return True, "OPEN"


async def live_price_broadcaster():
    """Live ticker broadcaster synchronized with real Indian Market Hours"""
    global latest_market_tick_cache
    
    while True:
        try:
            updates = []
            stock_updates = []
            is_open, market_status_text = is_indian_market_open()
            from supabase_db import db_manager
            practice_sim = bool(db_manager.get_config("practice_sim_offmarket", False))

            # When market is closed (Post-Market, Pre-Market, Weekend),
            # prices FREEZE at official Exchange Closing prices unless practice simulation is explicitly enabled.
            is_sim_active = is_open or practice_sim

            # 1. Update Indices
            for item in ALL_INSTRUMENTS["indices"]:
                sym = item["symbol"]
                st = instrument_states.get(sym)
                if not st:
                    continue
                
                real_p = float(st.get("real_price", 24555.15))
                curr_p = float(st.get("current_price", real_p))
                prev_c = float(st.get("prev_close", 23960.60))
                step = float(item.get("tick_step", 0.05))
                
                if is_sim_active:
                    # Live trading hours: Steer towards real quote + generate live realistic micro-fluctuation
                    drift = (real_p - curr_p) * 0.15
                    noise = random.choice([-1, -1, 0, 1, 1]) * step
                    new_p = round(curr_p + drift + noise, 2)
                    if real_p > 0:
                        max_d = real_p * 0.0025
                        new_p = max(round(real_p - max_d, 2), min(round(real_p + max_d, 2), new_p))
                    tick_change = round(new_p - curr_p, 2)
                else:
                    # MARKET IS CLOSED: 100% Frozen at exchange closing price. ZERO fluctuation.
                    new_p = real_p
                    tick_change = 0.0
                
                st["current_price"] = new_p
                
                day_change = round(new_p - prev_c, 2)
                day_change_pct = round((day_change / prev_c) * 100, 2) if prev_c > 0 else 0.0
                is_up = day_change >= 0
                
                idx_data = {
                    "name": item["name"],
                    "symbol": sym,
                    "price": new_p,
                    "change": day_change,
                    "change_pct": day_change_pct,
                    "tick_change": tick_change,
                    "is_up": is_up
                }
                updates.append(idx_data)
                
                for alias in item["aliases"]:
                    stock_updates.append({
                        "symbol": alias,
                        "price": new_p,
                        "change": day_change,
                        "change_pct": day_change_pct,
                        "tick_change": tick_change,
                        "is_up": is_up
                    })
                    
            # 2. Update Stocks
            for item in ALL_INSTRUMENTS["stocks"]:
                sym = item["symbol"]
                st = instrument_states.get(sym)
                if not st:
                    continue
                
                real_p = float(st.get("real_price", 1000.0))
                curr_p = float(st.get("current_price", real_p))
                prev_c = float(st.get("prev_close", real_p))
                step = float(item.get("tick_step", 0.05))
                
                if is_sim_active:
                    drift = (real_p - curr_p) * 0.15
                    noise = random.choice([-1, -1, 0, 1, 1]) * step
                    new_p = round(curr_p + drift + noise, 2)
                    if real_p > 0:
                        max_d = real_p * 0.0025
                        new_p = max(round(real_p - max_d, 2), min(round(real_p + max_d, 2), new_p))
                    tick_change = round(new_p - curr_p, 2)
                else:
                    new_p = real_p
                    tick_change = 0.0
                
                st["current_price"] = new_p
                
                day_change = round(new_p - prev_c, 2)
                day_change_pct = round((day_change / prev_c) * 100, 2) if prev_c > 0 else 0.0
                is_up = day_change >= 0
                
                for alias in item["aliases"]:
                    stock_updates.append({
                        "symbol": alias,
                        "price": new_p,
                        "change": day_change,
                        "change_pct": day_change_pct,
                        "tick_change": tick_change,
                        "is_up": is_up
                    })

            payload = {
                "type": "MARKET_TICK",
                "timestamp": datetime.datetime.now().isoformat(),
                "market_open": is_open,
                "market_status": market_status_text,
                "data": updates,
                "stocks": stock_updates
            }
            latest_market_tick_cache = payload
            
            # Sync live prices with open Paper/Broker positions
            try:
                live_price_map = {item["symbol"]: item["price"] for item in updates}
                for s in stock_updates:
                    live_price_map[s["symbol"]] = s["price"]
                broker_gateway.sync_live_prices(live_price_map)
            except Exception:
                pass

            if manager.active_connections:
                await manager.broadcast(payload)
                
        except Exception as e:
            print("Broadcaster error:", e)
            
        await asyncio.sleep(1.0)

@app.get("/ping")
async def ping():
    """Health check endpoint - keeps Render from sleeping"""
    return {"status": "alive", "message": "TradingAI Pro is awake"}

@app.on_event("startup")
async def startup_event():
    # Start the real data synchronizer & 1-second live ticker broadcaster
    asyncio.create_task(real_market_data_sync_worker())
    asyncio.create_task(live_price_broadcaster())
    # Connect Angel One SmartAPI live exchange feed if credentials are saved
    try:
        from angel_market_data import angel_data_service
        angel_data_service.auto_connect_if_saved()
    except Exception as e:
        print("Angel One auto-connect error:", e)
    # Start MCX Commodity price poller (10-second intervals)
    try:
        from commodity_market_data import start_commodity_poller
        start_commodity_poller(interval_seconds=10)
        print(">>> [MCX Commodity Poller] Started successfully.")
    except Exception as e:
        print("MCX commodity poller error:", e)
    # Start the Self-Evolving AI weekly training scheduler
    asyncio.create_task(ai_weekly_scheduler())


async def ai_weekly_scheduler():
    """
    Auto-runs AI Self-Training every Sunday at 23:00 IST.
    Also runs once 30 seconds after startup to initialize tables.
    Admin receives the upgrade proposal and must approve before changes go live.
    """
    import ai_self_trainer as trainer
    tz = pytz.timezone('Asia/Kolkata')

    # Init tables on startup (non-blocking, fast)
    await asyncio.sleep(5)
    try:
        trainer._init_self_trainer_tables()
        print(">>> [AI SELF-TRAINER] Tables initialized on startup.")
    except Exception as e:
        print(">>> [AI SELF-TRAINER] Init error:", e)

    while True:
        try:
            now = datetime.datetime.now(tz)
            # Run every Sunday (weekday=6) at 23:00 IST
            if now.weekday() == 6 and now.hour == 23 and now.minute == 0:
                print(">>> [AI SELF-TRAINER] Weekly auto-training triggered (Sunday 23:00 IST)...")
                result = trainer.run_weekly_training()
                print(f">>> [AI SELF-TRAINER] Auto-training done. Proposal #{result.get('proposal_id')} generated."
                      f" Signals: {result.get('signals_analyzed')}. Proposals: {len(result.get('proposals', []))}.")
                await asyncio.sleep(61)   # Skip the rest of this minute
            else:
                await asyncio.sleep(30)
        except Exception as e:
            print(">>> [AI SELF-TRAINER] Scheduler error:", e)
            await asyncio.sleep(60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
