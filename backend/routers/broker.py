"""
Broker & Paper Trading REST Router
==================================
Provides institutional endpoints for:
- 1-Click order execution
- Paper trading funds & margin tracking
- Positions & order book monitoring
- Broker connection setup (Zerodha, Angel One, Dhan, Fyers)
- Prop-desk risk management & Kill Switch controls
"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, Optional
from broker_gateway import broker_gateway

router = APIRouter()

@router.get("/status")
async def get_broker_status():
    """Returns active broker mode, net equity, available margin, and P&L metrics"""
    try:
        return broker_gateway.get_account_summary()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/positions")
async def get_active_positions():
    """Returns all active open trading positions with live unrealized P&L"""
    try:
        return broker_gateway.get_positions()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/orders")
async def get_order_history(limit: int = 50):
    """Returns historical order audit logs with slippage and exchange fee breakdown"""
    try:
        return broker_gateway.get_orders(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/order")
async def place_trade_order(payload: Dict[str, Any] = Body(...)):
    """
    Executes a market/limit trade order with stop loss and multi-tier profit targets.
    Live orders are strictly allowed during Indian Market Hours (Mon-Fri 09:15 - 15:30 IST);
    Paper trading is available 24/7 for practice.
    """
    from main import is_indian_market_open
    is_open, market_status_text = is_indian_market_open()
    from supabase_db import db_manager
    practice_sim = bool(db_manager.get_config("practice_sim_offmarket", True))
    target_mode = str(payload.get("broker_mode", broker_gateway.active_mode)).upper()

    if target_mode != "PAPER" and not is_open and not practice_sim:
        raise HTTPException(
            status_code=400,
            detail=f"Order Blocked: Indian Stock Market is currently {market_status_text}. Real broker orders can strictly be placed ONLY during Live Market Hours (09:15 AM - 03:30 PM IST)."
        )

    try:
        res = broker_gateway.place_order(payload)
        if res.get("status") == "REJECTED":
            raise HTTPException(status_code=400, detail=res.get("error", "Order rejected by risk manager"))
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/positions/close-all")
async def close_all_trade_positions():
    """Immediately squares off all active open positions in one click"""
    try:
        return broker_gateway.close_all_positions()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/position/close/{position_id}")
async def close_trade_position(position_id: str, payload: Optional[Dict[str, Any]] = Body(None)):
    """Closes an open position at market price or optional specified exit price"""
    try:
        exit_p = payload.get("exit_price") if payload else None
        res = broker_gateway.close_position(position_id, exit_price=exit_p)
        if res.get("status") == "ERROR":
            raise HTTPException(status_code=400, detail=res.get("error", "Failed to close position"))
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/position/trail-sl/{position_id}")
async def trail_position_stop(position_id: str, payload: Dict[str, Any] = Body(...)):
    """Updates the stop loss price to lock in profits (Trailing Stop)"""
    try:
        new_sl = float(payload.get("new_stop", 0.0))
        return broker_gateway.trail_stop_loss(position_id, new_sl)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/paper/reset")
async def reset_paper_funds(payload: Optional[Dict[str, Any]] = Body(None)):
    """Resets paper account funds back to ₹10,00,000 (or custom balance)"""
    try:
        balance = float(payload.get("balance", 1000000.0)) if payload else 1000000.0
        return broker_gateway.reset_paper_account(balance)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/switch-mode")
async def switch_trading_mode(payload: Dict[str, Any] = Body(...)):
    """Switches active broker mode (e.g. 'PAPER', 'ZERODHA', 'ANGEL', 'DHAN', 'FYERS')"""
    try:
        mode = str(payload.get("mode", "PAPER")).upper()
        broker_gateway.active_mode = mode
        return {"status": "SUCCESS", "active_mode": mode}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/risk-settings")
async def get_risk_guard_settings():
    """Returns prop-firm risk parameters and kill switch status"""
    try:
        return broker_gateway.get_risk_settings()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/risk-settings")
async def update_risk_guard_settings(payload: Dict[str, Any] = Body(...)):
    """Updates prop-desk risk limits (Max Risk %, Daily Loss Limit, Kill Switch)"""
    try:
        return broker_gateway.update_risk_settings(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/connect")
async def connect_broker_account(payload: Dict[str, Any] = Body(...)):
    """
    Connects an Indian broker account (Zerodha, Angel One, Dhan, Fyers)
    with API Key, Secret, Client ID, or Auth token.
    """
    try:
        b_name = payload.get("broker_name", "ZERODHA")
        return broker_gateway.save_broker_credentials(b_name, payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/connected")
async def get_connected_broker_list():
    """Returns list of configured Indian broker integrations"""
    try:
        return broker_gateway.get_connected_brokers()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
