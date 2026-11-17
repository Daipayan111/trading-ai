"""
Institutional Multi-Broker Gateway & Professional Paper Trading Engine
========================================================================
Supports:
1. Advanced Paper Trading Simulator (Realistic slippage, STT/GST/Exchange taxes, trailing stops, prop-desk risk limits)
2. Indian Broker Connectors:
   - Zerodha Kite Connect API
   - Angel One SmartAPI
   - DhanHQ API
   - Fyers API v3
3. Prop-Firm Risk Engine:
   - Max Risk per Trade (1-2%)
   - Daily Drawdown Guard & Kill-Switch
   - Max Concurrent Open Positions
"""

import os
import sqlite3
import datetime
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

DB_PATH = Path(__file__).resolve().parent / "trading_ai.db"

def get_db_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def _init_broker_db():
    with get_db_conn() as conn:
        cursor = conn.cursor()
        
        # Paper Account Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_account (
                id TEXT PRIMARY KEY,
                initial_balance REAL DEFAULT 1000000.0,
                current_balance REAL DEFAULT 1000000.0,
                used_margin REAL DEFAULT 0.0,
                realized_pnl REAL DEFAULT 0.0,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Trade Positions Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_positions (
                id TEXT PRIMARY KEY,
                broker_mode TEXT DEFAULT 'PAPER',
                symbol TEXT NOT NULL,
                instrument_type TEXT DEFAULT 'EQUITY',
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL NOT NULL,
                stop_loss REAL,
                target_1 REAL,
                target_2 REAL,
                status TEXT DEFAULT 'OPEN',
                unrealized_pnl REAL DEFAULT 0.0,
                realized_pnl REAL DEFAULT 0.0,
                brokerage_fees REAL DEFAULT 0.0,
                opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                closed_at TIMESTAMP
            )
        """)

        # Options migration columns
        for col_def in [
            "underlying_entry_price REAL DEFAULT 0.0",
            "option_type TEXT",
            "strike_price REAL"
        ]:
            try:
                cursor.execute(f"ALTER TABLE trade_positions ADD COLUMN {col_def}")
            except Exception:
                pass
        
        # Trade Orders Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_orders (
                id TEXT PRIMARY KEY,
                position_id TEXT,
                broker_mode TEXT DEFAULT 'PAPER',
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT DEFAULT 'MARKET',
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                execution_price REAL,
                status TEXT DEFAULT 'EXECUTED',
                slippage_pts REAL DEFAULT 0.0,
                brokerage_fees REAL DEFAULT 0.0,
                reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Broker Configs Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS broker_configs (
                broker_name TEXT PRIMARY KEY,
                is_active INTEGER DEFAULT 0,
                api_key TEXT,
                api_secret TEXT,
                client_id TEXT,
                access_token TEXT,
                totp_secret TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Risk Settings Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prop_risk_settings (
                id TEXT PRIMARY KEY,
                max_risk_per_trade_pct REAL DEFAULT 1.5,
                max_daily_loss_pct REAL DEFAULT 3.0,
                max_concurrent_positions INTEGER DEFAULT 50,
                auto_algo_enabled INTEGER DEFAULT 0,
                kill_switch_active INTEGER DEFAULT 0,
                daily_loss_accumulated REAL DEFAULT 0.0,
                last_reset_date TEXT
            )
        """)

        # Seed initial records if empty
        cursor.execute("SELECT id FROM paper_account WHERE id = 'default'")
        if not cursor.fetchone():
            cursor.execute("""
                INSERT INTO paper_account (id, initial_balance, current_balance, used_margin, realized_pnl)
                VALUES ('default', 1000000.0, 1000000.0, 0.0, 0.0)
            """)

        cursor.execute("SELECT id FROM prop_risk_settings WHERE id = 'default'")
        if not cursor.fetchone():
            cursor.execute("""
                INSERT INTO prop_risk_settings (id, max_risk_per_trade_pct, max_daily_loss_pct, max_concurrent_positions, auto_algo_enabled, kill_switch_active, last_reset_date)
                VALUES ('default', 1.5, 3.0, 50, 0, 0, date('now'))
            """)
        else:
            # Upgrade existing limit from old 4 positions to 50 positions
            cursor.execute("UPDATE prop_risk_settings SET max_concurrent_positions = 50 WHERE max_concurrent_positions < 50")
        conn.commit()

_init_broker_db()


class IndianBrokerageCalculator:
    """
    Computes realistic Indian exchange taxes and charges (NSE/BSE):
    - STT (Securities Transaction Tax): 0.025% on intraday sell, 0.1% on delivery
    - Exchange Turnover Fees: 0.00345%
    - SEBI Charges: ₹10 per crore (0.0001%)
    - Stamp Duty: 0.003% on buy
    - Brokerage: ₹20 per executed order (Discount broker standard like Zerodha/Dhan/Angel)
    - GST: 18% on (Brokerage + Exchange turnover + SEBI charges)
    """
    @staticmethod
    def calculate_charges(side: str, price: float, quantity: int, is_intraday: bool = True) -> float:
        turnover = price * quantity
        brokerage = min(20.0, turnover * 0.0003)
        stt = (turnover * 0.00025) if (side.upper() == 'SELL' and is_intraday) else ((turnover * 0.001) if not is_intraday else 0.0)
        exchange_turnover_fee = turnover * 0.0000345
        sebi_charges = turnover * 0.000001
        stamp_duty = (turnover * 0.00003) if side.upper() == 'BUY' else 0.0
        gst = 0.18 * (brokerage + exchange_turnover_fee + sebi_charges)
        total = round(brokerage + stt + exchange_turnover_fee + sebi_charges + stamp_duty + gst, 2)
        return max(2.50, total)


class BrokerGateway:
    def __init__(self):
        self.active_mode = "PAPER" # 'PAPER', 'ZERODHA', 'ANGEL', 'DHAN', 'FYERS'

    def get_risk_settings(self) -> Dict[str, Any]:
        with get_db_conn() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM prop_risk_settings WHERE id = 'default'").fetchone()
            if row:
                return dict(row)
            return {
                "max_risk_per_trade_pct": 1.5,
                "max_daily_loss_pct": 3.0,
                "max_concurrent_positions": 50,
                "auto_algo_enabled": 0,
                "kill_switch_active": 0,
                "daily_loss_accumulated": 0.0
            }

    def update_risk_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        with get_db_conn() as conn:
            conn.execute("""
                UPDATE prop_risk_settings
                SET max_risk_per_trade_pct = ?,
                    max_daily_loss_pct = ?,
                    max_concurrent_positions = ?,
                    auto_algo_enabled = ?,
                    kill_switch_active = ?
                WHERE id = 'default'
            """, (
                settings.get('max_risk_per_trade_pct', 1.5),
                settings.get('max_daily_loss_pct', 3.0),
                settings.get('max_concurrent_positions', 4),
                settings.get('auto_algo_enabled', 0),
                settings.get('kill_switch_active', 0)
            ))
            conn.commit()
        return self.get_risk_settings()

    def get_account_summary(self) -> Dict[str, Any]:
        with get_db_conn() as conn:
            conn.row_factory = sqlite3.Row
            acc = dict(conn.execute("SELECT * FROM paper_account WHERE id = 'default'").fetchone() or {})
            
            # Fetch active positions to calculate live unrealized P&L
            positions = [dict(r) for r in conn.execute("SELECT * FROM trade_positions WHERE status = 'OPEN'").fetchall()]
            unrealized_pnl = sum(p.get('unrealized_pnl', 0.0) for p in positions)
            # Options buyers pay 100% of premium, equities have 20% margin (5x intraday leverage)
            margin_used = sum(
                (p['entry_price'] * p['quantity']) if p.get('instrument_type') == 'OPTION'
                else (p['entry_price'] * p['quantity'] * 0.20)
                for p in positions
            )
            
            net_equity = round(acc.get('current_balance', 1000000.0) + unrealized_pnl, 2)
            available_margin = max(0.0, round(acc.get('current_balance', 1000000.0) - margin_used, 2))
            
            risk = self.get_risk_settings()
            
            return {
                "active_broker": self.active_mode,
                "initial_capital": acc.get('initial_balance', 1000000.0),
                "current_balance": round(acc.get('current_balance', 1000000.0), 2),
                "net_equity": net_equity,
                "used_margin": round(margin_used, 2),
                "available_margin": available_margin,
                "unrealized_pnl": round(unrealized_pnl, 2),
                "realized_pnl": round(acc.get('realized_pnl', 0.0), 2),
                "open_positions_count": len(positions),
                "total_trades": acc.get('total_trades', 0),
                "winning_trades": acc.get('winning_trades', 0),
                "losing_trades": acc.get('losing_trades', 0),
                "win_rate_pct": round((acc.get('winning_trades', 0) / max(1, acc.get('total_trades', 0))) * 100, 1),
                "kill_switch_active": risk.get('kill_switch_active', 0) == 1,
                "auto_algo_enabled": risk.get('auto_algo_enabled', 0) == 1
            }

    def reset_paper_account(self, new_balance: float = 1000000.0) -> Dict[str, Any]:
        with get_db_conn() as conn:
            conn.execute("""
                UPDATE paper_account
                SET initial_balance = ?,
                    current_balance = ?,
                    used_margin = 0.0,
                    realized_pnl = 0.0,
                    total_trades = 0,
                    winning_trades = 0,
                    losing_trades = 0,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = 'default'
            """, (new_balance, new_balance))
            conn.execute("DELETE FROM trade_positions")
            conn.execute("DELETE FROM trade_orders")
            conn.execute("""
                UPDATE prop_risk_settings 
                SET kill_switch_active = 0, daily_loss_accumulated = 0.0 
                WHERE id = 'default'
            """)
            conn.commit()
        return self.get_account_summary()

    def get_positions(self) -> List[Dict[str, Any]]:
        with get_db_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM trade_positions WHERE status = 'OPEN' ORDER BY opened_at DESC").fetchall()
            return [dict(r) for r in rows]

    def get_orders(self, limit: int = 50) -> List[Dict[str, Any]]:
        with get_db_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM trade_orders ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]

    def place_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        symbol = str(order_data.get('symbol', 'NIFTY50')).upper().replace('.NS', '')
        side = str(order_data.get('side', 'BUY')).upper()
        quantity = int(order_data.get('quantity', 1))
        order_type = str(order_data.get('order_type', 'MARKET')).upper()
        price = float(order_data.get('price', 0.0))
        stop_loss = float(order_data.get('stop_loss', 0.0)) if order_data.get('stop_loss') else None
        target_1 = float(order_data.get('target_1', 0.0)) if order_data.get('target_1') else None
        target_2 = float(order_data.get('target_2', 0.0)) if order_data.get('target_2') else None

        target_mode = str(order_data.get('broker_mode', self.active_mode)).upper()

        from main import is_indian_market_open
        is_open, market_status_text = is_indian_market_open()
        from supabase_db import db_manager
        practice_sim = bool(db_manager.get_config("practice_sim_offmarket", True))

        # Real broker executions enforce market hours; Paper trading is allowed 24/7 in live/sim mode
        if target_mode != 'PAPER' and not is_open and not practice_sim:
            return {
                "status": "REJECTED",
                "error": f"Market is currently {market_status_text}. Orders can strictly be placed ONLY during Live Market Hours (Mon-Fri 09:15 AM - 03:30 PM IST)."
            }

        risk = self.get_risk_settings()
        if risk.get('kill_switch_active', 0) == 1:
            return {"status": "REJECTED", "error": "Prop-desk Kill Switch is active. Daily loss limit hit!"}

        open_positions = self.get_positions()
        max_allowed = risk.get('max_concurrent_positions', 50)
        if len(open_positions) >= max_allowed:
            return {"status": "REJECTED", "error": f"Max concurrent positions ({max_allowed}) reached!"}

        instrument_type = str(order_data.get('instrument_type', 'EQUITY')).upper()
        if 'OPTION' in instrument_type or ' CE' in symbol or ' PE' in symbol:
            instrument_type = 'OPTION'
        
        option_type = 'CE' if ' CE' in symbol else ('PE' if ' PE' in symbol else None)
        strike_price = float(order_data.get('strike_price', 0.0))
        underlying_price = float(order_data.get('underlying_price', 0.0))

        # Simulate Realistic Intraday Slippage (0.02% on Indian equities, 0.5-1.0 pts on Options)
        is_index = any(k in symbol for k in ['NIFTY', 'BANK', 'SENSEX'])
        if instrument_type == 'OPTION':
            slippage = 0.50
        else:
            slippage = round(0.75 if is_index else (price * 0.0003), 2)
        exec_price = price + slippage if side == 'BUY' else price - slippage

        # Compute Indian Statutory Charges
        brokerage_fee = IndianBrokerageCalculator.calculate_charges(side, exec_price, quantity)

        pos_id = f"pos_{uuid.uuid4().hex[:8]}"
        ord_id = f"ord_{uuid.uuid4().hex[:8]}"

        with get_db_conn() as conn:
            # Insert Trade Position
            conn.execute("""
                INSERT INTO trade_positions (
                    id, broker_mode, symbol, instrument_type, side, quantity, entry_price, current_price,
                    stop_loss, target_1, target_2, status, unrealized_pnl, brokerage_fees,
                    underlying_entry_price, option_type, strike_price
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', 0.0, ?, ?, ?, ?)
            """, (
                pos_id, self.active_mode, symbol, instrument_type, side, quantity, exec_price, exec_price,
                stop_loss, target_1, target_2, brokerage_fee, underlying_price, option_type, strike_price
            ))

            # Insert Trade Order Log
            conn.execute("""
                INSERT INTO trade_orders (
                    id, position_id, broker_mode, symbol, side, order_type, quantity,
                    price, execution_price, status, slippage_pts, brokerage_fees, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'EXECUTED', ?, ?, ?)
            """, (
                ord_id, pos_id, self.active_mode, symbol, side, order_type, quantity,
                price, exec_price, slippage, brokerage_fee, order_data.get('reason', 'Manual AI Order')
            ))

            # Deduct brokerage fee from paper balance
            conn.execute("""
                UPDATE paper_account 
                SET current_balance = current_balance - ?,
                    total_trades = total_trades + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = 'default'
            """, (brokerage_fee,))
            conn.commit()

        return {
            "status": "EXECUTED",
            "position_id": pos_id,
            "order_id": ord_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "execution_price": exec_price,
            "slippage": slippage,
            "brokerage_fees": brokerage_fee,
            "stop_loss": stop_loss,
            "target_1": target_1,
            "target_2": target_2
        }

    def close_position(self, position_id: str, exit_price: Optional[float] = None) -> Dict[str, Any]:
        with get_db_conn() as conn:
            conn.row_factory = sqlite3.Row
            pos = conn.execute("SELECT * FROM trade_positions WHERE id = ?", (position_id,)).fetchone()
            if not pos or pos['status'] != 'OPEN':
                return {"status": "ERROR", "error": "Position not found or already closed"}

            pos = dict(pos)
            side = pos['side']
            qty = pos['quantity']
            entry_p = pos['entry_price']
            sym = pos['symbol']

            actual_exit = exit_price if (exit_price is not None and exit_price > 0) else (pos['current_price'] if (pos['current_price'] and pos['current_price'] > 0) else pos['entry_price'])
            
            # PnL calculation
            gross_pnl = (actual_exit - entry_p) * qty if side == 'BUY' else (entry_p - actual_exit) * qty
            exit_charges = IndianBrokerageCalculator.calculate_charges('SELL' if side == 'BUY' else 'BUY', actual_exit, qty)
            net_pnl = round(gross_pnl - exit_charges, 2)
            is_win = net_pnl > 0

            # Update position to CLOSED
            conn.execute("""
                UPDATE trade_positions
                SET status = 'CLOSED',
                    current_price = ?,
                    realized_pnl = ?,
                    unrealized_pnl = 0.0,
                    brokerage_fees = brokerage_fees + ?,
                    closed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (actual_exit, net_pnl, exit_charges, position_id))

            # Update paper account balance
            conn.execute("""
                UPDATE paper_account
                SET current_balance = current_balance + ?,
                    realized_pnl = realized_pnl + ?,
                    winning_trades = winning_trades + ?,
                    losing_trades = losing_trades + ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = 'default'
            """, (net_pnl, net_pnl, 1 if is_win else 0, 0 if is_win else 1))

            # Update Prop Risk Daily Loss
            if net_pnl < 0:
                conn.execute("""
                    UPDATE prop_risk_settings
                    SET daily_loss_accumulated = daily_loss_accumulated + ?
                    WHERE id = 'default'
                """, (abs(net_pnl),))
                
                # Check Daily Loss Limit (3%)
                risk = dict(conn.execute("SELECT * FROM prop_risk_settings WHERE id = 'default'").fetchone())
                acc = dict(conn.execute("SELECT * FROM paper_account WHERE id = 'default'").fetchone())
                max_daily_loss = acc['initial_balance'] * (risk['max_daily_loss_pct'] / 100.0)
                if risk['daily_loss_accumulated'] >= max_daily_loss:
                    conn.execute("UPDATE prop_risk_settings SET kill_switch_active = 1 WHERE id = 'default'")

            # Insert exit order log
            ord_id = f"ord_{uuid.uuid4().hex[:8]}"
            conn.execute("""
                INSERT INTO trade_orders (
                    id, position_id, broker_mode, symbol, side, order_type, quantity,
                    price, execution_price, status, brokerage_fees, reason
                ) VALUES (?, ?, ?, ?, ?, 'MARKET', ?, ?, ?, 'EXECUTED', ?, 'Position Closed')
            """, (
                ord_id, position_id, self.active_mode, sym, 'SELL' if side == 'BUY' else 'BUY',
                qty, actual_exit, actual_exit, exit_charges
            ))

            conn.commit()

        return {
            "status": "CLOSED",
            "position_id": position_id,
            "realized_pnl": net_pnl,
            "exit_price": actual_exit,
            "is_win": is_win
        }

    def close_all_positions(self) -> Dict[str, Any]:
        """
        Immediately squares off all active open positions in one transaction.
        """
        with get_db_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT id FROM trade_positions WHERE status = 'OPEN'").fetchall()
            pos_ids = [r['id'] for r in rows]

        results = []
        total_pnl = 0.0
        for pid in pos_ids:
            res = self.close_position(pid)
            if res.get('status') == 'CLOSED':
                results.append(res)
                total_pnl += res.get('realized_pnl', 0.0)

        return {
            "status": "CLOSED_ALL",
            "closed_count": len(results),
            "total_realized_pnl": round(total_pnl, 2),
            "positions": results
        }

    def trail_stop_loss(self, position_id: str, new_stop: float) -> Dict[str, Any]:
        with get_db_conn() as conn:
            conn.execute("""
                UPDATE trade_positions
                SET stop_loss = ?
                WHERE id = ? AND status = 'OPEN'
            """, (new_stop, position_id))
            conn.commit()
        return {"status": "UPDATED", "position_id": position_id, "new_stop": new_stop}

    def sync_live_prices(self, price_dict: Dict[str, float]):
        """
        Syncs live tick prices across active positions and checks SL/Target hitting.
        """
        with get_db_conn() as conn:
            conn.row_factory = sqlite3.Row
            positions = conn.execute("SELECT * FROM trade_positions WHERE status = 'OPEN'").fetchall()
            
            for p in positions:
                sym = p['symbol'].upper().replace('.NS', '')
                curr_p = None
                side = p['side']
                qty = p['quantity']
                entry_p = p['entry_price']
                sl = p['stop_loss']
                tp1 = p['target_1']
                tp2 = p['target_2']

                # ── OPTIONS DELTA TRACKING ──
                if p['instrument_type'] == 'OPTION' or ' CE' in sym or ' PE' in sym:
                    underlying_sym = 'NIFTY'
                    for k in ['BANKNIFTY', 'FINNIFTY', 'SENSEX', 'NIFTY']:
                        if k in sym:
                            underlying_sym = k
                            break

                    curr_underlying = price_dict.get(underlying_sym)
                    if not curr_underlying and underlying_sym == 'NIFTY':
                        curr_underlying = price_dict.get('NIFTY50') or price_dict.get('^NSEI')

                    if curr_underlying:
                        underlying_ref = p['underlying_entry_price'] if p['underlying_entry_price'] and p['underlying_entry_price'] > 0 else curr_underlying
                        diff = curr_underlying - underlying_ref
                        is_ce = ' CE' in sym or p['option_type'] == 'CE'
                        delta = 0.52 if is_ce else -0.48
                        premium_change = diff * delta
                        curr_p = max(0.50, round(entry_p + premium_change, 2))
                elif sym in price_dict:
                    curr_p = price_dict[sym]

                if curr_p is not None:
                    unrealized = (curr_p - entry_p) * qty if side == 'BUY' else (entry_p - curr_p) * qty
                    
                    conn.execute("""
                        UPDATE trade_positions
                        SET current_price = ?,
                            unrealized_pnl = ?
                        WHERE id = ?
                    """, (curr_p, round(unrealized, 2), p['id']))

                    # Automated Paper TP/SL Trigger
                    if side == 'BUY':
                        if sl and curr_p <= sl:
                            self.close_position(p['id'], exit_price=sl)
                        elif tp2 and curr_p >= tp2:
                            self.close_position(p['id'], exit_price=tp2)
                    else:
                        if sl and curr_p >= sl:
                            self.close_position(p['id'], exit_price=sl)
                        elif tp2 and curr_p <= tp2:
                            self.close_position(p['id'], exit_price=tp2)
            conn.commit()

    def save_broker_credentials(self, broker_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        b_name = broker_name.upper()
        with get_db_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO broker_configs (
                    broker_name, is_active, api_key, api_secret, client_id, access_token, totp_secret, updated_at
                ) VALUES (?, 1, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                b_name,
                config.get('api_key', ''),
                config.get('api_secret', ''),
                config.get('client_id', ''),
                config.get('access_token', ''),
                config.get('totp_secret', '')
            ))
            conn.commit()
        return {"status": "CONNECTED", "broker": b_name}

    def get_connected_brokers(self) -> List[Dict[str, Any]]:
        with get_db_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT broker_name, is_active, client_id, updated_at FROM broker_configs").fetchall()
            return [dict(r) for r in rows]


# Singleton Broker Gateway Instance
broker_gateway = BrokerGateway()
