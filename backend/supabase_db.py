"""
Supabase Cloud Database & Local Resilient Persistence Layer
===========================================================
Enterprise database adapter for TradingAI Pro:
- Connects to Supabase PostgreSQL Cloud instance (supabase.com)
- Built-in SQLite local cache & offline resilience (trading_ai.db)
- Manages Users, Subscription Tiers, Live AI Engine Controls, and Broadcast Logs
"""

import os
import json
import sqlite3
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

DB_DIR = Path(__file__).resolve().parent
LOCAL_DB_PATH = DB_DIR / "trading_ai.db"
ENV_PATH = DB_DIR / ".env"

try:
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH)
except Exception:
    pass

# Try importing official supabase client
try:
    from supabase import create_client, Client
    HAS_SUPABASE_LIB = True
except ImportError:
    HAS_SUPABASE_LIB = False
    Client = Any


class SupabaseDBManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SupabaseDBManager, cls).__new__(cls)
            cls._instance._init_manager()
        return cls._instance

    def _init_manager(self):
        self.supabase_url = os.getenv("SUPABASE_URL", "").strip()
        self.supabase_key = os.getenv("SUPABASE_KEY", "").strip()
        self.client: Optional[Client] = None
        self.is_connected = False
        
        # 1. Initialize resilient local DB first
        self._init_local_db()
        
        # 2. Attempt Supabase Cloud connection
        self._connect_supabase()

    def _init_local_db(self):
        """Initializes SQLite schema for users, AI configs, and audit logs"""
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Users Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    phone TEXT,
                    plan TEXT DEFAULT 'FREE',
                    status TEXT DEFAULT 'ACTIVE',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT
                )
            """)

            # System Engine Configs (Dynamic AI parameter controls)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    description TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Signal Archive
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signal_archive (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    target REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    conviction REAL,
                    confluence_score TEXT,
                    outcome TEXT DEFAULT 'PENDING',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Announcements & Alerts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS announcements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    level TEXT DEFAULT 'INFO',
                    is_active INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Seed default system configs if empty
            cursor.execute("SELECT COUNT(*) FROM system_config")
            if cursor.fetchone()[0] == 0:
                default_configs = [
                    ("min_confluence_score", "7", "Minimum confluence checks passed out of 9 required to issue trade"),
                    ("tp1_multiplier", "0.65", "Safe TP1 target ATR multiplier for momentum 50% lock"),
                    ("tp2_multiplier", "1.40", "Institutional primary supply zone target multiplier"),
                    ("tp3_multiplier", "2.60", "Trend runner target multiplier"),
                    ("sl_buffer", "1.60", "Stop loss ATR risk cushion buffer"),
                    ("liquidity_wick_shield", "true", "Ignore intraday wick stop-outs unless candle body closes beyond SL"),
                    ("traps_filter_active", "true", "Auto-disqualify trades when Bull/Bear or Consolidation Chop Traps detected"),
                    ("kill_switch", "false", "Emergency Kill-Switch: Halt all trade signals across all user apps immediately"),
                    ("darkpool_min_cr", "5.0", "Minimum transaction value in Crores to trigger Dark Pool / Block Deal badge"),
                    ("active_announcement", "🚀 Trading AI Pro v2.4 Live: 9-Point Institutional Confluence Active (82.9% Win Rate)", "Live global announcement on app header"),
                    ("supabase_url", self.supabase_url, "Supabase Project URL"),
                    ("supabase_key", self.supabase_key, "Supabase Project Public/Secret Key")
                ]
                cursor.executemany("INSERT OR REPLACE INTO system_config (key, value, description) VALUES (?, ?, ?)", default_configs)

            # Seed demo users if empty
            cursor.execute("SELECT COUNT(*) FROM users")
            if cursor.fetchone()[0] == 0:
                demo_users = [
                    ("usr_101", "Tapas Adhikary (Admin)", "tapas@tradingai.pro", "+91 98765 43210", "VIP_INSTITUTIONAL", "ACTIVE", "Founder / System Admin"),
                    ("usr_102", "Vikram Malhotra", "vikram.m@wealthcorp.in", "+91 98112 34567", "VIP_INSTITUTIONAL", "ACTIVE", "Hedge Fund Trader - Scalping TCS & BankNifty"),
                    ("usr_103", "Ananya Sharma", "ananya.s@traderdesk.com", "+91 97234 56789", "PRO", "ACTIVE", "Swing Trader"),
                    ("usr_104", "Rahul Verma", "rahul.v@gmail.com", "+91 91234 56780", "FREE", "ACTIVE", "Trial Trader (7-day evaluation)"),
                    ("usr_105", "Amitabh Sen", "amitabh.sen@nsefirm.co", "+91 99887 76655", "PRO", "ACTIVE", "Momentum Scalper"),
                    ("usr_106", "Karan Kapoor", "karan.k@badtrader.org", "+91 94567 89012", "FREE", "BANNED", "Suspicious device activity / Multiple IP logins"),
                ]
                cursor.executemany("INSERT INTO users (id, full_name, email, phone, plan, status, notes) VALUES (?, ?, ?, ?, ?, ?, ?)", demo_users)

            conn.commit()

    def _connect_supabase(self):
        """Attempts to create client connection to Supabase cloud instance"""
        # Read from config table if not in environment
        if not self.supabase_url:
            self.supabase_url = self.get_config("supabase_url", "")
        if not self.supabase_key:
            self.supabase_key = self.get_config("supabase_key", "")

        if HAS_SUPABASE_LIB and self.supabase_url and self.supabase_key and "supabase.co" in self.supabase_url:
            try:
                self.client = create_client(self.supabase_url, self.supabase_key)
                self.is_connected = True
                print(">>> [DATABASE] Connected successfully to Supabase Cloud instance:", self.supabase_url)
            except Exception as e:
                print(">>> [DATABASE] Supabase cloud connection notice:", e)
                self.is_connected = False
        else:
            self.is_connected = False

    def get_status(self) -> Dict[str, Any]:
        """Returns database engine health, provider (Supabase vs Local), and record counts"""
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM users")
            user_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM users WHERE plan LIKE '%VIP%'")
            vip_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM system_config")
            config_count = cursor.fetchone()[0]

        return {
            "provider": "Supabase Cloud (PostgreSQL)" if self.is_connected else "Local Resilient Engine (SQLite / Supabase Ready)",
            "is_supabase_connected": self.is_connected,
            "supabase_url": self.supabase_url if self.supabase_url else "Not Configured Yet",
            "has_supabase_key": bool(self.supabase_key),
            "stats": {
                "total_users": user_count,
                "vip_subscribers": vip_count,
                "active_configs": config_count,
                "database_size": f"{round(os.path.getsize(LOCAL_DB_PATH) / 1024, 1)} KB" if os.path.exists(LOCAL_DB_PATH) else "0 KB"
            },
            "server_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
        }

    def save_supabase_credentials(self, url: str, key: str) -> Dict[str, Any]:
        """Save Supabase credentials and attempt immediate test handshake"""
        url = url.strip()
        key = key.strip()
        
        self.set_config("supabase_url", url)
        self.set_config("supabase_key", key)
        self.supabase_url = url
        self.supabase_key = key
        
        # Save to .env file as well
        try:
            with open(ENV_PATH, "a") as f:
                f.write(f"\nSUPABASE_URL={url}\nSUPABASE_KEY={key}\n")
        except Exception:
            pass

        self._connect_supabase()
        
        return {
            "success": self.is_connected,
            "message": "Successfully authenticated and connected with Supabase Cloud!" if self.is_connected else "Credentials saved. Supabase client initialized (verify URL/Key format)."
        }

    # ── User Operations (Cloud-First with Local Fallback) ────────
    def get_users(self, search: str = "", plan_filter: str = "") -> List[Dict[str, Any]]:
        # 1. Attempt Supabase Cloud Query
        if self.is_connected and self.client:
            try:
                query = self.client.table("users").select("*").order("created_at", desc=True)
                if plan_filter and plan_filter != "ALL":
                    query = query.eq("plan", plan_filter)
                res = query.execute()
                users = res.data or []
                if search:
                    s_low = search.lower()
                    users = [u for u in users if s_low in str(u.get("full_name", "")).lower() or s_low in str(u.get("email", "")).lower() or s_low in str(u.get("phone", "")).lower()]
                return users
            except Exception as e:
                print("Supabase cloud user query notice:", e)

        # 2. Resilient Local SQLite Fallback
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            query = "SELECT * FROM users WHERE 1=1"
            params = []
            if search:
                query += " AND (full_name LIKE ? OR email LIKE ? OR phone LIKE ?)"
                s_param = f"%{search}%"
                params.extend([s_param, s_param, s_param])
            if plan_filter and plan_filter != "ALL":
                query += " AND plan = ?"
                params.append(plan_filter)
            query += " ORDER BY created_at DESC"
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def create_user(self, full_name: str, email: str, phone: str = "", plan: str = "FREE", notes: str = "") -> Dict[str, Any]:
        user_id = f"usr_{int(datetime.datetime.now().timestamp())}"
        user_record = {
            "id": user_id,
            "full_name": full_name,
            "email": email,
            "phone": phone,
            "plan": plan,
            "status": "ACTIVE",
            "notes": notes
        }

        # 1. Sync to Supabase Cloud
        if self.is_connected and self.client:
            try:
                self.client.table("users").insert(user_record).execute()
            except Exception as e:
                print("Supabase cloud insert notice:", e)

        # 2. Resilient Local SQLite
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO users (id, full_name, email, phone, plan, status, notes)
                VALUES (?, ?, ?, ?, ?, 'ACTIVE', ?)
            """, (user_id, full_name, email, phone, plan, notes))
            conn.commit()

        return user_record

    def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        allowed = ["full_name", "email", "phone", "plan", "status", "notes"]
        filtered = {k: v for k, v in updates.items() if k in allowed}
        if not filtered:
            return False

        # 1. Update Supabase Cloud
        if self.is_connected and self.client:
            try:
                self.client.table("users").update(filtered).eq("id", user_id).execute()
            except Exception as e:
                print("Supabase cloud update notice:", e)

        # 2. Update Local SQLite
        fields = [f"{k} = ?" for k in filtered.keys()]
        values = list(filtered.values()) + [user_id]
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE users SET {', '.join(fields)}, last_active = CURRENT_TIMESTAMP WHERE id = ?", values)
            conn.commit()
        return True

    def delete_user(self, user_id: str) -> bool:
        if self.is_connected and self.client:
            try:
                self.client.table("users").delete().eq("id", user_id).execute()
            except Exception as e:
                print("Supabase cloud delete notice:", e)

        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
        return True

    # ── AI Engine System Config Operations ───────────────────────
    def get_all_configs(self) -> Dict[str, Any]:
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM system_config")
            res = {}
            for k, v in cursor.fetchall():
                if v.lower() == "true":
                    res[k] = True
                elif v.lower() == "false":
                    res[k] = False
                else:
                    try:
                        res[k] = float(v) if "." in v else int(v)
                    except ValueError:
                        res[k] = v
            return res

    def get_config(self, key: str, default: Any = None) -> Any:
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM system_config WHERE key = ?", (key,))
            row = cursor.fetchone()
            if not row:
                return default
            val = row[0]
            if val.lower() == "true": return True
            if val.lower() == "false": return False
            try:
                return float(val) if "." in val else int(val)
            except ValueError:
                return val

    def set_config(self, key: str, value: Any):
        str_val = "true" if value is True else ("false" if value is False else str(value))
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO system_config (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)", (key, str_val))
            conn.commit()

    def update_multiple_configs(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            cursor = conn.cursor()
            for k, v in config_dict.items():
                str_val = "true" if v is True else ("false" if v is False else str(v))
                cursor.execute("INSERT OR REPLACE INTO system_config (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)", (k, str_val))
            conn.commit()
        return self.get_all_configs()


# Singleton Instance Export
db_manager = SupabaseDBManager()
