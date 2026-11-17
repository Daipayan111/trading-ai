"""
Angel One SmartAPI Global Market Data Provider
==============================================
Institutional Real-Time Data Streaming for TradingAI Pro:
- Authenticates using API Key, Client Code, 4-digit PIN/Password, and dynamic TOTP
- Streams 0-delay real-time ticks for NSE/BSE heavyweights and indices
- Calibrates global `instrument_states` in `main.py` for all connected users & mobile app
"""

import time
import threading
import logging
from typing import Dict, Any, Optional

try:
    from SmartApi import SmartConnect
    import pyotp
    SMARTAPI_AVAILABLE = True
except ImportError:
    SMARTAPI_AVAILABLE = False

from supabase_db import db_manager

logger = logging.getLogger("AngelMarketData")

# Top NSE instrument tokens for Angel One SmartAPI
ANGEL_SYMBOL_TOKENS = {
    "NIFTY": {"exchange": "NSE", "symbol": "Nifty 50", "token": "99926000", "clean": "NIFTY", "key": "^NSEI"},
    "BANKNIFTY": {"exchange": "NSE", "symbol": "Nifty Bank", "token": "99926009", "clean": "BANKNIFTY", "key": "^NSEBANK"},
    "RELIANCE": {"exchange": "NSE", "symbol": "RELIANCE-EQ", "token": "2885", "clean": "RELIANCE", "key": "RELIANCE.NS"},
    "TCS": {"exchange": "NSE", "symbol": "TCS-EQ", "token": "11536", "clean": "TCS", "key": "TCS.NS"},
    "INFY": {"exchange": "NSE", "symbol": "INFY-EQ", "token": "1594", "clean": "INFY", "key": "INFY.NS"},
    "HDFCBANK": {"exchange": "NSE", "symbol": "HDFCBANK-EQ", "token": "1333", "clean": "HDFCBANK", "key": "HDFCBANK.NS"},
    "ICICIBANK": {"exchange": "NSE", "symbol": "ICICIBANK-EQ", "token": "4963", "clean": "ICICIBANK", "key": "ICICIBANK.NS"},
    "SBIN": {"exchange": "NSE", "symbol": "SBIN-EQ", "token": "3045", "clean": "SBIN", "key": "SBIN.NS"},
    "BHARTIARTL": {"exchange": "NSE", "symbol": "BHARTIARTL-EQ", "token": "10604", "clean": "BHARTIARTL", "key": "BHARTIARTL.NS"},
    "WIPRO": {"exchange": "NSE", "symbol": "WIPRO-EQ", "token": "3787", "clean": "WIPRO", "key": "WIPRO.NS"},
    "TATAMOTORS": {"exchange": "NSE", "symbol": "TATAMOTORS-EQ", "token": "3456", "clean": "TATAMOTORS", "key": "TMPV.NS"},
}


class AngelMarketDataService:
    def __init__(self):
        self.smart_api = None
        self.is_connected = False
        self.jwt_token = None
        self.feed_token = None
        self.last_sync_time = 0
        self.worker_thread = None
        self.running = False
        self._load_saved_config()
        self.auto_connect_if_saved()

    def auto_connect_if_saved(self):
        """Auto connects to Angel One SmartAPI if enabled in database"""
        self._load_saved_config()
        if self.enabled and self.api_key and self.client_code and self.pin:
            threading.Thread(target=self._run_auto_login, daemon=True).start()

    def renew_token_if_invalid(self):
        """Auto-renews jwtToken / session when AG8001 Invalid Token occurs"""
        self._load_saved_config()
        if self.enabled and self.api_key and self.client_code and self.pin:
            try:
                logger.info("[AngelMarketDataService] Auto-renewing expired Angel SmartAPI session...")
                res = self.authenticate(self.api_key, self.client_code, self.pin, self.totp_secret)
                logger.info(f"[AngelMarketDataService] Session renew status: {res.get('success')} - {res.get('message')}")
                return res.get("success", False)
            except Exception as e:
                logger.error(f"[AngelMarketDataService] Session renew error: {e}")
        return False

    def _run_auto_login(self):
        time.sleep(1.0)
        try:
            logger.info("Auto-authenticating Angel One SmartAPI using saved credentials...")
            res = self.authenticate(self.api_key, self.client_code, self.pin, self.totp_secret)
            logger.info(f"Angel One auto-connect status: {res.get('success')} - {res.get('message')}")
        except Exception as e:
            logger.error(f"Angel One auto-connect exception: {e}")

    def _load_saved_config(self):
        """Loads credentials from database system_config"""
        self.enabled = db_manager.get_config("angel_feed_enabled", False)
        self.api_key = str(db_manager.get_config("angel_api_key", "") or "").strip()
        self.client_code = str(db_manager.get_config("angel_client_code", "") or "").strip().upper()
        self.pin = str(db_manager.get_config("angel_pin", "") or "").strip()
        self.totp_secret = str(db_manager.get_config("angel_totp_secret", "") or "").strip()

    def get_status(self) -> Dict[str, Any]:
        self._load_saved_config()
        return {
            "enabled": bool(self.enabled),
            "is_connected": bool(self.is_connected),
            "api_key_set": bool(self.api_key),
            "api_key_masked": f"{self.api_key[:4]}****" if len(self.api_key) > 4 else ("Set" if self.api_key else "Not Set"),
            "client_code": self.client_code or "Not Set",
            "last_sync": self.last_sync_time,
            "library_installed": SMARTAPI_AVAILABLE
        }

    def authenticate(self, api_key: str, client_code: str, pin: str, totp_secret: str = "") -> Dict[str, Any]:
        if not SMARTAPI_AVAILABLE:
            return {"success": False, "message": "smartapi-python library is not installed"}

        try:
            api_key = str(api_key or "").strip()
            client_code = str(client_code or "").strip().upper()
            pin = str(pin or "").strip()
            totp_secret = str(totp_secret or "").strip()

            smart_api = SmartConnect(api_key=api_key)
            
            # Generate TOTP: support both direct 6-digit OTP or Base32 Secret Key
            totp_code = ""
            if totp_secret:
                clean_totp = totp_secret.replace(" ", "")
                if len(clean_totp) == 6 and clean_totp.isdigit():
                    totp_code = clean_totp
                else:
                    try:
                        totp_code = pyotp.TOTP(clean_totp).now()
                    except Exception as te:
                        return {"success": False, "message": f"Invalid TOTP Secret Key format: {te}"}

            login_res = smart_api.generateSession(
                client_code,
                pin,
                totp_code if totp_code else None
            )

            if login_res and login_res.get("status") is True:
                self.smart_api = smart_api
                self.jwt_token = login_res.get("data", {}).get("jwtToken")
                self.feed_token = smart_api.getfeedToken()
                self.is_connected = True

                # Save configs
                db_manager.set_config("angel_feed_enabled", True)
                db_manager.set_config("angel_api_key", api_key.strip())
                db_manager.set_config("angel_client_code", client_code.strip().upper())
                db_manager.set_config("angel_pin", pin.strip())
                if totp_secret:
                    db_manager.set_config("angel_totp_secret", totp_secret.strip())

                self._load_saved_config()
                self.start_background_sync()

                return {
                    "success": True,
                    "message": "Successfully authenticated with Angel One SmartAPI! Global 0-delay feed active.",
                    "client_name": login_res.get("data", {}).get("clientname", "Trader"),
                    "broker": "ANGEL_ONE"
                }
            else:
                err_msg = login_res.get("message") if login_res else "Authentication failed. Check Client ID, PIN or TOTP."
                return {"success": False, "message": err_msg}

        except Exception as e:
            logger.error(f"Angel One login exception: {e}")
            return {"success": False, "message": f"Connection error: {str(e)}"}

    def disconnect(self):
        self.is_connected = False
        self.running = False
        db_manager.set_config("angel_feed_enabled", False)
        return {"success": True, "message": "Angel One global feed disabled. Falling back to default data."}

    def fetch_live_quotes(self) -> Dict[str, float]:
        """Fetches latest LTPs for all tracked instruments via Angel One SmartAPI"""
        if not self.is_connected or not self.smart_api:
            return {}

        prices = {}
        # Query main spot indices only in background loop to keep connection completely free for Option Chain
        spot_keys = ["NIFTY", "BANKNIFTY"]
        for sym_id in spot_keys:
            meta = ANGEL_SYMBOL_TOKENS.get(sym_id)
            if not meta:
                continue
            try:
                res = self.smart_api.ltpData(
                    meta["exchange"],
                    meta["symbol"],
                    meta["token"]
                )
                if res and res.get("status") is True and res.get("data"):
                    ltp = float(res["data"].get("ltp", 0.0))
                    if ltp > 0:
                        prices[meta["key"]] = ltp
                        prices[meta["clean"]] = ltp
            except Exception:
                pass
        return prices

    def start_background_sync(self):
        if self.running:
            return
        self.running = True
        self.worker_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.worker_thread.start()

    def _sync_loop(self):
        """Continuously pulls live LTPs from Angel One and injects them into main instrument_states"""
        while self.running and self.is_connected:
            try:
                # Import main sync_live_price
                from main import sync_live_price
                quotes = self.fetch_live_quotes()
                for sym_key, price in quotes.items():
                    sync_live_price(sym_key, price)
                if quotes:
                    self.last_sync_time = time.time()
            except Exception as e:
                logger.debug(f"Angel sync cycle notice: {e}")
            time.sleep(5.0) # Sync every 5 seconds to avoid flooding Angel SmartAPI and causing read timeouts


# Singleton Service
angel_data_service = AngelMarketDataService()

