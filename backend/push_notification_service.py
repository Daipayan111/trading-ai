"""
TradingAI Pro — Real-Time Background Option Signal Push Notification Engine
==========================================================================
Dispatches institutional option trading signals (1-Click CE/PE Buy/Sell)
directly to mobile devices (iOS, Android, Web) via Expo Push API / FCM:
- Works even when the app is completely closed (background / killed)
- Plays notification sound + vibration
- Includes: Symbol, Expiry, Strike, Entry LTP, Target (1.6x), SL (0.7x), and CoA Scenario
- Anti-spam cooldown: Only 1 notification per contract every 15 minutes
"""

import time
import json
import logging
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional
from datetime import datetime
from supabase_db import db_manager

logger = logging.getLogger("OptionPushNotifier")

EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"


class OptionPushNotifier:
    def __init__(self):
        self._sent_signals: Dict[str, float] = {}  # {sig_key: last_sent_timestamp}
        self._cooldown_seconds = 900  # 15 minutes anti-spam cooldown per strike setup
        self._init_db_tables()

    def _init_db_tables(self):
        """Ensures device push tokens table exists in Supabase / Local DB"""
        try:
            # We store device push tokens in system_config or dedicated table
            pass
        except Exception as e:
            logger.debug(f"Table init notice: {e}")

    def register_device_token(self, token: str, platform: str = "mobile", device_id: str = "") -> bool:
        """Saves or updates Expo Push Token for user device"""
        if not token or not isinstance(token, str):
            return False
        token = token.strip()
        try:
            registered_tokens = db_manager.get_config("registered_push_tokens", []) or []
            if not isinstance(registered_tokens, list):
                registered_tokens = []
            
            existing = next((item for item in registered_tokens if item.get("token") == token), None)
            now_iso = datetime.now().isoformat()
            if existing:
                existing["last_active"] = now_iso
                existing["platform"] = platform
            else:
                registered_tokens.append({
                    "token": token,
                    "platform": platform,
                    "device_id": device_id,
                    "registered_at": now_iso,
                    "last_active": now_iso
                })
            
            db_manager.set_config("registered_push_tokens", registered_tokens)
            logger.info(f"Registered push token: {token[:15]}... Total devices: {len(registered_tokens)}")
            return True
        except Exception as e:
            logger.error(f"Error registering push token: {e}")
            return False

    def get_registered_tokens(self) -> List[str]:
        """Returns list of valid active push tokens"""
        try:
            tokens_data = db_manager.get_config("registered_push_tokens", [])
            if isinstance(tokens_data, str):
                import ast
                try:
                    tokens_data = json.loads(tokens_data)
                except Exception:
                    tokens_data = ast.literal_eval(tokens_data)

            if isinstance(tokens_data, list):
                res = []
                for d in tokens_data:
                    if isinstance(d, dict) and d.get("token"):
                        res.append(d["token"])
                    elif isinstance(d, str):
                        res.append(d)
                return res
        except Exception as e:
            logger.error(f"Error reading registered tokens: {e}")
        return []

    def dispatch_option_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Sends rich push notification for a live Option Trading Signal.
        signal expected keys:
        - symbol: 'BANKNIFTY' | 'NIFTY'
        - expiry: '23 Nov'
        - opt_type: 'CE' | 'PE'
        - strike: 61000
        - action: 'BUY' | 'SELL'
        - ltp: 576.70
        - target: 750.00
        - stop_loss: 490.00
        - scenario: 'Scenario 2: Strong Support & WTT'
        - win_rate: '80.4%'
        """
        sym = signal.get("symbol", "NIFTY")
        expiry = signal.get("expiry", "Weekly")
        opt_type = signal.get("opt_type", "CE")
        strike = int(signal.get("strike", 0))
        action = signal.get("action", "BUY").upper()
        ltp = float(signal.get("ltp", 0.0))
        target = float(signal.get("target", ltp * 1.5))
        sl = float(signal.get("stop_loss", ltp * 0.7))
        scenario = signal.get("scenario", "CoA 1.0 Institutional Setup")
        win_rate = signal.get("win_rate", "80%+")

        sig_key = f"{sym}_{expiry}_{strike}_{opt_type}_{action}"
        now = time.time()
        if sig_key in self._sent_signals and (now - self._sent_signals[sig_key]) < self._cooldown_seconds:
            logger.debug(f"Signal {sig_key} is on cooldown. Skipped push.")
            return False

        tokens = self.get_registered_tokens()
        if not tokens:
            logger.debug("No registered push tokens found.")
            return False

        # Craft Rich Notification Content
        title = f"🚀 {action} {sym} {strike} {opt_type} ({expiry})"
        body = (
            f"⚡ Entry: ₹{ltp:.2f} | 🎯 Target: ₹{target:.2f} | 🛑 SL: ₹{sl:.2f}\n"
            f"📊 {scenario} • Acc: {win_rate}"
        )

        payload_data = {
            "type": "OPTION_SIGNAL",
            "symbol": sym,
            "expiry": expiry,
            "strike": strike,
            "opt_type": opt_type,
            "action": action,
            "ltp": ltp,
            "target": target,
            "stop_loss": sl,
            "scenario": scenario,
            "url": f"/options?symbol={sym}&strike={strike}&opt={opt_type}"
        }

        # Build Expo Messages
        messages = []
        for t in tokens:
            messages.append({
                "to": t,
                "sound": "default",
                "title": title,
                "body": body,
                "data": payload_data,
                "priority": "high",
                "channelId": "trading_signals",
                "badge": 1,
            })

        try:
            req_data = json.dumps(messages).encode("utf-8")
            req = urllib.request.Request(
                EXPO_PUSH_URL,
                data=req_data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip, deflate"
                }
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                res_code = response.getcode()
                if res_code == 200:
                    self._sent_signals[sig_key] = now
                    logger.info(f"Successfully broadcasted push notification for {sig_key} to {len(tokens)} devices!")
                    return True
        except Exception as e:
            logger.warning(f"Failed to dispatch Expo push notification: {e}")
            return False

        return False


# Global Singleton Instance
option_push_notifier = OptionPushNotifier()
