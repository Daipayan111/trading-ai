"""
TradingAI Pro — High-Performance In-Memory RAM Cache (Groww / Angel One Architecture)
====================================================================================
Sub-millisecond (< 1ms) in-memory TTL cache with thread-safe locking and LRU eviction.
Eliminates redundant external network trips (Yahoo Finance / DB) and on-the-fly ML retraining.
"""

import time
import threading
from collections import OrderedDict
from functools import wraps
from typing import Any, Optional, Callable


class FastMemoryCache:
    """Thread-safe In-Memory LRU Cache with Time-To-Live (TTL)."""

    def __init__(self, max_size: int = 1500, default_ttl: float = 60.0):
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value if key exists and is not expired."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            value, expire_at = self._cache[key]
            now = time.time()
            if now > expire_at:
                # Expired -> Evict
                del self._cache[key]
                self._misses += 1
                return None

            # LRU: Move to end on hit
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store value with custom or default TTL."""
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expire_at = time.time() + effective_ttl

        with self._lock:
            if key in self._cache:
                self._cache[key] = (value, expire_at)
                self._cache.move_to_end(key)
                return

            # Check capacity and evict oldest if needed
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            self._cache[key] = (value, expire_at)

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict:
        """Get cache performance stats."""
        with self._lock:
            total = self._hits + self._misses
            hit_ratio = round((self._hits / total * 100), 1) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio_pct": hit_ratio,
            }


# Global singleton cache instances
# 1. Candle & market data cache (TTL: 45s for intraday, 300s for daily)
market_cache = FastMemoryCache(max_size=1000, default_ttl=45.0)

# 2. AI prediction cache (TTL: 45s)
predict_cache = FastMemoryCache(max_size=500, default_ttl=45.0)

# 3. Option chain / LTP calculator cache (TTL: 15s)
options_cache = FastMemoryCache(max_size=200, default_ttl=15.0)


def fast_cached(cache_instance: FastMemoryCache, ttl: Optional[float] = None, key_prefix: str = ""):
    """Decorator for caching function return values with automatic key generation."""
    def decorator(func: Callable):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            key_parts = [key_prefix or func.__name__]
            if args:
                key_parts.extend(str(a) for a in args)
            if kwargs:
                for k in sorted(kwargs.keys()):
                    key_parts.append(f"{k}={kwargs[k]}")
            cache_key = ":".join(key_parts)

            cached_val = cache_instance.get(cache_key)
            if cached_val is not None:
                return cached_val

            result = func(*args, **kwargs)
            if result is not None:
                cache_instance.set(cache_key, result, ttl=ttl)
            return result

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            key_parts = [key_prefix or func.__name__]
            if args:
                key_parts.extend(str(a) for a in args)
            if kwargs:
                for k in sorted(kwargs.keys()):
                    key_parts.append(f"{k}={kwargs[k]}")
            cache_key = ":".join(key_parts)

            cached_val = cache_instance.get(cache_key)
            if cached_val is not None:
                return cached_val

            result = await func(*args, **kwargs)
            if result is not None:
                cache_instance.set(cache_key, result, ttl=ttl)
            return result

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator

