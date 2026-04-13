"""
cache.py — Redis cache helpers for PriceIQ Mini
Keys are SHA256(catalog_content + image_url) so identical products are instant.
Falls back gracefully when Redis is not available (e.g., local dev without Docker).
"""

import hashlib
import json
import os
import redis

_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
_TTL_SECONDS = 3600  # 1 hour

try:
    _redis = redis.from_url(_REDIS_URL, decode_responses=True)
    _redis.ping()
    _REDIS_AVAILABLE = True
    print("✅ Redis connected")
except Exception as e:
    _REDIS_AVAILABLE = False
    print(f"⚠️  Redis unavailable ({e}) — caching disabled")


def make_key(catalog_content: str, image_url: str) -> str:
    raw = f"{catalog_content.strip()}{image_url.strip()}"
    return "priceiq:" + hashlib.sha256(raw.encode()).hexdigest()


def get_cache(key: str):
    if not _REDIS_AVAILABLE:
        return None
    try:
        val = _redis.get(key)
        return json.loads(val) if val else None
    except Exception:
        return None


def set_cache(key: str, data: dict) -> None:
    if not _REDIS_AVAILABLE:
        return
    try:
        _redis.setex(key, _TTL_SECONDS, json.dumps(data))
    except Exception:
        pass