"""
security.py
===========
FastAPI dependencies for authentication and rate limiting.

Authentication
--------------
Every data route requires the caller to supply a matching X-API-Key
header. The comparison uses hmac.compare_digest() — a constant-time
function — to prevent timing side-channel attacks that could let an
attacker probe the key one character at a time by measuring response
latency.

Rate limiting
-------------
Per-IP sliding-window counter (60-second window).

  In-memory (asyncio.Lock):
    Default mode. Safe for single-worker deployments on Render/Railway
    free tiers. State is lost on restart — that is acceptable for a
    rate limiter (the window simply resets).

  Redis-backed (aioredis sorted-set):
    Activated when REDIS_URL is set in the environment. Required for
    multi-worker or multi-instance deployments where each worker would
    otherwise maintain separate, inconsistent counters.
    Install: pip install redis
"""

import asyncio
import hmac
import time
from collections import defaultdict
from typing import Optional

from fastapi import HTTPException, Request, status

from config import settings
from logging_setup import logger


# ---------------------------------------------------------------------------
# API-key authentication
# ---------------------------------------------------------------------------

def verify_api_key(request: Request) -> None:
    """
    FastAPI dependency. Raises HTTP 401 if the X-API-Key header is
    absent or doesn't match the configured secret.

    Uses hmac.compare_digest() for constant-time string comparison —
    a plain `==` comparison short-circuits on the first differing byte,
    which can leak information about the key via response-timing
    measurements.
    """
    incoming_key = request.headers.get("X-API-Key", "")
    # compare_digest requires both operands to be the same type (str or bytes)
    if not hmac.compare_digest(incoming_key, settings.api_key):
        client_ip = request.client.host if request.client else "unknown"
        logger.warning(
            "Rejected request from %s — invalid or missing X-API-Key header.",
            client_ip,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


# ---------------------------------------------------------------------------
# In-memory sliding-window rate limiter
# ---------------------------------------------------------------------------
# Maps client IP → list of monotonic timestamps of recent requests.
# Uses an asyncio.Lock so concurrent async handlers don't race on the dict.

_rate_windows: dict[str, list[float]] = defaultdict(list)
_rate_lock = asyncio.Lock()
_WINDOW_SECONDS = 60.0


async def _in_memory_rate_check(ip: str) -> None:
    now = time.monotonic()
    cutoff = now - _WINDOW_SECONDS

    async with _rate_lock:
        # Evict timestamps that have fallen outside the sliding window
        _rate_windows[ip] = [t for t in _rate_windows[ip] if t > cutoff]

        if len(_rate_windows[ip]) >= settings.rate_limit_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Rate limit exceeded. "
                    f"Maximum {settings.rate_limit_per_minute} requests per minute."
                ),
                headers={"Retry-After": "60"},
            )

        _rate_windows[ip].append(now)


# ---------------------------------------------------------------------------
# Redis-backed sliding-window rate limiter (optional)
# ---------------------------------------------------------------------------
# Uses a sorted set keyed by IP. Score = Unix timestamp. Members older
# than the window are pruned on each request via ZREMRANGEBYSCORE,
# then the remaining cardinality is checked against the limit.

_redis_client: Optional[object] = None


def _get_redis_client() -> Optional[object]:
    global _redis_client
    if _redis_client is None and settings.redis_url:
        try:
            import redis.asyncio as aioredis  # type: ignore[import]
            _redis_client = aioredis.from_url(
                settings.redis_url, decode_responses=True
            )
            logger.info("Redis rate-limiter backend connected.")
        except ImportError:
            logger.warning(
                "REDIS_URL is set but the 'redis' package is not installed. "
                "Falling back to in-memory rate limiter. "
                "Fix: pip install redis"
            )
    return _redis_client


async def _redis_rate_check(ip: str) -> None:
    client = _get_redis_client()
    if client is None:
        # Redis package unavailable — fall back gracefully
        await _in_memory_rate_check(ip)
        return

    now = time.time()
    key = f"rxguard:rl:{ip}"
    cutoff = now - _WINDOW_SECONDS

    # Pipelined to reduce round-trips:
    # 1. Remove entries older than the window
    # 2. Record this request
    # 3. Count remaining entries
    # 4. Set a TTL so idle keys are cleaned up automatically
    pipe = client.pipeline()
    pipe.zremrangebyscore(key, "-inf", cutoff)
    pipe.zadd(key, {f"{now}": now})
    pipe.zcard(key)
    pipe.expire(key, int(_WINDOW_SECONDS) + 10)
    results = await pipe.execute()

    count: int = results[2]
    if count > settings.rate_limit_per_minute:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded. "
                f"Maximum {settings.rate_limit_per_minute} requests per minute."
            ),
            headers={"Retry-After": "60"},
        )


# ---------------------------------------------------------------------------
# Public dependency — routes to the correct backend automatically
# ---------------------------------------------------------------------------

async def enforce_rate_limit(request: Request) -> None:
    """
    FastAPI dependency. Enforces per-IP sliding-window rate limiting.
    Automatically selects Redis if REDIS_URL is configured, otherwise
    uses the in-memory implementation.
    """
    ip = request.client.host if request.client else "unknown"

    if settings.redis_url:
        await _redis_rate_check(ip)
    else:
        await _in_memory_rate_check(ip)


def get_rate_limiter_backend() -> str:
    """Returns a human-readable label for the active rate-limiter backend."""
    return "redis" if settings.redis_url else "in-memory"
