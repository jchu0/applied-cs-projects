"""Production hardening for the REST API: API-key auth, rate limiting, timeouts.

All three concerns are opt-in via environment variables so the default
development / test experience (and the quick-start) keep working with zero
configuration. Everything here is stdlib-only -- no extra dependencies.

Environment variables
----------------------
``API_KEYS``               Comma-separated list of valid API keys. Unset/empty
                           disables auth entirely.
``RATE_LIMIT_PER_MINUTE``  Max requests per minute per client (default 120;
                           0 disables).
``REQUEST_TIMEOUT_SECONDS`` Per-request wall-clock budget in seconds
                           (default 30; 0 disables).
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import threading
import time
from collections import defaultdict, deque
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# Paths that must stay open regardless of auth/rate-limit configuration.
OPEN_PATHS = frozenset(
    {"/health", "/readiness", "/", "/docs", "/redoc", "/openapi.json"}
)


# --------------------------------------------------------------------- helpers

def _load_api_keys() -> list[str]:
    """Return the configured API keys (empty list => auth disabled)."""
    raw = os.environ.get("API_KEYS", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def _env_number(name: str, default: float, cast) -> float:
    """Read a numeric env var, falling back to ``default`` on unset/garbage."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default


def _extract_key(request: Request) -> Optional[str]:
    """Pull an API key from ``Authorization: Bearer`` or ``X-API-Key``."""
    header = request.headers.get("Authorization")
    if header and header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.headers.get("X-API-Key")


def _key_valid(candidate: str, valid_keys: list[str]) -> bool:
    """Constant-time membership test to avoid leaking key length/prefix."""
    return any(hmac.compare_digest(candidate, k) for k in valid_keys)


# ------------------------------------------------------------------- rate limit

class _SlidingWindowLimiter:
    """Thread-safe in-process sliding-window rate limiter (per client)."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._window = 60.0
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        """Record a hit for ``key``. Return ``(allowed, retry_after_seconds)``."""
        now = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.per_minute:
                retry_after = max(1, int(self._window - (now - hits[0])) + 1)
                return False, retry_after
            hits.append(now)
            return True, 0


# ------------------------------------------------------------------- wiring

def install_security(app: FastAPI) -> None:
    """Wire API-key auth, rate limiting, and request timeouts onto ``app``.

    All three are enforced in a single ASGI/HTTP middleware so they apply to
    every route (open paths are exempted below). Call this once, after the
    routes are registered.
    """
    api_keys = _load_api_keys()
    app.state.api_keys = api_keys
    if not api_keys:
        logger.warning("API auth disabled (set API_KEYS to enable)")

    rate_limit = int(_env_number("RATE_LIMIT_PER_MINUTE", 120, int))
    timeout = _env_number("REQUEST_TIMEOUT_SECONDS", 30.0, float)
    limiter = _SlidingWindowLimiter(rate_limit) if rate_limit > 0 else None

    @app.middleware("http")
    async def _hardening_middleware(request: Request, call_next):
        path = request.url.path
        exempt = path in OPEN_PATHS

        # 1) API-key auth.
        if api_keys and not exempt:
            candidate = _extract_key(request)
            if not candidate or not _key_valid(candidate, api_keys):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Missing or invalid API key"},
                    headers={"WWW-Authenticate": "Bearer"},
                )

        # 2) Rate limiting (key by API key if present, else client IP).
        if limiter is not None and not exempt:
            client_key = _extract_key(request) or (
                request.client.host if request.client else "unknown"
            )
            allowed, retry_after = limiter.check(client_key)
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"},
                    headers={"Retry-After": str(retry_after)},
                )

        # 3) Request timeout.
        if timeout > 0 and not exempt:
            try:
                return await asyncio.wait_for(call_next(request), timeout=timeout)
            except asyncio.TimeoutError:
                return JSONResponse(
                    status_code=504,
                    content={"detail": "Request timed out"},
                )
        return await call_next(request)
