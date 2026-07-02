"""Production hardening for the Data Observability REST API.

Provides three opt-in, stdlib-only cross-cutting concerns:

1. API-key authentication (``API_KEYS``) as a FastAPI dependency.
2. In-process rate limiting (``RATE_LIMIT_PER_MINUTE``) via a sliding window.
3. Per-request timeouts (``REQUEST_TIMEOUT_SECONDS``) as ASGI/HTTP middleware.

All three are disabled by default so existing tests and the quick-start keep
working with no configuration.
"""

import asyncio
import hmac
import logging
import os
import threading
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# Paths that are always open (no auth, no rate limit).
_OPEN_PATHS = frozenset({
    "/health",
    "/",
    "/docs",
    "/redoc",
    "/openapi.json",
})


def _is_open_path(path: str) -> bool:
    """Return True if the path is exempt from auth and rate limiting."""
    return path in _OPEN_PATHS or path.startswith("/docs") or path.startswith("/redoc")


def _load_api_keys() -> list[str]:
    """Parse valid API keys from the ``API_KEYS`` env var (comma-separated)."""
    raw = os.environ.get("API_KEYS", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def _extract_key(authorization: str | None, x_api_key: str | None) -> str | None:
    """Pull the presented key from ``Authorization: Bearer`` or ``X-API-Key``."""
    if authorization:
        parts = authorization.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
    if x_api_key:
        return x_api_key.strip()
    return None


async def require_api_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """FastAPI dependency enforcing API-key auth when ``API_KEYS`` is set.

    Auth is disabled (no-op) when ``API_KEYS`` is unset or empty. When enabled,
    a valid key must be supplied via ``Authorization: Bearer <key>`` or the
    ``X-API-Key`` header; otherwise a 401 is raised.
    """
    valid_keys = _load_api_keys()
    if not valid_keys:
        return  # Auth disabled.

    presented = _extract_key(authorization, x_api_key)
    if presented and any(hmac.compare_digest(presented, k) for k in valid_keys):
        return

    raise HTTPException(
        status_code=401,
        detail="Missing or invalid API key",
        headers={"WWW-Authenticate": "Bearer"},
    )


class _SlidingWindowRateLimiter:
    """Thread-safe in-process sliding-window rate limiter (stdlib only).

    The limit is read from the environment on each call so tests (and live
    reconfiguration) take effect without recreating the app.
    """

    def __init__(self) -> None:
        self.window = 60.0
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int) -> tuple[bool, int]:
        """Record a hit for ``key`` against ``limit`` requests per window.

        Returns ``(allowed, retry_after_seconds)``. ``retry_after`` is 0 when
        the request is allowed.
        """
        now = time.monotonic()
        cutoff = now - self.window
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= limit:
                retry_after = int(self.window - (now - hits[0])) + 1
                return False, max(retry_after, 1)
            hits.append(now)
            return True, 0

    def reset(self) -> None:
        """Clear all recorded hits (used by tests)."""
        with self._lock:
            self._hits.clear()


def _load_rate_limit() -> int:
    """Read ``RATE_LIMIT_PER_MINUTE`` (int, default 120; 0 disables)."""
    try:
        return int(os.environ.get("RATE_LIMIT_PER_MINUTE", "120"))
    except ValueError:
        return 120


def _load_timeout() -> float:
    """Read ``REQUEST_TIMEOUT_SECONDS`` (float, default 30; 0 disables)."""
    try:
        return float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "30"))
    except ValueError:
        return 30.0


def _client_key(request: Request) -> str:
    """Rate-limit key: the API key if present, else the client IP."""
    key = _extract_key(
        request.headers.get("authorization"),
        request.headers.get("x-api-key"),
    )
    if key:
        return f"key:{key}"
    client = request.client
    return f"ip:{client.host if client else 'unknown'}"


def install_hardening(app) -> None:
    """Wire auth, rate limiting, and request timeouts onto ``app``.

    Emits exactly one startup warning when auth is disabled. All three concerns
    read their env vars per request, so tests can toggle them via monkeypatch
    without recreating the app. Safe to call on a module-level FastAPI ``app``.
    """
    if not _load_api_keys():
        logger.warning("API auth disabled (set API_KEYS to enable)")

    limiter = _SlidingWindowRateLimiter()
    # Exposed for tests / operators to reset the in-process window.
    app.state.rate_limiter = limiter

    @app.middleware("http")
    async def _hardening_middleware(request: Request, call_next):
        path = request.url.path
        is_open = _is_open_path(path)

        # --- API-key auth (open paths exempt) ---------------------------
        if not is_open and _load_api_keys():
            key = _extract_key(
                request.headers.get("authorization"),
                request.headers.get("x-api-key"),
            )
            valid_keys = _load_api_keys()
            if not (key and any(hmac.compare_digest(key, k) for k in valid_keys)):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Missing or invalid API key"},
                    headers={"WWW-Authenticate": "Bearer"},
                )

        # --- Rate limiting (open paths exempt) --------------------------
        if not is_open:
            rate_limit = _load_rate_limit()
            if rate_limit > 0:
                allowed, retry_after = limiter.check(_client_key(request), rate_limit)
                if not allowed:
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "Rate limit exceeded"},
                        headers={"Retry-After": str(retry_after)},
                    )

        # --- Request timeout (open paths exempt) ------------------------
        timeout = _load_timeout()
        if not is_open and timeout > 0:
            try:
                return await asyncio.wait_for(call_next(request), timeout=timeout)
            except asyncio.TimeoutError:
                return JSONResponse(
                    status_code=504,
                    content={"detail": "Request timed out"},
                )

        return await call_next(request)
