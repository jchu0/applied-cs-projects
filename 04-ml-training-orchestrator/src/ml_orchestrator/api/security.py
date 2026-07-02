"""Production hardening: API-key auth, rate limiting, and request timeouts.

All three concerns are opt-in via environment variables so existing tests and the
quick-start keep working with a permissive default. Everything here is stdlib-only
(plus FastAPI/Starlette primitives already used by the app) -- no extra dependencies.
"""

from __future__ import annotations

import asyncio
import hmac
import os
import time
from collections import defaultdict, deque
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Environment / config helpers
# ---------------------------------------------------------------------------

# Path prefixes that are always open (never require auth and are exempt from
# rate limiting): health/readiness/root and the interactive docs + schema.
EXEMPT_PREFIXES: tuple[str, ...] = (
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
)


def _load_api_keys() -> list[str]:
    """Return valid API keys from the ``API_KEYS`` env var (comma-separated)."""
    raw = os.environ.get("API_KEYS", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def _rate_limit_per_minute() -> int:
    """Requests allowed per minute per client. 0 disables. Default 120."""
    try:
        return int(os.environ.get("RATE_LIMIT_PER_MINUTE", "120"))
    except ValueError:
        return 120


def _request_timeout_seconds() -> float:
    """Per-request timeout in seconds. 0 disables. Default 30."""
    try:
        return float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "30"))
    except ValueError:
        return 30.0


def _is_exempt(path: str) -> bool:
    """True if ``path`` is a health/docs/root path exempt from auth + limits."""
    if path == "/":
        return True
    return any(path == p or path.startswith(p + "/") or path == p for p in EXEMPT_PREFIXES)


# ---------------------------------------------------------------------------
# 1) API-key authentication
# ---------------------------------------------------------------------------

# auto_error=False so we can support either header and craft our own 401.
_bearer_scheme = HTTPBearer(auto_error=False)
_api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


def _valid_key(candidate: str, keys: list[str]) -> bool:
    """Constant-time membership check against configured keys."""
    return any(hmac.compare_digest(candidate, k) for k in keys)


async def require_api_key(
    request: Request,
    bearer: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    x_api_key: Optional[str] = Depends(_api_key_scheme),
) -> None:
    """FastAPI dependency enforcing API-key auth when enabled.

    Accepts either ``Authorization: Bearer <key>`` or ``X-API-Key: <key>``.
    When ``API_KEYS`` is unset/empty, auth is disabled and this is a no-op.
    """
    keys = _load_api_keys()
    if not keys:
        return  # auth disabled

    candidate: Optional[str] = None
    if bearer is not None and bearer.scheme.lower() == "bearer":
        candidate = bearer.credentials
    elif x_api_key:
        candidate = x_api_key

    if candidate and _valid_key(candidate, keys):
        # Stash the authenticated key for the rate limiter to key on.
        request.state.api_key = candidate
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
        headers={"WWW-Authenticate": "Bearer"},
    )


# ---------------------------------------------------------------------------
# 2) Rate limiting (in-process sliding window, thread/async safe)
# ---------------------------------------------------------------------------


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window per-client rate limiter using only the stdlib.

    Keyed by API key when present, otherwise the client IP. Health/docs are
    exempt. Returns 429 with a ``Retry-After`` header when the limit is exceeded.
    """

    def __init__(self, app, per_minute: int) -> None:
        super().__init__(app)
        self._per_minute = per_minute
        self._window = 60.0
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    def _client_key(self, request: Request) -> str:
        key = getattr(request.state, "api_key", None)
        if key:
            return f"key:{key}"
        client = request.client
        return f"ip:{client.host if client else 'unknown'}"

    async def dispatch(self, request: Request, call_next):
        if self._per_minute <= 0 or _is_exempt(request.url.path):
            return await call_next(request)

        now = time.monotonic()
        client_key = self._client_key(request)

        async with self._lock:
            hits = self._hits[client_key]
            cutoff = now - self._window
            while hits and hits[0] <= cutoff:
                hits.popleft()

            if len(hits) >= self._per_minute:
                retry_after = max(1, int(self._window - (now - hits[0])))
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": "Rate limit exceeded"},
                    headers={"Retry-After": str(retry_after)},
                )
            hits.append(now)

        return await call_next(request)


# ---------------------------------------------------------------------------
# 3) Request timeout
# ---------------------------------------------------------------------------


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Wrap each request handler in ``asyncio.wait_for``; 504 on timeout.

    There are no streaming/SSE/WebSocket endpoints in this app, so nothing is
    exempt. (WebSocket connections bypass BaseHTTPMiddleware entirely in any case.)
    """

    def __init__(self, app, timeout_seconds: float) -> None:
        super().__init__(app)
        self._timeout = timeout_seconds

    async def dispatch(self, request: Request, call_next):
        if self._timeout <= 0:
            return await call_next(request)
        try:
            return await asyncio.wait_for(call_next(request), timeout=self._timeout)
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content={"detail": "Request timed out"},
            )


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def configure_security(app: FastAPI) -> None:
    """Attach rate-limit + timeout middleware and log the auth-status warning.

    Auth itself is applied as a router-level dependency in the app factory so
    that only the protected routers (not health/docs) require a key.
    """
    if not _load_api_keys():
        logger.warning("API auth disabled (set API_KEYS to enable)")

    # Middleware added last runs first (outermost). We add timeout first so it
    # wraps the rate limiter and the handler.
    app.add_middleware(TimeoutMiddleware, timeout_seconds=_request_timeout_seconds())
    app.add_middleware(RateLimitMiddleware, per_minute=_rate_limit_per_minute())
