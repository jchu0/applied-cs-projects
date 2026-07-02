"""
Production hardening for the Feature Platform API.

Provides three opt-in, stdlib-only cross-cutting concerns:

- API-key authentication (``API_KEYS``)
- In-process rate limiting (``RATE_LIMIT_PER_MINUTE``)
- Request timeouts (``REQUEST_TIMEOUT_SECONDS``)

Everything is configured via environment variables so that existing tests
and the quick-start keep working with no configuration (auth disabled).
"""

import asyncio
import hmac
import logging
import os
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from fastapi.dependencies.utils import get_parameterless_sub_dependant
from fastapi.security import APIKeyHeader, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Paths that stay open regardless of auth/rate-limit configuration.
OPEN_PATHS = frozenset(
    {
        "/",
        "/health",
        "/ready",
        "/readiness",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
)

# --- Config helpers ---------------------------------------------------------


def _load_api_keys() -> List[str]:
    """Parse comma-separated ``API_KEYS`` into a list of non-empty keys."""
    raw = os.environ.get("API_KEYS", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def _load_rate_limit() -> int:
    """Requests-per-minute limit (``RATE_LIMIT_PER_MINUTE``, default 120, 0 disables)."""
    try:
        return int(os.environ.get("RATE_LIMIT_PER_MINUTE", "120"))
    except ValueError:
        return 120


def _load_timeout() -> float:
    """Per-request timeout in seconds (``REQUEST_TIMEOUT_SECONDS``, default 30, 0 disables)."""
    try:
        return float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "30"))
    except ValueError:
        return 30.0


def _is_open_path(path: str) -> bool:
    """True for health/readiness/root and docs paths that must stay public."""
    return path in OPEN_PATHS


# --- API-key authentication -------------------------------------------------

_bearer_scheme = HTTPBearer(auto_error=False)
_api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


def _key_is_valid(candidate: str, valid_keys: List[str]) -> bool:
    """Constant-time membership check against the configured keys."""
    matched = False
    for key in valid_keys:
        if hmac.compare_digest(candidate, key):
            matched = True
    return matched


async def require_api_key(
    request: Request,
    bearer=Depends(_bearer_scheme),
    header_key: Optional[str] = Depends(_api_key_scheme),
) -> None:
    """
    FastAPI dependency enforcing API-key auth when ``API_KEYS`` is set.

    Accepts ``Authorization: Bearer <key>`` or ``X-API-Key: <key>``. Raises
    401 on missing/invalid credentials. A no-op when auth is disabled.
    """
    valid_keys = _load_api_keys()
    if not valid_keys:
        return

    if _is_open_path(request.url.path):
        return

    candidate: Optional[str] = None
    if bearer is not None and bearer.credentials:
        candidate = bearer.credentials
    elif header_key:
        candidate = header_key

    if candidate and _key_is_valid(candidate, valid_keys):
        return

    raise HTTPException(
        status_code=401,
        detail="Missing or invalid API key",
        headers={"WWW-Authenticate": "Bearer"},
    )


# --- Rate limiting ----------------------------------------------------------


class _SlidingWindowRateLimiter:
    """Thread/async-safe in-process sliding-window rate limiter."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self.window = 60.0
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> Optional[int]:
        """
        Register a hit for ``key``.

        Returns ``None`` if allowed, else the Retry-After value in seconds.
        """
        if self.per_minute <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            cutoff = now - self.window
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.per_minute:
                retry_after = int(self.window - (now - hits[0])) + 1
                return max(retry_after, 1)
            hits.append(now)
            return None


def _client_key(request: Request) -> str:
    """Rate-limit bucket key: the API key if present, else client IP."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return "key:" + auth[7:].strip()
    header_key = request.headers.get("x-api-key")
    if header_key:
        return "key:" + header_key.strip()
    client = request.client
    return "ip:" + (client.host if client else "unknown")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforce a per-minute request budget, exempting open paths."""

    def __init__(self, app, per_minute: int) -> None:
        super().__init__(app)
        self._limiter = _SlidingWindowRateLimiter(per_minute)

    async def dispatch(self, request: Request, call_next):
        if self._limiter.per_minute > 0 and not _is_open_path(request.url.path):
            retry_after = self._limiter.check(_client_key(request))
            if retry_after is not None:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"},
                    headers={"Retry-After": str(retry_after)},
                )
        return await call_next(request)


# --- Request timeout --------------------------------------------------------


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Abort handlers that exceed ``REQUEST_TIMEOUT_SECONDS`` with a 504."""

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
                status_code=504,
                content={"detail": "Request timed out"},
            )


# --- Wiring -----------------------------------------------------------------


def configure_hardening(app: FastAPI) -> None:
    """
    Wire auth, rate limiting, and request timeouts onto ``app``.

    Reads configuration from the environment at call time. Auth is applied as
    a global dependency; rate limiting and timeouts are ASGI middleware. All
    three exempt the open (health/docs) paths.
    """
    valid_keys = _load_api_keys()
    if not valid_keys:
        logger.warning("API auth disabled (set API_KEYS to enable)")

    # Apply the auth dependency to every route except the open ones. Routes
    # capture their dependencies at registration time, so we attach to each
    # existing route rather than the router (which runs after registration).
    # The dependency self-exempts when auth is disabled.
    auth_dep = Depends(require_api_key)
    for route in app.router.routes:
        if isinstance(route, APIRoute) and not _is_open_path(route.path):
            route.dependant.dependencies.append(
                get_parameterless_sub_dependant(
                    depends=auth_dep, path=route.path_format
                )
            )

    # Middleware runs outermost-first; timeout wraps rate limiting.
    app.add_middleware(RateLimitMiddleware, per_minute=_load_rate_limit())
    app.add_middleware(TimeoutMiddleware, timeout_seconds=_load_timeout())
