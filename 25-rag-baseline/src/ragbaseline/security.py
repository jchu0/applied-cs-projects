"""Production hardening: API-key auth, rate limiting, and request timeouts.

All three concerns are stdlib-only (no slowapi / external limiter deps) and are
opt-in via environment variables so the existing tests and the keyless quick-start
keep working unchanged:

- ``API_KEYS``               comma-separated valid keys; unset/empty -> auth disabled.
- ``RATE_LIMIT_PER_MINUTE``  int, default 120; 0 disables rate limiting.
- ``REQUEST_TIMEOUT_SECONDS`` float, default 30; 0 disables the timeout.

These layer *in front of* the existing tenant isolation (the caller-supplied
``tenant_id``), which is left untouched.
"""

import asyncio
import hmac
import logging
import os
import time
from collections import defaultdict, deque
from typing import Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger("ragbaseline.security")

# Paths that stay open regardless of auth / rate limiting.
OPEN_PATHS = frozenset({
    "/",
    "/health",
    "/ready",
    "/readiness",
    "/docs",
    "/redoc",
    "/openapi.json",
})


def _is_open_path(path: str) -> bool:
    """Return True if a path is exempt from auth and rate limiting."""
    if path in OPEN_PATHS:
        return True
    # Swagger UI static assets live under /docs.
    return path.startswith("/docs") or path.startswith("/redoc")


# =============================================================================
# API-key auth
# =============================================================================

def load_api_keys() -> list[str]:
    """Parse ``API_KEYS`` (comma-separated) into a list of non-empty keys."""
    raw = os.environ.get("API_KEYS", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def auth_enabled() -> bool:
    """Auth is enabled only when at least one key is configured."""
    return bool(load_api_keys())


_bearer_scheme = HTTPBearer(auto_error=False)
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _key_is_valid(candidate: str, valid_keys: list[str]) -> bool:
    """Constant-time comparison of ``candidate`` against every valid key."""
    matched = False
    for key in valid_keys:
        if hmac.compare_digest(candidate, key):
            matched = True
    return matched


async def require_api_key(
    request: Request,
    bearer: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
    x_api_key: Optional[str] = Security(_api_key_header),
) -> None:
    """FastAPI dependency enforcing API-key auth when enabled.

    Accepts ``Authorization: Bearer <key>`` or ``X-API-Key: <key>``. Returns 401
    with a ``WWW-Authenticate`` header on a missing or invalid key. When auth is
    disabled (no ``API_KEYS``), or for open paths (health/readiness/root/docs),
    this is a no-op. Applied as an app-wide dependency.
    """
    if _is_open_path(request.url.path):
        return

    valid_keys = load_api_keys()
    if not valid_keys:
        return

    candidate = None
    if bearer is not None and bearer.credentials:
        candidate = bearer.credentials
    elif x_api_key:
        candidate = x_api_key

    if not candidate or not _key_is_valid(candidate, valid_keys):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


# =============================================================================
# Rate limiting (in-process sliding window, stdlib only)
# =============================================================================

DEFAULT_RATE_LIMIT_PER_MINUTE = 120


def rate_limit_per_minute() -> int:
    """Read ``RATE_LIMIT_PER_MINUTE`` (int, default 120; 0 disables)."""
    raw = os.environ.get("RATE_LIMIT_PER_MINUTE")
    if raw is None or raw.strip() == "":
        return DEFAULT_RATE_LIMIT_PER_MINUTE
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_RATE_LIMIT_PER_MINUTE


class SlidingWindowRateLimiter:
    """Async/thread-safe in-process sliding-window rate limiter.

    Tracks request timestamps per key over a 60s window. Keyed by API key when
    present, otherwise by client IP.
    """

    WINDOW_SECONDS = 60.0

    def __init__(self, limit_per_minute: int) -> None:
        self.limit = limit_per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, key: str) -> tuple[bool, int]:
        """Record a hit for ``key``.

        Returns ``(allowed, retry_after_seconds)``. ``retry_after_seconds`` is 0
        when allowed.
        """
        if self.limit <= 0:
            return True, 0

        now = time.monotonic()
        cutoff = now - self.WINDOW_SECONDS
        async with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= cutoff:
                hits.popleft()

            if len(hits) >= self.limit:
                retry_after = max(1, int(self.WINDOW_SECONDS - (now - hits[0])) + 1)
                return False, retry_after

            hits.append(now)
            return True, 0


def _client_key(request: Request) -> str:
    """Derive the rate-limit key: API key if present, else client IP."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        if token:
            return f"key:{token}"
    x_api_key = request.headers.get("x-api-key", "")
    if x_api_key.strip():
        return f"key:{x_api_key.strip()}"
    client = request.client
    return f"ip:{client.host if client else 'unknown'}"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforce the sliding-window rate limit, exempting open paths."""

    def __init__(self, app, limiter: SlidingWindowRateLimiter) -> None:
        super().__init__(app)
        self.limiter = limiter

    async def dispatch(self, request: Request, call_next):
        if self.limiter.limit <= 0 or _is_open_path(request.url.path):
            return await call_next(request)

        allowed, retry_after = await self.limiter.check(_client_key(request))
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(retry_after)},
            )
        return await call_next(request)


# =============================================================================
# Request timeout
# =============================================================================

DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0

# Endpoints exempt from the timeout because they stream responses (SSE); wrapping
# a streaming handler in asyncio.wait_for would cut the stream short.
STREAMING_PATHS = frozenset({"/query/stream"})


def request_timeout_seconds() -> float:
    """Read ``REQUEST_TIMEOUT_SECONDS`` (float, default 30; 0 disables)."""
    raw = os.environ.get("REQUEST_TIMEOUT_SECONDS")
    if raw is None or raw.strip() == "":
        return DEFAULT_REQUEST_TIMEOUT_SECONDS
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_REQUEST_TIMEOUT_SECONDS


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Wrap request handling in ``asyncio.wait_for``; 504 JSON on timeout.

    Streaming endpoints (see ``STREAMING_PATHS``) are exempt, since a streaming
    response completes its handler quickly and then yields over time.
    """

    def __init__(self, app, timeout_seconds: float) -> None:
        super().__init__(app)
        self.timeout_seconds = timeout_seconds

    async def dispatch(self, request: Request, call_next):
        if self.timeout_seconds <= 0 or request.url.path in STREAMING_PATHS:
            return await call_next(request)

        try:
            return await asyncio.wait_for(
                call_next(request), timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={"detail": "Request timed out"},
            )
