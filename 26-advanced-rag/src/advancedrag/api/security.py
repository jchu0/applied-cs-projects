"""Production hardening: API-key auth, rate limiting, and request timeouts.

All three concerns are opt-in via environment variables and use only the
standard library (no extra dependencies). They are wired into the FastAPI
app in :mod:`advancedrag.api.main`.

Environment variables
----------------------
- ``API_KEYS``               Comma-separated list of valid keys. Unset/empty
                             disables auth entirely (default: disabled).
- ``RATE_LIMIT_PER_MINUTE``  Requests/minute per caller (default 120; 0 disables).
- ``REQUEST_TIMEOUT_SECONDS``Per-request timeout in seconds (default 30; 0 disables).
"""

import asyncio
import hmac
import logging
import os
import time
from collections import defaultdict, deque

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Paths that stay open regardless of auth / rate limiting.
OPEN_PATHS = frozenset({"/", "/health", "/ready", "/docs", "/redoc", "/openapi.json"})


# =============================================================================
# API-key authentication
# =============================================================================

def load_api_keys() -> list[str]:
    """Read valid API keys from the ``API_KEYS`` env var (comma-separated)."""
    raw = os.environ.get("API_KEYS", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def auth_enabled() -> bool:
    """Whether API-key auth is enabled (i.e. at least one key configured)."""
    return len(load_api_keys()) > 0


_bearer_scheme = HTTPBearer(auto_error=False)
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _key_is_valid(candidate: str, valid_keys: list[str]) -> bool:
    """Constant-time membership check for an API key."""
    return any(hmac.compare_digest(candidate, key) for key in valid_keys)


async def require_api_key(
    request: Request,
    bearer: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    api_key: str | None = Depends(_api_key_header),
) -> None:
    """FastAPI dependency enforcing API-key auth when enabled.

    Accepts ``Authorization: Bearer <key>`` or ``X-API-Key: <key>``. When auth
    is disabled (no ``API_KEYS`` configured) this is a no-op so tests and the
    quick-start keep working. Open paths (health/readiness/root/docs) are
    always exempt.
    """
    if request.url.path in OPEN_PATHS:
        return

    valid_keys = load_api_keys()
    if not valid_keys:
        return  # auth disabled

    candidate = None
    if bearer is not None and bearer.credentials:
        candidate = bearer.credentials
    elif api_key:
        candidate = api_key

    if not candidate or not _key_is_valid(candidate, valid_keys):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


# =============================================================================
# Rate limiting (in-process sliding window)
# =============================================================================

def rate_limit_per_minute() -> int:
    """Read the per-minute rate limit from env (default 120; 0 disables)."""
    try:
        return int(os.environ.get("RATE_LIMIT_PER_MINUTE", "120"))
    except ValueError:
        return 120


class SlidingWindowRateLimiter:
    """Thread/async-safe in-process sliding-window rate limiter."""

    def __init__(self, limit_per_minute: int):
        self.limit = limit_per_minute
        self.window = 60.0
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, key: str) -> tuple[bool, int]:
        """Register a hit for ``key``.

        Returns ``(allowed, retry_after_seconds)``. ``retry_after`` is 0 when
        allowed.
        """
        if self.limit <= 0:
            return True, 0

        now = time.monotonic()
        cutoff = now - self.window
        async with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= cutoff:
                hits.popleft()

            if len(hits) >= self.limit:
                retry_after = max(1, int(self.window - (now - hits[0])) + 1)
                return False, retry_after

            hits.append(now)
            return True, 0


def _client_key(request: Request) -> str:
    """Rate-limit key: API key if present, else client IP."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return f"key:{auth[7:].strip()}"
    api_key = request.headers.get("x-api-key")
    if api_key:
        return f"key:{api_key.strip()}"
    client = request.client
    return f"ip:{client.host if client else 'unknown'}"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """ASGI middleware enforcing the sliding-window rate limit."""

    def __init__(self, app, limiter: SlidingWindowRateLimiter):
        super().__init__(app)
        self.limiter = limiter

    async def dispatch(self, request: Request, call_next):
        if request.url.path in OPEN_PATHS:
            return await call_next(request)

        allowed, retry_after = await self.limiter.check(_client_key(request))
        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(retry_after)},
            )
        return await call_next(request)


# =============================================================================
# Request timeout
# =============================================================================

def request_timeout_seconds() -> float:
    """Read the request timeout from env (default 30.0; 0 disables)."""
    try:
        return float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "30"))
    except ValueError:
        return 30.0


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Wrap each request handler in ``asyncio.wait_for``.

    On timeout returns a 504 JSON response. Open paths (health/docs/root) are
    exempt. There are currently no streaming/SSE/websocket endpoints in the
    app; if any are added they must be exempted here too.
    """

    def __init__(self, app, timeout: float):
        super().__init__(app)
        self.timeout = timeout

    async def dispatch(self, request: Request, call_next):
        if self.timeout <= 0 or request.url.path in OPEN_PATHS:
            return await call_next(request)

        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout)
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content={"detail": "Request timed out"},
            )


# =============================================================================
# Wiring helper
# =============================================================================

def install_hardening(app: FastAPI) -> None:
    """Wire auth, rate limiting, and timeouts into ``app``.

    Auth is applied as a global app-level dependency (``require_api_key``);
    open paths (health/readiness/root/docs) are exempted inside the dependency
    itself. Rate limiting and timeouts are ASGI middleware.
    """
    if not auth_enabled():
        logger.warning("API auth disabled (set API_KEYS to enable)")

    limiter = SlidingWindowRateLimiter(rate_limit_per_minute())
    # Middleware runs outermost-first; timeout wraps the handler innermost.
    app.add_middleware(TimeoutMiddleware, timeout=request_timeout_seconds())
    app.add_middleware(RateLimitMiddleware, limiter=limiter)
