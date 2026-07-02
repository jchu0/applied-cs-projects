"""HTTP-layer hardening for the FastAPI app: API-key auth, rate limiting, timeouts.

These are cross-cutting concerns enforced at the ASGI/HTTP boundary, distinct from the
per-tenant `Gateway`/`TenantAuthenticator`/`RateLimiter` (which key off `Tenant.api_key`
and feed the scheduler). This module is stdlib-only, opt-in via environment variables, and
applied globally so the served endpoints enforce auth + limits regardless of the router
internals.
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

# Paths that stay open (no auth, no rate limit, no timeout wrapping).
_EXEMPT_PATHS = frozenset(
    {"/health", "/ready", "/readiness", "/", "/docs", "/redoc", "/openapi.json"}
)


def _load_api_keys() -> list[str]:
    """Parse valid API keys from the ``API_KEYS`` env var (comma-separated)."""
    raw = os.environ.get("API_KEYS", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def _is_exempt(path: str) -> bool:
    """True if the request path is exempt from auth/limits/timeout."""
    return path in _EXEMPT_PATHS


# ---------------------------------------------------------------------------
# API-key auth
# ---------------------------------------------------------------------------

def _extract_key(authorization: str | None, x_api_key: str | None) -> str | None:
    """Pull a key from an ``Authorization: Bearer <key>`` or ``X-API-Key`` header."""
    if authorization:
        parts = authorization.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
    if x_api_key:
        return x_api_key.strip()
    return None


def _key_is_valid(candidate: str, valid_keys: list[str]) -> bool:
    """Constant-time membership check against the configured keys."""
    matched = False
    for key in valid_keys:
        # compare_digest over every key so timing does not leak which matched.
        if hmac.compare_digest(candidate, key):
            matched = True
    return matched


async def require_api_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """FastAPI dependency enforcing API-key auth when enabled.

    Auth is disabled (dependency is a no-op) when ``API_KEYS`` is unset/empty. When
    enabled, requires a valid key via ``Authorization: Bearer <key>`` or ``X-API-Key``,
    returning 401 with a ``WWW-Authenticate`` header otherwise. Uses
    ``hmac.compare_digest`` for constant-time comparison.
    """
    valid_keys = _load_api_keys()
    if not valid_keys:
        return  # auth disabled

    candidate = _extract_key(authorization, x_api_key)
    if not candidate or not _key_is_valid(candidate, valid_keys):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Rate limiting (in-process sliding window, stdlib only)
# ---------------------------------------------------------------------------

class SlidingWindowRateLimiter:
    """Thread/async-safe fixed-per-minute sliding-window limiter.

    Keyed by API key when present, else client IP. Not related to the per-tenant
    token-bucket ``RateLimiter`` in ``gateway/``.
    """

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._window = 60.0
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        """Record a hit for ``key``.

        Returns ``(allowed, retry_after_seconds)``. ``retry_after`` is 0 when allowed.
        """
        if self.per_minute <= 0:
            return True, 0

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


def _client_key(request: Request) -> str:
    """Rate-limit key: the presented API key if any, else the client IP."""
    auth = request.headers.get("authorization")
    x_api_key = request.headers.get("x-api-key")
    key = _extract_key(auth, x_api_key)
    if key:
        return f"key:{key}"
    client = request.client
    return f"ip:{client.host if client else 'unknown'}"


# ---------------------------------------------------------------------------
# Middleware installation
# ---------------------------------------------------------------------------

def install_middleware(app) -> None:
    """Install rate-limit and request-timeout middleware on ``app``.

    Reads ``RATE_LIMIT_PER_MINUTE`` (int, default 120; 0 disables) and
    ``REQUEST_TIMEOUT_SECONDS`` (float, default 30; 0 disables). Both exempt the paths in
    ``_EXEMPT_PATHS``. There are no streaming/SSE/websocket endpoints, so nothing else is
    exempted from the timeout.
    """
    per_minute = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "120"))
    timeout_seconds = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "30"))

    limiter = SlidingWindowRateLimiter(per_minute)

    @app.middleware("http")
    async def _rate_limit_and_timeout(request: Request, call_next):
        if _is_exempt(request.url.path):
            return await call_next(request)

        # Rate limit
        if per_minute > 0:
            allowed, retry_after = limiter.check(_client_key(request))
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"},
                    headers={"Retry-After": str(retry_after)},
                )

        # Request timeout
        if timeout_seconds > 0:
            try:
                return await asyncio.wait_for(
                    call_next(request), timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                return JSONResponse(
                    status_code=504,
                    content={"detail": "Request timed out"},
                )
        return await call_next(request)


def log_auth_status() -> None:
    """Emit the one-time startup warning when auth is disabled."""
    if not _load_api_keys():
        logger.warning("API auth disabled (set API_KEYS to enable)")
