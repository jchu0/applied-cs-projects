"""Production hardening for the FastAPI API: auth, rate limiting, timeouts.

Stdlib only (no slowapi / external deps). All three concerns are opt-in via env
so existing tests and the quick-start keep working unchanged.

Environment variables:
    API_KEYS                comma-separated valid keys; unset/empty -> auth disabled
    RATE_LIMIT_PER_MINUTE   int, default 120; 0 disables rate limiting
    REQUEST_TIMEOUT_SECONDS float, default 30; 0 disables request timeouts
"""

import asyncio
import hmac
import logging
import os
import time
from collections import defaultdict, deque
from typing import Optional

logger = logging.getLogger(__name__)

# Paths that stay open (no auth, no rate limiting) regardless of settings.
OPEN_PATHS = frozenset(
    {"/", "/health", "/docs", "/redoc", "/openapi.json"}
)


def _parse_api_keys() -> list[str]:
    """Return the list of valid API keys from the API_KEYS env var."""
    raw = os.environ.get("API_KEYS", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def _extract_key(authorization: Optional[str], x_api_key: Optional[str]) -> Optional[str]:
    """Pull the presented key from Authorization: Bearer <key> or X-API-Key."""
    if authorization:
        parts = authorization.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
    if x_api_key:
        return x_api_key.strip()
    return None


def _key_is_valid(presented: str, valid_keys: list[str]) -> bool:
    """Constant-time membership check for the presented key."""
    matched = False
    for key in valid_keys:
        # Compare against every key so timing does not leak which matched.
        if hmac.compare_digest(presented, key):
            matched = True
    return matched


def build_auth_dependency():
    """Build a FastAPI dependency enforcing API-key auth (when enabled).

    Reads API_KEYS at request time so tests can toggle it via monkeypatch.
    Applied as a global app dependency; open paths (health/root/docs) are
    exempted by inspecting the request path.
    """
    from fastapi import Header, HTTPException, Request

    async def require_api_key(
        request: Request,
        authorization: Optional[str] = Header(default=None),
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> None:
        if request.url.path in OPEN_PATHS:
            return

        valid_keys = _parse_api_keys()
        if not valid_keys:
            # Auth disabled -> allow everything.
            return

        presented = _extract_key(authorization, x_api_key)
        if not presented or not _key_is_valid(presented, valid_keys):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return require_api_key


class RateLimiter:
    """In-process sliding-window rate limiter (async/thread safe via a lock)."""

    def __init__(self, limit_per_minute: int):
        self.limit = limit_per_minute
        self.window = 60.0
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, identity: str) -> tuple[bool, int]:
        """Register a hit for ``identity``.

        Returns (allowed, retry_after_seconds). retry_after is 0 when allowed.
        """
        if self.limit <= 0:
            return True, 0

        now = time.monotonic()
        async with self._lock:
            hits = self._hits[identity]
            cutoff = now - self.window
            while hits and hits[0] <= cutoff:
                hits.popleft()

            if len(hits) >= self.limit:
                retry_after = int(self.window - (now - hits[0])) + 1
                return False, max(retry_after, 1)

            hits.append(now)
            return True, 0


def _client_identity(request) -> str:
    """Rate-limit key: API key if present, else client IP."""
    key = _extract_key(
        request.headers.get("authorization"),
        request.headers.get("x-api-key"),
    )
    if key:
        return f"key:{key}"
    client = request.client
    return f"ip:{client.host if client else 'unknown'}"


def install_hardening(app) -> None:
    """Wire rate limiting + request timeout middleware onto the app.

    Auth is applied separately as a router/app dependency by the caller.
    Emits the startup warning when auth is disabled.
    """
    from fastapi.responses import JSONResponse

    if not _parse_api_keys():
        logger.warning("API auth disabled (set API_KEYS to enable)")

    rate_limit = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "120"))
    timeout_seconds = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "30"))
    limiter = RateLimiter(rate_limit)

    @app.middleware("http")
    async def rate_limit_middleware(request, call_next):
        if rate_limit > 0 and request.url.path not in OPEN_PATHS:
            allowed, retry_after = await limiter.check(_client_identity(request))
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"},
                    headers={"Retry-After": str(retry_after)},
                )
        return await call_next(request)

    @app.middleware("http")
    async def timeout_middleware(request, call_next):
        if timeout_seconds <= 0:
            return await call_next(request)
        try:
            return await asyncio.wait_for(call_next(request), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={"detail": "Request timed out"},
            )
