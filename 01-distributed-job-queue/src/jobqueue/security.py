"""Production hardening: API-key auth, rate limiting, and request timeouts.

This module wires three cross-cutting concerns into the FastAPI app using only
the standard library (no external dependencies). All three are opt-in / tunable
via environment variables so the existing tests and the quick-start keep working
out of the box:

* ``API_KEYS``               -- comma-separated valid keys; unset/empty disables auth.
* ``RATE_LIMIT_PER_MINUTE``  -- int, default 120; ``0`` disables rate limiting.
* ``REQUEST_TIMEOUT_SECONDS``-- float, default 30; ``0`` disables the timeout.
"""

from __future__ import annotations

import asyncio
import hmac
import os
import threading
import time
from collections import defaultdict, deque

import structlog
from fastapi import HTTPException, Request
from starlette.responses import JSONResponse

logger = structlog.get_logger()

# Paths that are always reachable without auth / rate limiting.
_OPEN_PATHS = frozenset(
    {
        "/",
        "/health",
        "/ready",
        "/readiness",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
)


def _is_open_path(path: str) -> bool:
    """Return True for health/readiness/root and docs paths (always open)."""
    if path in _OPEN_PATHS:
        return True
    # Swagger UI serves supporting assets under /docs/... and /redoc/...
    return path.startswith("/docs/") or path.startswith("/redoc/")


def load_api_keys() -> set[str]:
    """Load valid API keys from the ``API_KEYS`` env var (comma-separated)."""
    raw = os.environ.get("API_KEYS", "")
    return {key.strip() for key in raw.split(",") if key.strip()}


def _extract_key(request: Request) -> str | None:
    """Pull an API key from ``Authorization: Bearer`` or ``X-API-Key``."""
    auth = request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[len("bearer ") :].strip()
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return api_key.strip()
    return None


def _key_is_valid(candidate: str, valid_keys: set[str]) -> bool:
    """Constant-time comparison of ``candidate`` against all valid keys."""
    matched = False
    for key in valid_keys:
        # hmac.compare_digest short-circuits on length but not on content; run
        # against every key so timing does not reveal which key matched.
        if hmac.compare_digest(candidate, key):
            matched = True
    return matched


def make_auth_dependency(valid_keys: set[str]):
    """Build a FastAPI dependency enforcing API-key auth for protected routes.

    Provided for callers that prefer a route/router dependency. The app itself
    enforces auth in the hardening middleware (see :func:`install_hardening`)
    so it reliably covers every route regardless of registration order.
    """

    async def require_api_key(request: Request) -> None:
        if _is_open_path(request.url.path):
            return
        candidate = _extract_key(request)
        if candidate is None or not _key_is_valid(candidate, valid_keys):
            raise HTTPException(
                status_code=401,
                detail="Missing or invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return require_api_key


class SlidingWindowRateLimiter:
    """In-process sliding-window rate limiter (thread- and async-safe)."""

    def __init__(self, limit_per_minute: int, window_seconds: float = 60.0):
        self.limit = limit_per_minute
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, identity: str) -> tuple[bool, int]:
        """Record a hit for ``identity``.

        Returns ``(allowed, retry_after_seconds)``. ``retry_after_seconds`` is 0
        when the request is allowed.
        """
        now = time.monotonic()
        cutoff = now - self.window
        with self._lock:
            hits = self._hits[identity]
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.limit:
                retry_after = max(1, int(self.window - (now - hits[0])) + 1)
                return False, retry_after
            hits.append(now)
            return True, 0


def _client_identity(request: Request) -> str:
    """Rate-limit key: the API key if present, else the client IP."""
    key = _extract_key(request)
    if key:
        return f"key:{key}"
    client = request.client
    return f"ip:{client.host if client else 'unknown'}"


def install_hardening(app) -> None:
    """Wire auth, rate limiting, and request timeouts into ``app``.

    Reads configuration from the environment at call time so tests can toggle
    behaviour with monkeypatch / os.environ.
    """
    valid_keys = load_api_keys()
    if valid_keys:
        logger.info("API auth enabled", key_count=len(valid_keys))
    else:
        logger.warning("API auth disabled (set API_KEYS to enable)")

    # --- Rate limiting -----------------------------------------------------
    try:
        rate_limit = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "120"))
    except ValueError:
        rate_limit = 120
    limiter = SlidingWindowRateLimiter(rate_limit) if rate_limit > 0 else None

    # --- Request timeout ---------------------------------------------------
    try:
        timeout_seconds = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "30"))
    except ValueError:
        timeout_seconds = 30.0

    @app.middleware("http")
    async def hardening_middleware(request: Request, call_next):
        path = request.url.path
        exempt = _is_open_path(path)

        # API-key auth (exempt health/readiness/root and docs).
        if valid_keys and not exempt:
            candidate = _extract_key(request)
            if candidate is None or not _key_is_valid(candidate, valid_keys):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Missing or invalid API key"},
                    headers={"WWW-Authenticate": "Bearer"},
                )

        # Rate limiting (exempt health/docs).
        if limiter is not None and not exempt:
            allowed, retry_after = limiter.check(_client_identity(request))
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"},
                    headers={"Retry-After": str(retry_after)},
                )

        # Request timeout. Streaming/SSE/websocket endpoints are exempt; this
        # app exposes none, so the timeout applies to every non-open route.
        if timeout_seconds > 0 and not exempt:
            try:
                return await asyncio.wait_for(call_next(request), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                return JSONResponse(
                    status_code=504,
                    content={"detail": "Request timed out"},
                )

        return await call_next(request)
