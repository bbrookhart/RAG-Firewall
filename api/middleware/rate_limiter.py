"""
api/middleware/rate_limiter.py
──────────────────────────────
In-memory sliding-window rate limiter middleware.
Uses client IP or X-API-Key as the bucket key.
Replace with Redis-backed implementation for multi-process deployments.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Callable

import structlog
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = structlog.get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding window counter per client key."""

    def __init__(self, app: ASGIApp, requests_per_minute: int = 60) -> None:
        super().__init__(app)
        self.limit = requests_per_minute
        self.window = 60  # seconds
        # key → deque of request timestamps
        self._buckets: dict[str, deque[float]] = defaultdict(deque)

    def _client_key(self, request: Request) -> str:
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"apikey:{api_key[:12]}"  # prefix only — don't log full key
        client = request.client
        return f"ip:{client.host if client else 'unknown'}"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip health check
        if request.url.path in {"/health", "/docs", "/openapi.json", "/redoc"}:
            return await call_next(request)

        key = self._client_key(request)
        now = time.monotonic()
        bucket = self._buckets[key]

        # Evict timestamps outside the window
        while bucket and now - bucket[0] > self.window:
            bucket.popleft()

        if len(bucket) >= self.limit:
            logger.warning("rate_limit.exceeded", client_key=key, count=len(bucket))
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after_seconds": int(self.window - (now - bucket[0])),
                },
                headers={"Retry-After": str(int(self.window - (now - bucket[0])))},
            )

        bucket.append(now)

        response = await call_next(request)
        remaining = max(0, self.limit - len(bucket))
        response.headers["X-RateLimit-Limit"] = str(self.limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
