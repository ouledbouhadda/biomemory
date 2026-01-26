from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Tuple
class RateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = defaultdict(list)
    def is_allowed(self, client_id: str) -> Tuple[bool, Dict[str, int]]:
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.window_seconds)
        self.requests[client_id] = [
            (ts, count) for ts, count in self.requests[client_id]
            if ts > window_start
        ]
        total_requests = sum(count for _, count in self.requests[client_id])
        rate_limit_info = {
            "limit": self.max_requests,
            "remaining": max(0, self.max_requests - total_requests),
            "reset": int((window_start + timedelta(seconds=self.window_seconds)).timestamp())
        }
        if total_requests >= self.max_requests:
            return False, rate_limit_info
        self.requests[client_id].append((now, 1))
        rate_limit_info["remaining"] -= 1
        return True, rate_limit_info
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate_limiter: RateLimiter):
        super().__init__(app)
        self.rate_limiter = rate_limiter
    async def dispatch(self, request: Request, call_next):
        exempt_paths = ["/health", "/api/docs", "/api/redoc", "/openapi.json"]
        if any(request.url.path.startswith(path) for path in exempt_paths):
            return await call_next(request)
        client_id = self._get_client_id(request)
        allowed, rate_info = self.rate_limiter.is_allowed(client_id)
        response = None
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
                headers={
                    "X-RateLimit-Limit": str(rate_info["limit"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(rate_info["reset"]),
                    "Retry-After": str(self.rate_limiter.window_seconds)
                }
            )
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(rate_info["reset"])
        return response
    def _get_client_id(self, request: Request) -> str:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                from jose import jwt
                from backend.config.settings import get_settings
                settings = get_settings()
                token = auth_header.split(" ")[1]
                payload = jwt.decode(
                    token,
                    settings.SECRET_KEY,
                    algorithms=[settings.ALGORITHM]
                )
                user_id = payload.get("sub")
                if user_id:
                    return f"user:{user_id}"
            except:
                pass
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0]}"
        return f"ip:{request.client.host}"