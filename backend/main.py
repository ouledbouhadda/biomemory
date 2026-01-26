from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
from backend.config.settings import get_settings
from backend.api.routes import auth, experiments, search, design, health
from backend.security.rate_limiting import RateLimiter, RateLimitMiddleware
from backend.security.audit_logger import AuditLogger, AuditMiddleware
from backend.services.qdrant_service import get_qdrant_service
from backend.database.user_repository import UserRepository
settings = get_settings()
_user_repository = None
def get_user_repository() -> UserRepository:
    return _user_repository
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _user_repository
    print(" Starting BioMemory API...")
    print(f"   Version: {settings.APP_VERSION}")
    print(f"   Debug: {settings.DEBUG}")
    try:
        qdrant = get_qdrant_service()
        await qdrant.init_collections()
        print("    Qdrant collections initialized")
        _user_repository = UserRepository(qdrant.private_client)
        print("   User repository initialized")
    except Exception as e:
        print(f"    Qdrant initialization warning: {e}")
    print(" BioMemory API ready!")
    print(f"   API Docs: http://localhost:8000/api/docs")
    yield
    print("Shutting down BioMemory API...")
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Multimodal Biological Design & Discovery Intelligence System",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-RateLimit-Limit", "X-RateLimit-Remaining"]
)
if not settings.DEBUG:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "*.biomemory.dev"]
    )
rate_limiter = RateLimiter(max_requests=settings.RATE_LIMIT_PER_MINUTE)
app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)
audit_logger = AuditLogger()
app.add_middleware(AuditMiddleware, audit_logger=audit_logger)
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    return response
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={
            "detail": str(exc),
            "error_type": "ValueError"
        }
    )
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    if settings.DEBUG:
        raise exc
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error_type": type(exc).__name__
        }
    )
app.include_router(
    health.router,
    prefix=settings.API_V1_PREFIX,
    tags=["Health"]
)
app.include_router(
    auth.router,
    prefix=f"{settings.API_V1_PREFIX}/auth",
    tags=["Authentication"]
)
app.include_router(
    experiments.router,
    prefix=f"{settings.API_V1_PREFIX}/experiments",
    tags=["Experiments"]
)
app.include_router(
    search.router,
    prefix=f"{settings.API_V1_PREFIX}/search",
    tags=["Search"]
)
app.include_router(
    design.router,
    prefix=f"{settings.API_V1_PREFIX}/design",
    tags=["Design"]
)
@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/api/docs",
        "health": "/health"
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )