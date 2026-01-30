from fastapi import APIRouter
from datetime import datetime
import time
from backend.models.responses import HealthResponse
from backend.config.settings import get_settings
from backend.services.qdrant_service import get_qdrant_service
router = APIRouter()
settings = get_settings()

_health_cache = {"result": None, "timestamp": 0}
HEALTH_CACHE_TTL = 30  # seconds

@router.get("/health", response_model=HealthResponse)
async def health_check():
    now = time.time()
    if _health_cache["result"] and (now - _health_cache["timestamp"]) < HEALTH_CACHE_TTL:
        return _health_cache["result"]

    qdrant = get_qdrant_service()
    try:
        cloud_status = await qdrant.health_check("cloud")
        qdrant_cloud = "healthy" if cloud_status else "unavailable"
    except:
        qdrant_cloud = "unavailable"
    try:
        private_status = await qdrant.health_check("private")
        qdrant_private = "healthy" if private_status else "unhealthy"
    except:
        qdrant_private = "unavailable"
    if qdrant_private == "healthy":
        overall_status = "healthy"
    elif qdrant_cloud == "healthy":
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"
    result = HealthResponse(
        status=overall_status,
        version=settings.APP_VERSION,
        qdrant_cloud=qdrant_cloud,
        qdrant_private=qdrant_private,
        timestamp=datetime.utcnow()
    )
    _health_cache["result"] = result
    _health_cache["timestamp"] = now
    return result
@router.get("/metrics")
async def get_metrics():
    return {
        "message": "Metrics endpoint - coming soon",
        "format": "prometheus"
    }