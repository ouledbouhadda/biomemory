from fastapi import APIRouter
from datetime import datetime
from backend.models.responses import HealthResponse
from backend.config.settings import get_settings
from backend.services.qdrant_service import get_qdrant_service
router = APIRouter()
settings = get_settings()
@router.get("/health", response_model=HealthResponse)
async def health_check():
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
    return HealthResponse(
        status=overall_status,
        version=settings.APP_VERSION,
        qdrant_cloud=qdrant_cloud,
        qdrant_private=qdrant_private,
        timestamp=datetime.utcnow()
    )
@router.get("/metrics")
async def get_metrics():
    return {
        "message": "Metrics endpoint - coming soon",
        "format": "prometheus"
    }