from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from typing import List, Optional
from datetime import datetime
from backend.models.requests import ExperimentUploadRequest
from backend.models.responses import ExperimentResponse, ExperimentListResponse
from backend.agents.orchestrator import OrchestratorAgent
from backend.services.qdrant_service import get_qdrant_service
from backend.services.embedding_service import get_embedding_service
from backend.services.file_service import get_file_service
from backend.security.audit_logger import get_audit_logger
from backend.api.routes.auth import get_current_user
import time
import uuid

router = APIRouter()
orchestrator = OrchestratorAgent()
qdrant = get_qdrant_service()

# Stats cache (10 min TTL)
_stats_cache = {"result": None, "timestamp": 0}
STATS_CACHE_TTL = 600
file_service = get_file_service()
audit_logger = get_audit_logger()
@router.post("/upload", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
async def upload_experiment(
    request: ExperimentUploadRequest,
    current_user: dict = Depends(get_current_user)
):
    """Fast direct upload - bypasses orchestrator for speed."""
    try:
        embedding_service = get_embedding_service()
        conditions = request.conditions.model_dump() if request.conditions else {}

        # Generate embedding directly (skips orchestrator overhead)
        embedding = await embedding_service.generate_multimodal_embedding(
            text=request.text,
            sequence=request.sequence or '',
            conditions=conditions,
            image_base64=request.image_base64
        )

        experiment_id = str(uuid.uuid4())

        # Upsert directly to private_experiments
        await qdrant.upsert(
            collection_name="private_experiments",
            points=[{
                'id': experiment_id,
                'vector': embedding.tolist(),
                'payload': {
                    'text': request.text,
                    'title': request.text[:100] if request.text else '',
                    'sequence': request.sequence,
                    'conditions': conditions,
                    'organism': conditions.get('organism'),
                    'temperature': conditions.get('temperature'),
                    'ph': conditions.get('ph'),
                    'success': request.success,
                    'source': 'user_upload',
                    'notes': request.notes,
                    'image_base64': request.image_base64,
                    'created_at': datetime.utcnow().isoformat(),
                    'user_id': current_user["email"]
                }
            }]
        )

        # Invalidate stats cache after new upload
        _stats_cache["result"] = None

        audit_logger.log_event(
            event_type="experiment",
            user_id=current_user["email"],
            action="upload",
            resource=f"experiment:{experiment_id}",
            success=True,
            details={
                "has_sequence": bool(request.sequence),
                "has_image": bool(request.image_base64),
                "has_conditions": bool(request.conditions)
            }
        )
        return ExperimentResponse(
            experiment_id=experiment_id,
            text=request.text,
            sequence=request.sequence,
            conditions=conditions,
            success=request.success,
            notes=request.notes,
            image_base64=request.image_base64,
            embedding_metadata={
                'text_dim': embedding_service.text_dim,
                'seq_dim': embedding_service.seq_dim,
                'cond_dim': embedding_service.cond_dim,
                'total_dim': len(embedding),
            },
            created_at=datetime.utcnow(),
            message="Experiment successfully uploaded and indexed"
        )
    except Exception as e:
        audit_logger.log_event(
            event_type="experiment",
            user_id=current_user["email"],
            action="upload_failed",
            resource="experiment",
            success=False,
            details={"error": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload experiment: {str(e)}"
        )
@router.post("/upload-file", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
async def upload_experiment_file(
    file: UploadFile = File(...),
    text: str = Form(...),
    sequence: Optional[str] = Form(None),
    success: bool = Form(True),
    notes: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user)
):
    try:
        image_data = await file.read()
        image_base64 = None
        if file.content_type and file.content_type.startswith('image/'):
            import base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        request = ExperimentUploadRequest(
            text=text,
            sequence=sequence,
            image_base64=image_base64,
            success=success,
            notes=notes
        )
        return await upload_experiment(request, current_user)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}"
        )
@router.get("/", response_model=ExperimentListResponse)
async def list_experiments(
    limit: int = 20,
    offset: int = 0,
    success_only: Optional[bool] = None,
    current_user: dict = Depends(get_current_user)
):
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        conditions = []
        if success_only is not None:
            conditions.append(
                FieldCondition(
                    key="success",
                    match=MatchValue(value=success_only)
                )
            )
        query_filter = Filter(must=conditions) if conditions else None
        experiments = await qdrant.scroll(
            collection_name="private_experiments",
            scroll_filter=query_filter,
            limit=limit,
            offset=offset
        )
        results = []
        for exp in experiments:
            payload = exp.get('payload', {})
            results.append({
                "experiment_id": exp['id'],
                "text": payload.get('text', ''),
                "sequence": payload.get('sequence'),
                "conditions": payload.get('conditions'),
                "success": payload.get('success'),
                "notes": payload.get('notes'),
                "created_at": payload.get('created_at')
            })
        return ExperimentListResponse(
            experiments=results,
            total=len(results),
            limit=limit,
            offset=offset
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve experiments: {str(e)}"
        )
@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        point = await qdrant.retrieve(
            collection_name="private_experiments",
            ids=[experiment_id]
        )
        if not point or len(point) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found"
            )
        payload = point[0].get('payload', {})
        return ExperimentResponse(
            experiment_id=experiment_id,
            text=payload.get('text', ''),
            sequence=payload.get('sequence'),
            conditions=payload.get('conditions'),
            success=payload.get('success'),
            notes=payload.get('notes'),
            created_at=payload.get('created_at'),
            message="Experiment retrieved successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve experiment: {str(e)}"
        )
@router.delete("/{experiment_id}")
async def delete_experiment(
    experiment_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        await qdrant.delete(
            collection_name="private_experiments",
            points_selector=[experiment_id]
        )
        audit_logger.log_event(
            event_type="experiment",
            user_id=current_user["email"],
            action="delete",
            resource=f"experiment:{experiment_id}",
            success=True
        )
        return {
            "message": f"Experiment {experiment_id} deleted successfully",
            "experiment_id": experiment_id
        }
    except Exception as e:
        audit_logger.log_event(
            event_type="experiment",
            user_id=current_user["email"],
            action="delete_failed",
            resource=f"experiment:{experiment_id}",
            success=False,
            details={"error": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete experiment: {str(e)}"
        )
@router.get("/stats/summary")
async def get_experiment_stats(current_user: dict = Depends(get_current_user)):
    now = time.time()
    if _stats_cache["result"] and (now - _stats_cache["timestamp"]) < STATS_CACHE_TTL:
        return _stats_cache["result"]

    try:
        # Get real stats from public_science (Qdrant Cloud)
        client = qdrant.cloud_client
        if not client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Qdrant Cloud not available"
            )

        # Total count from collection info
        info = client.get_collection("public_science")
        total = info.points_count or 0

        # Sample points to estimate success rate and organism distribution
        sample_size = 500
        sample_points, _ = client.scroll(
            collection_name="public_science",
            limit=sample_size,
            with_payload=True,
            with_vectors=False
        )

        if not sample_points:
            return {
                "total_experiments": total,
                "success_rate": 0.0,
                "organism_distribution": {},
                "average_conditions": {}
            }

        success_count = 0
        total_with_status = 0
        organisms = {}
        for p in sample_points:
            payload = p.payload or {}
            success_val = payload.get('success')
            if success_val is not None:
                total_with_status += 1
                if success_val is True:
                    success_count += 1
            organism = payload.get('organism', 'unknown')
            if organism:
                organisms[organism] = organisms.get(organism, 0) + 1

        success_rate = success_count / total_with_status if total_with_status > 0 else 0.0

        result = {
            "total_experiments": total,
            "success_rate": round(success_rate, 2),
            "successful_experiments": int(total * success_rate),
            "failed_experiments": int(total * (1 - success_rate)),
            "organism_distribution": organisms
        }
        _stats_cache["result"] = result
        _stats_cache["timestamp"] = now
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute stats: {str(e)}"
        )