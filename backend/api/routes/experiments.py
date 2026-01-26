from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from typing import List, Optional
from datetime import datetime
from backend.models.requests import ExperimentUploadRequest
from backend.models.responses import ExperimentResponse, ExperimentListResponse
from backend.agents.orchestrator import OrchestratorAgent
from backend.services.qdrant_service import get_qdrant_service
from backend.services.file_service import get_file_service
from backend.security.audit_logger import get_audit_logger
from backend.api.routes.auth import get_current_user
router = APIRouter()
orchestrator = OrchestratorAgent()
qdrant = get_qdrant_service()
file_service = get_file_service()
audit_logger = get_audit_logger()
@router.post("/upload", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
async def upload_experiment(
    request: ExperimentUploadRequest,
    current_user: dict = Depends(get_current_user)
):
    try:
        user_input = {
            "intent": "upload_experiment",
            "experiment": {
                "text": request.text,
                "sequence": request.sequence,
                "conditions": request.conditions.model_dump() if request.conditions else None,
                "image_base64": request.image_base64,
                "success": request.success,
                "notes": request.notes,
                "source": "user_upload"
            },
            "user_id": current_user["email"],
            "timestamp": datetime.utcnow().isoformat()
        }
        result = await orchestrator.process_request(user_input)
        audit_logger.log_event(
            event_type="experiment",
            user_id=current_user["email"],
            action="upload",
            resource=f"experiment:{result.get('experiment_id')}",
            success=True,
            details={
                "has_sequence": bool(request.sequence),
                "has_image": bool(request.image_base64),
                "has_conditions": bool(request.conditions)
            }
        )
        return ExperimentResponse(
            experiment_id=result["experiment_id"],
            text=request.text,
            sequence=request.sequence,
            conditions=request.conditions,
            success=request.success,
            notes=request.notes,
            embedding_metadata=result.get("embedding_metadata"),
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
    try:
        experiments = await qdrant.scroll(
            collection_name="private_experiments",
            limit=1000
        )
        total = len(experiments)
        if total == 0:
            return {
                "total_experiments": 0,
                "success_rate": 0.0,
                "organism_distribution": {},
                "average_conditions": {}
            }
        success_count = sum(1 for exp in experiments if exp.get('payload', {}).get('success'))
        success_rate = success_count / total if total > 0 else 0.0
        organisms = {}
        for exp in experiments:
            conditions = exp.get('payload', {}).get('conditions', {})
            organism = conditions.get('organism', 'unknown')
            organisms[organism] = organisms.get(organism, 0) + 1
        return {
            "total_experiments": total,
            "success_rate": round(success_rate, 2),
            "successful_experiments": success_count,
            "failed_experiments": total - success_count,
            "organism_distribution": organisms
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute stats: {str(e)}"
        )