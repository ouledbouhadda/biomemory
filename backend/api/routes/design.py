from fastapi import APIRouter, Depends, HTTPException, status
from typing import Optional, List
from backend.models.requests import DesignRequest
from backend.models.responses import DesignResponse, DesignVariantResponse
from backend.agents.orchestrator import OrchestratorAgent
from backend.security.audit_logger import get_audit_logger
from backend.security.rate_limiting import RateLimiter
from backend.api.routes.auth import get_current_user
router = APIRouter()
orchestrator = OrchestratorAgent()
audit_logger = get_audit_logger()
design_rate_limiter = RateLimiter(max_requests=10, window_seconds=60)
@router.post("/variants", response_model=DesignResponse)
async def generate_variants(
    request: DesignRequest,
    current_user: dict = Depends(get_current_user)
):
    allowed, info = design_rate_limiter.is_allowed(current_user["email"])
    if not allowed:
        audit_logger.log_event(
            event_type="design",
            user_id=current_user["email"],
            action="design_rate_limited",
            resource="design",
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Design rate limit exceeded. Try again in {info['reset_in']} seconds"
        )
    try:
        user_input = {
            "intent": "generate_design_variants",
            "base_experiment": {
                "text": request.text,
                "sequence": request.sequence,
                "conditions": request.conditions.model_dump() if request.conditions else None,
                "goal": request.goal or "optimize_protocol"
            },
            "num_variants": request.num_variants,
            "user_id": current_user["email"]
        }
        result = await orchestrator.process_request(user_input)
        variants = result.get('design_variants', [])
        reproducibility_risk = result.get('reproducibility_risk', 0.0)
        evidence_map = result.get('evidence_map', {})
        formatted_variants = []
        for variant in variants:
            variant_id = variant.get('id')
            evidence = evidence_map.get(variant_id, {})
            formatted_variants.append(DesignVariantResponse(
                variant_id=variant_id,
                text=variant.get('text', ''),
                sequence=variant.get('sequence'),
                conditions=variant.get('conditions'),
                modifications=variant.get('modifications', []),
                justification=variant.get('justification', ''),
                confidence=variant.get('confidence', 0.0),
                supporting_evidence=evidence.get('supporting_experiments', []),
                risk_factors=variant.get('risk_factors', [])
            ))
        audit_logger.log_event(
            event_type="design",
            user_id=current_user["email"],
            action="generate_variants",
            resource="design",
            success=True,
            details={
                "num_variants_requested": request.num_variants,
                "num_variants_generated": len(formatted_variants),
                "goal": request.goal
            }
        )
        return DesignResponse(
            variants=formatted_variants,
            reproducibility_risk=reproducibility_risk,
            base_experiment={
                "text": request.text,
                "sequence": request.sequence,
                "conditions": request.conditions.model_dump() if request.conditions else None
            },
            generation_metadata={
                "method": result.get('generation_method', 'gemini'),
                "model": result.get('model_used', 'gemini-pro'),
                "context_experiments": result.get('context_size', 0)
            }
        )
    except Exception as e:
        audit_logger.log_event(
            event_type="design",
            user_id=current_user["email"],
            action="generate_variants_failed",
            resource="design",
            success=False,
            details={"error": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Variant generation failed: {str(e)}"
        )
@router.post("/optimize")
async def optimize_protocol(
    experiment_id: str,
    goal: str = "increase_yield",
    current_user: dict = Depends(get_current_user)
):
    allowed, info = design_rate_limiter.is_allowed(current_user["email"])
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Design rate limit exceeded. Try again in {info['reset_in']} seconds"
        )
    try:
        from backend.services.qdrant_service import get_qdrant_service
        qdrant = get_qdrant_service()
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
        design_request = DesignRequest(
            text=payload.get('text', ''),
            sequence=payload.get('sequence'),
            conditions=payload.get('conditions'),
            num_variants=3,
            goal=goal
        )
        result = await generate_variants(design_request, current_user)
        result_dict = result.model_dump()
        result_dict['optimization_goal'] = goal
        result_dict['base_experiment_id'] = experiment_id
        audit_logger.log_event(
            event_type="design",
            user_id=current_user["email"],
            action="optimize_protocol",
            resource=f"experiment:{experiment_id}",
            success=True,
            details={"goal": goal}
        )
        return result_dict
    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_event(
            event_type="design",
            user_id=current_user["email"],
            action="optimize_protocol_failed",
            resource=f"experiment:{experiment_id}",
            success=False,
            details={"error": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Protocol optimization failed: {str(e)}"
        )
@router.post("/troubleshoot")
async def troubleshoot_failure(
    failed_experiment_id: str,
    current_user: dict = Depends(get_current_user)
):
    allowed, info = design_rate_limiter.is_allowed(current_user["email"])
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Design rate limit exceeded. Try again in {info['reset_in']} seconds"
        )
    try:
        from backend.services.qdrant_service import get_qdrant_service
        qdrant = get_qdrant_service()
        point = await qdrant.retrieve(
            collection_name="private_experiments",
            ids=[failed_experiment_id]
        )
        if not point or len(point) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {failed_experiment_id} not found"
            )
        payload = point[0].get('payload', {})
        if payload.get('success', True):
            return {
                "message": "This experiment was marked as successful. Troubleshooting is for failed experiments.",
                "experiment_id": failed_experiment_id,
                "success": True
            }
        user_input = {
            "intent": "troubleshoot_failure",
            "failed_experiment": {
                "id": failed_experiment_id,
                "text": payload.get('text', ''),
                "sequence": payload.get('sequence'),
                "conditions": payload.get('conditions'),
                "notes": payload.get('notes')
            },
            "user_id": current_user["email"]
        }
        result = await orchestrator.process_request(user_input)
        troubleshooting_report = {
            "failed_experiment_id": failed_experiment_id,
            "probable_causes": result.get('probable_causes', []),
            "suggested_fixes": result.get('design_variants', []),
            "successful_references": result.get('reranked_neighbors', [])[:5],
            "key_differences": result.get('key_differences', []),
            "reproducibility_analysis": {
                "risk": result.get('reproducibility_risk', 0.0),
                "similar_failures": result.get('similar_failures', 0),
                "similar_successes": result.get('similar_successes', 0)
            }
        }
        audit_logger.log_event(
            event_type="design",
            user_id=current_user["email"],
            action="troubleshoot_failure",
            resource=f"experiment:{failed_experiment_id}",
            success=True,
            details={
                "causes_identified": len(troubleshooting_report['probable_causes']),
                "fixes_suggested": len(troubleshooting_report['suggested_fixes'])
            }
        )
        return troubleshooting_report
    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_event(
            event_type="design",
            user_id=current_user["email"],
            action="troubleshoot_failure_failed",
            resource=f"experiment:{failed_experiment_id}",
            success=False,
            details={"error": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Troubleshooting failed: {str(e)}"
        )
@router.get("/templates")
async def get_design_templates(
    category: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    try:
        templates = {
            "protein_expression": {
                "name": "Protein Expression in E. coli",
                "text": "Express recombinant protein in E. coli BL21(DE3) cells",
                "conditions": {
                    "organism": "ecoli",
                    "temperature": 37.0,
                    "ph": 7.0,
                    "duration_hours": 4.0
                },
                "typical_parameters": {
                    "IPTG_concentration": "0.5mM",
                    "induction_temperature": "37°C or 18°C",
                    "growth_media": "LB or TB"
                }
            },
            "cell_culture": {
                "name": "Mammalian Cell Culture",
                "text": "Culture mammalian cells for protein production",
                "conditions": {
                    "organism": "human",
                    "temperature": 37.0,
                    "co2_percent": 5.0
                },
                "typical_parameters": {
                    "media": "DMEM + 10% FBS",
                    "passage_ratio": "1:3 to 1:6",
                    "confluence": "80-90%"
                }
            },
            "western_blot": {
                "name": "Western Blot Analysis",
                "text": "Protein detection via Western blot",
                "conditions": {
                    "temperature": 4.0,
                    "duration_hours": 16.0
                },
                "typical_parameters": {
                    "gel_percentage": "10-12%",
                    "transfer_method": "wet or semi-dry",
                    "blocking": "5% milk or BSA"
                }
            },
            "pcr": {
                "name": "PCR Amplification",
                "text": "Amplify DNA fragment using PCR",
                "conditions": {
                    "temperature": 95.0,
                    "duration_hours": 1.0
                },
                "typical_parameters": {
                    "annealing_temp": "55-65°C",
                    "extension_time": "1min/kb",
                    "cycles": "25-35"
                }
            }
        }
        if category:
            if category not in templates:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Template category '{category}' not found"
                )
            return {category: templates[category]}
        return templates
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve templates: {str(e)}"
        )