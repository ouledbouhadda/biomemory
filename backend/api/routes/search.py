from fastapi import APIRouter, Depends, HTTPException, status, Form
from typing import Optional
from backend.models.requests import SearchRequest
from backend.models.responses import SearchResponse
from backend.agents.orchestrator import OrchestratorAgent
from backend.security.audit_logger import get_audit_logger
from backend.security.rate_limiting import RateLimiter
from backend.api.routes.auth import get_current_user
router = APIRouter()
orchestrator = OrchestratorAgent()
audit_logger = get_audit_logger()
search_rate_limiter = RateLimiter(max_requests=30, window_seconds=60)
@router.post("/", response_model=SearchResponse)
async def search_experiments(
    request: SearchRequest,
    current_user: dict = Depends(get_current_user)
):
    allowed, info = search_rate_limiter.is_allowed(current_user["email"])
    if not allowed:
        audit_logger.log_event(
            event_type="search",
            user_id=current_user["email"],
            action="search_rate_limited",
            resource="search",
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Search rate limit exceeded. Try again in {info['reset_in']} seconds"
        )
    try:
        user_input = {
            "intent": "search_experiments",
            "query": {
                "text": request.text,
                "sequence": request.sequence,
                "image_base64": request.image_base64,
                "conditions": request.conditions.model_dump() if request.conditions else None
            },
            "limit": request.limit,
            "similarity_threshold": request.similarity_threshold,
            "user_id": current_user["email"]
        }
        result = await orchestrator.process_request(user_input)
        similar_experiments = result.get('reranked_neighbors', [])
        reproducibility_risk = result.get('reproducibility_risk', 0.0)
        evidence_map = result.get('evidence_map', {})
        search_metadata = result.get('search_metadata', {})
        formatted_experiments = []
        for exp in similar_experiments:
            exp_id = exp.get('id')
            payload = exp.get('payload', {})
            evidence = evidence_map.get(exp_id, {})
            formatted_experiments.append({
                "experiment_id": exp_id,
                "text": payload.get('text', ''),
                "sequence": payload.get('sequence'),
                "conditions": payload.get('conditions'),
                "success": payload.get('success'),
                "similarity_score": exp.get('score', 0.0),
                "reranked_score": exp.get('reranked_score', 0.0),
                "source": evidence.get('source', 'unknown'),
                "source_type": evidence.get('source_type', 'unknown'),
                "verification_status": evidence.get('verification_status', 'unverified'),
                "publication": evidence.get('publication'),
                "database_origin": evidence.get('database_origin', 'unknown')
            })
        audit_logger.log_event(
            event_type="search",
            user_id=current_user["email"],
            action="search_success",
            resource="search",
            success=True,
            details={
                "results_count": len(formatted_experiments),
                "modalities_used": search_metadata.get('modalities_used', {}),
                "search_strategy": search_metadata.get('search_strategy', 'unknown'),
                "reproducibility_risk": reproducibility_risk
            }
        )
        return SearchResponse(
            results=formatted_experiments,
            total_results=len(formatted_experiments),
            reproducibility_risk=reproducibility_risk,
            search_metadata=search_metadata,
            traceability_stats=result.get('traceability_stats', {})
        )
    except Exception as e:
        audit_logger.log_event(
            event_type="search",
            user_id=current_user["email"],
            action="search_failed",
            resource="search",
            success=False,
            details={"error": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )
@router.post("/image", response_model=SearchResponse)
async def search_by_image(
    image_base64: str = Form(..., description="Image encoded in base64"),
    limit: int = Form(10, ge=1, le=100, description="Maximum number of results"),
    include_failures: bool = Form(True, description="Include failed experiments"),
    current_user: dict = Depends(get_current_user)
):
    allowed, info = search_rate_limiter.is_allowed(current_user["email"])
    if not allowed:
        audit_logger.log_event(
            event_type="search",
            user_id=current_user["email"],
            action="image_search_rate_limited",
            resource="search",
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Search rate limit exceeded. Try again in {info['reset_in']} seconds"
        )
    try:
        user_input = {
            "intent": "search_image",
            "query": {
                "image_base64": image_base64
            },
            "limit": limit,
            "include_failures": include_failures,
            "user_id": current_user["email"]
        }
        result = await orchestrator.process_request(user_input)
        similar_experiments = result.get('reranked_neighbors', [])
        reproducibility_risk = result.get('reproducibility_risk', 0.0)
        evidence_map = result.get('evidence_map', {})
        search_metadata = result.get('search_metadata', {})
        formatted_experiments = []
        for exp in similar_experiments:
            exp_id = exp.get('id')
            payload = exp.get('payload', {})
            evidence = evidence_map.get(exp_id, {})
            formatted_experiments.append({
                "experiment_id": exp_id,
                "text": payload.get('text', ''),
                "sequence": payload.get('sequence'),
                "conditions": payload.get('conditions'),
                "image_base64": payload.get('image_base64'),
                "success": payload.get('success'),
                "similarity_score": exp.get('score', 0.0),
                "reranked_score": exp.get('reranked_score', 0.0),
                "source": evidence.get('source', 'unknown'),
                "source_type": evidence.get('source_type', 'unknown'),
                "verification_status": evidence.get('verification_status', 'unverified'),
                "publication": evidence.get('publication'),
                "database_origin": evidence.get('database_origin', 'unknown')
            })
        audit_logger.log_event(
            event_type="search",
            user_id=current_user["email"],
            action="image_search_success",
            resource="search",
            success=True,
            details={
                "results_count": len(formatted_experiments),
                "search_type": "image_similarity",
                "reproducibility_risk": reproducibility_risk
            }
        )
        return SearchResponse(
            results=formatted_experiments,
            total_results=len(formatted_experiments),
            reproducibility_risk=reproducibility_risk,
            search_metadata={
                **search_metadata,
                "search_type": "image_similarity",
                "modalities_used": {"image": True}
            },
            traceability_stats=result.get('traceability_stats', {})
        )
    except Exception as e:
        audit_logger.log_event(
            event_type="search",
            user_id=current_user["email"],
            action="image_search_failed",
            resource="search",
            success=False,
            details={"error": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image search failed: {str(e)}"
        )
@router.post("/advanced", response_model=SearchResponse)
async def advanced_search(
    request: SearchRequest,
    include_failures: bool = False,
    min_success_rate: Optional[float] = None,
    organism_filter: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    allowed, info = search_rate_limiter.is_allowed(current_user["email"])
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Search rate limit exceeded. Try again in {info['reset_in']} seconds"
        )
    try:
        user_input = {
            "intent": "search_experiments",
            "query": {
                "text": request.text,
                "sequence": request.sequence,
                "image_base64": request.image_base64,
                "conditions": request.conditions.model_dump() if request.conditions else None
            },
            "limit": request.limit,
            "similarity_threshold": request.similarity_threshold,
            "advanced_filters": {
                "include_failures": include_failures,
                "min_success_rate": min_success_rate,
                "organism_filter": organism_filter
            },
            "user_id": current_user["email"]
        }
        result = await orchestrator.process_request(user_input)
        similar_experiments = result.get('reranked_neighbors', [])
        if not include_failures:
            similar_experiments = [
                exp for exp in similar_experiments
                if exp.get('payload', {}).get('success', False)
            ]
        if organism_filter:
            similar_experiments = [
                exp for exp in similar_experiments
                if exp.get('payload', {}).get('conditions', {}).get('organism') == organism_filter
            ]
        evidence_map = result.get('evidence_map', {})
        formatted_experiments = []
        for exp in similar_experiments:
            exp_id = exp.get('id')
            payload = exp.get('payload', {})
            evidence = evidence_map.get(exp_id, {})
            formatted_experiments.append({
                "experiment_id": exp_id,
                "text": payload.get('text', ''),
                "sequence": payload.get('sequence'),
                "conditions": payload.get('conditions'),
                "success": payload.get('success'),
                "similarity_score": exp.get('score', 0.0),
                "reranked_score": exp.get('reranked_score', 0.0),
                "source": evidence.get('source', 'unknown'),
                "source_type": evidence.get('source_type', 'unknown'),
                "verification_status": evidence.get('verification_status', 'unverified'),
                "publication": evidence.get('publication'),
                "database_origin": evidence.get('database_origin', 'unknown')
            })
        audit_logger.log_event(
            event_type="search",
            user_id=current_user["email"],
            action="advanced_search_success",
            resource="search",
            success=True,
            details={
                "results_count": len(formatted_experiments),
                "filters_applied": {
                    "include_failures": include_failures,
                    "organism_filter": organism_filter
                }
            }
        )
        return SearchResponse(
            results=formatted_experiments,
            total_results=len(formatted_experiments),
            reproducibility_risk=result.get('reproducibility_risk', 0.0),
            search_metadata=result.get('search_metadata', {}),
            traceability_stats=result.get('traceability_stats', {})
        )
    except Exception as e:
        audit_logger.log_event(
            event_type="search",
            user_id=current_user["email"],
            action="advanced_search_failed",
            resource="search",
            success=False,
            details={"error": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Advanced search failed: {str(e)}"
        )
@router.get("/suggestions")
async def get_search_suggestions(
    query: str,
    limit: int = 5,
    current_user: dict = Depends(get_current_user)
):
    try:
        suggestions = {
            "terms": [
                f"{query} protein expression",
                f"{query} cell culture",
                f"{query} western blot",
                f"{query} PCR amplification",
                f"{query} cloning"
            ][:limit],
            "organisms": ["human", "mouse", "ecoli", "yeast", "drosophila"][:limit],
            "protocols": [
                "Western Blot",
                "PCR",
                "Cell Culture",
                "Protein Purification",
                "CRISPR"
            ][:limit]
        }
        return suggestions
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get suggestions: {str(e)}"
        )
@router.post("/rag", response_model=SearchResponse)
async def search_experiments_rag(
    request: SearchRequest,
    current_user: dict = Depends(get_current_user)
):
    allowed, info = search_rate_limiter.is_allowed(current_user["email"])
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {info['reset_in']} seconds."
        )
    try:
        audit_logger.log_event(
            event_type="rag_search",
            user_email=current_user["email"],
            details={
                "query_text": request.text,
                "has_sequence": bool(request.sequence),
                "has_conditions": bool(request.conditions),
                "has_image": bool(request.image)
            }
        )
        from backend.agents.bio_rag_agent import BioRAGAgent
        rag_agent = BioRAGAgent()
        context = {
            'user_input': {
                'query': {
                    'text': request.text,
                    'sequence': request.sequence,
                    'conditions': request.conditions,
                    'image': request.image,
                    'limit': request.limit or 15
                }
            },
            'user': current_user
        }
        result = await rag_agent.execute(context)
        if result.get('rag_error'):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"RAG search failed: {result['rag_error']}"
            )
        rag_response = result.get('rag_response', '')
        similar_experiments = result.get('similar_experiments', [])
        exp_count = result.get('similar_experiments_count', 0)
        experiments = []
        if similar_experiments:
            for exp in similar_experiments[:5]:
                payload = exp.get('payload', {})
                experiments.append({
                    'id': exp.get('id', ''),
                    'experiment_id': payload.get('experiment_id', ''),
                    'title': f"Expérience similaire (score: {exp.get('score', 0):.2f})",
                    'description': payload.get('text', '')[:200] + '...' if payload.get('text') else '',
                    'conditions': payload.get('conditions', {}),
                    'outcome': payload.get('outcome', {}),
                    'similarity_score': exp.get('score', 0),
                    'reproducibility_risk': 'low',
                    'evidence_sources': [{
                        'type': 'similarity_match',
                        'confidence': exp.get('score', 0),
                        'description': f"Similarité cosinus: {exp.get('score', 0):.3f}"
                    }]
                })
        response = SearchResponse(
            query=request.text,
            total_results=exp_count,
            experiments=experiments,
            search_metadata={
                'search_type': 'rag_intelligent',
                'rag_response': rag_response,
                'modalities_used': result.get('modalities_detected', {}),
                'processing_time': 0,
                'method': result.get('rag_method', 'similarity_rag')
            },
            suggestions={
                'related_terms': [
                    "optimisation conditions",
                    "paramètres expérimentaux",
                    "taux de succès",
                    "reproductibilité"
                ][:5],
                'technique_suggestions': [
                    "Expression protéique",
                    "Culture cellulaire",
                    "Purification",
                    "Analyse conditions"
                ][:5]
            }
        )
        return response
    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_event(
            event_type="rag_search_error",
            user_email=current_user["email"],
            details={"error": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG search failed: {str(e)}"
        )
@router.post("/recommend")
async def recommend_experiments(
    experiment_ids: list[str],
    negative_ids: Optional[list[str]] = None,
    limit: int = 10,
    current_user: dict = Depends(get_current_user)
):
    try:
        from backend.agents.similarity_agent import SimilarityAgent
        similarity_agent = SimilarityAgent()
        recommendations = await similarity_agent.recommend_similar_experiments(
            experiment_ids=experiment_ids,
            negative_ids=negative_ids,
            limit=limit
        )
        formatted_recommendations = []
        for rec in recommendations:
            payload = rec.get('payload', {})
            formatted_recommendations.append({
                "experiment_id": rec.get('id'),
                "score": rec.get('score'),
                "text": payload.get('text', ''),
                "sequence": payload.get('sequence'),
                "conditions": payload.get('conditions'),
                "source": "qdrant_recommendation"
            })
        audit_logger.log_event(
            event_type="recommendations",
            user_email=current_user["email"],
            details={
                "positive_examples": len(experiment_ids),
                "negative_examples": len(negative_ids or []),
                "recommendations_count": len(formatted_recommendations)
            }
        )
        return {
            "recommendations": formatted_recommendations,
            "total_found": len(formatted_recommendations),
            "search_strategy": "qdrant_recommendations"
        }
    except Exception as e:
        audit_logger.log_event(
            event_type="recommendations_error",
            user_email=current_user["email"],
            details={"error": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommendations failed: {str(e)}"
        )