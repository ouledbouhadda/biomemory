from fastapi import APIRouter, Depends, HTTPException, status, Form, Query
from typing import Optional, List
from backend.models.requests import SearchRequest
from backend.models.responses import SearchResponse
from backend.agents.orchestrator import OrchestratorAgent
from backend.agents.similarity_agent import SimilarityAgent
from backend.services.qdrant_service import get_qdrant_service
from backend.services.embedding_service import get_embedding_service
from backend.security.audit_logger import get_audit_logger
from backend.security.rate_limiting import RateLimiter
from backend.api.routes.auth import get_current_user
from pydantic import BaseModel

router = APIRouter()
orchestrator = OrchestratorAgent()
similarity_agent = SimilarityAgent()
qdrant_service = get_qdrant_service()
audit_logger = get_audit_logger()
search_rate_limiter = RateLimiter(max_requests=30, window_seconds=60)



class DiscoverRequest(BaseModel):
    target_id: str
    positive_context: List[str]
    negative_context: Optional[List[str]] = None
    limit: int = 10


class BatchSearchRequest(BaseModel):
    queries: List[dict]
    limit: int = 10


class GroupedSearchRequest(BaseModel):
    vector: List[float]
    group_by: str = "organism"
    group_size: int = 3
    limit: int = 10
    conditions: Optional[dict] = None


class TemporalSearchRequest(BaseModel):
    vector: List[float]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    limit: int = 10
    conditions: Optional[dict] = None


class BoostSearchRequest(BaseModel):
    vector: List[float]
    boost_factors: dict
    limit: int = 10
    conditions: Optional[dict] = None


class FullSearchRequest(BaseModel):
    vector: List[float]
    text: Optional[str] = None
    conditions: Optional[dict] = None
    limit: int = 20
    use_hybrid: bool = True
    use_grouping: bool = False
    group_by: str = "organism"
    use_temporal: bool = False
    date_range: Optional[dict] = None
    boost_factors: Optional[dict] = None
    order_by: Optional[str] = None


@router.post("/direct", response_model=SearchResponse)
async def search_direct(
    request: SearchRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    ENDPOINT DE RECHERCHE DIRECTE ET SIMPLE
    Retourne les résultats de Qdrant Cloud sans orchestration complexe
    Utilisation: POST /api/v1/search/direct
    Token: Bearer <access_token> (optionnel pour demo)
    """
    try:
        qdrant = get_qdrant_service()
        embedding = get_embedding_service()

        embed = await embedding.generate_multimodal_embedding(
            text=request.text,
            sequence=request.sequence or '',
            conditions=request.conditions.model_dump() if request.conditions else {},
            image_base64=request.image_base64
        )

        if not qdrant.cloud_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Qdrant Cloud not available"
            )

        results = qdrant.cloud_client.query_points(
            collection_name='public_science',
            query=embed.tolist(),
            limit=request.limit or 10,
            with_payload=True
        )

        formatted_experiments = []
        for result in results.points:
            payload = result.payload or {}

            formatted_experiments.append({
                "experiment_id": str(result.id),
                "text": payload.get('title', ''),
                "sequence": payload.get('sequence'),
                "conditions": {
                    "organism": payload.get('organism'),
                    "temperature": payload.get('temperature'),
                    "ph": payload.get('ph')
                },
                "success": payload.get('success'),
                "similarity_score": result.score,
                "reranked_score": result.score,
                "source": payload.get('contact', 'unknown'),
                "source_type": payload.get('type', 'unknown'),
                "verification_status": "verified",
                "publication": {
                    "title": payload.get('title', ''),
                    "reference": payload.get('reference', ''),
                    "document_id": payload.get('document_id', '')
                },
                "database_origin": payload.get('source', 'public_science'),
                "assay": payload.get('assay'),
                "organism": payload.get('organism')
            })

        return SearchResponse(
            results=formatted_experiments,
            total_results=len(formatted_experiments),
            reproducibility_risk=0.0,
            search_metadata={
                "search_type": "direct_qdrant",
                "collection": "public_science",
                "modalities_used": {
                    "text": bool(request.text),
                    "sequence": bool(request.sequence),
                    "image": bool(request.image_base64)
                }
            },
            traceability_stats={}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


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



@router.post("/discover")
async def discover_experiments(
    request: DiscoverRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Découverte contextuelle d'expériences avec Qdrant Discover API.
    """
    try:
        result = await similarity_agent.discover_experiments(
            target_id=request.target_id,
            positive_context=request.positive_context,
            negative_context=request.negative_context,
            limit=request.limit,
            user_id=current_user["email"]
        )

        audit_logger.log_event(
            event_type="discover_search",
            user_id=current_user["email"],
            action="discover_success",
            resource="search",
            success=True,
            details={"results_count": len(result.get("discovered_experiments", []))}
        )

        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Discover search failed: {str(e)}"
        )


@router.post("/batch")
async def batch_search_experiments(
    request: BatchSearchRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Recherche batch pour plusieurs requêtes simultanées.
    """
    try:
        result = await similarity_agent.batch_search_experiments(
            queries=request.queries,
            limit=request.limit,
            user_id=current_user["email"]
        )

        audit_logger.log_event(
            event_type="batch_search",
            user_id=current_user["email"],
            action="batch_search_success",
            resource="search",
            success=True,
            details={"queries_count": len(request.queries)}
        )

        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch search failed: {str(e)}"
        )


@router.post("/grouped")
async def grouped_search_experiments(
    request: GroupedSearchRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Recherche groupée par champ (organisme, type d'expérience, etc.).
    """
    try:
        query_filter = qdrant_service.build_metadata_filter(request.conditions)

        result = await similarity_agent.search_with_grouping(
            query_vector=request.vector,
            group_by=request.group_by,
            group_size=request.group_size,
            limit=request.limit,
            query_filter=query_filter,
            user_id=current_user["email"]
        )

        audit_logger.log_event(
            event_type="grouped_search",
            user_id=current_user["email"],
            action="grouped_search_success",
            resource="search",
            success=True,
            details={
                "groups_count": result.get("groups_count", 0),
                "group_by": request.group_by
            }
        )

        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Grouped search failed: {str(e)}"
        )


@router.post("/temporal")
async def temporal_search_experiments(
    request: TemporalSearchRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Recherche temporelle avec plage de dates.
    """
    try:
        query_filter = qdrant_service.build_metadata_filter(request.conditions)

        result = await similarity_agent.search_temporal_advanced(
            query_vector=request.vector,
            start_date=request.start_date,
            end_date=request.end_date,
            limit=request.limit,
            query_filter=query_filter,
            user_id=current_user["email"]
        )

        audit_logger.log_event(
            event_type="temporal_search",
            user_id=current_user["email"],
            action="temporal_search_success",
            resource="search",
            success=True,
            details={
                "results_count": len(result.get("temporal_results", [])),
                "date_range": {"start": request.start_date, "end": request.end_date}
            }
        )

        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Temporal search failed: {str(e)}"
        )


@router.post("/boosted")
async def boosted_search_experiments(
    request: BoostSearchRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Recherche avec boosting dynamique.
    """
    try:
        query_filter = qdrant_service.build_metadata_filter(request.conditions)

        result = await similarity_agent.search_with_qdrant_boosting(
            query_vector=request.vector,
            boost_factors=request.boost_factors,
            limit=request.limit,
            query_filter=query_filter,
            user_id=current_user["email"]
        )

        audit_logger.log_event(
            event_type="boosted_search",
            user_id=current_user["email"],
            action="boosted_search_success",
            resource="search",
            success=True,
            details={
                "results_count": len(result.get("boosted_results", [])),
                "boost_factors": request.boost_factors
            }
        )

        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Boosted search failed: {str(e)}"
        )


@router.post("/full")
async def full_qdrant_search(
    request: FullSearchRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Recherche complète utilisant toutes les fonctionnalités Qdrant.
    """
    allowed, info = search_rate_limiter.is_allowed(current_user["email"])
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {info['reset_in']} seconds"
        )

    try:
        result = await similarity_agent.full_qdrant_search(
            query_vector=request.vector,
            query_text=request.text,
            conditions=request.conditions,
            user_id=current_user["email"],
            limit=request.limit,
            use_hybrid=request.use_hybrid,
            use_grouping=request.use_grouping,
            group_by=request.group_by,
            use_temporal=request.use_temporal,
            date_range=request.date_range,
            boost_factors=request.boost_factors,
            order_by=request.order_by
        )

        audit_logger.log_event(
            event_type="full_qdrant_search",
            user_id=current_user["email"],
            action="full_search_success",
            resource="search",
            success=True,
            details={
                "results_count": result.get("total_found", 0),
                "features_used": result.get("search_metadata", {}).get("features_used", [])
            }
        )

        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Full Qdrant search failed: {str(e)}"
        )


@router.get("/aggregate/{group_by}")
async def aggregate_experiments(
    group_by: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Agrégation et statistiques sur les expériences.
    """
    try:
        result = await similarity_agent.aggregate_experiments(
            group_by=group_by,
            user_id=current_user["email"]
        )

        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Aggregation failed: {str(e)}"
        )


@router.get("/count")
async def count_experiments(
    collection: str = Query("public_science", description="Collection name"),
    current_user: dict = Depends(get_current_user)
):
    """
    Comptage des expériences dans une collection.
    """
    try:
        result = await similarity_agent.count_experiments(
            collection_name=collection
        )

        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Count failed: {str(e)}"
        )


@router.get("/experiment/{experiment_id}")
async def get_experiment(
    experiment_id: str,
    collection: str = Query("public_science", description="Collection name"),
    current_user: dict = Depends(get_current_user)
):
    """
    Récupération d'une expérience par ID.
    """
    try:
        result = await similarity_agent.get_experiment_by_id(
            experiment_id=experiment_id,
            collection_name=collection
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found"
            )

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Get experiment failed: {str(e)}"
        )


@router.get("/scroll")
async def scroll_experiments(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    collection: str = Query("public_science", description="Collection name"),
    current_user: dict = Depends(get_current_user)
):
    """
    Pagination des expériences.
    """
    try:
        result = await similarity_agent.scroll_experiments(
            limit=limit,
            offset=offset,
            collection_name=collection
        )

        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scroll failed: {str(e)}"
        )


@router.delete("/experiments")
async def delete_experiments(
    experiment_ids: List[str],
    current_user: dict = Depends(get_current_user)
):
    """
    Suppression d'expériences.
    """
    try:
        collection = f"private_experiments_{current_user['email']}"

        result = await similarity_agent.delete_experiments(
            experiment_ids=experiment_ids,
            collection_name=collection
        )

        audit_logger.log_event(
            event_type="delete_experiments",
            user_id=current_user["email"],
            action="delete_success",
            resource="experiments",
            success=result.get("success", False),
            details={"deleted_count": len(experiment_ids)}
        )

        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Delete failed: {str(e)}"
        )


@router.post("/multi-organism")
async def search_multi_organism(
    vector: List[float],
    organisms: List[str],
    limit: int = 10,
    current_user: dict = Depends(get_current_user)
):
    """
    Recherche filtrée par plusieurs organismes.
    """
    try:
        result = await similarity_agent.search_by_multiple_organisms(
            query_vector=vector,
            organisms=organisms,
            limit=limit,
            user_id=current_user["email"]
        )

        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multi-organism search failed: {str(e)}"
        )


@router.get("/facets/{field}")
async def get_facets(
    field: str,
    collection: str = Query("public_science", description="Collection name"),
    current_user: dict = Depends(get_current_user)
):
    """
    Obtenir les facettes (valeurs uniques et comptages) pour un champ.
    """
    try:
        result = await qdrant_service.faceted_count(
            collection_name=collection,
            facet_field=field
        )

        return {
            "field": field,
            "facets": result,
            "total_facets": len(result)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Facets failed: {str(e)}"
        )


@router.get("/collection/stats/{collection_name}")
async def get_collection_stats(
    collection_name: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Statistiques d'une collection Qdrant.
    """
    try:
        result = await qdrant_service.get_collection_stats(
            collection_name=collection_name
        )

        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Collection stats failed: {str(e)}"
        )


@router.post("/collection/user")
async def create_user_collection(
    current_user: dict = Depends(get_current_user)
):
    """
    Création d'une collection personnelle pour l'utilisateur.
    """
    try:
        collection_name = await qdrant_service.create_user_collection(
            user_id=current_user["email"]
        )

        audit_logger.log_event(
            event_type="create_collection",
            user_id=current_user["email"],
            action="collection_created",
            resource="collections",
            success=True,
            details={"collection_name": collection_name}
        )

        return {
            "collection_name": collection_name,
            "status": "created",
            "user_id": current_user["email"]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Create user collection failed: {str(e)}"
        )


@router.post("/snapshot/{collection_name}")
async def create_snapshot(
    collection_name: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Création d'un snapshot pour backup.
    """
    try:
        snapshot_name = await qdrant_service.create_snapshot(
            collection_name=collection_name
        )

        if snapshot_name:
            return {
                "snapshot_name": snapshot_name,
                "collection": collection_name,
                "status": "created"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Snapshot creation failed"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Create snapshot failed: {str(e)}"
        )


@router.get("/snapshots/{collection_name}")
async def list_snapshots(
    collection_name: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Liste les snapshots disponibles.
    """
    try:
        snapshots = await qdrant_service.list_snapshots(
            collection_name=collection_name
        )

        return {
            "collection": collection_name,
            "snapshots": snapshots,
            "count": len(snapshots)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"List snapshots failed: {str(e)}"
        )


@router.get("/health")
async def qdrant_health_check(
    current_user: dict = Depends(get_current_user)
):
    """
    Vérification de la santé des instances Qdrant.
    """
    try:
        private_health = await qdrant_service.health_check("private")
        cloud_health = await qdrant_service.health_check("cloud")

        return {
            "private_instance": {
                "status": "healthy" if private_health else "unhealthy",
                "available": private_health
            },
            "cloud_instance": {
                "status": "healthy" if cloud_health else "unhealthy",
                "available": cloud_health
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )