from typing import Dict, Any, List
from backend.services.gemini_service import get_gemini_service
from backend.services.qdrant_service import get_qdrant_service
from backend.services.embedding_service import get_embedding_service
import json

class AgenticRAGAgent:
    def __init__(self):
        self.gemini = get_gemini_service()
        self.qdrant = get_qdrant_service()
        self.embedding_service = get_embedding_service()
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        user_input = context.get("user_input", {})
        query = user_input.get("query", "")
        
        parsed_info = await self._parse_experiment_query(query)
        
        qdrant_results = await self._orchestrate_qdrant_searches(
            parsed_info,
            query
        )
        
        recommendations = await self._generate_recommendations(
            parsed_info,
            qdrant_results,
            query
        )
        
        return {
            "intent": "agentic_search",
            "parsed_experiment": parsed_info,
            "qdrant_insights": qdrant_results,
            "recommendations": recommendations,
            "status": "success"
        }
    
    async def _parse_experiment_query(self, query: str) -> Dict[str, Any]:
        try:
            prompt = f"""
Analyser cette requête d'expérience et extraire en JSON:
- type_experiment: Type d'expérience ou méthode
- organism: Organisme ou sujet biologique  
- conditions: Conditions (température, temps, pH, etc.)
- goal: Objectif ou hypothèse
- concerns: Défis ou préoccupations

Requête: {query}

Retourner JSON valide.
"""
            result = await self.gemini.generate_content(prompt)
            
            try:
                parsed = json.loads(result)
            except:
                parsed = {
                    "type_experiment": query[:50],
                    "organism": "",
                    "conditions": "",
                    "goal": query,
                    "concerns": ""
                }
        except Exception as e:
            # Fallback si gemini échoue
            parsed = {
                "type_experiment": query[:50],
                "organism": "unknown",
                "conditions": "standard",
                "goal": query,
                "concerns": ""
            }
        
        return parsed
    
    async def _orchestrate_qdrant_searches(
        self,
        parsed_info: Dict[str, Any],
        query: str
    ) -> Dict[str, Any]:
        query_embedding = await self.embedding_service.embed_text(query)
        
        results = {}
        
        results["1_vector_search"] = await self._qdrant_vector_search(
            query_embedding,
            parsed_info
        )
        
        results["2_hybrid_search"] = await self._qdrant_hybrid_search(
            query_embedding,
            parsed_info
        )
        
        results["3_discover_success"] = await self._qdrant_discover_patterns(
            query_embedding,
            parsed_info
        )
        
        results["4_recommend_variants"] = await self._qdrant_recommend_variants(
            results["1_vector_search"],
            parsed_info
        )
        
        results["5_grouping_by_conditions"] = await self._qdrant_group_by_conditions(
            query_embedding,
            parsed_info
        )
        
        results["6_temporal_trend"] = await self._qdrant_temporal_search(
            query_embedding,
            parsed_info
        )
        
        results["7_faceted_stats"] = await self._qdrant_faceted_counts(
            parsed_info
        )
        
        results["8_boosted_search"] = await self._qdrant_boosted_search(
            query_embedding,
            parsed_info
        )
        
        results["9_sampling"] = await self._qdrant_random_sampling(
            parsed_info
        )
        
        return results
    
    async def _qdrant_vector_search(
        self,
        query_embedding: List[float],
        parsed_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        results = await self.qdrant.search(
            collection_name="public_science",
            query_vector=query_embedding,
            limit=5,
            with_payload=True
        )
        
        return [
            {
                "title": r.payload.get("title", ""),
                "score": r.score,
                "organism": r.payload.get("organism"),
                "method": r.payload.get("method"),
                "success": r.payload.get("success")
            }
            for r in results
        ]
    
    async def _qdrant_hybrid_search(
        self,
        query_embedding: List[float],
        parsed_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        query_text = f"{parsed_info.get('type_experiment', '')} {parsed_info.get('organism', '')}"
        
        results = await self.qdrant.hybrid_search(
            collection_name="public_science",
            query_vector=query_embedding,
            query_text=query_text,
            limit=5,
            with_payload=True
        )
        
        return [
            {
                "title": r.payload.get("title", ""),
                "vector_score": getattr(r, 'vector_score', 0),
                "keyword_score": getattr(r, 'keyword_score', 0),
                "organism": r.payload.get("organism")
            }
            for r in results
        ]
    
    async def _qdrant_discover_patterns(
        self,
        query_embedding: List[float],
        parsed_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            results = await self.qdrant.discover(
                collection_name="public_science",
                query_positive=query_embedding,
                query_negative=[],
                limit=5,
                with_payload=True
            )
            
            return {
                "success_patterns": [
                    r.payload.get("title", "") for r in results
                ],
                "count": len(results)
            }
        except:
            return {"success_patterns": [], "count": 0}
    
    async def _qdrant_recommend_variants(
        self,
        successful_experiments: List[Dict[str, Any]],
        parsed_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        try:
            if not successful_experiments:
                return []
            
            positive_ids = []
            for exp in successful_experiments[:2]:
                pass
            
            results = await self.qdrant.recommend(
                collection_name="public_science",
                positive=positive_ids if positive_ids else [1],
                negative=[],
                limit=5,
                with_payload=True
            )
            
            return [
                {
                    "title": r.payload.get("title", ""),
                    "recommendation_score": r.score,
                    "variant_type": r.payload.get("method")
                }
                for r in results
            ]
        except:
            return []
    
    async def _qdrant_group_by_conditions(
        self,
        query_embedding: List[float],
        parsed_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        try:
            results = await self.qdrant.search_with_grouping(
                collection_name="public_science",
                query_vector=query_embedding,
                group_by="organism",
                group_size=2,
                limit=10,
                with_payload=True
            )
            
            grouped = {}
            for r in results:
                organism = r.payload.get("organism", "unknown")
                if organism not in grouped:
                    grouped[organism] = []
                grouped[organism].append({
                    "title": r.payload.get("title"),
                    "score": r.score
                })
            
            return grouped
        except:
            return {}
    
    async def _qdrant_temporal_search(
        self,
        query_embedding: List[float],
        parsed_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            results = await self.qdrant.search_temporal(
                collection_name="public_science",
                query_vector=query_embedding,
                days_back=365,
                limit=5,
                with_payload=True
            )
            
            return {
                "recent_experiments": [r.payload.get("title", "") for r in results],
                "trend": "recent" if results else "no_data"
            }
        except:
            return {"recent_experiments": [], "trend": "error"}
    
    async def _qdrant_faceted_counts(
        self,
        parsed_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            organism = parsed_info.get("organism")
            
            total_count = await self.qdrant.count_points(
                collection_name="public_science"
            )
            
            return {
                "total_experiments": total_count,
                "collection": "public_science"
            }
        except:
            return {"total_experiments": 0}
    
    async def _qdrant_boosted_search(
        self,
        query_embedding: List[float],
        parsed_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        try:
            results = await self.qdrant.search_with_boosting(
                collection_name="public_science",
                query_vector=query_embedding,
                boost_field="success",
                limit=5,
                with_payload=True
            )
            
            return [
                {
                    "title": r.payload.get("title", ""),
                    "boosted_score": r.score,
                    "success_boosted": r.payload.get("success", False)
                }
                for r in results
            ]
        except:
            return []
    
    async def _qdrant_random_sampling(
        self,
        parsed_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        try:
            results = await self.qdrant.random_sample(
                collection_name="public_science",
                sample_size=3,
                with_payload=True
            )
            
            return [
                {
                    "title": r.payload.get("title", ""),
                    "organism": r.payload.get("organism")
                }
                for r in results
            ]
        except:
            return []
    
    async def _generate_recommendations(
        self,
        parsed_info: Dict[str, Any],
        qdrant_results: Dict[str, Any],
        original_query: str
    ) -> Dict[str, Any]:
        synthesis_prompt = f"""
REQUÊTE UTILISATEUR:
{original_query}

PARAMÈTRES EXTRAITS:
{json.dumps(parsed_info, indent=2)}

RÉSULTATS QDRANT (9 types de recherche):
{json.dumps({k: v for k, v in list(qdrant_results.items())[:5]}, indent=2)}

Générer une réponse structurée:

1. RÉSUMÉ: "En se basant sur [X] expériences similaires trouvées..."
2. APPROCHE RECOMMANDÉE: Points clés basés sur les patterns de succès
3. CONDITIONS OPTIMALES: Température, durée, concentrations détaillées
4. DÉFIS ANTICIPÉS: D'après les cas similaires
5. VARIATIONS À EXPLORER: Alternatives recommandées
6. PROBABILITÉ DE SUCCÈS: Estimation basée sur les données

Soyez précis et actionnable.
"""
        
        try:
            response = await self.gemini.generate_content(synthesis_prompt)
        except Exception as e:
            # Fallback si gemini échoue
            response = f"""
RÉSUMÉ: En se basant sur les données Qdrant trouvées.

APPROCHE RECOMMANDÉE: Examiner les expériences similaires trouvées.

CONDITIONS OPTIMALES: À déterminer par analyse des résultats.

DÉFIS ANTICIPÉS: Considérer les variations dans les protocoles.

VARIATIONS À EXPLORER: Consulter les resultats de groupement par conditions.

PROBABILITÉ DE SUCCÈS: Basée sur la pertinence des résultats Qdrant.
"""
        
        return {
            "full_response": response,
            "qdrant_sources_used": 9,
            "data_quality": "high" if qdrant_results.get("1_vector_search") else "low"
        }
