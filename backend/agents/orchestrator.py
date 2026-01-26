from typing import Dict, Any, List
from backend.services.gemini_service import get_gemini_service
from backend.services.qdrant_service import get_qdrant_service
from backend.services.embedding_service import get_embedding_service
import uuid
class OrchestratorAgent:
    def __init__(self):
        self.gemini = get_gemini_service()
        self.qdrant = get_qdrant_service()
        self.embedding_service = get_embedding_service()
        self._agents = {}
    def _get_agent(self, agent_name: str):
        if agent_name not in self._agents:
            if agent_name == "ingestion":
                from backend.agents.ingestion_agent import IngestionAgent
                self._agents[agent_name] = IngestionAgent()
            elif agent_name == "embedding":
                from backend.agents.embedding_agent import EmbeddingAgent
                self._agents[agent_name] = EmbeddingAgent()
            elif agent_name == "multimodal_search":
                from backend.agents.multimodal_search_agent import MultimodalSearchAgent
                self._agents[agent_name] = MultimodalSearchAgent()
            elif agent_name == "similarity":
                from backend.agents.similarity_agent import SimilarityAgent
                self._agents[agent_name] = SimilarityAgent()
            elif agent_name == "reranking":
                from backend.agents.reranking_agent import RerankingAgent
                self._agents[agent_name] = RerankingAgent()
            elif agent_name == "failure":
                from backend.agents.failure_agent import FailureAgent
                self._agents[agent_name] = FailureAgent()
            elif agent_name == "design":
                from backend.agents.design_agent import DesignAgent
                self._agents[agent_name] = DesignAgent()
            elif agent_name == "bio_rag":
                from backend.agents.bio_rag_agent import BioRAGAgent
                self._agents[agent_name] = BioRAGAgent()
        return self._agents.get(agent_name)
    async def process_request(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        intent = await self._analyze_intent(user_input)
        execution_plan = await self._create_execution_plan(intent)
        results = await self._execute_plan_a2a(execution_plan, user_input)
        final_response = await self._aggregate_results(results, intent)
        return final_response
    async def _analyze_intent(self, user_input: Dict[str, Any]) -> Dict[str, str]:
        if "intent" in user_input:
            return {
                "primary_intent": user_input["intent"],
                "entities": user_input.get("query", {}),
                "context": {}
            }
        intent = await self.gemini.analyze_intent(user_input)
        return intent
    async def _create_execution_plan(self, intent: Dict[str, str]) -> List[str]:
        plans = {
            "search_similar": [
                "multimodal_search",
                "similarity",
                "reranking",
                "failure",
                "evidence"
            ],
            "search_image": [
                "multimodal_search",
                "similarity",
                "reranking",
                "failure",
                "evidence"
            ],
            "search_rag": [
                "multimodal_search",
                "similarity",
                "reranking",
                "bio_rag",
                "evidence"
            ],
            "design_variant": [
                "similarity",
                "reranking",
                "design",
                "evidence"
            ],
            "upload_experiment": [
                "ingestion",
                "embedding"
            ],
            "analyze_failure": [
                "similarity",
                "failure",
                "evidence"
            ]
        }
        primary_intent = intent.get('primary_intent', 'search_similar')
        return plans.get(primary_intent, ["similarity", "evidence"])
    async def _execute_plan_a2a(
        self,
        plan: List[str],
        user_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        context = {
            "user_input": user_input,
            "intent": user_input.get("intent"),
            "results": {}
        }
        for agent_name in plan:
            try:
                agent = self._get_agent(agent_name)
                if agent:
                    agent_result = await agent.execute(context)
                    context.update(agent_result)
                    context["results"][agent_name] = agent_result
            except Exception as e:
                context["results"][agent_name] = {
                    "error": str(e),
                    "status": "failed"
                }
        return context
    async def _aggregate_results(
        self,
        results: Dict[str, Any],
        intent: Dict[str, str]
    ) -> Dict[str, Any]:
        primary_intent = intent.get('primary_intent', 'unknown')
        if primary_intent == "search_similar":
            return {
                "intent": primary_intent,
                "modalities_used": results.get("modalities_used", {}),
                "search_strategy": results.get("search_strategy", "hybrid"),
                "reranked_neighbors": results.get("reranked_neighbors", []),
                "reproducibility_risk": results.get("reproducibility_risk", 0.0),
                "risk_level": results.get("risk_level", "UNKNOWN"),
                "failures": results.get("failures", []),
                "failure_patterns": results.get("failure_patterns", {}),
                "evidence_map": results.get("evidence_map", {})
            }
        elif primary_intent == "design_variant":
            return {
                "intent": primary_intent,
                "design_variants": results.get("design_variants", []),
                "reranked_neighbors": results.get("reranked_neighbors", []),
                "evidence_map": results.get("evidence_map", {})
            }
        elif primary_intent == "upload_experiment":
            return {
                "intent": primary_intent,
                "experiment_id": results.get("experiment_id", str(uuid.uuid4())),
                "status": "success",
                "embedding_metadata": results.get("embedding_metadata", {})
            }
        elif primary_intent == "analyze_failure":
            return {
                "intent": primary_intent,
                "failure_analysis": results.get("failure_patterns", {}),
                "similar_failures": results.get("failures", []),
                "recommendations": results.get("recommendations", []),
                "evidence_map": results.get("evidence_map", {})
            }
        return {
            "intent": primary_intent,
            "results": results
        }