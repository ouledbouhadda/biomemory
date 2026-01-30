import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import OrderedDict
from typing import Dict, Any, List

from backend.services.gemini_service import get_gemini_service
from backend.services.qdrant_service import get_qdrant_service
from backend.services.embedding_service import get_embedding_service

logger = logging.getLogger("biomemory.orchestrator")


class SearchResultsCache:
    """TTL cache for search results to avoid re-executing identical pipelines."""

    def __init__(self, max_size: int = 128, ttl_seconds: int = 300):
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    def _make_key(self, user_input: Dict[str, Any]) -> str:
        # Create a stable key from search parameters (exclude volatile fields)
        key_data = {
            "text": user_input.get("query", {}).get("text", ""),
            "sequence": user_input.get("query", {}).get("sequence", ""),
            "conditions": user_input.get("query", {}).get("conditions", {}),
            "limit": user_input.get("query", {}).get("limit", 10),
            "intent": user_input.get("intent", ""),
        }
        raw = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> Any:
        if key in self._cache:
            # Check TTL
            if time.time() - self._timestamps[key] < self._ttl:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            else:
                # Expired
                del self._cache[key]
                del self._timestamps[key]
        self._misses += 1
        return None

    def put(self, key: str, value: Any):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                oldest_key, _ = self._cache.popitem(last=False)
                self._timestamps.pop(oldest_key, None)
        self._cache[key] = value
        self._timestamps[key] = time.time()

    @property
    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0
        }


class OrchestratorAgent:
    def __init__(self):
        self.gemini = get_gemini_service()
        self.qdrant = get_qdrant_service()
        self.embedding_service = get_embedding_service()
        self._agents = {}
        self._results_cache = SearchResultsCache(max_size=128, ttl_seconds=300)

    def _get_agent(self, agent_name: str):
        if agent_name not in self._agents:
            if agent_name == "ingestion":
                from backend.agents.ingestion_agent import IngestionAgent
                self._agents[agent_name] = IngestionAgent()
            elif agent_name == "embedding":
                from backend.agents.embedding_agent import EmbeddingAgent
                self._agents[agent_name] = EmbeddingAgent()
            elif agent_name == "chunking":
                from backend.agents.chunking_agent import ChunkingAgent
                self._agents[agent_name] = ChunkingAgent()
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
            elif agent_name == "evidence":
                from backend.agents.evidence_agent import EvidenceAgent
                self._agents[agent_name] = EvidenceAgent()
            elif agent_name == "agentic_rag":
                from backend.agents.agentic_rag_agent import AgenticRAGAgent
                self._agents[agent_name] = AgenticRAGAgent()
        return self._agents.get(agent_name)

    async def process_request(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        # Check results cache
        cache_key = self._results_cache._make_key(user_input)
        cached = self._results_cache.get(cache_key)
        if cached is not None:
            logger.info("Search results cache HIT")
            return cached

        start_time = time.time()
        intent = await self._analyze_intent(user_input)
        execution_plan = await self._create_execution_plan(intent)
        results = await self._execute_plan_a2a(execution_plan, user_input)
        final_response = await self._aggregate_results(results, intent)

        elapsed = time.time() - start_time
        final_response["_pipeline_time_ms"] = round(elapsed * 1000, 1)
        logger.info("Pipeline completed in %.0fms (intent=%s)", elapsed * 1000, intent.get("primary_intent"))

        # Cache the results
        self._results_cache.put(cache_key, final_response)

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
                "multimodal_search", "similarity", "reranking",
                ["failure", "evidence"]
            ],
            "search_image": [
                "multimodal_search", "similarity", "reranking",
                ["failure", "evidence"]
            ],
            "search_rag": [
                "multimodal_search", "chunking", "similarity", "reranking",
                "bio_rag", "evidence"
            ],
            "design_variant": [
                "similarity", "reranking", "design", ["failure", "evidence"]
            ],
            "generate_design_variants": [
                "similarity", "reranking", "design", ["failure", "evidence"]
            ],
            "upload_experiment": [
                "ingestion", "chunking", "embedding"
            ],
            "analyze_failure": [
                "similarity", ["failure", "evidence"]
            ],
            "search_experiments": [
                "multimodal_search", "similarity", "reranking",
                ["failure", "evidence"]
            ],
            "agentic_planning": [
                "agentic_rag"
            ]
        }
        primary_intent = intent.get('primary_intent', 'search_similar')
        return plans.get(primary_intent, ["similarity", "evidence"])

    async def _execute_plan_a2a(
        self,
        plan: List,
        user_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        context = {
            "user_input": user_input,
            "intent": user_input.get("intent"),
            "results": {}
        }

        # For design intents, generate embedding from base experiment text
        # so the similarity agent can find similar experiments
        if user_input.get("intent") == "generate_design_variants":
            base_exp = user_input.get("base_experiment", {})
            try:
                embedding = await self.embedding_service.generate_multimodal_embedding(
                    text=base_exp.get("text", ""),
                    sequence=base_exp.get("sequence"),
                    conditions=base_exp.get("conditions"),
                )
                context["query_embedding"] = embedding
                context["query_conditions"] = base_exp.get("conditions") or {}
                context["limit"] = 20
                logger.info("Design: generated embedding from base experiment text")
            except Exception as e:
                logger.error("Design: failed to generate embedding: %s", e)

        for step in plan:
            if isinstance(step, list):
                # Parallel execution of independent agents
                await self._execute_parallel_agents(step, context)
            else:
                # Sequential execution
                try:
                    agent = self._get_agent(step)
                    if agent:
                        step_start = time.time()
                        agent_result = await agent.execute(context)
                        step_elapsed = time.time() - step_start
                        context.update(agent_result)
                        context["results"][step] = agent_result
                        logger.debug("Agent '%s' completed in %.0fms", step, step_elapsed * 1000)
                except Exception as e:
                    logger.error("Agent '%s' failed: %s", step, e)
                    context["results"][step] = {
                        "error": str(e),
                        "status": "failed"
                    }
        return context

    async def _execute_parallel_agents(
        self,
        agent_names: List[str],
        context: Dict[str, Any]
    ):
        """Execute multiple agents in parallel using asyncio.gather."""
        async def run_agent(name: str) -> tuple:
            try:
                agent = self._get_agent(name)
                if agent:
                    step_start = time.time()
                    result = await agent.execute(context)
                    elapsed = time.time() - step_start
                    logger.debug("Agent '%s' (parallel) completed in %.0fms", name, elapsed * 1000)
                    return name, result
                return name, {}
            except Exception as e:
                logger.error("Agent '%s' (parallel) failed: %s", name, e)
                return name, {"error": str(e), "status": "failed"}

        results = await asyncio.gather(
            *[run_agent(name) for name in agent_names],
            return_exceptions=False
        )

        for name, result in results:
            if isinstance(result, dict):
                context.update(result)
                context["results"][name] = result

    async def _aggregate_results(
        self,
        results: Dict[str, Any],
        intent: Dict[str, str]
    ) -> Dict[str, Any]:
        primary_intent = intent.get('primary_intent', 'unknown')
        if primary_intent in ("search_similar", "search_experiments"):
            return {
                "intent": primary_intent,
                "modalities_used": results.get("modalities_used", {}),
                "search_strategy": results.get("search_strategy", "hybrid"),
                "reranked_neighbors": results.get("reranked_neighbors", []),
                "reproducibility_risk": results.get("reproducibility_risk", 0.0),
                "risk_level": results.get("risk_level", "UNKNOWN"),
                "failures": results.get("failures", []),
                "failure_patterns": results.get("failure_patterns", {}),
                "evidence_map": results.get("evidence_map", {}),
                "cache_stats": {
                    "embedding_cache": self.embedding_service.cache_stats,
                    "results_cache": self._results_cache.stats,
                }
            }
        elif primary_intent in ("design_variant", "generate_design_variants"):
            return {
                "intent": primary_intent,
                "design_variants": results.get("design_variants", []),
                "reproducibility_risk": results.get("reproducibility_risk", 0.0),
                "reranked_neighbors": results.get("reranked_neighbors", []),
                "evidence_map": results.get("evidence_map", {}),
                "generation_method": "gemini",
                "model_used": "gemini-pro",
                "context_size": len(results.get("reranked_neighbors", []))
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
