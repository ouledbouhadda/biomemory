import logging
from typing import Dict, Any, List, Optional
from backend.services.qdrant_service import get_qdrant_service
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, Range, MatchText, MatchAny,
    HasIdCondition, IsEmptyCondition, IsNullCondition,
    DatetimeRange, OrderBy, Fusion, ContextExamplePair
)
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger("biomemory.similarity")


class SimilarityAgent:
    def __init__(self):
        self.qdrant = get_qdrant_service()
    def _create_contextual_filter(self, query_context: Dict[str, Any]) -> Filter:
        conditions = []
        if query_context.get("domain"):
            conditions.append(
                FieldCondition(
                    key="domain",
                    match=MatchValue(value=query_context["domain"])
                )
            )
        if query_context.get("experiment_type"):
            conditions.append(
                FieldCondition(
                    key="experiment_type",
                    match=MatchValue(value=query_context["experiment_type"])
                )
            )
        if query_context.get("recent_years"):
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=365 * query_context["recent_years"])
            conditions.append(
                FieldCondition(
                    key="publication_date",
                    range=Range(gte=cutoff_date.isoformat())
                )
            )
        if query_context.get("min_confidence"):
            conditions.append(
                FieldCondition(
                    key="confidence_score",
                    range=Range(gte=query_context["min_confidence"])
                )
            )
        return Filter(must=conditions) if conditions else None
    def _search_with_hybrid_ranking(self, collection_name: str, query_vector: List[float],
                                   query_text: str, limit: int = 20,
                                   filter_conditions: Filter = None) -> List[Dict[str, Any]]:
        try:
            hybrid_results = self.qdrant.hybrid_search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_text=query_text,
                limit=limit,
                query_filter=filter_conditions,
                score_threshold=0.0,
                fusion=Fusion.RRF
            )
            if hybrid_results:
                logger.info("Hybrid search succeeded: %d results", len(hybrid_results))
                return hybrid_results
            logger.info("Fallback to vector search only")
            return self.qdrant.search_sync(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=filter_conditions
            )
        except Exception as e:
            logger.warning("Hybrid search failed: %s, fallback to basic search", e)
            return self.qdrant.search_sync(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=filter_conditions
            )
    def search(self, query_vector: List[float], query_text: str = "",
               user_id: str = None, limit: int = 20,
               query_context: Dict[str, Any] = None) -> Dict[str, Any]:
        query_context = query_context or {}
        contextual_filter = self._create_advanced_filter(query_context)
        results = []
        if user_id:
            private_results = self._search_private_hybrid(
                query_vector, query_text, user_id, limit,
                contextual_filter, query_context
            )
            results.extend(private_results)
        public_results = self._search_public_hybrid(
            query_vector, query_text, limit, contextual_filter, query_context
        )
        results.extend(public_results)
        merged_results = self._merge_results_advanced(results, limit, query_context)
        return {
            "results": merged_results,
            "total_found": len(merged_results),
            "search_type": "hybrid_advanced_qdrant",
            "search_metadata": {
                "used_hybrid_search": bool(query_text),
                "used_temporal_filter": bool(query_context.get("date_range")),
                "used_grouping": bool(query_context.get("group_by")),
                "used_boosting": bool(query_context.get("boost_factors"))
            }
        }
    def search_by_image(self, image_base64: str, user_id: str = None,
                       limit: int = 20, query_context: Dict[str, Any] = None) -> Dict[str, Any]:
        from backend.services.embedding_service import get_embedding_service
        import asyncio
        embedding_service = get_embedding_service()
        async def generate_image_embedding():
            return await embedding_service.generate_multimodal_embedding(
                image_base64=image_base64
            )
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, generate_image_embedding())
                    query_vector = future.result()
            else:
                query_vector = loop.run_until_complete(generate_image_embedding())
        except RuntimeError:
            query_vector = asyncio.run(generate_image_embedding())
        query_context = query_context or {}
        contextual_filter = self._create_contextual_filter(query_context)
        results = []
        if user_id:
            private_results = self._search_with_hybrid_ranking(
                f"private_experiments_{user_id}", query_vector, "", limit,
                contextual_filter
            )
            results.extend(private_results)
        public_results = self._search_with_hybrid_ranking(
            "public_science", query_vector, "", limit, contextual_filter
        )
        results.extend(public_results)
        merged_results = self._merge_results_advanced(results, limit)
        return {
            "results": merged_results,
            "total_found": len(merged_results),
            "search_type": "image_similarity"
        }
    def _search_private_hybrid(self, query_vector: List[float], query_text: str,
                               user_id: str, limit: int,
                               filter_conditions: Filter = None,
                               query_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        collection_name = f"private_experiments_{user_id}"
        query_context = query_context or {}
        try:
            if query_text and query_text.strip():
                results = self.qdrant.hybrid_search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    query_text=query_text,
                    limit=limit * 2,
                    query_filter=filter_conditions
                )
            else:
                results = self.qdrant.search_sync(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit * 2,
                    query_filter=filter_conditions
                )
            if query_context.get("boost_factors"):
                results = self._apply_boosting(results, query_context["boost_factors"])
            reranked_results = self._rerank_with_context(results, query_text, query_context)
            for result in reranked_results:
                result["source"] = "private"
            return reranked_results[:limit]
        except Exception as e:
            logger.error("Private hybrid search error: %s", e)
            return []
    def _search_public_hybrid(self, query_vector: List[float], query_text: str,
                             limit: int, filter_conditions: Filter = None,
                             query_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        query_context = query_context or {}
        try:
            if query_context.get("group_by"):
                grouped_results = self.qdrant.search_with_grouping(
                    collection_name="public_science",
                    query_vector=query_vector,
                    group_by=query_context["group_by"],
                    limit=limit,
                    group_size=query_context.get("group_size", 3),
                    query_filter=filter_conditions
                )
                results = []
                for group_results in grouped_results.values():
                    results.extend(group_results)
            elif query_context.get("date_range"):
                results = self.qdrant.search_temporal(
                    collection_name="public_science",
                    query_vector=query_vector,
                    date_range=query_context["date_range"],
                    limit=limit * 2,
                    query_filter=filter_conditions
                )
            elif query_text and query_text.strip():
                results = self.qdrant.hybrid_search(
                    collection_name="public_science",
                    query_vector=query_vector,
                    query_text=query_text,
                    limit=limit * 2,
                    query_filter=filter_conditions
                )
            else:
                results = self.qdrant.search_sync(
                    collection_name="public_science",
                    query_vector=query_vector,
                    limit=limit * 2,
                    query_filter=filter_conditions
                )
            if query_context.get("boost_factors"):
                results = self._apply_boosting(results, query_context["boost_factors"])
            reranked_results = self._rerank_with_context(results, query_text, query_context)
            for result in reranked_results:
                result["source"] = "public"
            return reranked_results[:limit]
        except Exception as e:
            logger.error("Public hybrid search error: %s", e)
            return []
    def _merge_results_advanced(self, all_results: List[Dict[str, Any]],
                               limit: int) -> List[Dict[str, Any]]:
        if not all_results:
            return []
        all_results.sort(key=lambda x: x["score"], reverse=True)
        seen_ids = set()
        deduplicated = []
        for result in all_results:
            result_id = result["id"]
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                deduplicated.append(result)
            elif result["source"] == "private":
                existing_idx = next((i for i, r in enumerate(deduplicated) if r["id"] == result_id), None)
                if existing_idx is not None and result["score"] > deduplicated[existing_idx]["score"]:
                    deduplicated[existing_idx] = result
        return deduplicated[:limit]
    def _create_advanced_filter(self, query_context: Dict[str, Any]) -> Filter:
        conditions = []
        if query_context.get("domain"):
            conditions.append(
                FieldCondition(
                    key="domain",
                    match=MatchValue(value=query_context["domain"])
                )
            )
        if query_context.get("experiment_type"):
            conditions.append(
                FieldCondition(
                    key="experiment_type",
                    match=MatchValue(value=query_context["experiment_type"])
                )
            )
        if query_context.get("organism"):
            conditions.append(
                FieldCondition(
                    key="organism",
                    match=MatchValue(value=query_context["organism"])
                )
            )
        if query_context.get("temperature_range"):
            temp_range = query_context["temperature_range"]
            conditions.append(
                FieldCondition(
                    key="temperature",
                    range=Range(
                        gte=temp_range.get("min"),
                        lte=temp_range.get("max")
                    )
                )
            )
        if query_context.get("ph_range"):
            ph_range = query_context["ph_range"]
            conditions.append(
                FieldCondition(
                    key="ph",
                    range=Range(
                        gte=ph_range.get("min"),
                        lte=ph_range.get("max")
                    )
                )
            )
        if query_context.get("min_confidence"):
            conditions.append(
                FieldCondition(
                    key="confidence_score",
                    range=Range(gte=query_context["min_confidence"])
                )
            )
        if query_context.get("success_only") is not None:
            conditions.append(
                FieldCondition(
                    key="success",
                    match=MatchValue(value=query_context["success_only"])
                )
            )
        if query_context.get("keywords"):
            keyword_conditions = []
            for keyword in query_context["keywords"]:
                keyword_conditions.extend([
                    FieldCondition(key="title", match=MatchText(text=keyword)),
                    FieldCondition(key="abstract", match=MatchText(text=keyword)),
                    FieldCondition(key="keywords", match=MatchText(text=keyword))
                ])
            if keyword_conditions:
                conditions.append(Filter(should=keyword_conditions))
        if query_context.get("recent_years"):
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=365 * query_context["recent_years"])
            conditions.append(
                FieldCondition(
                    key="publication_date",
                    range=Range(gte=cutoff_date.isoformat())
                )
            )
        if query_context.get("source_filter"):
            conditions.append(
                FieldCondition(
                    key="source",
                    match=MatchValue(value=query_context["source_filter"])
                )
            )
        return Filter(must=conditions) if conditions else None
    def _apply_boosting(self, results: List[Dict[str, Any]],
                       boost_factors: Dict[str, float]) -> List[Dict[str, Any]]:
        for result in results:
            payload = result["payload"]
            boost_score = 1.0
            for field, factor in boost_factors.items():
                if field in payload:
                    field_value = payload[field]
                    if isinstance(field_value, (int, float)):
                        boost_score *= (1.0 + field_value * factor)
                    elif isinstance(field_value, str):
                        important_keywords = ["novel", "breakthrough", "important", "significant"]
                        if any(keyword in field_value.lower() for keyword in important_keywords):
                            boost_score *= factor
                    elif isinstance(field_value, bool):
                        if field_value:
                            boost_score *= factor
            result["score"] *= boost_score
        return results
    def _rerank_with_context(self, results: List[Dict[str, Any]],
                           query_text: str, query_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not results:
            return results
        for result in results:
            payload = result["payload"]
            context_score = 0.0
            if query_text:
                text_relevance = self._calculate_text_relevance(query_text, payload)
                context_score += text_relevance * 0.3
            if query_context.get("prefer_recent"):
                pub_date = payload.get("publication_date")
                if pub_date:
                    try:
                        from datetime import datetime
                        days_old = (datetime.now() - datetime.fromisoformat(pub_date.replace('Z', '+00:00'))).days
                        recency_score = max(0, 1.0 - (days_old / 365.0))
                        context_score += recency_score * 0.2
                    except:
                        pass
            quality_score = payload.get("confidence_score", 0.5)
            context_score += quality_score * 0.2
            if payload.get("success"):
                context_score += 0.1
            if query_context.get("preferred_organism"):
                if payload.get("organism") == query_context["preferred_organism"]:
                    context_score += 0.2
            original_score = result["score"]
            result["score"] = 0.7 * original_score + 0.3 * context_score
            result["context_score"] = context_score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        embedding = context.get('query_embedding')
        conditions = context.get('query_conditions', {})
        limit = context.get('limit', 10)
        include_failures = context.get('include_failures', True)
        similarity_threshold = context.get('similarity_threshold', 0.0)
        private_neighbors = await self._search_private(
            embedding,
            conditions,
            limit,
            similarity_threshold
        )
        cloud_neighbors = await self._search_cloud(
            embedding,
            conditions,
            limit,
            similarity_threshold
        )
        all_neighbors = self._merge_results(
            private_neighbors,
            cloud_neighbors,
            limit
        )
        if not include_failures:
            all_neighbors = [
                n for n in all_neighbors
                if n['payload'].get('success', True)
            ]
        return {
            'neighbors': all_neighbors,
            'neighbor_count': len(all_neighbors),
            'private_count': len(private_neighbors),
            'cloud_count': len(cloud_neighbors)
        }
    async def _search_private(
        self,
        embedding: Any,
        conditions: Dict[str, Any],
        limit: int,
        threshold: float
    ) -> List[Dict]:
        try:
            metadata_filter = self.qdrant.build_metadata_filter(conditions)
            results = await self.qdrant.search(
                collection_name="private_experiments",
                query_vector=embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                limit=limit,
                query_filter=metadata_filter,
                with_payload=True,
                score_threshold=threshold if threshold > 0 else None
            )
            return results
        except Exception as e:
            logger.error(" Private search failed: %s", e)
            return []
    async def _search_cloud(
        self,
        embedding: Any,
        conditions: Dict[str, Any],
        limit: int,
        threshold: float
    ) -> List[Dict]:
        logger.debug("_search_cloud: cloud_client=%s", self.qdrant.cloud_client is not None)


        try:
            query_vector = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
            logger.debug("Cloud search: vector_dim=%d, limit=%d, threshold=%s", len(query_vector), limit, threshold)

            metadata_filter = self.qdrant.build_metadata_filter(conditions)
            results = await self.qdrant.search(
                collection_name="public_science",
                query_vector=query_vector,
                limit=limit,
                query_filter=metadata_filter,
                with_payload=True,
                score_threshold=threshold if threshold > 0 else None
            )
            logger.info("Cloud search returned %d results", len(results))
            return results
        except Exception as e:
            logger.error(" Cloud search failed: %s", e)
            
            
            return []
    def _merge_results(
        self,
        private: List[Dict],
        cloud: List[Dict],
        limit: int
    ) -> List[Dict]:
        for item in private:
            item['source_db'] = 'private'
        for item in cloud:
            item['source_db'] = 'cloud'
        all_results = private + cloud
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return all_results[:limit]
    def recommend_similar_experiments(self, experiment_ids: List[str],
                                    negative_ids: List[str] = None,
                                    limit: int = 10,
                                    query_vector: List[float] = None) -> List[Dict[str, Any]]:
        try:
            recommendations = self.qdrant.recommend(
                collection_name="private_experiments",
                positive_ids=experiment_ids,
                negative_ids=negative_ids or [],
                query_vector=query_vector,
                limit=limit,
                score_threshold=0.1
            )
            if recommendations:
                logger.info("Recommandations générées: %d expériences", len(recommendations))
                return recommendations
            logger.info("Fallback to public recommendations")
            return self.qdrant.recommend(
                collection_name="public_science",
                positive_ids=experiment_ids,
                negative_ids=negative_ids or [],
                query_vector=query_vector,
                limit=limit,
                score_threshold=0.1
            )
        except Exception as e:
            logger.error("Recommendations failed: %s", e)
            return []
    def search_by_criteria(self, criteria: Dict[str, Any], limit: int = 20) -> List[Dict[str, Any]]:
        try:
            query_parts = []
            if criteria.get("organism"):
                query_parts.append(f"expérience avec {criteria['organism']}")
            if criteria.get("method"):
                query_parts.append(f"méthode {criteria['method']}")
            if criteria.get("target"):
                query_parts.append(f"ciblant {criteria['target']}")
            if criteria.get("conditions"):
                cond = criteria["conditions"]
                if cond.get("temperature"):
                    query_parts.append(f"température {cond['temperature']}°C")
                if cond.get("ph"):
                    query_parts.append(f"pH {cond['ph']}")
            query_text = " ".join(query_parts) if query_parts else "expérience biologique"
            conditions = []
            if criteria.get("organism"):
                conditions.append(
                    FieldCondition(
                        key="conditions.organism",
                        match=MatchValue(value=criteria["organism"])
                    )
                )
            if criteria.get("conditions", {}).get("temperature"):
                temp = criteria["conditions"]["temperature"]
                conditions.append(
                    FieldCondition(
                        key="temperature",
                        range=Range(gte=temp - 2, lte=temp + 2)
                    )
                )
            if criteria.get("conditions", {}).get("ph"):
                ph = criteria["conditions"]["ph"]
                conditions.append(
                    FieldCondition(
                        key="ph",
                        range=Range(gte=ph - 0.5, lte=ph + 0.5)
                    )
                )
            filter_conditions = Filter(must=conditions) if conditions else None
            results = self._search_with_hybrid_ranking(
                collection_name="private_experiments",
                query_vector=[],
                query_text=query_text,
                limit=limit,
                filter_conditions=filter_conditions
            )
            return results
        except Exception as e:
            logger.error(" Recherche par critères failed: %s", e)
            return []





    async def discover_experiments(
        self,
        target_id: str,
        positive_context: List[str],
        negative_context: List[str] = None,
        limit: int = 10,
        user_id: str = None
    ) -> Dict[str, Any]:
        """
        Découverte contextuelle d'expériences avec Qdrant Discover API.
        Trouve des expériences similaires au target mais dans le contexte défini.
        """
        try:
            context_pairs = []
            for pos_id in positive_context:
                neg_id = negative_context[0] if negative_context else None
                if neg_id:
                    context_pairs.append(
                        ContextExamplePair(positive=pos_id, negative=neg_id)
                    )

            collection = f"private_experiments_{user_id}" if user_id else "public_science"

            results = await self.qdrant.discover(
                collection_name=collection,
                target=target_id,
                context=context_pairs,
                limit=limit
            )

            logger.info("Discover Qdrant: %d expériences découvertes", len(results))
            return {
                "discovered_experiments": results,
                "discovery_context": {
                    "target": target_id,
                    "positive_examples": positive_context,
                    "negative_examples": negative_context or []
                },
                "search_method": "qdrant_discover"
            }
        except Exception as e:
            logger.error(" Discover failed: %s", e)
            return {"discovered_experiments": [], "error": str(e)}

    async def batch_search_experiments(
        self,
        queries: List[Dict[str, Any]],
        limit: int = 10,
        user_id: str = None
    ) -> Dict[str, Any]:
        """
        Recherche batch pour plusieurs requêtes simultanées avec Qdrant Batch Search.
        """
        try:
            collection = f"private_experiments_{user_id}" if user_id else "public_science"

            batch_queries = []
            for query in queries:
                batch_queries.append({
                    "vector": query.get("vector", [0.0] * self.qdrant.vector_size),
                    "filter": self.qdrant.build_metadata_filter(query.get("conditions"))
                })

            results = await self.qdrant.batch_search(
                collection_name=collection,
                queries=batch_queries,
                limit=limit
            )

            logger.info("Batch Search Qdrant: %d ensembles de résultats", len(results))
            return {
                "batch_results": results,
                "queries_count": len(queries),
                "search_method": "qdrant_batch_search"
            }
        except Exception as e:
            logger.error(" Batch search failed: %s", e)
            return {"batch_results": [], "error": str(e)}

    async def search_with_grouping(
        self,
        query_vector: List[float],
        group_by: str = "organism",
        group_size: int = 3,
        limit: int = 10,
        query_filter: Optional[Filter] = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """
        Recherche groupée par champ (organisme, type d'expérience, etc.)
        """
        try:
            collection = f"private_experiments_{user_id}" if user_id else "public_science"

            grouped_results = await self.qdrant.search_with_grouping(
                collection_name=collection,
                query_vector=query_vector,
                group_by=group_by,
                group_size=group_size,
                limit=limit,
                query_filter=query_filter
            )

            total_results = sum(len(items) for items in grouped_results.values())
            logger.info("Grouped Search Qdrant: %d groupes, {total_results} résultats", len(grouped_results))

            return {
                "grouped_results": grouped_results,
                "groups_count": len(grouped_results),
                "total_results": total_results,
                "group_by": group_by,
                "search_method": "qdrant_grouped_search"
            }
        except Exception as e:
            logger.error(" Grouped search failed: %s", e)
            return {"grouped_results": {}, "error": str(e)}

    async def search_with_ordering(
        self,
        query_vector: List[float],
        order_by_field: str = "publication_date",
        order_direction: str = "desc",
        limit: int = 10,
        query_filter: Optional[Filter] = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """
        Recherche avec tri par champ spécifique.
        """
        try:
            collection = f"private_experiments_{user_id}" if user_id else "public_science"

            order_by = OrderBy(
                key=order_by_field,
                direction=order_direction
            )

            results = await self.qdrant.search_with_ordering(
                collection_name=collection,
                query_vector=query_vector,
                order_by=order_by,
                limit=limit,
                query_filter=query_filter
            )

            logger.info("Ordered Search Qdrant: %d résultats triés par {order_by_field}", len(results))
            return {
                "ordered_results": results,
                "order_by": order_by_field,
                "order_direction": order_direction,
                "search_method": "qdrant_ordered_search"
            }
        except Exception as e:
            logger.error(" Ordered search failed: %s", e)
            return {"ordered_results": [], "error": str(e)}

    async def search_with_qdrant_boosting(
        self,
        query_vector: List[float],
        boost_factors: Dict[str, float],
        limit: int = 10,
        query_filter: Optional[Filter] = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """
        Recherche avec boosting dynamique via Qdrant.
        """
        try:
            collection = f"private_experiments_{user_id}" if user_id else "public_science"

            results = await self.qdrant.search_with_boosting(
                collection_name=collection,
                query_vector=query_vector,
                boost_factors=boost_factors,
                limit=limit,
                query_filter=query_filter
            )

            logger.info("Boosted Search Qdrant: %d résultats avec boosting", len(results))
            return {
                "boosted_results": results,
                "boost_factors_applied": boost_factors,
                "search_method": "qdrant_boosted_search"
            }
        except Exception as e:
            logger.error(" Boosted search failed: %s", e)
            return {"boosted_results": [], "error": str(e)}

    async def aggregate_experiments(
        self,
        group_by: str = "organism",
        query_filter: Optional[Filter] = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """
        Agrégation et statistiques sur les expériences.
        """
        try:
            collection = f"private_experiments_{user_id}" if user_id else "public_science"

            aggregation = await self.qdrant.aggregate(
                collection_name=collection,
                group_by=group_by,
                query_filter=query_filter
            )

            logger.info("Aggregate Qdrant: %d groupes agrégés", len(aggregation))
            return {
                "aggregation": aggregation,
                "group_by": group_by,
                "total_groups": len(aggregation),
                "search_method": "qdrant_aggregate"
            }
        except Exception as e:
            logger.error(" Aggregation failed: %s", e)
            return {"aggregation": {}, "error": str(e)}

    async def search_temporal_advanced(
        self,
        query_vector: List[float],
        start_date: str = None,
        end_date: str = None,
        limit: int = 10,
        query_filter: Optional[Filter] = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """
        Recherche temporelle avancée avec plage de dates.
        """
        try:
            collection = f"private_experiments_{user_id}" if user_id else "public_science"

            date_range = {}
            if start_date:
                date_range["start_date"] = start_date
            if end_date:
                date_range["end_date"] = end_date

            if not date_range:
                cutoff = datetime.now() - timedelta(days=365)
                date_range["start_date"] = cutoff.isoformat()

            results = await self.qdrant.search_temporal_advanced(
                collection_name=collection,
                query_vector=query_vector,
                date_range=date_range,
                limit=limit,
                query_filter=query_filter
            )

            logger.info("Temporal Advanced Search Qdrant: %d résultats", len(results))
            return {
                "temporal_results": results,
                "date_range": date_range,
                "search_method": "qdrant_temporal_advanced"
            }
        except Exception as e:
            logger.error(" Temporal advanced search failed: %s", e)
            return {"temporal_results": [], "error": str(e)}

    async def get_experiment_by_id(
        self,
        experiment_id: str,
        collection_name: str = "public_science"
    ) -> Optional[Dict[str, Any]]:
        """
        Récupération d'une expérience par ID avec Qdrant Retrieve.
        """
        try:
            result = await self.qdrant.retrieve(
                collection_name=collection_name,
                point_id=experiment_id
            )

            if result:
                logger.info("Retrieve Qdrant: expérience {experiment_id} found")
            return result
        except Exception as e:
            logger.error(" Retrieve failed: %s", e)
            return None

    async def scroll_experiments(
        self,
        limit: int = 20,
        offset: int = 0,
        collection_name: str = "public_science"
    ) -> Dict[str, Any]:
        """
        Pagination des expériences avec Qdrant Scroll.
        """
        try:
            results = await self.qdrant.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset
            )

            logger.info("Scroll Qdrant: %d expériences (offset={offset})", len(results))
            return {
                "experiments": results,
                "offset": offset,
                "limit": limit,
                "count": len(results),
                "search_method": "qdrant_scroll"
            }
        except Exception as e:
            logger.error(" Scroll failed: %s", e)
            return {"experiments": [], "error": str(e)}

    async def delete_experiments(
        self,
        experiment_ids: List[str],
        collection_name: str = "private_experiments"
    ) -> Dict[str, Any]:
        """
        Suppression d'expériences avec Qdrant Delete.
        """
        try:
            await self.qdrant.delete(
                collection_name=collection_name,
                points_selector=experiment_ids
            )

            logger.info("Delete Qdrant: %d expériences supprimées", len(experiment_ids))
            return {
                "deleted_ids": experiment_ids,
                "count": len(experiment_ids),
                "success": True
            }
        except Exception as e:
            logger.error(" Delete failed: %s", e)
            return {"deleted_ids": [], "success": False, "error": str(e)}

    async def count_experiments(
        self,
        query_filter: Optional[Filter] = None,
        collection_name: str = "public_science"
    ) -> Dict[str, Any]:
        """
        Comptage des expériences avec Qdrant Count.
        """
        try:
            count = await self.qdrant.count_points(
                collection_name=collection_name,
                query_filter=query_filter,
                exact=True
            )

            logger.info("Count Qdrant: %d experiments", count)
            return {
                "count": count,
                "collection": collection_name,
                "filtered": query_filter is not None
            }
        except Exception as e:
            logger.error(" Count failed: %s", e)
            return {"count": 0, "error": str(e)}

    async def search_by_multiple_organisms(
        self,
        query_vector: List[float],
        organisms: List[str],
        limit: int = 10,
        user_id: str = None
    ) -> Dict[str, Any]:
        """
        Recherche filtrée par plusieurs organismes avec MatchAny.
        """
        try:
            collection = f"private_experiments_{user_id}" if user_id else "public_science"

            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="organism",
                        match=MatchAny(any=organisms)
                    )
                ]
            )

            results = await self.qdrant.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter
            )

            logger.info("Multi-organism Search: %d résultats pour {organisms}", len(results))
            return {
                "results": results,
                "organisms_filter": organisms,
                "search_method": "qdrant_match_any"
            }
        except Exception as e:
            logger.error(" Multi-organism search failed: %s", e)
            return {"results": [], "error": str(e)}

    async def search_exclude_ids(
        self,
        query_vector: List[float],
        exclude_ids: List[str],
        limit: int = 10,
        user_id: str = None
    ) -> Dict[str, Any]:
        """
        Recherche en excluant certains IDs avec HasIdCondition.
        """
        try:
            collection = f"private_experiments_{user_id}" if user_id else "public_science"

            query_filter = Filter(
                must_not=[
                    HasIdCondition(has_id=exclude_ids)
                ]
            )

            results = await self.qdrant.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter
            )

            logger.info("Exclude IDs Search: %d résultats (exclu {len(exclude_ids)} IDs)", len(results))
            return {
                "results": results,
                "excluded_ids": exclude_ids,
                "search_method": "qdrant_exclude_ids"
            }
        except Exception as e:
            logger.error(" Exclude IDs search failed: %s", e)
            return {"results": [], "error": str(e)}

    async def search_with_empty_field_filter(
        self,
        query_vector: List[float],
        field_name: str,
        exclude_empty: bool = True,
        limit: int = 10,
        user_id: str = None
    ) -> Dict[str, Any]:
        """
        Recherche en filtrant les champs vides/null avec IsEmptyCondition/IsNullCondition.
        """
        try:
            collection = f"private_experiments_{user_id}" if user_id else "public_science"

            if exclude_empty:
                query_filter = Filter(
                    must_not=[
                        IsEmptyCondition(is_empty={"key": field_name}),
                        IsNullCondition(is_null={"key": field_name})
                    ]
                )
            else:
                query_filter = Filter(
                    should=[
                        IsEmptyCondition(is_empty={"key": field_name}),
                        IsNullCondition(is_null={"key": field_name})
                    ]
                )

            results = await self.qdrant.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter
            )

            action = "excluant" if exclude_empty else "incluant seulement"
            logger.info("Empty Field Filter: %d résultats {action} {field_name} vide", len(results))
            return {
                "results": results,
                "field_filter": field_name,
                "exclude_empty": exclude_empty,
                "search_method": "qdrant_empty_field_filter"
            }
        except Exception as e:
            logger.error(" Empty field filter search failed: %s", e)
            return {"results": [], "error": str(e)}

    async def full_qdrant_search(
        self,
        query_vector: List[float],
        query_text: str = "",
        conditions: Dict[str, Any] = None,
        user_id: str = None,
        limit: int = 20,
        use_hybrid: bool = True,
        use_grouping: bool = False,
        group_by: str = "organism",
        use_temporal: bool = False,
        date_range: Dict[str, str] = None,
        boost_factors: Dict[str, float] = None,
        order_by: str = None
    ) -> Dict[str, Any]:
        """
        Recherche complète utilisant toutes les fonctionnalités Qdrant disponibles.
        """
        try:
            collection = f"private_experiments_{user_id}" if user_id else "public_science"
            query_filter = self.qdrant.build_metadata_filter(conditions)

            search_metadata = {
                "collection": collection,
                "features_used": [],
                "search_strategy": "full_qdrant"
            }


            if use_grouping:
                grouped = await self.search_with_grouping(
                    query_vector=query_vector,
                    group_by=group_by,
                    limit=limit,
                    query_filter=query_filter,
                    user_id=user_id
                )
                search_metadata["features_used"].append("grouping")
                search_metadata["grouped_results"] = grouped


            if use_temporal and date_range:
                temporal = await self.search_temporal_advanced(
                    query_vector=query_vector,
                    start_date=date_range.get("start"),
                    end_date=date_range.get("end"),
                    limit=limit,
                    query_filter=query_filter,
                    user_id=user_id
                )
                search_metadata["features_used"].append("temporal")
                search_metadata["temporal_results"] = temporal


            if use_hybrid and query_text:
                results = await self.qdrant.hybrid_search(
                    collection_name=collection,
                    query_vector=query_vector,
                    query_text=query_text,
                    limit=limit * 2,
                    query_filter=query_filter,
                    fusion=Fusion.RRF
                )
                search_metadata["features_used"].append("hybrid_rrf")
            else:
                results = await self.qdrant.search(
                    collection_name=collection,
                    query_vector=query_vector,
                    limit=limit * 2,
                    query_filter=query_filter
                )
                search_metadata["features_used"].append("vector_search")


            if boost_factors:
                boosted = await self.search_with_qdrant_boosting(
                    query_vector=query_vector,
                    boost_factors=boost_factors,
                    limit=limit,
                    query_filter=query_filter,
                    user_id=user_id
                )
                results = boosted.get("boosted_results", results)
                search_metadata["features_used"].append("boosting")


            if order_by:
                ordered = await self.search_with_ordering(
                    query_vector=query_vector,
                    order_by_field=order_by,
                    limit=limit,
                    query_filter=query_filter,
                    user_id=user_id
                )
                results = ordered.get("ordered_results", results)
                search_metadata["features_used"].append("ordering")


            if results:
                top_ids = [r.get("id") for r in results[:3] if r.get("id")]
                if top_ids:
                    recommendations = await self.qdrant.recommend(
                        collection_name=collection,
                        positive_ids=top_ids,
                        limit=5
                    )
                    search_metadata["recommendations"] = recommendations
                    search_metadata["features_used"].append("recommendations")


            aggregation = await self.aggregate_experiments(
                group_by="organism",
                user_id=user_id
            )
            search_metadata["aggregation_stats"] = aggregation
            search_metadata["features_used"].append("aggregation")

            logger.info("Full Qdrant Search: %d résultats avec {len(search_metadata['features_used'])} fonctionnalités", len(results))

            return {
                "results": results[:limit],
                "total_found": len(results),
                "search_metadata": search_metadata
            }
        except Exception as e:
            logger.error(" Full Qdrant search failed: %s", e)
            return {"results": [], "error": str(e)}