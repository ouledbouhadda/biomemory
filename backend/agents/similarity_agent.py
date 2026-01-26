from typing import Dict, Any, List
from backend.services.qdrant_service import get_qdrant_service
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range, MatchText
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
                print(f"✅ Recherche hybride réussie: {len(hybrid_results)} résultats")
                return hybrid_results
            print("ℹ️ Fallback vers recherche vectorielle seule")
            return self.qdrant.search_sync(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=filter_conditions
            )
        except Exception as e:
            print(f"⚠️ Recherche hybride échouée: {e}, fallback vers recherche basique")
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
            "public_experiments", query_vector, "", limit, contextual_filter
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
            print(f"Erreur recherche privée hybride: {e}")
            return []
    def _search_public_hybrid(self, query_vector: List[float], query_text: str,
                             limit: int, filter_conditions: Filter = None,
                             query_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        query_context = query_context or {}
        try:
            if query_context.get("group_by"):
                grouped_results = self.qdrant.search_with_grouping(
                    collection_name="public_experiments",
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
                    collection_name="public_experiments",
                    query_vector=query_vector,
                    date_range=query_context["date_range"],
                    limit=limit * 2,
                    query_filter=filter_conditions
                )
            elif query_text and query_text.strip():
                results = self.qdrant.hybrid_search(
                    collection_name="public_experiments",
                    query_vector=query_vector,
                    query_text=query_text,
                    limit=limit * 2,
                    query_filter=filter_conditions
                )
            else:
                results = self.qdrant.search_sync(
                    collection_name="public_experiments",
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
            print(f"Erreur recherche publique hybride: {e}")
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
            print(f"⚠️ Private search failed: {e}")
            return []
    async def _search_cloud(
        self,
        embedding: Any,
        conditions: Dict[str, Any],
        limit: int,
        threshold: float
    ) -> List[Dict]:
        if not self.qdrant.cloud_client:
            return []
        try:
            metadata_filter = self.qdrant.build_metadata_filter(conditions)
            results = await self.qdrant.search(
                collection_name="public_science",
                query_vector=embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                limit=limit,
                query_filter=metadata_filter,
                with_payload=True,
                score_threshold=threshold if threshold > 0 else None
            )
            return results
        except Exception as e:
            print(f"⚠️ Cloud search failed: {e}")
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
                print(f"✅ Recommandations générées: {len(recommendations)} expériences")
                return recommendations
            print("ℹ️ Fallback vers recommandations dans données publiques")
            return self.qdrant.recommend(
                collection_name="public_science",
                positive_ids=experiment_ids,
                negative_ids=negative_ids or [],
                query_vector=query_vector,
                limit=limit,
                score_threshold=0.1
            )
        except Exception as e:
            print(f"⚠️ Recommandations échouées: {e}")
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
            print(f"⚠️ Recherche par critères échouée: {e}")
            return []