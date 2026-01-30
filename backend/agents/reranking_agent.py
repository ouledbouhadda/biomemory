import logging
from typing import Dict, Any, List
import numpy as np
from datetime import datetime, timedelta
from backend.services.qdrant_service import get_qdrant_service

logger = logging.getLogger("biomemory.reranking")


class RerankingAgent:
    def __init__(self):
        self.qdrant = get_qdrant_service()

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        neighbors = context.get('neighbors', [])
        query_conditions = context.get('query_conditions', {})
        user_id = context.get('user_id')
        search_query = context.get('search_query', {})
        modalities_used = context.get('modalities_used', {})

        if not neighbors:
            return {'reranked_neighbors': []}

        # Dynamic score threshold: filter out very low scores
        neighbors = self._apply_dynamic_threshold(neighbors)

        reranked = await self._multi_stage_reranking(
            neighbors, query_conditions, user_id, search_query, modalities_used
        )
        enriched_results = await self._enrich_with_recommendations(
            reranked, user_id
        )
        temporal_boosted = await self._apply_temporal_boosting(
            enriched_results, query_conditions
        )
        final_reranked = self._apply_rrf_fusion(temporal_boosted)

        logger.info("Reranking: %d -> %d results", len(neighbors), len(final_reranked))
        return {
            'reranked_neighbors': final_reranked,
            'reranking_strategy': 'advanced_qdrant',
            'stages_applied': ['dynamic_threshold', 'multi_stage', 'recommendations', 'temporal', 'rrf_fusion']
        }

    def _apply_dynamic_threshold(self, neighbors: List[Dict]) -> List[Dict]:
        """Filter out results with very low similarity scores dynamically."""
        if len(neighbors) <= 3:
            return neighbors

        scores = [n.get('score', 0.0) for n in neighbors]
        if not scores:
            return neighbors

        max_score = max(scores)
        mean_score = sum(scores) / len(scores)

        # Dynamic threshold: at least 30% of max score, or mean - 1 std dev
        std_dev = np.std(scores)
        threshold = max(
            max_score * 0.3,
            mean_score - std_dev,
            0.1  # absolute minimum
        )

        filtered = [n for n in neighbors if n.get('score', 0.0) >= threshold]
        if len(filtered) < 3:
            # Always keep at least top 3
            filtered = sorted(neighbors, key=lambda x: x.get('score', 0), reverse=True)[:3]

        if len(filtered) < len(neighbors):
            logger.debug("Dynamic threshold %.3f: %d -> %d results", threshold, len(neighbors), len(filtered))

        return filtered

    async def _multi_stage_reranking(
        self,
        neighbors: List[Dict],
        query_conditions: Dict,
        user_id: str = None,
        search_query: Dict = None,
        modalities_used: Dict = None
    ) -> List[Dict]:
        # Adapt weights based on which modalities were used in the query
        weights = self._compute_adaptive_weights(modalities_used or {})

        scored_neighbors = []
        for idx, neighbor in enumerate(neighbors):
            composite_score = self._compute_advanced_composite_score(
                neighbor, query_conditions, idx, len(neighbors), weights
            )
            neighbor_copy = dict(neighbor)
            neighbor_copy['composite_score'] = composite_score
            neighbor_copy['stage_1_score'] = composite_score
            scored_neighbors.append(neighbor_copy)

        quality_boosted = self._apply_quality_boosting(scored_neighbors)
        diversified = self._apply_diversity_filtering(quality_boosted)
        return diversified

    def _compute_adaptive_weights(self, modalities_used: Dict) -> Dict[str, float]:
        """Adapt reranking weights based on which modalities the user provided."""
        has_text = modalities_used.get('has_text', False)
        has_sequence = modalities_used.get('has_sequence', False)
        has_conditions = modalities_used.get('has_conditions', False)
        has_image = modalities_used.get('has_image', False)

        # Default weights
        weights = {
            'similarity': 0.3,
            'condition_match': 0.2,
            'success_weight': 0.15,
            'quality_score': 0.15,
            'diversity': 0.1,
            'reproducibility': 0.1
        }

        if has_text and not has_conditions and not has_sequence:
            # Text-only: boost similarity, reduce condition_match
            weights['similarity'] = 0.45
            weights['condition_match'] = 0.05
            weights['quality_score'] = 0.20
            weights['success_weight'] = 0.15
            weights['diversity'] = 0.10
            weights['reproducibility'] = 0.05
        elif has_conditions and not has_text:
            # Conditions-focused: boost condition matching
            weights['similarity'] = 0.15
            weights['condition_match'] = 0.40
            weights['success_weight'] = 0.15
            weights['quality_score'] = 0.10
            weights['diversity'] = 0.10
            weights['reproducibility'] = 0.10
        elif has_sequence:
            # Sequence present: boost similarity (sequence similarity is critical)
            weights['similarity'] = 0.40
            weights['condition_match'] = 0.15
            weights['success_weight'] = 0.15
            weights['quality_score'] = 0.10
            weights['diversity'] = 0.10
            weights['reproducibility'] = 0.10

        return weights

    async def _enrich_with_recommendations(
        self,
        neighbors: List[Dict],
        user_id: str = None
    ) -> List[Dict]:
        enriched = []
        for neighbor in neighbors:
            neighbor_copy = dict(neighbor)
            try:
                point_id = neighbor.get('id')
                if point_id:
                    recommendations = await self.qdrant.recommend(
                        collection_name="private_experiments" if user_id else "public_science",
                        positive_ids=[point_id],
                        limit=3,
                        score_threshold=0.7
                    )
                    rec_list = recommendations if isinstance(recommendations, list) else recommendations.get('result', [])
                    neighbor_copy['recommendations'] = [
                        rec.get('id') for rec in rec_list
                    ]
                    neighbor_copy['recommendation_score'] = len(neighbor_copy['recommendations']) * 0.1
                else:
                    neighbor_copy['recommendations'] = []
                    neighbor_copy['recommendation_score'] = 0.0
            except Exception:
                neighbor_copy['recommendations'] = []
                neighbor_copy['recommendation_score'] = 0.0
            enriched.append(neighbor_copy)
        return enriched

    async def _apply_temporal_boosting(
        self,
        neighbors: List[Dict],
        query_conditions: Dict
    ) -> List[Dict]:
        boosted = []
        recent_threshold = datetime.now() - timedelta(days=365)
        for neighbor in neighbors:
            neighbor_copy = dict(neighbor)
            indexed_at = neighbor_copy.get('payload', {}).get('indexed_at')
            if indexed_at:
                try:
                    indexed_date = datetime.fromisoformat(indexed_at.replace('Z', '+00:00'))
                    if indexed_date > recent_threshold:
                        neighbor_copy['temporal_boost'] = 0.2
                    else:
                        neighbor_copy['temporal_boost'] = 0.0
                except Exception:
                    neighbor_copy['temporal_boost'] = 0.0
            else:
                neighbor_copy['temporal_boost'] = 0.0

            temporal_score = await self._temporal_similarity_search(
                neighbor_copy, query_conditions
            )
            neighbor_copy['temporal_similarity'] = temporal_score
            boosted.append(neighbor_copy)
        return boosted

    def _apply_rrf_fusion(self, neighbors: List[Dict]) -> List[Dict]:
        for idx, neighbor in enumerate(neighbors):
            neighbor['rrf_components'] = {
                'composite_rank': idx + 1,
                'quality_rank': self._calculate_quality_rank(neighbor),
                'temporal_rank': self._calculate_temporal_rank(neighbor),
                'recommendation_rank': self._calculate_recommendation_rank(neighbor)
            }
        k = 60
        for neighbor in neighbors:
            rrf_score = 0.0
            components = neighbor['rrf_components']
            for rank in components.values():
                if rank > 0:
                    rrf_score += 1.0 / (k + rank)
            neighbor['rrf_score'] = rrf_score
            neighbor['final_score'] = rrf_score
        neighbors.sort(key=lambda x: x['rrf_score'], reverse=True)
        return neighbors

    def _compute_advanced_composite_score(
        self,
        neighbor: Dict,
        query_conditions: Dict,
        position: int,
        total: int,
        weights: Dict[str, float] = None
    ) -> float:
        if weights is None:
            weights = {
                'similarity': 0.3,
                'condition_match': 0.2,
                'success_weight': 0.15,
                'quality_score': 0.15,
                'diversity': 0.1,
                'reproducibility': 0.1
            }

        similarity = neighbor.get('score', 0.0)
        condition_match = self._advanced_condition_similarity(
            neighbor.get('payload', {}).get('conditions', {}),
            query_conditions
        )
        success = neighbor.get('payload', {}).get('success', False)
        success_weight = 1.0 if success else 0.3
        quality_score = neighbor.get('payload', {}).get('data_quality_score', 0.5)
        diversity_score = 1.0 - (position / max(total, 1))
        repro_score = self._estimate_reproducibility(neighbor)

        score = (
            weights['similarity'] * similarity +
            weights['condition_match'] * condition_match +
            weights['success_weight'] * success_weight +
            weights['quality_score'] * quality_score +
            weights['diversity'] * diversity_score +
            weights['reproducibility'] * repro_score
        )
        return max(0.0, min(1.0, score))

    def _advanced_condition_similarity(
        self,
        cond1: Dict[str, Any],
        cond2: Dict[str, Any]
    ) -> float:
        if not cond1 or not cond2:
            return 0.3
        score = 0.0
        total_weight = 0.0

        org1 = cond1.get('organism', '').lower()
        org2 = cond2.get('organism', '').lower()
        if org1 and org2:
            if org1 == org2:
                score += 0.4
            elif org1 in org2 or org2 in org1:
                score += 0.2
            total_weight += 0.4

        temp1 = cond1.get('temperature')
        temp2 = cond2.get('temperature')
        if temp1 is not None and temp2 is not None:
            try:
                temp1 = float(temp1) if not isinstance(temp1, (int, float)) else temp1
                temp2 = float(temp2) if not isinstance(temp2, (int, float)) else temp2
                temp_diff = abs(temp1 - temp2)
                tolerance = max(5.0, abs(temp1) * 0.1)
                temp_similarity = max(0, 1 - temp_diff / tolerance)
                score += 0.3 * temp_similarity
                total_weight += 0.3
            except (ValueError, TypeError):
                pass

        ph1 = cond1.get('ph')
        ph2 = cond2.get('ph')
        if ph1 is not None and ph2 is not None:
            try:
                ph1 = float(ph1) if not isinstance(ph1, (int, float)) else ph1
                ph2 = float(ph2) if not isinstance(ph2, (int, float)) else ph2
                ph_diff = abs(ph1 - ph2)
                ph_similarity = max(0, 1 - ph_diff / 1.0)
                score += 0.3 * ph_similarity
                total_weight += 0.3
            except (ValueError, TypeError):
                pass

        return score / total_weight if total_weight > 0 else 0.0

    def _apply_quality_boosting(self, neighbors: List[Dict]) -> List[Dict]:
        for neighbor in neighbors:
            quality_score = neighbor.get('payload', {}).get('data_quality_score', 0.5)
            completeness = neighbor.get('payload', {}).get('completeness_score', 0.5)
            quality_boost = 1.0 + (quality_score - 0.5) * 0.4
            completeness_boost = 1.0 + (completeness - 0.5) * 0.2
            neighbor['quality_boost'] = quality_boost * completeness_boost
            neighbor['composite_score'] *= neighbor['quality_boost']
        return neighbors

    def _apply_diversity_filtering(self, neighbors: List[Dict]) -> List[Dict]:
        if len(neighbors) <= 5:
            return neighbors
        by_organism = {}
        for neighbor in neighbors:
            organism = neighbor.get('payload', {}).get('conditions', {}).get('organism', 'unknown')
            if organism not in by_organism:
                by_organism[organism] = []
            by_organism[organism].append(neighbor)
        diversified = []
        for organism_neighbors in by_organism.values():
            organism_neighbors.sort(key=lambda x: x['composite_score'], reverse=True)
            diversified.extend(organism_neighbors[:3])
        diversified.sort(key=lambda x: x['composite_score'], reverse=True)
        return diversified[:len(neighbors)]

    async def _temporal_similarity_search(
        self,
        neighbor: Dict,
        query_conditions: Dict
    ) -> float:
        try:
            recent_date = (datetime.now() - timedelta(days=180)).isoformat()
            temporal_results = await self.qdrant.search_temporal(
                collection_name="public_science",
                query_vector=neighbor.get('vector', []),
                date_range={"start": recent_date},
                limit=5,
                query_filter=None
            )
            result_list = temporal_results if isinstance(temporal_results, list) else temporal_results.get('result', [])
            temporal_score = min(len(result_list), 5) / 5.0
            return temporal_score
        except Exception:
            return 0.0

    def _estimate_reproducibility(self, neighbor: Dict) -> float:
        payload = neighbor.get('payload', {})
        score = 0.5
        if payload.get('completeness_score', 0) > 0.7:
            score += 0.2
        conditions = payload.get('conditions', {})
        if conditions.get('organism') and conditions.get('organism') != 'unknown':
            score += 0.1
        if conditions.get('temperature') is not None:
            score += 0.1
        if conditions.get('ph') is not None:
            score += 0.1
        if payload.get('success'):
            score += 0.1
        return min(score, 1.0)

    def _calculate_quality_rank(self, neighbor: Dict) -> int:
        quality = neighbor.get('payload', {}).get('data_quality_score', 0.5)
        return int((1.0 - quality) * 100) + 1

    def _calculate_temporal_rank(self, neighbor: Dict) -> int:
        temporal_boost = neighbor.get('temporal_boost', 0.0)
        return int((1.0 - temporal_boost) * 50) + 1

    def _calculate_recommendation_rank(self, neighbor: Dict) -> int:
        rec_score = neighbor.get('recommendation_score', 0.0)
        return int((1.0 - rec_score) * 30) + 1
