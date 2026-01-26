from typing import Dict, Any, List
from backend.services.embedding_service import get_embedding_service
class MultimodalSearchAgent:
    def __init__(self):
        self.embedding_service = get_embedding_service()
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        user_input = context.get('user_input', {})
        query = user_input.get('query', user_input.get('experiment', {}))
        modalities = self._detect_modalities(query)
        embedding = await self._generate_adaptive_embedding(query, modalities)
        search_strategy = self._determine_advanced_search_strategy(modalities, query)
        search_context = {
            'query_embedding': embedding,
            'modalities_used': modalities,
            'search_strategy': search_strategy,
            'query_conditions': query.get('conditions', {}),
            'include_failures': query.get('include_failures', True),
            'similarity_threshold': query.get('similarity_threshold', 0.0),
            'limit': query.get('limit', 10),
            'boost_factors': self._calculate_boost_factors(modalities, query),
            'group_by': query.get('group_by'),
            'group_size': query.get('group_size', 3),
            'date_range': query.get('date_range'),
            'prefer_recent': query.get('prefer_recent', False),
            'min_confidence': query.get('min_confidence', 0.0),
            'keywords': query.get('keywords', []),
            'temperature_range': query.get('temperature_range'),
            'ph_range': query.get('ph_range'),
            'success_only': query.get('success_only'),
            'source_filter': query.get('source_filter')
        }
        return search_context
    def _detect_modalities(self, query: Dict[str, Any]) -> Dict[str, bool]:
        return {
            'has_text': bool(query.get('text')),
            'has_sequence': bool(query.get('sequence')),
            'has_image': bool(query.get('image_base64')),
            'has_conditions': bool(query.get('conditions'))
        }
    async def _generate_adaptive_embedding(
        self,
        query: Dict[str, Any],
        modalities: Dict[str, bool]
    ) -> Any:
        if modalities['has_text'] and not modalities['has_sequence'] and not modalities['has_image']:
            return await self.embedding_service.generate_multimodal_embedding(
                text=query.get('text'),
                conditions=query.get('conditions')
            )
        elif modalities['has_image'] and not modalities['has_text'] and not modalities['has_sequence']:
            return await self.embedding_service.generate_multimodal_embedding(
                image_base64=query.get('image_base64'),
                conditions=query.get('conditions')
            )
        elif modalities['has_text'] and modalities['has_sequence']:
            return await self.embedding_service.generate_multimodal_embedding(
                text=query.get('text'),
                sequence=query.get('sequence'),
                conditions=query.get('conditions')
            )
        elif modalities['has_text'] and modalities['has_image']:
            return await self.embedding_service.generate_multimodal_embedding(
                text=query.get('text'),
                image_base64=query.get('image_base64'),
                conditions=query.get('conditions')
            )
        elif modalities['has_image'] and modalities['has_sequence']:
            return await self.embedding_service.generate_multimodal_embedding(
                sequence=query.get('sequence'),
                image_base64=query.get('image_base64'),
                conditions=query.get('conditions')
            )
        else:
            return await self.embedding_service.generate_multimodal_embedding(
                text=query.get('text'),
                sequence=query.get('sequence'),
                conditions=query.get('conditions'),
                image_base64=query.get('image_base64')
            )
    def _determine_search_strategy(self, modalities: Dict[str, bool]) -> str:
        modal_count = sum(1 for v in modalities.values() if v)
        if modal_count == 1:
            if modalities['has_text']:
                return "text_similarity"
            elif modalities['has_sequence']:
                return "sequence_similarity"
            elif modalities['has_image']:
                return "image_similarity"
            elif modalities['has_conditions']:
                return "conditions_filtering"
        elif modal_count == 2:
            if modalities['has_text'] and modalities['has_sequence']:
                return "text_sequence_hybrid"
            elif modalities['has_text'] and modalities['has_image']:
                return "text_image_hybrid"
            elif modalities['has_sequence'] and modalities['has_image']:
                return "sequence_image_hybrid"
            elif modalities['has_text'] and modalities['has_conditions']:
                return "text_conditions_filter"
            elif modalities['has_image'] and modalities['has_conditions']:
                return "image_conditions_filter"
            else:
                return "bimodal_fusion"
        elif modal_count == 3:
            return "trimodal_fusion"
        else:
            return "multimodal_complete"
    def _analyze_text_quality(self, text: str) -> Dict[str, Any]:
        if not text:
            return {"score": 0, "length": 0, "has_keywords": False}
        words = text.split()
        return {
            "score": min(len(words) / 10, 1.0),
            "length": len(words),
            "has_keywords": any(word.lower() in ["expression", "protein", "gene", "cell", "assay"]
                              for word in words)
        }
    def _extract_keywords(self, text: str) -> List[str]:
        if not text:
            return []
        bio_keywords = [
            "expression", "protein", "gene", "cell", "assay", "pcr", "western",
            "transfection", "transduction", "cloning", "sequencing", "microscopy",
            "flow cytometry", "mass spectrometry", "crispr", "rna", "dna"
        ]
        words = text.lower().split()
        return [word for word in words if word in bio_keywords]
    def _detect_sequence_type(self, sequence: str) -> str:
        if not sequence:
            return "unknown"
        bases_dna = set('ATGC')
        bases_rna = set('AUGC')
        amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
        sequence_set = set(sequence.upper())
        if sequence_set.issubset(bases_dna):
            return "dna"
        elif sequence_set.issubset(bases_rna):
            return "rna"
        elif sequence_set.issubset(amino_acids):
            return "protein"
        else:
            return "mixed"
    def _detect_image_format(self, image_base64: str) -> str:
        if not image_base64:
            return "unknown"
        if image_base64.startswith('data:image/'):
            format_part = image_base64.split(';')[0]
            return format_part.split('/')[1]
        return "unknown"
    def _analyze_conditions_completeness(self, conditions: Dict[str, Any]) -> float:
        required_fields = ['organism', 'temperature', 'ph']
        provided_fields = [field for field in required_fields if field in conditions]
        return len(provided_fields) / len(required_fields)
    def _calculate_boost_factors(self, modalities: Dict[str, Any], query: Dict[str, Any]) -> Dict[str, float]:
        boost_factors = {}
        if query.get('prefer_successful'):
            boost_factors['success'] = 1.5
        boost_factors['confidence_score'] = 0.3
        if query.get('prefer_recent'):
            boost_factors['recency'] = 0.2
        return boost_factors
    def _determine_advanced_search_strategy(self, modalities: Dict[str, Any], query: Dict[str, Any]) -> str:
        modality_count = modalities.get('modality_count', 0)
        if modalities.get('has_image') and not modalities.get('has_text') and not modalities.get('has_sequence'):
            return "image_similarity_advanced"
        if query.get('group_by'):
            return f"grouped_{query['group_by']}_search"
        if query.get('date_range') or query.get('prefer_recent'):
            return "temporal_hybrid_search"
        if query.get('boost_factors') or query.get('prefer_successful'):
            return "boosted_hybrid_search"
        if modality_count >= 2:
            return "multimodal_hybrid_fusion"
        if modalities.get('has_text') and modalities.get('text_quality', {}).get('has_keywords'):
            return "keyword_enhanced_search"
        if modalities.get('has_sequence'):
            sequence_type = modalities.get('sequence_type', 'unknown')
            return f"{sequence_type}_sequence_search"
        return "adaptive_similarity_search"