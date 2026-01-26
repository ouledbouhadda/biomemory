from typing import Dict, Any, List
from backend.services.chunking_service import get_chunking_service
class ChunkingAgent:
    def __init__(self):
        self.chunking_service = get_chunking_service()
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        experiment = context.get('cleaned_experiment', {})
        text = experiment.get('text', '')
        should_chunk = self._should_chunk(text)
        if not should_chunk:
            return {
                'chunks': [{
                    'text': text,
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'strategy': 'no_chunking',
                    **experiment
                }],
                'chunking_applied': False,
                'chunk_count': 1
            }
        strategy = self.chunking_service.auto_select_strategy(text)
        metadata = {
            'sequence': experiment.get('sequence'),
            'conditions': experiment.get('conditions', {}),
            'success': experiment.get('success'),
            'source': experiment.get('source', 'user_upload'),
            'notes': experiment.get('notes')
        }
        chunks = self.chunking_service.chunk_with_metadata(
            text=text,
            metadata=metadata,
            strategy=strategy,
            max_chunk_size=512,
            overlap=50
        )
        return {
            'chunks': chunks,
            'chunking_applied': True,
            'chunk_count': len(chunks),
            'strategy_used': strategy,
            'original_text_length': len(text),
            'avg_chunk_length': sum(len(c['text']) for c in chunks) / len(chunks) if chunks else 0
        }
    def _should_chunk(self, text: str, threshold: int = 1500) -> bool:
        if not text or len(text.strip()) == 0:
            return False
        if len(text) > threshold:
            return True
        if self._is_structured_protocol(text):
            return True
        return False
    def _is_structured_protocol(self, text: str) -> bool:
        import re
        protocol_patterns = [
            r'\bstep\s+\d+',
            r'^\d+\.',
            r'\n\d+\.\s+',
            r'materials:\s*\n',
            r'methods:\s*\n',
            r'procedure:\s*\n'
        ]
        matches = sum(
            1 for pattern in protocol_patterns
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        )
        return matches >= 3
    async def chunk_for_embedding(
        self,
        context: Dict[str, Any],
        embedding_dim: int = 384
    ) -> Dict[str, Any]:
        optimal_chunk_size = min(512, embedding_dim * 2)
        experiment = context.get('cleaned_experiment', {})
        text = experiment.get('text', '')
        strategy = self.chunking_service.auto_select_strategy(text)
        metadata = {
            'sequence': experiment.get('sequence'),
            'conditions': experiment.get('conditions', {}),
            'success': experiment.get('success'),
            'source': experiment.get('source', 'user_upload')
        }
        chunks = self.chunking_service.chunk_with_metadata(
            text=text,
            metadata=metadata,
            strategy=strategy,
            max_chunk_size=optimal_chunk_size,
            overlap=50
        )
        return {
            'embedding_chunks': chunks,
            'chunk_strategy': strategy,
            'optimal_chunk_size': optimal_chunk_size,
            'ready_for_embedding': True
        }
    async def merge_chunk_results(
        self,
        chunks_results: List[Dict[str, Any]],
        aggregation_method: str = "average"
    ) -> Dict[str, Any]:
        if not chunks_results:
            return {}
        if aggregation_method == "average":
            return self._average_aggregation(chunks_results)
        elif aggregation_method == "max":
            return self._max_aggregation(chunks_results)
        elif aggregation_method == "weighted":
            return self._weighted_aggregation(chunks_results)
        else:
            return self._average_aggregation(chunks_results)
    def _average_aggregation(self, chunks_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not chunks_results:
            return {}
        all_neighbors = []
        for chunk_result in chunks_results:
            neighbors = chunk_result.get('neighbors', [])
            all_neighbors.extend(neighbors)
        neighbor_scores = {}
        for neighbor in all_neighbors:
            nid = neighbor.get('id')
            score = neighbor.get('score', 0.0)
            if nid not in neighbor_scores:
                neighbor_scores[nid] = {
                    'scores': [],
                    'neighbor': neighbor
                }
            neighbor_scores[nid]['scores'].append(score)
        aggregated = []
        for nid, data in neighbor_scores.items():
            avg_score = sum(data['scores']) / len(data['scores'])
            neighbor = data['neighbor'].copy()
            neighbor['score'] = avg_score
            neighbor['chunk_matches'] = len(data['scores'])
            aggregated.append(neighbor)
        aggregated.sort(key=lambda x: x['score'], reverse=True)
        return {
            'neighbors': aggregated,
            'aggregation_method': 'average',
            'total_chunk_results': len(chunks_results)
        }
    def _max_aggregation(self, chunks_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_neighbors = []
        for chunk_result in chunks_results:
            neighbors = chunk_result.get('neighbors', [])
            all_neighbors.extend(neighbors)
        best_neighbors = {}
        for neighbor in all_neighbors:
            nid = neighbor.get('id')
            score = neighbor.get('score', 0.0)
            if nid not in best_neighbors or score > best_neighbors[nid]['score']:
                best_neighbors[nid] = neighbor
        aggregated = sorted(best_neighbors.values(), key=lambda x: x['score'], reverse=True)
        return {
            'neighbors': aggregated,
            'aggregation_method': 'max',
            'total_chunk_results': len(chunks_results)
        }
    def _weighted_aggregation(self, chunks_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        weighted_neighbors = []
        for i, chunk_result in enumerate(chunks_results):
            total_chunks = len(chunks_results)
            position_weight = 1.0 - abs((i - total_chunks / 2) / (total_chunks / 2)) * 0.3
            neighbors = chunk_result.get('neighbors', [])
            for neighbor in neighbors:
                weighted_neighbor = neighbor.copy()
                original_score = neighbor.get('score', 0.0)
                weighted_neighbor['score'] = original_score * position_weight
                weighted_neighbor['original_score'] = original_score
                weighted_neighbor['weight'] = position_weight
                weighted_neighbors.append(weighted_neighbor)
        neighbor_groups = {}
        for neighbor in weighted_neighbors:
            nid = neighbor.get('id')
            if nid not in neighbor_groups:
                neighbor_groups[nid] = {
                    'scores': [],
                    'weights': [],
                    'neighbor': neighbor
                }
            neighbor_groups[nid]['scores'].append(neighbor['score'])
            neighbor_groups[nid]['weights'].append(neighbor.get('weight', 1.0))
        aggregated = []
        for nid, data in neighbor_groups.items():
            weighted_avg = sum(data['scores']) / len(data['scores'])
            neighbor = data['neighbor'].copy()
            neighbor['score'] = weighted_avg
            neighbor['chunk_matches'] = len(data['scores'])
            aggregated.append(neighbor)
        aggregated.sort(key=lambda x: x['score'], reverse=True)
        return {
            'neighbors': aggregated,
            'aggregation_method': 'weighted',
            'total_chunk_results': len(chunks_results)
        }