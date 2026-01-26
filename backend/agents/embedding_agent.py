from typing import Dict, Any
from backend.services.embedding_service import get_embedding_service
from backend.services.qdrant_service import get_qdrant_service
import uuid
class EmbeddingAgent:
    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.qdrant = get_qdrant_service()
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        experiment = context.get('cleaned_experiment', {})
        embedding = await self.embedding_service.generate_multimodal_embedding(
            text=experiment.get('text'),
            sequence=experiment.get('sequence'),
            conditions=experiment.get('conditions'),
            image_base64=experiment.get('image_base64')
        )
        experiment_id = str(uuid.uuid4())
        collection_name = "private_experiments"
        await self.qdrant.upsert(
            collection_name=collection_name,
            points=[{
                'id': experiment_id,
                'vector': embedding.tolist(),
                'payload': {
                    'text': experiment.get('text', ''),
                    'sequence': experiment.get('sequence'),
                    'conditions': experiment.get('conditions', {}),
                    'success': experiment.get('success'),
                    'source': experiment.get('source', 'user_upload'),
                    'notes': experiment.get('notes'),
                    'created_at': context.get('user_input', {}).get('timestamp', None)
                }
            }]
        )
        return {
            'experiment_id': experiment_id,
            'embedding': embedding,
            'embedding_metadata': {
                'text_dim': self.embedding_service.text_dim,
                'seq_dim': self.embedding_service.seq_dim,
                'cond_dim': self.embedding_service.cond_dim,
                'total_dim': len(embedding)
            },
            'collection': collection_name
        }