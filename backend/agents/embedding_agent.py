from typing import Dict, Any, List
from backend.services.embedding_service import get_embedding_service
from backend.services.qdrant_service import get_qdrant_service
import uuid


class EmbeddingAgent:
    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.qdrant = get_qdrant_service()

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        chunks = context.get('chunks', [])
        chunking_applied = context.get('chunking_applied', False)

        if chunking_applied and chunks:
            return await self._embed_chunks(context, chunks)
        else:
            return await self._embed_single(context)

    async def _embed_single(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Embedding classique pour un seul document."""
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
                    'created_at': context.get('user_input', {}).get('timestamp', None),
                    'chunked': False
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
                'total_dim': len(embedding),
                'chunking_used': False
            },
            'collection': collection_name
        }

    async def _embed_chunks(self, context: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Embedding avec Chonkie: créer un embedding par chunk.
        Permet une recherche sémantique plus fine sur les documents longs.
        """
        experiment = context.get('cleaned_experiment', {})
        parent_id = str(uuid.uuid4())
        collection_name = "private_experiments"
        chunk_ids = []
        embeddings = []

        print(f"Chonkie: Embedding de {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            chunk_id = f"{parent_id}_chunk_{i}"
            chunk_text = chunk.get('text', '')


            embedding = await self.embedding_service.generate_multimodal_embedding(
                text=chunk_text,
                sequence=experiment.get('sequence'),
                conditions=experiment.get('conditions'),
                image_base64=experiment.get('image_base64') if i == 0 else None
            )


            chunk_payload = {
                'text': chunk_text,
                'sequence': experiment.get('sequence'),
                'conditions': experiment.get('conditions', {}),
                'success': experiment.get('success'),
                'source': experiment.get('source', 'user_upload'),
                'notes': experiment.get('notes'),
                'created_at': context.get('user_input', {}).get('timestamp', None),

                'chunked': True,
                'parent_id': parent_id,
                'chunk_index': chunk.get('chunk_index', i),
                'total_chunks': chunk.get('total_chunks', len(chunks)),
                'chunk_strategy': chunk.get('strategy', context.get('strategy_used', 'unknown')),
                'original_length': chunk.get('original_length', 0),
                'chunk_length': len(chunk_text)
            }

            chunk_ids.append(chunk_id)
            embeddings.append({
                'id': chunk_id,
                'vector': embedding.tolist(),
                'payload': chunk_payload
            })


        await self.qdrant.upsert(
            collection_name=collection_name,
            points=embeddings
        )


        parent_embedding = await self.embedding_service.generate_multimodal_embedding(
            text=experiment.get('text', '')[:1000],
            sequence=experiment.get('sequence'),
            conditions=experiment.get('conditions'),
            image_base64=experiment.get('image_base64')
        )

        await self.qdrant.upsert(
            collection_name=collection_name,
            points=[{
                'id': parent_id,
                'vector': parent_embedding.tolist(),
                'payload': {
                    'text': experiment.get('text', ''),
                    'sequence': experiment.get('sequence'),
                    'conditions': experiment.get('conditions', {}),
                    'success': experiment.get('success'),
                    'source': experiment.get('source', 'user_upload'),
                    'notes': experiment.get('notes'),
                    'created_at': context.get('user_input', {}).get('timestamp', None),
                    'chunked': True,
                    'is_parent': True,
                    'chunk_ids': chunk_ids,
                    'total_chunks': len(chunks),
                    'chunk_strategy': context.get('strategy_used', 'unknown')
                }
            }]
        )

        print(f"Chonkie: {len(chunk_ids)} chunks + 1 parent indexés")

        return {
            'experiment_id': parent_id,
            'chunk_ids': chunk_ids,
            'embedding_count': len(chunk_ids) + 1,
            'embedding_metadata': {
                'text_dim': self.embedding_service.text_dim,
                'seq_dim': self.embedding_service.seq_dim,
                'cond_dim': self.embedding_service.cond_dim,
                'total_dim': self.embedding_service.total_dim,
                'chunking_used': True,
                'chunk_count': len(chunks),
                'chunk_strategy': context.get('strategy_used', 'unknown'),
                'avg_chunk_length': context.get('avg_chunk_length', 0)
            },
            'collection': collection_name
        }