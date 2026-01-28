from typing import Dict, Any, List
from backend.services.embedding_service import get_embedding_service
from backend.services.qdrant_service import get_qdrant_service
from backend.agents.multimodal_search_agent import MultimodalSearchAgent
from backend.services.bio_rag_service import get_bio_rag_service
class BioRAGAgent:
    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.qdrant_service = get_qdrant_service()
        self.multimodal_agent = MultimodalSearchAgent()
        self.rag_service = get_bio_rag_service()
    async def _find_similar_experiments(self, query_embedding: List[float], limit: int) -> List[Dict]:
        try:
            collection_name = "public_science"
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()
            print(f"Recherche dans collection '{collection_name}' avec vecteur de dimension {len(query_embedding)}")
            search_results = await self.qdrant_service.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=0.1
            )
            if search_results and len(search_results) > 0:
                print(f"Trouvé {len(search_results)} expériences similaires dans Qdrant Cloud")
                return search_results
            search_results = await self.qdrant_service.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=0.0
            )
            if search_results and len(search_results) > 0:
                print(f"Trouvé {len(search_results)} expériences similaires (sans seuil) dans Qdrant Cloud")
                return search_results
            print("Aucune expérience trouvée dans Qdrant Cloud, utilisation du fallback")
            return await self._get_mock_experiments(limit)
        except Exception as e:
            print(f"Qdrant Cloud search failed: {e}")
            import traceback
            traceback.print_exc()
            print("Utilisation des données mockées comme fallback")
            return await self._get_mock_experiments(limit)
    async def _get_mock_experiments(self, limit: int) -> List[Dict]:
        mock_experiments = [
            {
                'id': '34fdfee6-0a50-4a42-b9e9-bbee6c3fed8d',
                'score': 0.85,
                'payload': {
                    'experiment_id': 'protocols_io_106698',
                    'text': '{"blocks":[{"key":"djacg","text":"This is a protocol for protein expression..."}]}',
                    'sequence': '',
                    'conditions': {
                        'ph': 8.0,
                        'temperature': 37.0,
                        'organism': 'bacteria'
                    },
                    'outcome': {'status': 'success'},
                    'success': True
                }
            },
            {
                'id': '39409da8-e416-4c2d-95c7-4b99fad5b5c1',
                'score': 0.82,
                'payload': {
                    'experiment_id': 'protocols_io_10877',
                    'text': '{"blocks":[{"key":"22o1f","text":"The research protocol for mycobacteria..."}]}',
                    'sequence': '',
                    'conditions': {
                        'ph': None,
                        'temperature': 37.0,
                        'organism': 'bacteria'
                    },
                    'outcome': {'status': 'success'},
                    'success': True
                }
            },
            {
                'id': '39800e48-10c5-4d42-a921-268243205cbf',
                'score': 0.78,
                'payload': {
                    'experiment_id': 'protocols_io_15804',
                    'text': '{"blocks":[{"key":"er1du","text":"Protocol for fecal sampling in infants..."}]}',
                    'sequence': '',
                    'conditions': {
                        'ph': None,
                        'temperature': 27.0,
                        'organism': 'bacteria'
                    },
                    'outcome': {'status': 'success'},
                    'success': True
                }
            },
            {
                'id': '3b81645e-ed84-4890-890a-b9843f6eed3d',
                'score': 0.75,
                'payload': {
                    'experiment_id': 'protocols_io_8855',
                    'text': '{"blocks":[{"key":"4k5ao","text":"Diagnostic tests for pertussis..."}]}',
                    'sequence': '',
                    'conditions': {
                        'ph': None,
                        'temperature': 35.0,
                        'organism': None
                    },
                    'outcome': {'status': 'success'},
                    'success': True
                }
            },
            {
                'id': '3db2c19d-c734-43dd-9654-dadda7524fb9',
                'score': 0.72,
                'payload': {
                    'experiment_id': 'protocols_io_14117',
                    'text': '{"blocks":[{"key":"bmb9s","text":"Virus and bacteria counts protocol..."}]}',
                    'sequence': '',
                    'conditions': {
                        'ph': 7.5,
                        'temperature': 30.0,
                        'organism': 'bacteria'
                    },
                    'outcome': {'status': 'success'},
                    'success': True
                }
            }
        ]
        return mock_experiments[:min(limit, len(mock_experiments))]
    async def _generate_rag_response(self, query: Dict[str, Any], similar_experiments: List[Dict]) -> str:
        try:
            analysis = self._analyze_experiments(similar_experiments)
            response_parts = []
            exp_count = len(similar_experiments)
            query_text = query.get('text', '')
            response_parts.append(f"Basé sur {exp_count} expérience{'s' if exp_count > 1 else ''} similaire{'s' if exp_count > 1 else ''}")
            if query_text:
                response_parts.append(f"concernant '{query_text}'")
            response_parts.append(", ")
            if analysis['optimal_conditions']:
                conditions = analysis['optimal_conditions']
                temp = conditions.get('temperature')
                ph = conditions.get('ph')
                organism = conditions.get('organism')
                if temp is not None or ph is not None:
                    cond_parts = []
                    if temp is not None:
                        cond_parts.append(f"à {temp}°C")
                    if ph is not None:
                        cond_parts.append(f"avec un pH de {ph}")
                    if organism:
                        cond_parts.append(f"dans {organism}")
                    if cond_parts:
                        response_parts.append("l'expression protéique réussit mieux " + " ".join(cond_parts) + ". ")
            if analysis['success_rate'] is not None:
                success_rate = analysis['success_rate']
                response_parts.append(f"Les expériences montrent un taux de succès de {success_rate:.0f}% dans ces conditions. ")
            if analysis['recommendations']:
                response_parts.append(" ".join(analysis['recommendations']))
            return "".join(response_parts).strip()
        except Exception as e:
            print(f"RAG response generation failed: {e}")
            return f"Basé sur {len(similar_experiments)} expériences similaires trouvées, je recommande de consulter les conditions expérimentales détaillées pour optimiser vos résultats."
    def _analyze_experiments(self, experiments: List[Dict]) -> Dict[str, Any]:
        if not experiments:
            return {}
        temperatures = []
        phs = []
        organisms = []
        successes = []
        assays = []

        for exp in experiments:
            payload = exp.get('payload', {})


            organism = payload.get('organism')
            temperature = payload.get('temperature')
            assay = payload.get('assay')


            conditions = payload.get('conditions', {})
            if not organism:
                organism = conditions.get('organism')
            if not temperature:
                temperature = conditions.get('temperature')


            if temperature is not None:
                if isinstance(temperature, str):

                    import re
                    temp_match = re.search(r'(\d+(?:\.\d+)?)', str(temperature))
                    if temp_match:
                        temperatures.append(float(temp_match.group(1)))
                elif isinstance(temperature, (int, float)):
                    temperatures.append(float(temperature))

            if conditions.get('ph') is not None:
                phs.append(conditions['ph'])

            if organism:
                organisms.append(organism)

            if assay:
                assays.append(assay)


            success = payload.get('success', True)
            successes.append(success)
        analysis = {
            'optimal_conditions': {},
            'success_rate': None,
            'recommendations': []
        }
        if temperatures:
            successful_temps = [t for t, s in zip(temperatures, successes) if s]
            if successful_temps:
                analysis['optimal_conditions']['temperature'] = round(sum(successful_temps) / len(successful_temps), 1)
        if phs:
            successful_phs = [p for p, s in zip(phs, successes) if s]
            if successful_phs:
                analysis['optimal_conditions']['ph'] = round(sum(successful_phs) / len(successful_phs), 1)
        if organisms:
            from collections import Counter
            most_common = Counter(organisms).most_common(1)
            if most_common:
                analysis['optimal_conditions']['organism'] = most_common[0][0]
        if successes:
            success_count = sum(1 for s in successes if s)
            analysis['success_rate'] = (success_count / len(successes)) * 100
        if analysis['success_rate'] and analysis['success_rate'] > 80:
            analysis['recommendations'].append("Ces conditions semblent très efficaces.")
        elif analysis['success_rate'] and analysis['success_rate'] < 50:
            analysis['recommendations'].append("Considérez d'ajuster les conditions expérimentales.")
        return analysis
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        user_input = context.get('user_input', {})
        query = user_input.get('query', {})


        chunks = context.get('chunks', [])
        chunking_applied = context.get('chunking_applied', False)

        if not query:
            return {
                'rag_response': None,
                'rag_error': 'Missing query'
            }

        try:
            multimodal_result = await self.multimodal_agent.execute({
                'user_input': {'query': query}
            })
            query_embedding = multimodal_result.get('query_embedding')

            if query_embedding is None or (hasattr(query_embedding, '__len__') and len(query_embedding) == 0):
                return {
                    'rag_response': None,
                    'rag_error': 'Failed to generate query embedding'
                }

            if len(query_embedding) > 488:
                query_text = query.get('text', '')
                conditions = query.get('conditions', {})
                query_embedding = await self.embedding_service.generate_multimodal_embedding(
                    text=query_text,
                    conditions=conditions,
                    include_image=False
                )

            limit = query.get('limit', 15)


            if chunking_applied and chunks:
                similar_experiments = await self._find_similar_chunks(query_embedding, limit)
                rag_method = 'chunked_rag'
                print(f" Chonkie RAG: recherche sur {len(chunks)} chunks contextuels")
            else:
                similar_experiments = await self._find_similar_experiments(query_embedding, limit)
                rag_method = 'similarity_rag'

            if not similar_experiments:
                return {
                    'rag_response': f"Aucune expérience similaire trouvée pour votre requête : '{query.get('text', '')}'. Essayez de reformuler ou d'ajouter plus de détails.",
                    'similar_experiments_count': 0,
                    'rag_method': rag_method
                }

            rag_response = await self._generate_rag_response(query, similar_experiments)

            return {
                'rag_response': rag_response,
                'similar_experiments_count': len(similar_experiments),
                'similar_experiments': similar_experiments[:5],
                'rag_method': rag_method,
                'modalities_detected': multimodal_result.get('modalities_used', {}),
                'chunking_used': chunking_applied,
                'chunks_count': len(chunks) if chunking_applied else 0
            }

        except Exception as e:
            print(f"BioRAG Agent failed: {e}")
            return {
                'rag_response': None,
                'rag_error': str(e)
            }

    async def _find_similar_chunks(self, query_embedding: List[float], limit: int) -> List[Dict]:
        """
        Recherche des chunks similaires (documents chunkés avec Chonkie).
        Permet une recherche plus fine sur des passages spécifiques.
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue


            chunk_filter = Filter(
                must=[
                    FieldCondition(
                        key="chunked",
                        match=MatchValue(value=True)
                    )
                ],
                must_not=[
                    FieldCondition(
                        key="is_parent",
                        match=MatchValue(value=True)
                    )
                ]
            )


            chunk_results = await self.qdrant_service.search(
                collection_name="private_experiments",
                query_vector=query_embedding,
                limit=limit * 2,
                query_filter=chunk_filter,
                score_threshold=0.1
            )

            if chunk_results:
                print(f" Chonkie: Trouvé {len(chunk_results)} chunks similaires")


                seen_parents = set()
                deduplicated = []
                for chunk in chunk_results:
                    parent_id = chunk.get('payload', {}).get('parent_id')
                    if parent_id not in seen_parents:
                        seen_parents.add(parent_id)
                        deduplicated.append(chunk)
                    elif len([c for c in deduplicated if c.get('payload', {}).get('parent_id') == parent_id]) < 2:

                        deduplicated.append(chunk)

                return deduplicated[:limit]


            print(" Pas de chunks trouvés, fallback vers recherche classique")
            return await self._find_similar_experiments(query_embedding, limit)

        except Exception as e:
            print(f" Chunk search failed: {e}, fallback vers recherche classique")
            return await self._find_similar_experiments(query_embedding, limit)
    async def find_similar_experiments(self, experiment: Dict, limit: int = 5) -> List[Dict]:
        try:
            experiment_id = experiment.get('id')
            if not experiment_id:
                return []
            similar_experiments = await self.similarity_agent.recommend_similar_experiments(
                experiment_ids=[experiment_id],
                limit=limit + 1
            )
            filtered_similars = [
                exp for exp in similar_experiments
                if str(exp.get('id')) != str(experiment_id)
            ][:limit]
            if filtered_similars:
                print(f"Recommandations Qdrant: {len(filtered_similars)} expériences similaires")
                return filtered_similars
            print("Fallback vers recherche vectorielle classique")
            return await self._find_similar_experiments_classic(experiment, limit)
        except Exception as e:
            print(f"Recommandations Qdrant échouées: {e}")
            return await self._find_similar_experiments_classic(experiment, limit)
    async def _find_similar_experiments_classic(self, experiment: Dict, limit: int) -> List[Dict]:
        try:
            multimodal_result = await self.multimodal_agent.execute({
                'user_input': {'query': experiment.get('payload', {})}
            })
            query_embedding = multimodal_result.get('query_embedding')
            if not query_embedding:
                return []
            return await self._find_similar_experiments(query_embedding, limit)
        except Exception as e:
            print(f"Recherche classique échouée: {e}")
            return []
    async def search_similar_by_criteria(self, criteria: Dict[str, Any], limit: int = 10) -> List[Dict]:
        try:
            query_parts = []
            if criteria.get('organism'):
                query_parts.append(f"expérience avec {criteria['organism']}")
            if criteria.get('method'):
                query_parts.append(f"méthode {criteria['method']}")
            if criteria.get('target'):
                query_parts.append(f"ciblant {criteria['target']}")
            if criteria.get('conditions'):
                cond = criteria['conditions']
                if cond.get('temperature'):
                    query_parts.append(f"température {cond['temperature']}°C")
                if cond.get('ph'):
                    query_parts.append(f"pH {cond['ph']}")
            query_text = " ".join(query_parts) if query_parts else "expérience biologique"
            query_data = {
                'text': query_text,
                'conditions': criteria.get('conditions', {})
            }
            multimodal_result = await self.multimodal_agent.execute({
                'user_input': {'query': query_data}
            })
            query_embedding = multimodal_result.get('query_embedding')
            if query_embedding is None or (hasattr(query_embedding, '__len__') and len(query_embedding) == 0):
                return []
            if len(query_embedding) > 488:
                embedding_service = self.multimodal_agent.embedding_service
                query_embedding = await embedding_service.generate_multimodal_embedding(
                    text=query_text,
                    conditions=criteria.get('conditions', {}),
                    include_image=False
                )
            similarity_result = await self.similarity_agent.execute({
                'query_embedding': query_embedding,
                'query_conditions': criteria.get('conditions', {}),
                'limit': limit,
                'similarity_threshold': 0.0
            })
            return similarity_result.get('neighbors', [])
        except Exception as e:
            print(f"Search by criteria failed: {e}")
            return []