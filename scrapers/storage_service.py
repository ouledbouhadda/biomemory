from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import uuid
from backend.models.scraped_experiment import ScrapedExperiment
from backend.services.embedding_service import get_embedding_service
class ScrapedExperimentStorage:
    def __init__(self, qdrant_client: QdrantClient, collection_name: str = "private_experiments"):
        self.client = qdrant_client
        self.collection_name = collection_name
        self.embedding_service = get_embedding_service()
    async def store_experiment(self, experiment: ScrapedExperiment) -> str:
        point_id = str(uuid.uuid4())
        embedding_text = self._create_embedding_text(experiment)
        try:
            embedding = await self.embedding_service.generate_multimodal_embedding(
                text=embedding_text,
                sequence=experiment.sequence,
                conditions={
                    "ph": experiment.conditions.ph,
                    "temperature": experiment.conditions.temperature,
                    "organism": experiment.conditions.organism
                }
            )
        except Exception as e:
            print(f"Embedding failed for {experiment.experiment_id}: {e}")
            import numpy as np
            embedding = np.zeros(768)
        payload = {
            "experiment_id": experiment.experiment_id,
            "text": experiment.description,
            "sequence": experiment.sequence,
            "conditions": {
                "ph": experiment.conditions.ph,
                "temperature": experiment.conditions.temperature,
                "organism": experiment.conditions.organism,
                "assay": experiment.conditions.assay,
                "additional": experiment.conditions.additional
            },
            "outcome": {
                "status": experiment.outcome.status,
                "notes": experiment.outcome.notes,
                "metrics": experiment.outcome.metrics
            },
            "evidence": {
                "paper_title": experiment.evidence.paper_title,
                "doi": experiment.evidence.doi,
                "arxiv_link": experiment.evidence.arxiv_link,
                "protocol_url": experiment.evidence.protocol_url,
                "authors": experiment.evidence.authors,
                "publication_date": experiment.evidence.publication_date
            },
            "source": experiment.source,
            "scraped_at": experiment.scraped_at.isoformat(),
            "success": experiment.outcome.status == "success"
        }
        point = PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload=payload
        )
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        print(f"Expérience {experiment.experiment_id} stockée dans Qdrant (ID: {point_id})")
        return point_id
    async def store_batch(self, experiments: List[ScrapedExperiment], batch_size: int = 10) -> List[str]:
        point_ids = []
        print(f"Stockage de {len(experiments)} expériences dans Qdrant...")
        for i in range(0, len(experiments), batch_size):
            batch = experiments[i:i + batch_size]
            batch_points = []
            for experiment in batch:
                try:
                    point_id = str(uuid.uuid4())
                    embedding_text = self._create_embedding_text(experiment)
                    try:
                        embedding = await self.embedding_service.generate_multimodal_embedding(
                            text=embedding_text,
                            sequence=experiment.sequence,
                            conditions={
                                "ph": experiment.conditions.ph,
                                "temperature": experiment.conditions.temperature,
                                "organism": experiment.conditions.organism
                            }
                        )
                    except Exception as e:
                        print(f"Embedding failed for {experiment.experiment_id}: {e}")
                        import numpy as np
                        embedding = np.zeros(768)
                    payload = {
                        "experiment_id": experiment.experiment_id,
                        "text": experiment.description,
                        "sequence": experiment.sequence,
                        "conditions": {
                            "ph": experiment.conditions.ph,
                            "temperature": experiment.conditions.temperature,
                            "organism": experiment.conditions.organism,
                            "assay": experiment.conditions.assay,
                            "additional": experiment.conditions.additional
                        },
                        "outcome": {
                            "status": experiment.outcome.status,
                            "notes": experiment.outcome.notes,
                            "metrics": experiment.outcome.metrics
                        },
                        "evidence": {
                            "paper_title": experiment.evidence.paper_title,
                            "doi": experiment.evidence.doi,
                            "arxiv_link": experiment.evidence.arxiv_link,
                            "protocol_url": experiment.evidence.protocol_url,
                            "authors": experiment.evidence.authors,
                            "publication_date": experiment.evidence.publication_date
                        },
                        "source": experiment.source,
                        "scraped_at": experiment.scraped_at.isoformat(),
                        "success": experiment.outcome.status == "success"
                    }
                    point = PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload=payload
                    )
                    batch_points.append(point)
                    point_ids.append(point_id)
                except Exception as e:
                    print(f"Erreur pour {experiment.experiment_id}: {e}")
                    continue
            if batch_points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch_points
                )
                print(f"   Batch {i//batch_size + 1}: {len(batch_points)} expériences stockées")
        print(f"Stockage terminé: {len(point_ids)}/{len(experiments)} expériences")
        return point_ids
    def _create_embedding_text(self, experiment: ScrapedExperiment) -> str:
        parts = [experiment.description]
        if experiment.conditions.organism:
            parts.append(f"Organism: {experiment.conditions.organism}")
        if experiment.conditions.assay:
            parts.append(f"Assay: {experiment.conditions.assay}")
        if experiment.evidence.paper_title:
            parts.append(f"Reference: {experiment.evidence.paper_title}")
        return " ".join(parts)
    def count_scraped_experiments(self) -> int:
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception as e:
            print(f"Erreur lors du comptage: {e}")
            return 0