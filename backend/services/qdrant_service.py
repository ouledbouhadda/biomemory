from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    MatchText,
    MatchAny,
    HasIdCondition,
    IsEmptyCondition,
    IsNullCondition,
    GeoBoundingBox,
    GeoPoint,
    GeoRadius,
    DatetimeRange,
    OrderBy,
    OrderValue,
    Fusion,
    SparseVector,
    SparseIndexParams,
    Modifier,
    QuantizationConfig,
    HnswConfigDiff,
    OptimizersConfigDiff,
    WalConfigDiff,
    CollectionConfig,
    PayloadSchemaType,
    TextIndexParams,
    TokenizerType,
    FilterSelector,
    LookupLocation,
    RecommendRequest,
    SearchRequest,
    Query,
    Prefetch,
    ContextExamplePair,
    ContextQuery
)
from typing import List, Dict, Any, Optional, Union
from backend.config.settings import get_settings
import numpy as np
from datetime import datetime, timedelta
settings = get_settings()
settings = get_settings()
class QdrantService:
    def __init__(self):
        self.cloud_client = None
        print(f"QDRANT_CLOUD_URL: {settings.QDRANT_CLOUD_URL}")
        print(f"QDRANT_CLOUD_API_KEY: {'SET' if settings.QDRANT_CLOUD_API_KEY else 'NOT SET'}")

        if (settings.QDRANT_CLOUD_URL and settings.QDRANT_CLOUD_API_KEY and
            settings.QDRANT_CLOUD_URL != "https://your-cluster.qdrant.io" and
            settings.QDRANT_CLOUD_API_KEY != "your-qdrant-cloud-api-key"):
            try:
                self.cloud_client = QdrantClient(
                    url=settings.QDRANT_CLOUD_URL,
                    api_key=settings.QDRANT_CLOUD_API_KEY,
                    timeout=30
                )
                collections = self.cloud_client.get_collections()
                print(f"Qdrant Cloud client initialized - {len(collections.collections)} collections found")
                for col in collections.collections:
                    print(f"   Collection: {col.name}")
            except Exception as e:
                print(f"Qdrant Cloud unavailable: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Qdrant Cloud not configured (using placeholder values")
        try:
            self.private_client = QdrantClient(
                host=settings.QDRANT_PRIVATE_HOST,
                port=settings.QDRANT_PRIVATE_PORT,
                timeout=30
            )
        except Exception as e:
            print(f"Qdrant Private unavailable: {e}")
            self.private_client = None
        self.vector_size = settings.total_vector_dim
        self.hnsw_config = {
            "m": 32,
            "ef_construct": 200,
            "full_scan_threshold": 10000,
            "max_indexing_threads": 4,
            "on_disk": False
        }
        self.quantization_config = {
            "scalar": {
                "type": "int8",
                "quantile": 0.99,
                "always_ram": True
            }
        }
        self.optimizers_config = {
            "deleted_threshold": 0.2,
            "vacuum_min_vector_number": 1000,
            "default_segment_number": 2,
            "max_segment_size": None,
            "memmap_threshold": None,
            "indexing_threshold": 20000,
            "flush_interval_sec": 5,
            "max_optimization_threads": 4
        }
    async def init_collections(self):
        if self.cloud_client:
            try:
                collection_config = {
                    "vectors": VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                        hnsw_config=HnswConfigDiff(**self.hnsw_config),
                        quantization_config=QuantizationConfig(**self.quantization_config),
                        on_disk=False
                    ),
                    "optimizers_config": OptimizersConfigDiff(**self.optimizers_config),
                    "wal_config": WalConfigDiff(wal_capacity_mb=32, wal_segments_ahead=2),
                    "sparse_vectors": {
                        "text_keywords": SparseIndexParams(
                            index_type="keyword",
                            tokenizer=TokenizerType.WORD,
                            lowercase=True,
                            min_token_len=2,
                            max_token_len=20
                        )
                    }
                }
                self.cloud_client.create_collection(
                    collection_name="public_experiments",
                    **collection_config
                )
                self._setup_payload_schema(self.cloud_client, "public_experiments")
                print("Collection 'public_experiments' created with advanced features")
                try:
                    self.cloud_client.create_collection(
                        collection_name="biomemory_experiments",
                        **collection_config
                    )
                    self._setup_payload_schema(self.cloud_client, "biomemory_experiments")
                    print("Collection 'biomemory_experiments' created with advanced features")
                except Exception as e:
                    print(f"Collection 'biomemory_experiments' already exists: {e}")
            except Exception as e:
                print(f"Collection 'public_experiments' already exists or error: {e}")
        if self.private_client:
            try:
                collection_config = {
                    "vectors": VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                        hnsw_config=HnswConfigDiff(**self.hnsw_config),
                        quantization_config=QuantizationConfig(**self.quantization_config),
                        on_disk=False
                    ),
                    "optimizers_config": OptimizersConfigDiff(**self.optimizers_config),
                    "wal_config": WalConfigDiff(wal_capacity_mb=32, wal_segments_ahead=2),
                    "shard_number": 2,
                    "replication_factor": 1,
                    "sparse_vectors": {
                        "text_keywords": SparseIndexParams(
                            index_type="keyword",
                            tokenizer=TokenizerType.WORD,
                            lowercase=True,
                            min_token_len=2,
                            max_token_len=20
                        )
                    }
                }
                self.private_client.create_collection(
                    collection_name="private_experiments_template",
                    **collection_config
                )
                self._setup_payload_schema(self.private_client, "private_experiments_template")
                print("✓ Collection template 'private_experiments_template' créée")
            except Exception as e:
                print(f"ℹCollection template existe déjà ou erreur: {e}")
    def _setup_payload_schema(self, client: QdrantClient, collection_name: str):
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name="title",
                field_schema=TextIndexParams(
                    type=PayloadSchemaType.TEXT,
                    tokenizer=TokenizerType.WORD,
                    lowercase=True,
                    min_token_len=2
                )
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="abstract",
                field_schema=TextIndexParams(
                    type=PayloadSchemaType.TEXT,
                    tokenizer=TokenizerType.WORD,
                    lowercase=True,
                    min_token_len=2
                )
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="domain",
                field_schema=PayloadSchemaType.KEYWORD
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="experiment_type",
                field_schema=PayloadSchemaType.KEYWORD
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="organism",
                field_schema=PayloadSchemaType.KEYWORD
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="temperature",
                field_schema=PayloadSchemaType.FLOAT
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="ph",
                field_schema=PayloadSchemaType.FLOAT
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="confidence_score",
                field_schema=PayloadSchemaType.FLOAT
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="publication_date",
                field_schema=PayloadSchemaType.DATETIME
            )
            print(f"✓ Schéma de payload configuré pour {collection_name}")
        except Exception as e:
            print(f" Erreur lors de la configuration du schéma: {e}")
            try:
                client.create_collection(
                    collection_name="private_experiments",
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                print("✓ Collection 'private_experiments' créée (fallback)")
            except Exception as e2:
                print(f" Échec création collection fallback: {e2}")
    async def init_collections(self):
        if self.cloud_client:
            try:
                collection_config = {
                    "vectors": VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                        hnsw_config=HnswConfigDiff(**self.hnsw_config),
                        quantization_config=QuantizationConfig(**self.quantization_config),
                        on_disk=False
                    ),
                    "optimizers_config": OptimizersConfigDiff(**self.optimizers_config),
                    "wal_config": WalConfigDiff(wal_capacity_mb=32, wal_segments_ahead=2),
                    "sparse_vectors": {
                        "text_keywords": SparseIndexParams(
                            index_type="keyword",
                            tokenizer=TokenizerType.WORD,
                            lowercase=True,
                            min_token_len=2,
                            max_token_len=20
                        )
                    }
                }
                self.cloud_client.create_collection(
                    collection_name="public_science",
                    **collection_config
                )
                self._setup_payload_schema(self.cloud_client, "public_science")
                print("✓ Collection 'public_science' créée avec fonctionnalités avancées")
            except Exception as e:
                print(f"ℹ️ Collection 'public_science' existe déjà ou erreur: {e}")
        if self.private_client:
            try:
                collection_config = {
                    "vectors": VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                        hnsw_config=HnswConfigDiff(**self.hnsw_config),
                        quantization_config=QuantizationConfig(**self.quantization_config),
                        on_disk=False
                    ),
                    "optimizers_config": OptimizersConfigDiff(**self.optimizers_config),
                    "wal_config": WalConfigDiff(wal_capacity_mb=32, wal_segments_ahead=2),
                    "shard_number": 2,
                    "replication_factor": 1,
                    "sparse_vectors": {
                        "text_keywords": SparseIndexParams(
                            index_type="keyword",
                            tokenizer=TokenizerType.WORD,
                            lowercase=True,
                            min_token_len=2,
                            max_token_len=20
                        )
                    }
                }
                self.private_client.create_collection(
                    collection_name="private_experiments",
                    **collection_config
                )
                self._setup_payload_schema(self.private_client, "private_experiments")
                print("✓ Collection 'private_experiments' créée avec fonctionnalités avancées")
                self.private_client.create_collection(
                    collection_name="biomemory_users",
                    vectors_config=VectorParams(
                        size=128,
                        distance=Distance.COSINE
                    )
                )
                print("✓ Collection 'biomemory_users' créée")
            except Exception as e:
                print(f"ℹ️ Collections privées existent déjà ou erreur: {e}")
    def _setup_payload_schema(self, client: QdrantClient, collection_name: str):
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name="title",
                field_schema=TextIndexParams(
                    type=PayloadSchemaType.TEXT,
                    tokenizer=TokenizerType.WORD,
                    lowercase=True,
                    min_token_len=2
                )
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="abstract",
                field_schema=TextIndexParams(
                    type=PayloadSchemaType.TEXT,
                    tokenizer=TokenizerType.WORD,
                    lowercase=True,
                    min_token_len=2
                )
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="text",
                field_schema=TextIndexParams(
                    type=PayloadSchemaType.TEXT,
                    tokenizer=TokenizerType.WORD,
                    lowercase=True,
                    min_token_len=2
                )
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="domain",
                field_schema=PayloadSchemaType.KEYWORD
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="experiment_type",
                field_schema=PayloadSchemaType.KEYWORD
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="organism",
                field_schema=PayloadSchemaType.KEYWORD
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="conditions.organism",
                field_schema=PayloadSchemaType.KEYWORD
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="temperature",
                field_schema=PayloadSchemaType.FLOAT
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="ph",
                field_schema=PayloadSchemaType.FLOAT
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="confidence_score",
                field_schema=PayloadSchemaType.FLOAT
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="score",
                field_schema=PayloadSchemaType.FLOAT
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="publication_date",
                field_schema=PayloadSchemaType.DATETIME
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="scraped_at",
                field_schema=PayloadSchemaType.DATETIME
            )
            print(f"✓ Schéma de payload configuré pour {collection_name}")
        except Exception as e:
            print(f" Erreur lors de la configuration du schéma: {e}")
            try:
                client.create_collection(
                    collection_name="private_experiments",
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                print("✓ Collection 'private_experiments' créée (fallback)")
            except Exception as e2:
                print(f" Échec création collection fallback: {e2}")
                print(f"ℹ Collection 'private_experiments' already exists or error: {e}")
    async def upsert(
        self,
        collection_name: str,
        points: List[Dict[str, Any]]
    ):
        client = self._get_client(collection_name)
        point_structs = [
            PointStruct(
                id=point["id"],
                vector=point["vector"],
                payload=point.get("payload", {})
            )
            for point in points
        ]
        client.upsert(
            collection_name=collection_name,
            points=point_structs
        )
    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        query_filter: Optional[Filter] = None,
        with_payload: bool = True,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        client = self._get_client(collection_name)
        print(f" Qdrant search: collection={collection_name}, vector_dim={len(query_vector)}, limit={limit}")

        try:
            results = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=with_payload,
                score_threshold=score_threshold
            )
            print(f" Qdrant search returned {len(results.points)} results")
            return [
                {
                    "id": str(point.id),
                    "score": point.score,
                    "payload": point.payload if with_payload else {}
                }
                for point in results.points
            ]
        except Exception as e:
            print(f" Qdrant search error: {e}")
            import traceback
            traceback.print_exc()
            return []
    async def hybrid_search(
        self,
        collection_name: str,
        query_vector: List[float],
        query_text: str = "",
        limit: int = 10,
        query_filter: Optional[Filter] = None,
        score_threshold: Optional[float] = None,
        fusion: Fusion = Fusion.RRF
    ) -> List[Dict[str, Any]]:
        client = self._get_client(collection_name)
        prefetch_queries = []
        if query_vector:
            prefetch_queries.append(
                Prefetch(
                    query=query_vector,
                    using="",
                    filter=query_filter,
                    limit=limit * 2
                )
            )
        if query_text and query_text.strip():
            text_filter = Filter(
                should=[
                    FieldCondition(
                        key="title",
                        match=MatchText(text=query_text)
                    ),
                    FieldCondition(
                        key="abstract",
                        match=MatchText(text=query_text)
                    ),
                    FieldCondition(
                        key="keywords",
                        match=MatchText(text=query_text)
                    )
                ]
            )
            combined_filter = query_filter
            if combined_filter and text_filter:
                combined_filter = Filter(
                    must=[combined_filter, text_filter]
                )
            elif text_filter:
                combined_filter = text_filter
            prefetch_queries.append(
                Prefetch(
                    query=query_vector or [0.0] * self.vector_size,
                    filter=combined_filter,
                    limit=limit * 2
                )
            )
        try:

            results = client.query_points(
                collection_name=collection_name,
                prefetch=prefetch_queries,
                query=Query(fusion=fusion),
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True
            )
            print(f" Hybrid search returned {len(results.points)} results")
            return [
                {
                    "id": str(hit.id),
                    "score": hit.score,
                    "payload": hit.payload or {}
                }
                for hit in results.points
            ]
        except Exception as e:
            print(f" Hybrid search error: {e}")
            import traceback
            traceback.print_exc()

            return await self.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter,
                score_threshold=score_threshold
            )
    async def recommend(
        self,
        collection_name: str,
        positive_ids: List[str],
        negative_ids: Optional[List[str]] = None,
        query_vector: Optional[List[float]] = None,
        limit: int = 10,
        query_filter: Optional[Filter] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        from qdrant_client.models import RecommendInput, RecommendQuery
        
        client = self._get_client(collection_name)
        

        recommend_input = RecommendInput(
            positive=[int(pid) if pid.isdigit() else pid for pid in positive_ids],
            negative=[int(nid) if nid.isdigit() else nid for nid in (negative_ids or [])]
        )
        recommend_query = RecommendQuery(recommend=recommend_input)
        
        results = client.query_points(
            collection_name=collection_name,
            query=recommend_query,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold,
            with_payload=True
        )
        
        return [
            {
                "id": str(point.id),
                "score": point.score,
                "payload": point.payload or {}
            }
            for point in results.points
        ]
    async def discover(
        self,
        collection_name: str,
        target: str,
        context: List[ContextExamplePair],
        limit: int = 10,
        query_filter: Optional[Filter] = None
    ) -> List[Dict[str, Any]]:
        from qdrant_client.models import DiscoverInput, DiscoverQuery
        
        client = self._get_client(collection_name)
        

        discover_input = DiscoverInput(
            target=target,
            context=context
        )
        discover_query = DiscoverQuery(discover=discover_input)
        
        results = client.query_points(
            collection_name=collection_name,
            query=discover_query,
            limit=limit,
            query_filter=query_filter,
            with_payload=True
        )
        
        return [
            {
                "id": str(point.id),
                "score": point.score,
                "payload": point.payload or {}
            }
            for point in results.points
        ]
    async def search_with_grouping(
        self,
        collection_name: str,
        query_vector: List[float],
        group_by: str,
        limit: int = 10,
        group_size: int = 3,
        query_filter: Optional[Filter] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        client = self._get_client(collection_name)
        results = client.query_points_groups(
            collection_name=collection_name,
            query=query_vector,
            group_by=group_by,
            limit=limit,
            group_size=group_size,
            query_filter=query_filter,
            with_payload=True
        )
        grouped_results = {}
        for group in results.groups:
            group_name = group.hits[0].payload.get(group_by, "unknown") if group.hits else "unknown"
            grouped_results[group_name] = [
                {
                    "id": str(point.id),
                    "score": point.score,
                    "payload": point.payload or {}
                }
                for point in group.hits
            ]
        return grouped_results
    async def search_with_ordering(
        self,
        collection_name: str,
        query_vector: List[float],
        order_by: OrderBy,
        limit: int = 10,
        query_filter: Optional[Filter] = None
    ) -> List[Dict[str, Any]]:
        client = self._get_client(collection_name)
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True
        )
        return [
            {
                "id": str(point.id),
                "score": point.score,
                "payload": point.payload or {}
            }
            for point in results.points
        ]
    async def batch_search(
        self,
        collection_name: str,
        queries: List[Dict[str, Any]],
        limit: int = 10
    ) -> List[List[Dict[str, Any]]]:
        from qdrant_client.models import QueryRequest
        
        client = self._get_client(collection_name)
        query_requests = []
        for query in queries:
            query_requests.append(
                QueryRequest(
                    query=query.get("vector", [0.0] * self.vector_size),
                    filter=query.get("filter"),
                    limit=limit,
                    with_payload=True
                )
            )
        results = client.query_batch_points(
            collection_name=collection_name,
            requests=query_requests
        )
        batch_results = []
        for result_set in results:
            batch_results.append([
                {
                    "id": str(point.id),
                    "score": point.score,
                    "payload": point.payload or {}
                }
                for point in result_set.points
            ])
        return batch_results
    async def count_points(
        self,
        collection_name: str,
        query_filter: Optional[Filter] = None,
        exact: bool = True
    ) -> int:
        client = self._get_client(collection_name)
        count_result = client.count(
            collection_name=collection_name,
            filter=query_filter,
            exact=exact
        )
        return count_result.count
    async def aggregate(
        self,
        collection_name: str,
        group_by: str,
        query_filter: Optional[Filter] = None
    ) -> Dict[str, Any]:
        client = self._get_client(collection_name)
        scroll_results = await self.scroll(
            collection_name=collection_name,
            limit=10000,
            with_payload=True
        )
        groups = {}
        for point in scroll_results:
            payload = point["payload"]
            group_key = payload.get(group_by, "unknown")
            if group_key not in groups:
                groups[group_key] = {
                    "count": 0,
                    "avg_confidence": 0.0,
                    "total_experiments": 0
                }
            groups[group_key]["count"] += 1
            groups[group_key]["avg_confidence"] += payload.get("confidence_score", 0.0)
            groups[group_key]["total_experiments"] += 1
        for group in groups.values():
            if group["count"] > 0:
                group["avg_confidence"] /= group["count"]
        return groups
    async def search_temporal(
        self,
        collection_name: str,
        query_vector: List[float],
        date_range: Dict[str, str],
        limit: int = 10,
        query_filter: Optional[Filter] = None
    ) -> List[Dict[str, Any]]:
        temporal_filter = Filter(
            must=[
                FieldCondition(
                    key="publication_date",
                    range=Range(
                        gte=date_range.get("start"),
                        lte=date_range.get("end")
                    )
                )
            ]
        )
        combined_filter = query_filter
        if combined_filter:
            combined_filter = Filter(
                must=[combined_filter, temporal_filter]
            )
        else:
            combined_filter = temporal_filter
        return await self.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=combined_filter
        )
    async def search_with_boosting(
        self,
        collection_name: str,
        query_vector: List[float],
        boost_factors: Dict[str, float],
        limit: int = 10,
        query_filter: Optional[Filter] = None
    ) -> List[Dict[str, Any]]:
        results = await self.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit * 2,
            query_filter=query_filter
        )
        for result in results:
            payload = result["payload"]
            boost_score = 1.0
            for field, factor in boost_factors.items():
                if field in payload:
                    field_value = payload[field]
                    if isinstance(field_value, (int, float)):
                        boost_score *= (1.0 + field_value * factor)
                    elif isinstance(field_value, str):
                        if any(keyword in field_value.lower() for keyword in ["novel", "breakthrough", "important"]):
                            boost_score *= factor
            result["score"] *= boost_score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    async def create_user_collection(
        self,
        user_id: str,
        collection_name: Optional[str] = None
    ) -> str:
        if not self.private_client:
            raise Exception("Private Qdrant client not available")
        collection_name = collection_name or f"private_experiments_{user_id}"
        try:
            self.private_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(**self.hnsw_config),
                    quantization_config=QuantizationConfig(**self.quantization_config)
                ),
                optimizers_config=OptimizersConfigDiff(**self.optimizers_config)
            )
            print(f"✓ Collection utilisateur '{collection_name}' créée")
            return collection_name
        except Exception as e:
            print(f"ℹCollection '{collection_name}' existe déjà ou erreur: {e}")
            return collection_name
    async def get_collection_stats(
        self,
        collection_name: str
    ) -> Dict[str, Any]:
        client = self._get_client(collection_name)
        try:
            info = client.get_collection(collection_name)
            count = await self.count_points(collection_name)
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "points_count": count,
                "status": info.status,
                "config": {
                    "vectors": info.config.params.vectors,
                    "optimizers": info.config.optimizer_config,
                    "quantization": info.config.quantization_config
                }
            }
        except Exception as e:
            return {"error": str(e)}
    def search_sync(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        query_filter: Optional[Filter] = None,
        with_payload: bool = True,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        client = self._get_client(collection_name)
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=with_payload,
            score_threshold=score_threshold
        )
        return [
            {
                "id": str(point.id),
                "score": point.score,
                "payload": point.payload if with_payload else {}
            }
            for point in results.points
        ]
    async def retrieve(
        self,
        collection_name: str,
        point_id: str
    ) -> Optional[Dict[str, Any]]:
        client = self._get_client(collection_name)
        try:
            results = client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=False
            )
            if results:
                point = results[0]
                return {
                    "id": str(point.id),
                    "payload": point.payload
                }
        except:
            pass
        return None
    async def scroll(
        self,
        collection_name: str,
        limit: int = 20,
        offset: int = 0,
        with_payload: bool = True
    ) -> List[Dict[str, Any]]:
        client = self._get_client(collection_name)
        results, next_page_offset = client.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            with_payload=with_payload,
            with_vectors=False
        )
        return [
            {
                "id": str(point.id),
                "payload": point.payload if with_payload else {}
            }
            for point in results
        ]
    async def delete(
        self,
        collection_name: str,
        points_selector: List[str]
    ):
        client = self._get_client(collection_name)
        client.delete(
            collection_name=collection_name,
            points_selector=points_selector
        )
    async def health_check(self, instance: str = "private") -> bool:
        try:
            client = self.cloud_client if instance == "cloud" else self.private_client
            if client:
                collections = client.get_collections()
                return True
        except Exception as e:
            print(f"Qdrant {instance} health check failed: {e}")
            return False
        return False





    async def upsert_with_sparse(
        self,
        collection_name: str,
        points: List[Dict[str, Any]]
    ):
        """
        Upsert avec vecteurs denses ET sparse pour recherche hybride.
        """
        client = self._get_client(collection_name)
        point_structs = []
        for point in points:
            sparse_vector = None
            text = point.get("payload", {}).get("text", "")
            if text:
                sparse_vector = self._generate_sparse_vector(text)

            point_struct = PointStruct(
                id=point["id"],
                vector={
                    "": point["vector"],
                },
                payload=point.get("payload", {})
            )
            if sparse_vector:
                point_struct.vector["text_keywords"] = sparse_vector

            point_structs.append(point_struct)

        client.upsert(
            collection_name=collection_name,
            points=point_structs
        )
        print(f" Upsert with sparse vectors: {len(point_structs)} points")

    def _generate_sparse_vector(self, text: str) -> SparseVector:
        """
        Génère un vecteur sparse à partir du texte (BM25-like).
        """
        words = text.lower().split()
        word_counts = {}
        for word in words:
            if len(word) >= 2:
                word_counts[word] = word_counts.get(word, 0) + 1

        indices = []
        values = []
        for idx, (word, count) in enumerate(sorted(word_counts.items())):
            indices.append(hash(word) % 100000)
            values.append(float(count))

        return SparseVector(indices=indices, values=values)

    async def hybrid_search_with_sparse(
        self,
        collection_name: str,
        query_vector: List[float],
        query_text: str,
        limit: int = 10,
        query_filter: Optional[Filter] = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Recherche hybride utilisant vecteurs denses ET sparse.
        """
        client = self._get_client(collection_name)

        prefetch_queries = [
            Prefetch(
                query=query_vector,
                using="",
                limit=limit * 3
            )
        ]

        if query_text:
            sparse_vector = self._generate_sparse_vector(query_text)
            prefetch_queries.append(
                Prefetch(
                    query=sparse_vector,
                    using="text_keywords",
                    limit=limit * 3
                )
            )

        results = client.query_points(
            collection_name=collection_name,
            prefetch=prefetch_queries,
            query=Query(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True
        )

        return [
            {
                "id": str(hit.id),
                "score": hit.score,
                "payload": hit.payload or {}
            }
            for hit in results.points
        ]





    async def create_collection_alias(
        self,
        collection_name: str,
        alias_name: str
    ) -> bool:
        """
        Crée un alias pour une collection (utile pour blue-green deployments).
        """
        try:
            client = self._get_client(collection_name)
            client.update_collection_aliases(
                change_aliases_operations=[
                    {
                        "create_alias": {
                            "collection_name": collection_name,
                            "alias_name": alias_name
                        }
                    }
                ]
            )
            print(f" Alias '{alias_name}' créé pour collection '{collection_name}'")
            return True
        except Exception as e:
            print(f" Création alias échouée: {e}")
            return False

    async def switch_collection_alias(
        self,
        old_collection: str,
        new_collection: str,
        alias_name: str
    ) -> bool:
        """
        Bascule un alias d'une collection à une autre (zero-downtime).
        """
        try:
            client = self._get_client(old_collection)
            client.update_collection_aliases(
                change_aliases_operations=[
                    {
                        "delete_alias": {
                            "alias_name": alias_name
                        }
                    },
                    {
                        "create_alias": {
                            "collection_name": new_collection,
                            "alias_name": alias_name
                        }
                    }
                ]
            )
            print(f" Alias '{alias_name}' basculé de '{old_collection}' vers '{new_collection}'")
            return True
        except Exception as e:
            print(f" Switch alias échoué: {e}")
            return False

    async def create_snapshot(
        self,
        collection_name: str
    ) -> Optional[str]:
        """
        Crée un snapshot de la collection pour backup.
        """
        try:
            client = self._get_client(collection_name)
            snapshot_info = client.create_snapshot(collection_name=collection_name)
            print(f" Snapshot créé: {snapshot_info.name}")
            return snapshot_info.name
        except Exception as e:
            print(f" Création snapshot échouée: {e}")
            return None

    async def list_snapshots(
        self,
        collection_name: str
    ) -> List[Dict[str, Any]]:
        """
        Liste les snapshots disponibles pour une collection.
        """
        try:
            client = self._get_client(collection_name)
            snapshots = client.list_snapshots(collection_name=collection_name)
            return [
                {
                    "name": s.name,
                    "creation_time": s.creation_time,
                    "size": s.size
                }
                for s in snapshots
            ]
        except Exception as e:
            print(f" Liste snapshots échouée: {e}")
            return []

    async def restore_snapshot(
        self,
        collection_name: str,
        snapshot_name: str
    ) -> bool:
        """
        Restaure une collection à partir d'un snapshot.
        """
        try:
            client = self._get_client(collection_name)
            client.recover_snapshot(
                collection_name=collection_name,
                location=snapshot_name
            )
            print(f" Snapshot '{snapshot_name}' restauré pour '{collection_name}'")
            return True
        except Exception as e:
            print(f" Restauration snapshot échouée: {e}")
            return False





    async def set_payload(
        self,
        collection_name: str,
        point_id: str,
        payload: Dict[str, Any]
    ) -> bool:
        """
        Met à jour le payload d'un point sans réindexer.
        """
        try:
            client = self._get_client(collection_name)
            client.set_payload(
                collection_name=collection_name,
                payload=payload,
                points=[point_id]
            )
            print(f" Payload mis à jour pour point {point_id}")
            return True
        except Exception as e:
            print(f" Set payload échoué: {e}")
            return False

    async def overwrite_payload(
        self,
        collection_name: str,
        point_id: str,
        payload: Dict[str, Any]
    ) -> bool:
        """
        Remplace entièrement le payload d'un point.
        """
        try:
            client = self._get_client(collection_name)
            client.overwrite_payload(
                collection_name=collection_name,
                payload=payload,
                points=[point_id]
            )
            print(f" Payload remplacé pour point {point_id}")
            return True
        except Exception as e:
            print(f" Overwrite payload échoué: {e}")
            return False

    async def delete_payload_keys(
        self,
        collection_name: str,
        point_id: str,
        keys: List[str]
    ) -> bool:
        """
        Supprime des clés spécifiques du payload.
        """
        try:
            client = self._get_client(collection_name)
            client.delete_payload(
                collection_name=collection_name,
                keys=keys,
                points=[point_id]
            )
            print(f" Clés {keys} supprimées du payload pour point {point_id}")
            return True
        except Exception as e:
            print(f" Delete payload keys échoué: {e}")
            return False





    async def random_sample(
        self,
        collection_name: str,
        limit: int = 10,
        query_filter: Optional[Filter] = None
    ) -> List[Dict[str, Any]]:
        """
        Échantillonnage aléatoire de points pour diversité.
        """
        try:
            client = self._get_client(collection_name)
            results, _ = client.scroll(
                collection_name=collection_name,
                limit=limit * 10,
                with_payload=True,
                filter=query_filter
            )

            import random
            sampled = random.sample(results, min(limit, len(results)))

            return [
                {
                    "id": str(point.id),
                    "payload": point.payload or {}
                }
                for point in sampled
            ]
        except Exception as e:
            print(f" Random sample échoué: {e}")
            return []

    async def search_with_oversampling(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        oversampling_factor: float = 2.0,
        query_filter: Optional[Filter] = None
    ) -> List[Dict[str, Any]]:
        """
        Recherche avec sur-échantillonnage pour améliorer la précision avec quantization.
        """
        client = self._get_client(collection_name)

        oversample_limit = int(limit * oversampling_factor)

        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=oversample_limit,
            query_filter=query_filter,
            with_payload=True
        )

        return [
            {
                "id": str(point.id),
                "score": point.score,
                "payload": point.payload or {}
            }
            for point in results.points[:limit]
        ]

    async def faceted_count(
        self,
        collection_name: str,
        facet_field: str,
        query_filter: Optional[Filter] = None
    ) -> Dict[str, int]:
        """
        Comptage par facettes (pour filtres dans l'UI).
        """
        try:
            all_points = await self.scroll(
                collection_name=collection_name,
                limit=10000,
                with_payload=True
            )

            facet_counts = {}
            for point in all_points:
                value = point.get("payload", {}).get(facet_field, "unknown")
                if isinstance(value, dict):
                    value = value.get("organism", "unknown")
                facet_counts[str(value)] = facet_counts.get(str(value), 0) + 1

            return dict(sorted(facet_counts.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            print(f" Faceted count échoué: {e}")
            return {}
    def build_metadata_filter(
        self,
        conditions: Optional[Dict[str, Any]] = None
    ) -> Optional[Filter]:
        if not conditions:
            return None
        must_conditions = []


        if conditions.get("organism"):
            organism_value = conditions["organism"]

            must_conditions.append(
                FieldCondition(
                    key="organism",
                    match=MatchText(text=organism_value)
                )
            )


        if conditions.get("assay"):
            must_conditions.append(
                FieldCondition(
                    key="assay",
                    match=MatchText(text=conditions["assay"])
                )
            )


        if conditions.get("type"):
            must_conditions.append(
                FieldCondition(
                    key="type",
                    match=MatchValue(value=conditions["type"])
                )
            )


        if conditions.get("source"):
            must_conditions.append(
                FieldCondition(
                    key="source",
                    match=MatchValue(value=conditions["source"])
                )
            )


        if conditions.get("contact"):
            must_conditions.append(
                FieldCondition(
                    key="contact",
                    match=MatchText(text=conditions["contact"])
                )
            )




        if not must_conditions:
            return None
        return Filter(must=must_conditions)
    def _get_client(self, collection_name: str) -> QdrantClient:
        print(f" _get_client: collection={collection_name}, cloud={self.cloud_client is not None}, private={self.private_client is not None}")

        if collection_name in ["public_science", "biomemory_experiments"]:
            if self.cloud_client:
                print(f" Using CLOUD client for {collection_name}")
                return self.cloud_client
            if self.private_client:
                print(f" Fallback to PRIVATE client for {collection_name}")
                return self.private_client
        if collection_name == "private_experiments" or collection_name.startswith("private_experiments_"):
            if self.private_client:
                print(f" Using PRIVATE client for {collection_name}")
                return self.private_client

            if self.cloud_client:
                print(f" Fallback to CLOUD client for {collection_name}")
                return self.cloud_client
        raise ValueError(f"No Qdrant client available for collection {collection_name}")
    async def search_temporal_advanced(
        self,
        collection_name: str,
        query_vector: List[float],
        date_range: Dict[str, str],
        limit: int = 10,
        query_filter: Optional[Filter] = None
    ) -> List[Dict[str, Any]]:
        client = self._get_client(collection_name)
        temporal_conditions = []
        if date_range.get("start_date"):
            temporal_conditions.append(
                FieldCondition(
                    key="scraped_at",
                    range=Range(gte=date_range["start_date"])
                )
            )
        if date_range.get("end_date"):
            temporal_conditions.append(
                FieldCondition(
                    key="scraped_at",
                    range=Range(lte=date_range["end_date"])
                )
            )
        if temporal_conditions:
            temporal_filter = Filter(must=temporal_conditions)
            if query_filter:
                combined_filter = Filter(must=[query_filter, temporal_filter])
            else:
                combined_filter = temporal_filter
        else:
            combined_filter = query_filter
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            query_filter=combined_filter,
            with_payload=True
        )
        return [
            {
                "id": str(point.id),
                "score": point.score,
                "payload": point.payload or {},
                "temporal_match": True
            }
            for point in results.points
        ]
    async def search_with_grouping(
        self,
        collection_name: str,
        query_vector: List[float],
        group_by: str,
        group_size: int = 3,
        limit: int = 10,
        query_filter: Optional[Filter] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        client = self._get_client(collection_name)
        extended_results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit * 5,
            query_filter=query_filter,
            with_payload=True
        )
        groups = {}
        for point in extended_results.points:
            payload = point.payload or {}
            group_key = payload.get(group_by, "unknown")
            if group_key not in groups:
                groups[group_key] = []
            if len(groups[group_key]) < group_size:
                groups[group_key].append({
                    "id": str(point.id),
                    "score": point.score,
                    "payload": payload,
                    "group": group_key
                })
        if len(groups) > limit:
            group_scores = {}
            for group_key, items in groups.items():
                avg_score = sum(item["score"] for item in items) / len(items)
                group_scores[group_key] = avg_score
            top_groups = sorted(group_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
            groups = {group_key: groups[group_key] for group_key, _ in top_groups}
        return groups
    async def search_with_boosting(
        self,
        collection_name: str,
        query_vector: List[float],
        boost_factors: Dict[str, float],
        limit: int = 10,
        query_filter: Optional[Filter] = None
    ) -> List[Dict[str, Any]]:
        client = self._get_client(collection_name)
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit * 2,
            query_filter=query_filter,
            with_payload=True
        )
        boosted_results = []
        for point in results.points:
            payload = point.payload or {}
            score = point.score
            for boost_key, boost_value in boost_factors.items():
                if payload.get(boost_key):
                    score *= boost_value
            boosted_results.append({
                "id": str(point.id),
                "original_score": point.score,
                "boosted_score": score,
                "payload": payload,
                "boosting_applied": boost_factors
            })
        boosted_results.sort(key=lambda x: x["boosted_score"], reverse=True)
        return boosted_results[:limit]
_qdrant_service = None
def get_qdrant_service() -> QdrantService:
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService()
    return _qdrant_service