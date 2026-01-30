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
from collections import OrderedDict
from backend.config.settings import get_settings
import numpy as np
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta

logger = logging.getLogger("biomemory.qdrant")
settings = get_settings()


class QdrantCache:
    """TTL-based LRU cache for Qdrant query results."""

    def __init__(self, max_size: int = 256, ttl_seconds: int = 300):
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    def _make_key(self, *args) -> str:
        raw = json.dumps(args, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str):
        if key in self._cache:
            if time.time() - self._timestamps[key] < self._ttl:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            else:
                del self._cache[key]
                del self._timestamps[key]
        self._misses += 1
        return None

    def put(self, key: str, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                oldest_key, _ = self._cache.popitem(last=False)
                self._timestamps.pop(oldest_key, None)
        self._cache[key] = value
        self._timestamps[key] = time.time()

    def invalidate(self):
        self._cache.clear()
        self._timestamps.clear()

    @property
    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0
        }


class CircuitBreaker:
    """Circuit breaker to handle Qdrant Cloud outages gracefully."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self._failure_count = 0
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._last_failure_time = 0.0
        self._state = "closed"  # closed, open, half-open

    @property
    def is_open(self) -> bool:
        if self._state == "open":
            if time.time() - self._last_failure_time > self._recovery_timeout:
                self._state = "half-open"
                return False
            return True
        return False

    def record_success(self):
        self._failure_count = 0
        self._state = "closed"

    def record_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self._failure_threshold:
            self._state = "open"
            logger.warning("Circuit breaker OPEN after %d failures", self._failure_count)

    @property
    def state(self) -> str:
        # Re-check for half-open transition
        if self._state == "open" and time.time() - self._last_failure_time > self._recovery_timeout:
            self._state = "half-open"
        return self._state


class QdrantService:
    def __init__(self):
        self.cloud_client = None
        self._cloud_circuit = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        self._search_cache = QdrantCache(max_size=256, ttl_seconds=300)
        self._stats_cache = QdrantCache(max_size=32, ttl_seconds=600)
        self._facets_cache = QdrantCache(max_size=64, ttl_seconds=600)
        self._scroll_cache = QdrantCache(max_size=64, ttl_seconds=300)

        if settings.DEBUG:
            logger.debug("QDRANT_CLOUD_URL: %s", settings.QDRANT_CLOUD_URL)
            logger.debug("QDRANT_CLOUD_API_KEY: %s", 'SET' if settings.QDRANT_CLOUD_API_KEY else 'NOT SET')

        if (settings.QDRANT_CLOUD_URL and settings.QDRANT_CLOUD_API_KEY and
            settings.QDRANT_CLOUD_URL != "https://your-cluster.qdrant.io" and
            settings.QDRANT_CLOUD_API_KEY != "your-qdrant-cloud-api-key"):
            try:
                self.cloud_client = QdrantClient(
                    url=settings.QDRANT_CLOUD_URL,
                    api_key=settings.QDRANT_CLOUD_API_KEY,
                    timeout=30,
                    prefer_grpc=True
                )
                collections = self.cloud_client.get_collections()
                logger.info("Qdrant Cloud initialized - %d collections", len(collections.collections))
                for col in collections.collections:
                    logger.info("  Collection: %s", col.name)
            except Exception as e:
                logger.error("Qdrant Cloud unavailable: %s", e)
        else:
            logger.info("Qdrant Cloud not configured")
        try:
            self.private_client = QdrantClient(
                host=settings.QDRANT_PRIVATE_HOST,
                port=settings.QDRANT_PRIVATE_PORT,
                timeout=30
            )
        except Exception as e:
            logger.warning("Qdrant Private unavailable: %s", e)
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
                    collection_name="public_science",
                    **collection_config
                )
                self._setup_payload_schema(self.cloud_client, "public_science")
                logger.info("Collection 'public_science' created with advanced features")
            except Exception as e:
                logger.info("Collection 'public_science' already exists or error: %s", e)
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
                logger.info("Collection 'private_experiments' created with advanced features")
                self.private_client.create_collection(
                    collection_name="biomemory_users",
                    vectors_config=VectorParams(
                        size=128,
                        distance=Distance.COSINE
                    )
                )
                logger.info("Collection 'biomemory_users' created")
            except Exception as e:
                logger.info("Private collections already exist or error: %s", e)
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
            logger.info("Payload schema configured for %s", collection_name)
        except Exception as e:
            logger.error("Payload schema configuration error: %s", e)
            try:
                client.create_collection(
                    collection_name="private_experiments",
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info("Collection 'private_experiments' created (fallback)")
            except Exception as e2:
                logger.error("Fallback collection creation failed: %s", e2)
                logger.error("Collection 'private_experiments' already exists or error: %s", e)
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
        # Invalidate caches after data mutation
        self._search_cache.invalidate()
        self._stats_cache.invalidate()
        self._facets_cache.invalidate()
    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        query_filter: Optional[Filter] = None,
        with_payload: bool = True,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        # Check cache first
        vec_hash = hashlib.sha256(np.array(query_vector).tobytes()).hexdigest()[:16]
        filter_str = str(query_filter) if query_filter else ""
        cache_key = self._search_cache._make_key(
            "search", collection_name, vec_hash, limit, filter_str, score_threshold
        )
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            logger.info("Search cache HIT for %s (key=%s)", collection_name, cache_key[:8])
            return cached

        client = self._get_client(collection_name)
        logger.debug("Qdrant search: collection=%s, vector_dim=%d, limit=%d",
                      collection_name, len(query_vector), limit)

        try:
            results = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=with_payload,
                score_threshold=score_threshold
            )
            # Record success for circuit breaker
            if collection_name in ["public_science", "biomemory_experiments"]:
                self._cloud_circuit.record_success()

            logger.info("Qdrant search returned %d results from %s", len(results.points), collection_name)
            formatted = [
                {
                    "id": str(point.id),
                    "score": point.score,
                    "payload": point.payload if with_payload else {}
                }
                for point in results.points
            ]
            self._search_cache.put(cache_key, formatted)
            return formatted
        except Exception as e:
            # Record failure for circuit breaker
            if collection_name in ["public_science", "biomemory_experiments"]:
                self._cloud_circuit.record_failure()
            logger.error("Qdrant search error on %s: %s", collection_name, e)
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
        # Check cache
        vec_hash = hashlib.sha256(np.array(query_vector).tobytes()).hexdigest()[:16]
        filter_str = str(query_filter) if query_filter else ""
        cache_key = self._search_cache._make_key(
            "hybrid", collection_name, vec_hash, query_text, limit, filter_str
        )
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            logger.info("Hybrid search cache HIT for %s", collection_name)
            return cached

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
            logger.info("Hybrid search returned %s results", len(results.points))
            formatted = [
                {
                    "id": str(hit.id),
                    "score": hit.score,
                    "payload": hit.payload or {}
                }
                for hit in results.points
            ]
            self._search_cache.put(cache_key, formatted)
            return formatted
        except Exception as e:
            logger.error("Hybrid search error: %s", e, exc_info=True)

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
            logger.info("User collection '%s' created", collection_name)
            return collection_name
        except Exception as e:
            logger.info("Collection '%s' already exists or error: %s", collection_name, e)
            return collection_name
    async def get_collection_stats(
        self,
        collection_name: str
    ) -> Dict[str, Any]:
        cache_key = self._stats_cache._make_key("stats", collection_name)
        cached = self._stats_cache.get(cache_key)
        if cached is not None:
            logger.info("Stats cache HIT for %s", collection_name)
            return cached

        client = self._get_client(collection_name)
        try:
            info = client.get_collection(collection_name)
            count = await self.count_points(collection_name)
            result = {
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
            self._stats_cache.put(cache_key, result)
            return result
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
            logger.error("Qdrant %s health check failed: %s", instance, e)
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
        logger.info("Upsert with sparse vectors: %s points", len(point_structs))

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
            logger.info("Alias '%s' created for collection '%s'", alias_name, collection_name)
            return True
        except Exception as e:
            logger.error("Alias creation failed: %s", e)
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
            logger.info("Alias '%s' switched from '%s' to '%s'", alias_name, old_collection, new_collection)
            return True
        except Exception as e:
            logger.error("Switch alias failed: %s", e)
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
            logger.info("Snapshot created: %s", snapshot_info.name)
            return snapshot_info.name
        except Exception as e:
            logger.error("Snapshot creation failed: %s", e)
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
            logger.error("List snapshots failed: %s", e)
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
            logger.info("Snapshot '%s' restored for '%s'", snapshot_name, collection_name)
            return True
        except Exception as e:
            logger.error("Snapshot restoration failed: %s", e)
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
            logger.info("Payload updated for point %s", point_id)
            return True
        except Exception as e:
            logger.error("Set payload failed: %s", e)
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
            logger.info("Payload replaced for point %s", point_id)
            return True
        except Exception as e:
            logger.error("Overwrite payload failed: %s", e)
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
            logger.info("Keys %s deleted from payload for point %s", keys, point_id)
            return True
        except Exception as e:
            logger.error("Delete payload keys failed: %s", e)
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
            logger.error("Random sample failed: %s", e)
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
        cache_key = self._facets_cache._make_key("facets", collection_name, facet_field)
        cached = self._facets_cache.get(cache_key)
        if cached is not None:
            logger.info("Facets cache HIT for %s.%s", collection_name, facet_field)
            return cached

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

            result = dict(sorted(facet_counts.items(), key=lambda x: x[1], reverse=True))
            self._facets_cache.put(cache_key, result)
            return result
        except Exception as e:
            logger.error("Faceted count failed: %s", e)
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

        # Range filter for temperature (+/- 5 degrees tolerance)
        if conditions.get("temperature") is not None:
            try:
                temp = float(conditions["temperature"])
                tolerance = conditions.get("temperature_tolerance", 5.0)
                must_conditions.append(
                    FieldCondition(
                        key="temperature",
                        range=Range(gte=str(temp - tolerance), lte=str(temp + tolerance))
                    )
                )
            except (ValueError, TypeError):
                pass

        # Range filter for pH (+/- 1.0 tolerance)
        if conditions.get("ph") is not None:
            try:
                ph = float(conditions["ph"])
                tolerance = conditions.get("ph_tolerance", 1.0)
                must_conditions.append(
                    FieldCondition(
                        key="ph",
                        range=Range(gte=str(ph - tolerance), lte=str(ph + tolerance))
                    )
                )
            except (ValueError, TypeError):
                pass

        # Success filter
        if conditions.get("success_only") is True:
            must_conditions.append(
                FieldCondition(
                    key="success",
                    match=MatchValue(value=True)
                )
            )

        if not must_conditions:
            return None
        return Filter(must=must_conditions)
    def _get_client(self, collection_name: str) -> QdrantClient:
        # Circuit breaker check for cloud
        if collection_name in ["public_science", "biomemory_experiments"]:
            if self.cloud_client and not self._cloud_circuit.is_open:
                logger.debug("Using CLOUD client for %s", collection_name)
                return self.cloud_client
            if self._cloud_circuit.is_open:
                logger.warning("Circuit breaker OPEN - falling back for %s", collection_name)
            if self.private_client:
                logger.debug("Fallback to PRIVATE client for %s", collection_name)
                return self.private_client
        if collection_name == "private_experiments" or collection_name.startswith("private_experiments_"):
            if self.private_client:
                return self.private_client
            if self.cloud_client:
                logger.debug("Fallback to CLOUD client for %s", collection_name)
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

    async def record_feedback(
        self,
        experiment_id: str,
        feedback: str,
        query_text: str = "",
        collection_name: str = "public_science"
    ) -> Dict[str, Any]:
        """Record user feedback (like/dislike) on a search result."""
        try:
            client = self._get_client(collection_name)
            # Store feedback as payload update
            client.set_payload(
                collection_name=collection_name,
                payload={
                    "user_feedback": feedback,
                    "feedback_query": query_text,
                    "feedback_at": datetime.now().isoformat()
                },
                points=[experiment_id]
            )
            logger.info("Feedback recorded: %s for %s", feedback, experiment_id)
            return {"status": "success", "experiment_id": experiment_id, "feedback": feedback}
        except Exception as e:
            logger.error("Feedback recording failed: %s", e)
            return {"status": "error", "error": str(e)}

    @property
    def cache_stats(self) -> Dict[str, Any]:
        return {
            "search_cache": self._search_cache.stats,
            "stats_cache": self._stats_cache.stats,
            "facets_cache": self._facets_cache.stats,
            "scroll_cache": self._scroll_cache.stats,
        }

    @property
    def circuit_breaker_state(self) -> str:
        return self._cloud_circuit.state

_qdrant_service = None
def get_qdrant_service() -> QdrantService:
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService()
    return _qdrant_service