import logging
import hashlib
import numpy as np
from typing import Dict, Any, Optional
from collections import OrderedDict
from backend.config.settings import get_settings
import base64
from PIL import Image
from io import BytesIO
import torch

logger = logging.getLogger("biomemory.embedding")
settings = get_settings()


class LRUEmbeddingCache:
    """Cache LRU pour les embeddings avec TTL."""

    def __init__(self, max_size: int = 512):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def _make_key(self, text: Optional[str], sequence: Optional[str],
                  conditions: Optional[Dict], include_image: bool) -> str:
        raw = f"{text or ''}|{sequence or ''}|{sorted(conditions.items()) if conditions else ''}|{include_image}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> Optional[np.ndarray]:
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: str, value: np.ndarray):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
        self._cache[key] = value

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 3)
        }


# Synonymes biologiques pour query expansion
BIO_SYNONYMS = {
    "pcr": ["polymerase chain reaction", "amplification"],
    "polymerase chain reaction": ["pcr", "amplification"],
    "western blot": ["immunoblot", "western blotting", "protein blot"],
    "immunoblot": ["western blot", "western blotting"],
    "qpcr": ["quantitative pcr", "real-time pcr", "rt-pcr"],
    "rt-pcr": ["reverse transcription pcr", "qpcr"],
    "elisa": ["enzyme-linked immunosorbent assay", "immunoassay"],
    "facs": ["flow cytometry", "fluorescence-activated cell sorting"],
    "flow cytometry": ["facs", "fluorescence-activated cell sorting"],
    "crispr": ["crispr-cas9", "gene editing", "genome editing"],
    "gene editing": ["crispr", "crispr-cas9", "genome editing"],
    "transfection": ["gene delivery", "dna delivery"],
    "cloning": ["molecular cloning", "gene cloning"],
    "sequencing": ["dna sequencing", "ngs", "next-generation sequencing"],
    "ngs": ["next-generation sequencing", "sequencing"],
    "mass spectrometry": ["ms", "mass spec", "proteomics"],
    "microscopy": ["imaging", "fluorescence microscopy"],
    "cell culture": ["cell line", "in vitro culture"],
    "protein expression": ["recombinant protein", "protein production"],
    "gel electrophoresis": ["agarose gel", "page", "sds-page"],
    "sds-page": ["gel electrophoresis", "protein gel"],
    "southern blot": ["dna blot", "southern blotting"],
    "northern blot": ["rna blot", "northern blotting"],
    "rna extraction": ["rna isolation", "rna purification"],
    "dna extraction": ["dna isolation", "dna purification"],
    "human": ["homo sapiens", "h. sapiens"],
    "mouse": ["mus musculus", "m. musculus", "murine"],
    "ecoli": ["e. coli", "escherichia coli"],
    "yeast": ["saccharomyces cerevisiae", "s. cerevisiae"],
    "drosophila": ["fruit fly", "d. melanogaster"],
    "arabidopsis": ["a. thaliana", "arabidopsis thaliana"],
    "rat": ["rattus norvegicus", "r. norvegicus"],
}


def expand_query(text: str) -> str:
    """Expand query text with biological synonyms for better recall."""
    if not text:
        return text
    words_lower = text.lower()
    expansions = []
    for term, synonyms in BIO_SYNONYMS.items():
        if term in words_lower:
            for syn in synonyms[:2]:  # max 2 synonyms per term
                if syn not in words_lower:
                    expansions.append(syn)
    if expansions:
        expanded = text + " " + " ".join(expansions)
        logger.debug("Query expanded: '%s' -> '%s'", text[:50], expanded[:100])
        return expanded
    return text


class EmbeddingService:
    def __init__(self):
        self._embedding_cache = LRUEmbeddingCache(max_size=512)

        try:
            self.text_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info("SentenceTransformer model loaded: %s", settings.EMBEDDING_MODEL)
        except Exception as e:
            logger.warning("Failed to load SentenceTransformer: %s", e)
            self.text_model = None

        # Lazy-loaded CLIP model
        self._clip_model = None
        self._clip_processor = None

        self.kmer_size = 3
        self.text_dim = settings.TEXT_EMBEDDING_DIM
        self.seq_dim = settings.SEQUENCE_EMBEDDING_DIM
        self.cond_dim = settings.CONDITIONS_EMBEDDING_DIM
        self.image_dim = 512

    @property
    def clip_model(self):
        """Lazy load CLIP model only when needed (saves ~600MB at startup)."""
        if self._clip_model is None:
            try:
                from transformers import CLIPModel
                self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                logger.info("CLIP model loaded (lazy)")
            except Exception as e:
                logger.warning("Failed to load CLIP model: %s", e)
        return self._clip_model

    @property
    def clip_processor(self):
        """Lazy load CLIP processor only when needed."""
        if self._clip_processor is None:
            try:
                from transformers import CLIPProcessor
                self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                logger.info("CLIP processor loaded (lazy)")
            except Exception as e:
                logger.warning("Failed to load CLIP processor: %s", e)
        return self._clip_processor

    @property
    def total_dim(self) -> int:
        """Dimension totale sans image (488 = 384 + 100 + 4)."""
        return self.text_dim + self.seq_dim + self.cond_dim

    @property
    def cache_stats(self) -> Dict[str, Any]:
        return self._embedding_cache.stats

    async def generate_multimodal_embedding(
        self,
        text: Optional[str] = None,
        sequence: Optional[str] = None,
        conditions: Optional[Dict[str, Any]] = None,
        image_base64: Optional[str] = None,
        include_image: bool = False
    ) -> np.ndarray:
        # Check cache (skip for image queries since base64 is too large for key)
        cache_key = None
        if not image_base64:
            cache_key = self._embedding_cache._make_key(text, sequence, conditions, include_image)
            cached = self._embedding_cache.get(cache_key)
            if cached is not None:
                logger.debug("Embedding cache HIT (rate: %.1f%%)", self._embedding_cache.hit_rate * 100)
                return cached

        text_emb = self._encode_text(text)
        seq_emb = self._encode_sequence(sequence)
        cond_emb = self._encode_conditions(conditions)

        if image_base64 and include_image:
            image_emb = self._encode_image(image_base64)
        else:
            image_emb = np.zeros(self.image_dim)

        if include_image:
            unified_emb = self._fuse_embeddings(text_emb, seq_emb, cond_emb, image_emb)
        else:
            unified_emb = self._fuse_embeddings_no_image(text_emb, seq_emb, cond_emb)

        # Store in cache
        if cache_key:
            self._embedding_cache.put(cache_key, unified_emb)

        return unified_emb

    def _encode_text(self, text: Optional[str]) -> np.ndarray:
        if not text or text.strip() == "":
            return np.zeros(self.text_dim)
        if self.text_model is None:
            hash_obj = hashlib.md5(text.encode())
            hash_bytes = hash_obj.digest()
            embedding = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)
            if len(embedding) < self.text_dim:
                embedding = np.pad(embedding, (0, self.text_dim - len(embedding)))
            else:
                embedding = embedding[:self.text_dim]
            return embedding
        embedding = self.text_model.encode(text)
        if len(embedding) < self.text_dim:
            embedding = np.pad(embedding, (0, self.text_dim - len(embedding)))
        elif len(embedding) > self.text_dim:
            embedding = embedding[:self.text_dim]
        return embedding

    def _encode_sequence(self, sequence: Optional[str]) -> np.ndarray:
        if not sequence or sequence.strip() == "":
            return np.zeros(self.seq_dim)
        kmers = {}
        for i in range(len(sequence) - self.kmer_size + 1):
            kmer = sequence[i:i + self.kmer_size]
            kmers[kmer] = kmers.get(kmer, 0) + 1
        kmer_vector = np.zeros(self.seq_dim)
        sorted_kmers = sorted(kmers.items(), key=lambda x: x[1], reverse=True)
        for idx, (kmer, count) in enumerate(sorted_kmers[:self.seq_dim]):
            kmer_vector[idx] = count
        norm = np.linalg.norm(kmer_vector)
        if norm > 0:
            kmer_vector = kmer_vector / norm
        return kmer_vector

    def _encode_conditions(self, conditions: Optional[Dict[str, Any]]) -> np.ndarray:
        if not conditions:
            return np.zeros(self.cond_dim)
        organism_map = {
            'human': 0, 'mouse': 1, 'ecoli': 2, 'yeast': 3,
            'arabidopsis': 4, 'drosophila': 5, 'other': 6
        }
        organism_code = organism_map.get(
            conditions.get('organism', 'other'), 6
        ) / 7.0
        temp_val = conditions.get('temperature', 37.0)
        temperature = (temp_val if temp_val is not None else 37.0) / 100.0
        ph_val = conditions.get('ph', 7.0)
        ph = (ph_val if ph_val is not None else 7.0) / 14.0
        success = 1.0 if conditions.get('success') else 0.0
        features = [organism_code, temperature, ph, success]
        while len(features) < self.cond_dim:
            features.append(0.0)
        return np.array(features[:self.cond_dim])

    def _encode_image(self, image_base64: Optional[str]) -> np.ndarray:
        if not image_base64 or self.clip_model is None or self.clip_processor is None:
            return np.zeros(self.image_dim)
        try:
            image = self._decode_image(image_base64)
            inputs = self.clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            embedding = image_features.squeeze().numpy()
            embedding = self._normalize(embedding)
            if len(embedding) < self.image_dim:
                embedding = np.pad(embedding, (0, self.image_dim - len(embedding)))
            elif len(embedding) > self.image_dim:
                embedding = embedding[:self.image_dim]
            return embedding
        except Exception as e:
            logger.error("Image encoding failed: %s", e)
            return np.zeros(self.image_dim)

    def describe_image(self, image_base64: str) -> str:
        """Use CLIP zero-shot classification to generate a text description from an image.
        Returns the best matching biological/scientific labels as a text query."""
        if not image_base64 or self.clip_model is None or self.clip_processor is None:
            return ""

        labels = [
            "DNA double helix structure",
            "DNA gel electrophoresis",
            "DNA sequencing results",
            "DNA amplification PCR",
            "ADN acide desoxyribonucleique",
            "gene expression analysis",
            "protein structure",
            "protein gel electrophoresis SDS-PAGE",
            "western blot results",
            "microscopy cell imaging",
            "fluorescence microscopy",
            "flow cytometry results",
            "bacterial culture E. coli",
            "cell culture tissue",
            "CRISPR gene editing",
            "RNA extraction",
            "chromosome karyotype",
            "plasmid vector map",
            "phylogenetic tree",
            "mass spectrometry proteomics",
            "biological laboratory experiment",
            "molecular biology technique",
        ]

        try:
            image = self._decode_image(image_base64)
            inputs = self.clip_processor(
                text=labels, images=image, return_tensors="pt", padding=True
            )
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
            logits = outputs.logits_per_image.squeeze()
            probs = torch.softmax(logits, dim=0).numpy()

            # Take top 3 labels
            top_indices = probs.argsort()[-3:][::-1]
            top_labels = [labels[i] for i in top_indices if probs[i] > 0.05]
            description = " ".join(top_labels) if top_labels else labels[top_indices[0]]
            logger.info("CLIP image description: %s", description[:100])
            return description
        except Exception as e:
            logger.error("CLIP image description failed: %s", e)
            return ""

    def _fuse_embeddings(
        self, text_emb: np.ndarray, seq_emb: np.ndarray,
        cond_emb: np.ndarray, image_emb: np.ndarray
    ) -> np.ndarray:
        text_emb = self._normalize(text_emb)
        seq_emb = self._normalize(seq_emb)
        cond_emb = self._normalize(cond_emb)
        image_emb = self._normalize(image_emb)
        weights = [0.4, 0.3, 0.15, 0.15]
        weighted_embeddings = [
            text_emb * weights[0],
            seq_emb * weights[1],
            cond_emb * weights[2],
            image_emb * weights[3]
        ]
        unified = np.concatenate(weighted_embeddings)
        unified = self._normalize(unified)
        return unified

    def _fuse_embeddings_no_image(
        self, text_emb: np.ndarray, seq_emb: np.ndarray,
        cond_emb: np.ndarray
    ) -> np.ndarray:
        text_emb = self._normalize(text_emb)
        seq_emb = self._normalize(seq_emb)
        cond_emb = self._normalize(cond_emb)
        weights = [0.5, 0.35, 0.15]
        weighted_embeddings = [
            text_emb * weights[0],
            seq_emb * weights[1],
            cond_emb * weights[2]
        ]
        unified = np.concatenate(weighted_embeddings)
        unified = self._normalize(unified)
        return unified

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector

    def _decode_image(self, image_base64: str) -> Image.Image:
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))
        return image


# Import at module level to avoid issues with lazy loading
from sentence_transformers import SentenceTransformer

_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
