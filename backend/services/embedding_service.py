from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, Any, Optional
from backend.config.settings import get_settings
import base64
from PIL import Image
from io import BytesIO
import torch
from transformers import CLIPProcessor, CLIPModel
settings = get_settings()
class EmbeddingService:
    def __init__(self):
        try:
            self.text_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        except Exception as e:
            print(f"⚠️ Failed to load SentenceTransformer: {e}")
            self.text_model = None
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("✓ CLIP model loaded for image embeddings")
        except Exception as e:
            print(f"⚠️ Failed to load CLIP model: {e}")
            self.clip_model = None
            self.clip_processor = None
        self.kmer_size = 3
        self.text_dim = settings.TEXT_EMBEDDING_DIM
        self.seq_dim = settings.SEQUENCE_EMBEDDING_DIM
        self.cond_dim = settings.CONDITIONS_EMBEDDING_DIM
        self.image_dim = 512
    async def generate_multimodal_embedding(
        self,
        text: Optional[str] = None,
        sequence: Optional[str] = None,
        conditions: Optional[Dict[str, Any]] = None,
        image_base64: Optional[str] = None,
        include_image: bool = True
    ) -> np.ndarray:
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
        return unified_emb
    def _encode_text(self, text: Optional[str]) -> np.ndarray:
        if not text or text.strip() == "":
            return np.zeros(self.text_dim)
        if self.text_model is None:
            import hashlib
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
            'human': 0,
            'mouse': 1,
            'ecoli': 2,
            'yeast': 3,
            'arabidopsis': 4,
            'drosophila': 5,
            'other': 6
        }
        organism_code = organism_map.get(
            conditions.get('organism', 'other'),
            6
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
            print(f"Erreur lors de l'encodage d'image: {e}")
            return np.zeros(self.image_dim)
    def _fuse_embeddings(
        self,
        text_emb: np.ndarray,
        seq_emb: np.ndarray,
        cond_emb: np.ndarray,
        image_emb: np.ndarray
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
        self,
        text_emb: np.ndarray,
        seq_emb: np.ndarray,
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
_embedding_service = None
def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service