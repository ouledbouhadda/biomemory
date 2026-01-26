from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from backend.config.settings import get_settings
settings = get_settings()
class BioRAGService:
    def __init__(self):
        self.embedding_model = None
        self.generation_model = None
        self.tokenizer = None
        self._load_models()
    def _load_models(self):
        try:
            self.embedding_model = SentenceTransformer(
                "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
            )
            model_name = "microsoft/BioGPT-Large"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.generation_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        except Exception as e:
            print(f"⚠️ Failed to load BioRAG models: {e}")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    async def search_and_generate(
        self,
        query: str,
        retrieved_experiments: List[Dict],
        context_window: int = 5
    ) -> Dict[str, Any]:
        context = self._prepare_context(retrieved_experiments[:context_window])
        response = await self._generate_with_context(query, context)
        confidence = self._calculate_confidence(response, retrieved_experiments)
        return {
            "answer": response,
            "confidence": confidence,
            "supporting_experiments": retrieved_experiments[:context_window],
            "method": "bio_rag"
        }
    def _prepare_context(self, experiments: List[Dict]) -> str:
        context_parts = []
        for exp in experiments:
            payload = exp.get('payload', {})
            context_parts.append(f)
        return "\n".join(context_parts)
    async def _generate_with_context(self, query: str, context: str) -> str:
        prompt = f
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.generation_model.device)
            with torch.no_grad():
                outputs = self.generation_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Answer:" in response:
                response = response.split("Answer:")[1].strip()
            return response
        except Exception as e:
            print(f"⚠️ BioRAG generation failed: {e}")
            return f"Based on similar experiments, I cannot provide a definitive answer due to technical issues. However, the retrieved experiments may contain relevant information."
    def _calculate_confidence(self, response: str, experiments: List[Dict]) -> float:
        if not experiments:
            return 0.0
        avg_similarity = np.mean([exp.get('score', 0) for exp in experiments])
        num_experiments = len(experiments)
        response_length = len(response.split())
        confidence = min(1.0, (avg_similarity * 0.7 + min(num_experiments/5, 1.0) * 0.3))
        return round(confidence, 2)
    async def fine_tune_on_domain(
        self,
        training_data: List[Dict[str, str]],
        epochs: int = 3
    ):
        pass
_bio_rag_service = None
def get_bio_rag_service() -> BioRAGService:
    global _bio_rag_service
    if _bio_rag_service is None:
        _bio_rag_service = BioRAGService()
    return _bio_rag_service