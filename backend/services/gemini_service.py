import google.generativeai as genai
from backend.config.gemini_config import get_gemini_model, SAFETY_SETTINGS, GENERATION_CONFIG
from typing import Dict, Any, Optional
import json
class GeminiService:
    def __init__(self):
        self.model = get_gemini_model()
    async def generate_content(
        self,
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> str:
        try:
            if system_instruction:
                full_prompt = f"{system_instruction}\n\n{prompt}"
            else:
                full_prompt = prompt
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")
    async def analyze_intent(
        self,
        user_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        prompt = f
        response = await self.generate_content(prompt)
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            intent = json.loads(json_str)
            return intent
        except json.JSONDecodeError:
            return {
                "primary_intent": "search_similar",
                "entities": {},
                "context": {}
            }
    async def generate_design_variants(
        self,
        successes: list,
        failures: list
    ) -> list:
        success_summary = self._summarize_experiments(successes)
        failure_summary = self._summarize_experiments(failures)
        prompt = f
        response = await self.generate_content(prompt)
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            variants = json.loads(json_str)
            return variants
        except json.JSONDecodeError:
            return []
    async def analyze_failure_patterns(
        self,
        failed_experiments: list
    ) -> Dict[str, Any]:
        failure_summary = self._summarize_experiments(failed_experiments)
        prompt = f
        response = await self.generate_content(prompt)
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            analysis = json.loads(json_str)
            return analysis
        except:
            return {
                "common_conditions": [],
                "failure_correlations": {},
                "recommendations": []
            }
    def _summarize_experiments(self, experiments: list) -> str:
        if not experiments:
            return "None"
        summaries = []
        for exp in experiments[:5]:
            payload = exp.get("payload", {})
            summary = f
            summaries.append(summary)
        return "\n".join(summaries)
_gemini_service = None
def get_gemini_service() -> GeminiService:
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service