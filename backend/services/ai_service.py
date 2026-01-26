from typing import Dict, Any, Optional
from backend.config.settings import get_settings
settings = get_settings()
class AIService:
    def __init__(self):
        self.use_groq = settings.USE_GROQ
        if self.use_groq:
            from backend.services.groq_service import get_groq_service
            self.service = get_groq_service()
            self.provider = "Groq"
        else:
            from backend.services.gemini_service import get_gemini_service
            self.service = get_gemini_service()
            self.provider = "Gemini"
    async def generate_content(
        self,
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> str:
        return await self.service.generate_content(prompt, system_instruction)
    async def analyze_intent(
        self,
        user_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        if hasattr(self.service, 'analyze_intent'):
            return await self.service.analyze_intent(user_input)
        else:
            return {
                "primary_intent": "search_similar",
                "modalities": ["text"],
                "confidence": 0.5
            }
_ai_service = None
def get_ai_service() -> AIService:
    global _ai_service
    if _ai_service is None:
        _ai_service = AIService()
    return _ai_service