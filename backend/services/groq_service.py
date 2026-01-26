import requests
from typing import Dict, Any, Optional
from backend.config.settings import get_settings
settings = get_settings()
class GroqService:
    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.model = settings.GROQ_MODEL
        self.base_url = "https://api.groq.com/openai/v1"
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not configured")
    async def generate_content(
        self,
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> str:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            messages = []
            if system_instruction:
                messages.append({
                    "role": "system",
                    "content": system_instruction
                })
            messages.append({
                "role": "user",
                "content": prompt
            })
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2048,
                "top_p": 0.95
            }
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            if response.status_code != 200:
                raise Exception(f"Groq API error: {response.status_code} - {response.text}")
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"Groq API error: {str(e)}")
    async def analyze_intent(
        self,
        user_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        prompt = f
        try:
            response = await self.generate_content(prompt)
            import json
            return json.loads(response.strip())
        except:
            return {
                "primary_intent": "search_similar",
                "modalities": ["text"],
                "confidence": 0.5
            }
_groq_service = None
def get_groq_service() -> GroqService:
    global _groq_service
    if _groq_service is None:
        _groq_service = GroqService()
    return _groq_service