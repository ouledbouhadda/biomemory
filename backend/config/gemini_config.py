import google.generativeai as genai
from .settings import get_settings
settings = get_settings()
genai.configure(api_key=settings.GEMINI_API_KEY)
SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    }
]
GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 2048,
}
def get_gemini_model(model_name: str = None):
    model_name = model_name or settings.GEMINI_MODEL
    return genai.GenerativeModel(
        model_name=model_name,
        safety_settings=SAFETY_SETTINGS,
        generation_config=GENERATION_CONFIG
    )