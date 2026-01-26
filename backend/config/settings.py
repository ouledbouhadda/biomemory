from pydantic_settings import BaseSettings
from typing import Optional, List
from functools import lru_cache
import json
class Settings(BaseSettings):
    APP_NAME: str = "BioMemory API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ENCRYPTION_KEY: str
    ALLOWED_ORIGINS: str = '["http://localhost:3000"]'
    @property
    def allowed_origins_list(self) -> List[str]:
        try:
            return json.loads(self.ALLOWED_ORIGINS)
        except:
            return ["http://localhost:3000"]
    QDRANT_CLOUD_URL: Optional[str] = None
    QDRANT_CLOUD_API_KEY: Optional[str] = None
    QDRANT_PRIVATE_HOST: str = "localhost"
    QDRANT_PRIVATE_PORT: int = 6333
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-1.5-pro"
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama3-70b-8192"
    USE_GROQ: bool = False
    PROTOCOLS_IO_API_TOKEN: Optional[str] = None
    DATABASE_URL: Optional[str] = "sqlite:///./biomemory.db"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024
    UPLOAD_DIR: str = "/tmp/biomemory/uploads"
    RATE_LIMIT_PER_MINUTE: int = 100
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    TEXT_EMBEDDING_DIM: int = 384
    SEQUENCE_EMBEDDING_DIM: int = 100
    CONDITIONS_EMBEDDING_DIM: int = 4
    @property
    def total_vector_dim(self) -> int:
        return self.TEXT_EMBEDDING_DIM + self.SEQUENCE_EMBEDDING_DIM + self.CONDITIONS_EMBEDDING_DIM
    CHUNKING_ENABLED: bool = True
    CHUNKING_THRESHOLD: int = 1500
    CHUNKING_MAX_SIZE: int = 512
    CHUNKING_OVERLAP: int = 50
    CHUNKING_STRATEGY: str = "auto"
    class Config:
        env_file = ".env"
        case_sensitive = True
@lru_cache()
def get_settings() -> Settings:
    return Settings()