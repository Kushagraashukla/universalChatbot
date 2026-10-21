"""API settings, read from environment variables (or .env) and validated at startup.

RAG_CORS_ORIGINS       comma-separated browser origins allowed by CORS
RAG_MAX_QUESTION_CHARS longest question accepted (characters)
RAG_LOG_LEVEL          DEBUG / INFO / WARNING / ERROR
Engine settings (EMBEDDING_MODEL, LLM_MODEL, CHROMA_DIR, APPS_CONFIG, OLLAMA_HOST)
are read by the engine modules themselves; see .env.example.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", extra="ignore")

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    max_question_chars: int = Field(2000, ge=50, le=20000)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip().rstrip("/") for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
