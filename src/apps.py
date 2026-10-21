"""Per-application configuration, kept apart from the RAG engine.

Each app (website/product that embeds the chatbot) gets its own knowledge base,
collection, instructions and retrieval settings. The server picks the collection
from this file, so a client can never choose which index it searches.
File: config/apps.json (override with APPS_CONFIG env var).
"""

import json
import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPS_CONFIG = Path(os.environ.get("APPS_CONFIG", PROJECT_ROOT / "config" / "apps.json"))


class AppConfig(BaseModel):
    app_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    name: str
    enabled: bool = True
    knowledge_base_dir: str  # server-only; relative to the project root
    collection: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{2,62}$")  # ChromaDB name rules
    instructions: str = ""
    top_k: int = Field(4, ge=1, le=20)
    max_distance: float | None = Field(None, gt=0, le=2)  # cosine distance; null = no filter
    exclude_sources: list[str] = []  # KB files never searched for this app (e.g. design notes about the chatbot)
    roles: dict[str, list[str]] = {}  # role -> words naming it; used to mark which role a document section is about
    allowed_origins: list[str] = []  # browser origins allowed to call this app; [] = any origin
    api_key_env: str | None = None  # name of an env var holding this app's API key; null = no key
    welcome_message: str = ""
    example_questions: list[str] = []

    @field_validator("allowed_origins")
    @classmethod
    def _no_trailing_slash(cls, origins: list[str]) -> list[str]:
        return [o.rstrip("/") for o in origins]

    @property
    def kb_path(self) -> Path:
        return (PROJECT_ROOT / self.knowledge_base_dir).resolve()

    def public_info(self) -> dict:
        """The only fields safe to send to a browser."""
        return {
            "app_id": self.app_id,
            "name": self.name,
            "welcome_message": self.welcome_message,
            "example_questions": self.example_questions,
        }


@lru_cache
def load_apps() -> dict[str, AppConfig]:
    data = json.loads(APPS_CONFIG.read_text(encoding="utf-8"))
    apps = [AppConfig(**a) for a in data["apps"]]
    ids = [a.app_id for a in apps]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate app_id in {APPS_CONFIG}")
    collections = [a.collection for a in apps]
    if len(collections) != len(set(collections)):
        raise ValueError(f"Two apps share one collection in {APPS_CONFIG}; each app needs its own index.")
    return {a.app_id: a for a in apps}


def get_app(app_id: str) -> AppConfig | None:
    """Return the app only if it exists and is enabled."""
    app = load_apps().get(app_id)
    return app if app and app.enabled else None
