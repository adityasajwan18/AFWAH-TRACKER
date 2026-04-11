# ============================================================
# backend/core/config.py
# Central configuration using Pydantic Settings.
# Reads from .env file automatically.
# ============================================================

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Pydantic validates types automatically — no more config bugs.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Silently ignore unknown env vars
    )

    # --- App ---
    APP_NAME: str = "Afwaah Tracker"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # CORS: Which origins can call our API
    # In production, lock this down to your frontend domain
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:5500",
        "null",  # Allows file:// origin for local HTML demo
    ]

    # --- MongoDB ---
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "afwaah_tracker"

    # --- Neo4j ---
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password123"

    # --- ML Model ---
    HF_MODEL_NAME: str = "facebook/bart-large-mnli"
    HF_CACHE_DIR: str = "./ml_models_cache"


# Singleton: import this object everywhere instead of re-instantiating
settings = Settings()
