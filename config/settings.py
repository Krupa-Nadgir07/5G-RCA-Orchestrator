"""
5G gNB RCA Multi-Agent Orchestrator
Configuration management using Pydantic Settings.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "5g-rca-orchestrator"
    app_env: str = "development"
    app_debug: bool = False
    app_port: int = 8000
    app_host: str = "0.0.0.0"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_password: Optional[str] = None
    redis_max_connections: int = 50
    redis_working_memory_ttl: int = 3600

    # MongoDB
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_database: str = "rca_orchestrator"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "3gpp_knowledge"
    qdrant_api_key: Optional[str] = None

    # Ollama Inference
    ollama_base_url: str = "http://localhost:11434"
    slm_model: str = "llama-3.3-70b-versatile"
    slm_max_tokens: int = 1024
    slm_temperature: float = 0.3

    # Embedding
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384

    # RAG
    rag_top_k: int = 5
    rag_max_hops: int = 2
    rag_similarity_threshold: float = 0.7
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 64

    # Agent
    agent_confidence_threshold: float = 0.75
    agent_escalation_threshold: float = 0.5
    agent_max_reasoning_steps: int = 5
    agent_timeout_seconds: int = 30

    # Rate Limiting
    rate_limit_requests_per_minute: int = 100
    rate_limit_burst: int = 20

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
