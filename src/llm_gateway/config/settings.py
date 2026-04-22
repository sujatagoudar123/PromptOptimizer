"""Centralized, env-driven configuration. No hardcoded secrets."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration for the gateway.

    Values are loaded from (in order of precedence):
      1. actual environment variables
      2. a .env file in the working directory
      3. defaults defined here
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="LLMGW_",
        extra="ignore",
    )

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    environment: Literal["development", "staging", "production"] = "development"

    # --- Governance ---
    max_prompt_tokens: int = 8000
    reject_oversize: bool = False  # if False, we try to compress instead

    # --- Optimization ---
    optimization_enabled: bool = True
    compression_confidence_warn_threshold: float = 0.6
    aggressive_pruning: bool = True  # entropy-based stopword removal

    # --- Prompt Coaching (quality layer) ---
    coaching_enabled: bool = True
    prompt_quality_threshold: float = 0.55  # rewrite if below this

    # --- Caching ---
    cache_backend: Literal["memory", "redis"] = "memory"
    cache_ttl_seconds: int = 3600
    cache_max_entries: int = 10_000
    semantic_cache_enabled: bool = True
    semantic_similarity_threshold: float = 0.92
    embedding_backend: Literal["hashing", "openai"] = "hashing"
    embedding_dim: int = 256

    # --- Providers ---
    # Comma-separated provider chain. First = primary; rest = failover order.
    provider_chain: str = "echo"
    default_model: str = "echo-1"

    # OpenAI (only used when openai is in provider_chain or embedding_backend)
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_timeout_seconds: float = 30.0

    # --- Redis (only used when cache_backend=redis) ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Request limits ---
    request_timeout_seconds: float = 60.0

    @property
    def provider_list(self) -> list[str]:
        return [p.strip() for p in self.provider_chain.split(",") if p.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor. Reset by calling .cache_clear() in tests."""
    return Settings()
