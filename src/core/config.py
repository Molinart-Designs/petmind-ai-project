from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Project identity
    project_name: str = Field(default="petmind-ai")
    environment: str = Field(default="development")
    app_version: str = Field(default="0.1.0")

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_prefix: str = Field(default="/api/v1")
    api_key_header_name: str = Field(default="X-API-Key")
    api_key: str = Field(default="replace_with_a_secure_value")

    # LLM
    llm_provider: str = Field(default="openai")
    openai_api_key: str = Field(default="replace_with_your_key")
    llm_model: str = Field(default="gpt-4o-mini")
    embedding_model: str = Field(default="text-embedding-3-small")

    # Vector store / database
    vector_store_provider: str = Field(default="pgvector")
    database_url: str = Field(
        default="postgresql+psycopg://petmind:petmind_dev@localhost:5432/petmind"
    )

    # Local postgres
    postgres_user: str = Field(default="petmind")
    postgres_password: str = Field(default="petmind_dev")
    postgres_db: str = Field(default="petmind")
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)

    # RAG
    chunk_size: int = Field(default=700, ge=100, le=4000)
    chunk_overlap: int = Field(default=100, ge=0, le=1000)
    retriever_top_k: int = Field(default=4, ge=1, le=20)
    similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)

    # Logging
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("api_prefix")
    @classmethod
    def validate_api_prefix(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("API_PREFIX must not be empty.")
        if not value.startswith("/"):
            value = f"/{value}"
        return value.rstrip("/") or "/"

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        value = value.strip().lower()
        allowed = {"development", "test", "staging", "production"}
        if value not in allowed:
            raise ValueError(
                f"ENVIRONMENT must be one of: {', '.join(sorted(allowed))}."
            )
        return value

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, value: str) -> str:
        value = value.strip().lower()
        allowed = {"openai"}
        if value not in allowed:
            raise ValueError(
                f"LLM_PROVIDER must be one of: {', '.join(sorted(allowed))}."
            )
        return value

    @field_validator("vector_store_provider")
    @classmethod
    def validate_vector_store_provider(cls, value: str) -> str:
        value = value.strip().lower()
        allowed = {"pgvector", "chromadb"}
        if value not in allowed:
            raise ValueError(
                f"VECTOR_STORE_PROVIDER must be one of: {', '.join(sorted(allowed))}."
            )
        return value

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        value = value.strip().upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if value not in allowed:
            raise ValueError(
                f"LOG_LEVEL must be one of: {', '.join(sorted(allowed))}."
            )
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, value: int, info) -> int:
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and value >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        return value

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()