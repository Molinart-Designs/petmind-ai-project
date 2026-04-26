from functools import lru_cache

from pydantic import AliasChoices, Field, field_validator
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
    legacy_api_key_fallback_enabled: bool = Field(
        default=False,
        description="If true, valid X-API-Key grants access without JWT (temporary migration path).",
    )

    # Auth0 (JWT access tokens + RBAC permissions claim)
    auth0_domain: str = Field(default="", description="Tenant domain, e.g. your-tenant.auth0.com")
    auth0_audience: str = Field(default="", description="API identifier audience for access tokens")
    auth0_issuer: str = Field(default="", description="JWT iss claim value, e.g. https://your-tenant.auth0.com/")
    auth0_algorithms: str = Field(
        default="RS256",
        description="Comma-separated JWT algs, e.g. RS256",
    )

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

    # Trusted external research fallback (Layer 2 → Layer 3)
    enable_trusted_external_retrieval: bool = Field(
        default=False,
        description="Master switch: allow trusted external retrieval stack to be used (still requires allow_provisional_in_query for /query).",
    )
    allow_provisional_in_query: bool = Field(
        default=False,
        description="When false, /query never augments with provisional external evidence (internal-only RAG).",
    )
    external_research_trigger_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description=(
            "Mínimo ``top_score`` del RAG interno para considerar el contexto interno 'fuerte' y **no** "
            "disparar L2 cuando ya hay al menos un chunk ≥ ``SIMILARITY_THRESHOLD``. Si "
            "``retrieval_count==0``, ``matched_count==0``, o ``top_score`` es inferior a este valor, "
            "puede activarse el fallback externo (ver ``RAGOrchestrator._internal_context_reasonably_covers_question``)."
        ),
    )
    trusted_external_allowlist_domains: str = Field(
        default="",
        description=(
            "Comma-separated hostnames trusted for Layer 2 search/scrape (e.g. avma.org,wsava.org). "
            "Use a single ``*`` to allow any http(s) hostname (high risk; see WildcardTrustedSourceRegistry). "
            "Empty registry yields zero hits when not using wildcard."
        ),
    )
    trusted_external_denylist: str = Field(
        default="",
        validation_alias=AliasChoices("TRUSTED_EXTERNAL_DENYLIST", "TRUSTED_EXTERNAL_BLOCKLIST"),
        description=(
            "Comma-separated hosts or URLs never used in Layer 2 (overrides allowlist). "
            "Hostname only blocks that host and subdomains. Full URL blocks that path prefix on the host "
            "(e.g. https://example.org/promotions/spam blocks /promotions/spam and deeper paths)."
        ),
    )
    enable_auto_save_provisional_knowledge: bool = Field(
        default=False,
        description="When true, persist external research bundles as provisional research_candidates after a successful external-augmented answer.",
    )
    trusted_search_provider: str = Field(
        default="stub",
        description="External search backend for Layer 2 allowlisted retrieval: ``stub`` (no network) or ``tavily``.",
    )
    tavily_api_key: str = Field(
        default="",
        description="API key for Tavily Search (https://api.tavily.com/search). Required when TRUSTED_SEARCH_PROVIDER=tavily.",
    )
    tavily_max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Tavily ``max_results`` (bounded allowlisted search, not crawling).",
        validation_alias=AliasChoices("TAVILY_MAX_RESULTS", "TRUSTED_EXTERNAL_MAX_RESULTS"),
    )
    tavily_search_depth: str = Field(
        default="basic",
        description="Tavily ``search_depth`` (basic, advanced, fast, ultra-fast).",
    )
    tavily_topic: str = Field(
        default="general",
        description="Tavily ``topic``: general, news, or finance (see Tavily API).",
    )
    tavily_api_base_url: str = Field(
        default="https://api.tavily.com",
        description="Tavily API origin (override in tests only).",
    )
    trusted_content_provider: str = Field(
        default="stub",
        description="Content fetch for Layer 2: ``stub`` (no Firecrawl) or ``firecrawl`` (markdown scrape per URL).",
    )
    firecrawl_api_key: str = Field(
        default="",
        description="Bearer token for Firecrawl API (https://api.firecrawl.dev). Required when TRUSTED_CONTENT_PROVIDER=firecrawl.",
    )
    firecrawl_api_base_url: str = Field(
        default="https://api.firecrawl.dev",
        description="Firecrawl API origin (v2 scrape path appended in client).",
    )
    firecrawl_max_urls_per_request: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Max distinct URLs to scrape with Firecrawl per trusted research run.",
        validation_alias=AliasChoices("FIRECRAWL_MAX_URLS_PER_REQUEST", "TRUSTED_EXTERNAL_MAX_URLS_TO_SCRAPE"),
    )
    firecrawl_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=180,
        description="Per-URL client timeout; also passed to Firecrawl scrape ``timeout`` (ms capped at 300s).",
        validation_alias=AliasChoices("FIRECRAWL_TIMEOUT_SECONDS", "TRUSTED_EXTERNAL_TIMEOUT_SECONDS"),
    )
    expose_review_draft_in_query_api: bool = Field(
        default=False,
        description=(
            "Deprecated: /query always returns review_draft=null regardless of this flag. "
            "Kept for backward-compatible environment parsing only."
        ),
    )

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

    @field_validator("trusted_search_provider")
    @classmethod
    def validate_trusted_search_provider(cls, value: str) -> str:
        value = value.strip().lower()
        allowed = {"stub", "tavily"}
        if value not in allowed:
            raise ValueError(
                f"TRUSTED_SEARCH_PROVIDER must be one of: {', '.join(sorted(allowed))}."
            )
        return value

    @field_validator("tavily_search_depth")
    @classmethod
    def validate_tavily_search_depth(cls, value: str) -> str:
        value = value.strip().lower()
        allowed = {"advanced", "basic", "fast", "ultra-fast"}
        if value not in allowed:
            raise ValueError(
                f"TAVILY_SEARCH_DEPTH must be one of: {', '.join(sorted(allowed))}."
            )
        return value

    @field_validator("tavily_topic")
    @classmethod
    def validate_tavily_topic(cls, value: str) -> str:
        value = value.strip().lower()
        allowed = {"general", "news", "finance"}
        if value not in allowed:
            raise ValueError(
                f"TAVILY_TOPIC must be one of: {', '.join(sorted(allowed))}."
            )
        return value

    @field_validator("trusted_content_provider")
    @classmethod
    def validate_trusted_content_provider(cls, value: str) -> str:
        value = value.strip().lower()
        allowed = {"stub", "firecrawl"}
        if value not in allowed:
            raise ValueError(
                f"TRUSTED_CONTENT_PROVIDER must be one of: {', '.join(sorted(allowed))}."
            )
        return value

    @field_validator("firecrawl_api_base_url")
    @classmethod
    def validate_firecrawl_api_base_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value:
            raise ValueError("FIRECRAWL_API_BASE_URL must not be empty.")
        if not value.lower().startswith("https://"):
            raise ValueError("FIRECRAWL_API_BASE_URL must use https://")
        return value

    @field_validator("tavily_api_base_url")
    @classmethod
    def validate_tavily_api_base_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value:
            raise ValueError("TAVILY_API_BASE_URL must not be empty.")
        if not value.lower().startswith("https://"):
            raise ValueError("TAVILY_API_BASE_URL must use https://")
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

    @property
    def auth0_algorithm_list(self) -> list[str]:
        parts = [p.strip() for p in self.auth0_algorithms.split(",") if p.strip()]
        return parts or ["RS256"]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()