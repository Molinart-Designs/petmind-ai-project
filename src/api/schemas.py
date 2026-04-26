from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict, field_serializer, field_validator


class PetProfile(BaseModel):
    species: str = Field(..., description="Pet species, e.g. dog or cat")
    breed: str | None = Field(default=None, description="Breed of the pet")
    age_years: float | None = Field(default=None, ge=0, description="Age in years")
    life_stage: str | None = Field(
        default=None,
        description="Life stage, e.g. puppy, adult, senior",
    )
    weight_kg: float | None = Field(default=None, ge=0, description="Weight in kilograms")
    sex: Literal["male", "female", "unknown"] | None = Field(default=None)
    neutered: bool | None = Field(default=None)
    conditions: list[str] = Field(
        default_factory=list,
        description="Known conditions or sensitivities, if any",
    )
    notes: str | None = Field(default=None, max_length=1000)

    model_config = ConfigDict(extra="forbid")


class QueryFilters(BaseModel):
    category: str | None = Field(default=None, description="Knowledge category filter")
    species: str | None = Field(default=None, description="Species filter")
    life_stage: str | None = Field(default=None, description="Life stage filter")

    model_config = ConfigDict(extra="forbid")


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Natural language question for PetMind AI",
    )
    pet_profile: PetProfile | None = Field(
        default=None,
        description="Optional structured pet profile for personalization",
    )
    filters: QueryFilters | None = Field(
        default=None,
        description="Optional metadata filters for retrieval",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="Optional override for retriever top-k",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question must not be empty.")
        return value


class SourceItem(BaseModel):
    document_id: str
    chunk_id: str
    title: str | None = None
    source: str | None = None
    category: str | None = None
    species: str | None = None
    breed: str | None = None
    life_stage: str | None = None
    similarity_score: float | None = Field(default=None, ge=0, le=1)
    snippet: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


QueryAnswerSource = Literal["internal", "external_trusted", "fallback"]
QueryKnowledgeStatus = Literal["approved", "provisional", "none"]


def query_response_dict_from_orchestrator(result: dict[str, Any]) -> dict[str, Any]:
    """
    Subconjunto seguro del dict devuelto por ``RAGOrchestrator.answer`` para construir ``QueryResponse``.

    El orquestador puede incluir claves solo para logging u orquestación; ``QueryResponse`` usa
    ``extra='forbid'``, así que aquí se descartan campos que no formen parte del contrato HTTP público
    (por ejemplo ``used_external``).
    """
    public = frozenset(QueryResponse.model_fields.keys())
    return {k: v for k, v in result.items() if k in public}


class QueryResponse(BaseModel):
    """
    Contrato público de ``POST /query``: solo los campos declarados aquí deben serializarse.

    ``answer_source`` y ``knowledge_status`` indican procedencia del contexto; no incluir campos
    internos (p. ej. flags de orquestación) en la respuesta HTTP.
    """

    answer: str = Field(
        ...,
        description="Concise user-facing answer; safe for direct display to pet owners.",
    )
    review_draft: str | None = Field(
        default=None,
        description=(
            "Reserved for backward-compatible schema shape. The public ``/query`` HTTP API always "
            "returns ``null`` here; internal review text is never exposed to API clients."
        ),
    )
    needs_vet_followup: bool = Field(
        ...,
        description="Whether the response recommends consulting a veterinarian",
    )
    confidence: Literal["high", "medium", "low"]
    sources: list[SourceItem] = Field(default_factory=list)
    retrieval_count: int = Field(default=0, ge=0)
    used_filters: dict[str, Any] = Field(default_factory=dict)
    disclaimers: list[str] = Field(default_factory=list)
    generated_at: datetime
    answer_source: QueryAnswerSource = Field(
        default="internal",
        description=(
            "Where grounding for the answer came from: curated internal retrieval only, "
            "trusted allowlisted external augmentation, or a safe fallback when context was insufficient."
        ),
    )
    knowledge_status: QueryKnowledgeStatus = Field(
        default="approved",
        description=(
            "Lifecycle of knowledge backing the reply: fully curated internal KB, includes provisional "
            "external bundles, or none when no reliable context was available."
        ),
    )

    @field_serializer("review_draft")
    def _review_draft_never_serialized_to_clients(self, _value: str | None) -> None:
        """Defensa en profundidad: el JSON de respuesta nunca incluye borrador interno (solo null)."""
        return None

    model_config = ConfigDict(extra="forbid")


class IngestDocument(BaseModel):
    external_id: str | None = Field(
        default=None,
        description="Optional external identifier from upstream source",
    )
    title: str = Field(..., min_length=3, max_length=300)
    content: str = Field(..., min_length=20, description="Raw curated document content")
    category: str = Field(..., min_length=2, max_length=100)
    species: str = Field(..., min_length=2, max_length=50)
    life_stage: str | None = Field(default=None, max_length=50)
    source_url: str | None = Field(default=None, max_length=500)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @field_validator("title", "content", "category", "species")
    @classmethod
    def validate_text_fields(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Field must not be empty.")
        return value


class IngestRequest(BaseModel):
    source: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Logical source name for the ingestion batch",
    )
    documents: list[IngestDocument] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Documents to ingest into the knowledge base",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Source must not be empty.")
        return value


class IngestResponse(BaseModel):
    status: Literal["accepted", "completed"]
    source: str
    documents_received: int = Field(..., ge=0)
    documents_processed: int = Field(..., ge=0)
    chunks_created: int = Field(..., ge=0)
    document_ids: list[str] = Field(default_factory=list)
    message: str
    ingested_at: datetime

    model_config = ConfigDict(extra="forbid")


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    service: str
    version: str
    environment: str
    database: Literal["healthy", "unhealthy", "unknown"]
    timestamp: datetime

    model_config = ConfigDict(extra="forbid")