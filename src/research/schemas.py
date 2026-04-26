"""
Pydantic contracts for trusted external research and candidate knowledge rows.

Field names and enums align with planned SQLAlchemy models / Alembic migrations
(VARCHAR/ENUM/JSONB/TIMESTAMPTZ columns).
"""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict, HttpUrl


class KnowledgeRecordStatus(StrEnum):
    """Lifecycle state for candidate or curated external knowledge (DB check constraint)."""

    provisional = "provisional"
    approved = "approved"
    expired = "expired"
    needs_review = "needs_review"


class ExternalSourceType(StrEnum):
    """How the external origin is reached (stored as string in DB)."""

    allowlisted_web = "allowlisted_web"
    trusted_api = "trusted_api"


class ExternalSource(BaseModel):
    """
    One allowlisted origin used in Layer 2, suitable as a row or JSON fragment.

    Maps to columns: id, source_key, base_url, authority_score, source_type, topic,
    species, breed, life_stage, retrieved_at, review_after.
    """

    id: str = Field(..., min_length=1, max_length=36, description="UUID primary key as string")
    source_key: str = Field(..., min_length=1, max_length=64)
    base_url: HttpUrl = Field(..., description="Registered root; fetched URLs must be under this path")
    authority_score: float = Field(..., ge=0.0, le=1.0, description="Trust weight for ranking, not clinical certainty")
    source_type: ExternalSourceType
    topic: str | None = Field(default=None, max_length=200)
    species: str | None = Field(default=None, max_length=50)
    breed: str | None = Field(default=None, max_length=120)
    life_stage: str | None = Field(default=None, max_length=50)
    retrieved_at: datetime = Field(..., description="When this source snapshot was recorded (TIMESTAMPTZ)")
    review_after: datetime | None = Field(default=None, description="When this source metadata should be revalidated")

    model_config = ConfigDict(extra="forbid")


class ExtractedSnippet(BaseModel):
    """
    One attributable excerpt from an allowlisted page (evidence only).

    Maps to columns including external_source_id FK, text, authority_score, source_type,
    topic, species, breed, life_stage, retrieved_at, review_after.
    """

    id: str = Field(..., min_length=1, max_length=128)
    external_source_id: str = Field(..., min_length=1, max_length=36, description="FK to external_source.id")
    text: str = Field(..., min_length=1, description="Verbatim or clearly attributed excerpt")
    evidence_page_url: str | None = Field(
        default=None,
        max_length=2000,
        description="Canonical HTTP(S) page URL for this snippet (preserved when rebuilding from capped evidence).",
    )
    authority_score: float = Field(..., ge=0.0, le=1.0)
    source_type: ExternalSourceType
    topic: str | None = Field(default=None, max_length=200)
    species: str | None = Field(default=None, max_length=50)
    breed: str | None = Field(default=None, max_length=120)
    life_stage: str | None = Field(default=None, max_length=50)
    retrieved_at: datetime
    review_after: datetime | None = None
    page_title: str | None = Field(
        default=None,
        max_length=500,
        description="Article/page title from retrieval or fetch (for citations; not clinical metadata).",
    )
    provider_relevance: float | None = Field(
        default=None,
        description="Normalized provider search relevance when available (e.g. Tavily score).",
    )
    ranking_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Composite rank: relevance + domain authority + query overlap (used for ordering).",
    )

    model_config = ConfigDict(extra="forbid")


class ResearchEvidence(BaseModel):
    """Structured Layer 2 output: snippets plus the external sources they draw from."""

    snippets: list[ExtractedSnippet] = Field(default_factory=list)
    sources: list[ExternalSource] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class ResearchResult(BaseModel):
    """
    Outcome of one trusted research run (evidence bundle plus run-level scope metadata).

    Intended columns: id (nullable until persist), evidence JSONB, authority_score, source_type,
    topic, species, breed, life_stage, retrieved_at, review_after.
    """

    id: str | None = Field(default=None, max_length=36)
    evidence: ResearchEvidence
    authority_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Top external authority score observed on snippets (or sources if none)",
    )
    source_type: ExternalSourceType
    topic: str | None = Field(default=None, max_length=200)
    species: str | None = Field(default=None, max_length=50)
    breed: str | None = Field(default=None, max_length=120)
    life_stage: str | None = Field(default=None, max_length=50)
    retrieved_at: datetime
    review_after: datetime | None = Field(
        default=None,
        description="Suggested earliest review/refresh time for this provisional external bundle",
    )
    evidence_summary: str = Field(
        default="",
        max_length=4000,
        description="Human-readable summary of retrieval + extraction (not a user answer)",
    )
    external_confidence: Literal["high", "medium", "low"] = Field(
        default="low",
        description="Heuristic confidence tier from snippet depth and authority scores",
    )
    expanded_queries: list[str] = Field(
        default_factory=list,
        description="Expanded research strings passed as retrieval context (allowlisted flow only)",
    )

    model_config = ConfigDict(extra="forbid")


class CandidateKnowledgeRecord(BaseModel):
    """
    Provisional or curated external knowledge row (Layer 3).

    Maps to table candidate_knowledge: id, status, evidence JSONB, authority_score,
    source_type, topic, species, breed, life_stage, retrieved_at, review_after.
    """

    id: str = Field(..., min_length=1, max_length=36)
    status: KnowledgeRecordStatus
    evidence: ResearchEvidence
    authority_score: float = Field(..., ge=0.0, le=1.0)
    source_type: ExternalSourceType
    topic: str | None = Field(default=None, max_length=200)
    species: str | None = Field(default=None, max_length=50)
    breed: str | None = Field(default=None, max_length=120)
    life_stage: str | None = Field(default=None, max_length=50)
    retrieved_at: datetime
    review_after: datetime | None = None

    model_config = ConfigDict(extra="forbid")


class KnowledgeRefreshJob(BaseModel):
    """
    One scheduled or ad-hoc refresh batch (maps to knowledge_refresh_jobs).

    target_status selects which knowledge rows this job is meant to process.
    """

    id: str = Field(..., min_length=1, max_length=36)
    as_of: datetime = Field(..., description="Reference time for the batch (TIMESTAMPTZ)")
    limit: int = Field(..., ge=1, le=500)
    target_status: KnowledgeRecordStatus = Field(
        ...,
        description="Lifecycle filter: rows considered in scope for this job",
    )
    created_at: datetime

    model_config = ConfigDict(extra="forbid")
