from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import UserDefinedType

from src.db.session import Base


class VectorType(UserDefinedType):
    """
    Minimal PostgreSQL pgvector column type.
    We keep it simple here because vector operations are handled in raw SQL
    inside src/rag/vector_store.py.
    """

    cache_ok = True

    def get_col_spec(self, **kwargs) -> str:
        return "vector"


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    chunk_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    title: Mapped[str | None] = mapped_column(String(300), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    source: Mapped[str | None] = mapped_column(String(500), nullable=True)
    category: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    species: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    life_stage: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )

    embedding = mapped_column(VectorType(), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        Index(
            "ix_document_chunks_filters",
            "category",
            "species",
            "life_stage",
        ),
    )
    
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    auth0_sub: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    role_label: Mapped[str | None] = mapped_column(String(50), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    pets = relationship("Pet", back_populates="user")
    query_history = relationship("QueryHistory", back_populates="user")
    research_candidates = relationship("ResearchCandidate", back_populates="user")


class Pet(Base):
    __tablename__ = "pets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)

    name: Mapped[str] = mapped_column(String(120), nullable=False)
    species: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    breed: Mapped[str | None] = mapped_column(String(120), nullable=True)
    age_years: Mapped[float | None] = mapped_column(nullable=True)
    life_stage: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    weight_kg: Mapped[float | None] = mapped_column(nullable=True)
    sex: Mapped[str | None] = mapped_column(String(20), nullable=True)
    neutered: Mapped[bool | None] = mapped_column(nullable=True)
    conditions_json: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    user = relationship("User", back_populates="pets")
    query_history = relationship("QueryHistory", back_populates="pet")


class QueryHistory(Base):
    __tablename__ = "query_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    pet_id: Mapped[int | None] = mapped_column(ForeignKey("pets.id"), nullable=True, index=True)

    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[str] = mapped_column(String(20), nullable=False)
    needs_vet_followup: Mapped[bool] = mapped_column(nullable=False, default=False)
    sources_json: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="[]")
    filters_json: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    user = relationship("User", back_populates="query_history")
    pet = relationship("Pet", back_populates="query_history")


class KnowledgeSource(Base):
    """
    Persisted trusted external origin (allowlist registry row or mirrored config).

    Distinct from ``document_chunks`` (internal RAG chunks); links to research candidates via
    ``research_candidate_sources``.
    """

    __tablename__ = "knowledge_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    base_url: Mapped[str] = mapped_column(String(2000), nullable=False)
    category: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    authority_score: Mapped[float] = mapped_column(Float, nullable=False, server_default="0.5")
    medical_sensitivity: Mapped[str] = mapped_column(String(20), nullable=False, server_default="none")
    auto_ingest_allowed: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default="active",
        index=True,
        comment="active | deprecated | suspended",
    )
    last_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    review_after: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    candidate_links = relationship("ResearchCandidateSource", back_populates="knowledge_source")


class ResearchCandidate(Base):
    """
    Provisional or curated bundle from trusted external research (Layer 2/3 handoff).

    ``evidence_json`` stores schema v2: ``evidence`` (retrieval snippets, URLs, authority) and
    ``synthesis`` (API-safe frontend answer vs marked AI/provisional review draft). Not a final clinical answer.
    """

    __tablename__ = "research_candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default="provisional",
        index=True,
        comment="provisional | approved | expired | needs_review",
    )
    question_fingerprint: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    provider_id: Mapped[str] = mapped_column(String(64), nullable=False, server_default="unknown")
    evidence_json: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    synthesis_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    topic: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)
    species: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    breed: Mapped[str | None] = mapped_column(String(120), nullable=True)
    life_stage: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    authority_score: Mapped[float] = mapped_column(Float, nullable=False, server_default="0.0")
    review_after: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    last_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    user = relationship("User", back_populates="research_candidates")
    source_links = relationship("ResearchCandidateSource", back_populates="research_candidate")


class ResearchCandidateSource(Base):
    """Associates a research candidate with one or more knowledge sources (attribution graph)."""

    __tablename__ = "research_candidate_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    research_candidate_id: Mapped[int] = mapped_column(
        ForeignKey("research_candidates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    knowledge_source_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_sources.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False, server_default="citation")
    snippet_ids_json: Mapped[list[Any]] = mapped_column(
        "snippet_ids",
        JSONB,
        nullable=False,
        server_default="[]",
    )
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    research_candidate = relationship("ResearchCandidate", back_populates="source_links")
    knowledge_source = relationship("KnowledgeSource", back_populates="candidate_links")

    __table_args__ = (
        Index(
            "ix_research_candidate_sources_unique_pair",
            "research_candidate_id",
            "knowledge_source_id",
            unique=True,
        ),
    )


class KnowledgeRefreshJobRow(Base):
    """
    Scheduled or ad-hoc refresh batch for provisional knowledge (operational row).

    Table name ``knowledge_refresh_jobs``; class name avoids clashing with the Pydantic
    ``KnowledgeRefreshJob`` request DTO in ``src.research.schemas``.
    """

    __tablename__ = "knowledge_refresh_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    public_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False, index=True)
    target_status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    batch_limit: Mapped[int] = mapped_column(Integer, nullable=False)
    job_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default="pending",
        index=True,
        comment="pending | running | completed | failed",
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        server_default="{}",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
