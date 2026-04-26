"""
Persist provisional candidate knowledge from external research (Layer 3).

Stores structured payloads in ``research_candidates`` / ``knowledge_sources`` / ``research_candidate_sources``.
Does not write to ``document_chunks`` unless explicitly requested (bridge not implemented yet).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import KnowledgeSource, ResearchCandidate, ResearchCandidateSource
from src.db.session import get_db_session
from src.research.evidence_extractor import (
    EvidenceExtractionResult,
    FrontendEvidenceBundle,
    ReviewDraftEvidenceBundle,
)
from src.research.evidence_envelope import build_normalized_evidence_records
from src.research.evidence_quality import evidence_bundle_eligible_for_persistence
from src.research.knowledge_promotion_policy import (
    assert_sensitive_medical_bundle_policies,
    initial_status_for_external_fallback,
)
from src.research.schemas import CandidateKnowledgeRecord, ExternalSource, KnowledgeRecordStatus, ResearchResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExternalResearchIngestInput(BaseModel):
    """
    Payload to persist external research as a single research candidate row.

    ``extraction`` carries separated frontend vs review-draft bundles plus canonical snippets/sources.
    """

    extraction: EvidenceExtractionResult
    research_result: ResearchResult | None = None
    user_id: int | None = Field(default=None, description="Optional owning user FK")
    question_fingerprint: str | None = Field(default=None, max_length=128)
    content_sensitivity: Literal["general", "medical", "behavioral", "nutrition", "other"] = Field(
        default="general",
        description="When medical or heuristics match, status becomes needs_review",
    )
    synthesis_text: str | None = Field(
        default=None,
        description=(
            "Legacy: LLM review narrative. Prefer ``internal_review_llm_draft``. "
            "Persisted only inside ``evidence_json.synthesis`` (never mixed into evidence records). "
            "Not written to ``research_candidates.synthesis_text`` (column reserved / unused for v2)."
        ),
    )
    frontend_answer_text: str | None = Field(
        default=None,
        description="User-facing answer after guardrails; stored only under ``evidence_json.synthesis.frontend_answer``.",
    )
    internal_review_llm_draft: str | None = Field(
        default=None,
        description=(
            "LLM-generated staff review narrative; stored only under ``evidence_json.synthesis.review_draft`` "
            "(marked ai_generated / provisional / not_evidence)."
        ),
    )
    allow_document_chunk_ingest: bool = Field(
        default=False,
        description="Reserved for future bridge into document_chunks (currently never ingests)",
    )

    model_config = ConfigDict(extra="forbid")


def classify_external_candidate_state(
    content_sensitivity: Literal["general", "medical", "behavioral", "nutrition", "other"],
    *,
    topic: str | None,
    species: str | None,
    life_stage: str | None,
    category: str | None,
) -> tuple[str, bool]:
    """
    Return (``research_candidates.status`` value, auto_promotion_blocked).

    Delegates to :func:`initial_status_for_external_fallback` (backward-compatible name).
    """
    return initial_status_for_external_fallback(
        content_sensitivity,
        topic=topic,
        species=species,
        life_stage=life_stage,
        category=category,
    )


def _review_draft_synthesis_text(inp: ExternalResearchIngestInput) -> str | None:
    """Resolve the internal LLM review narrative (legacy ``synthesis_text`` as fallback)."""
    raw = inp.internal_review_llm_draft if inp.internal_review_llm_draft is not None else inp.synthesis_text
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped or None


def build_separated_evidence_json(
    inp: ExternalResearchIngestInput,
) -> dict[str, Any]:
    """
    Build ``evidence_json`` (schema v2): retrieval evidence vs LLM synthesis.

    - ``evidence`` holds normalized snippet rows (URL, domain, title, text, authority) plus
      canonical ``research_evidence`` and the structured retrieval bundle
      (``retrieval_extraction_bundle`` — not user-facing synthesis).
    - ``synthesis`` holds ``frontend_answer`` (API-safe) and ``review_draft`` (AI, provisional, not evidence).
    """
    ex = inp.extraction
    evidence_block: dict[str, Any] = {
        "records": build_normalized_evidence_records(ex),
        "research_evidence": ex.research_evidence.model_dump(mode="json"),
        "retrieval_extraction_bundle": ex.review_draft.model_dump(mode="json"),
    }
    if inp.research_result is not None:
        evidence_block["research_result_meta"] = inp.research_result.model_dump(mode="json")

    fe = (inp.frontend_answer_text or "").strip() or None
    rd = _review_draft_synthesis_text(inp)

    return {
        "schema_version": 2,
        "evidence": evidence_block,
        "synthesis": {
            "frontend_answer": {
                "text": fe,
                "safe_for_api": True,
            },
            "review_draft": {
                "text": rd,
                "ai_generated": True,
                "provisional": True,
                "not_evidence": True,
            },
        },
    }


def _scope_from_extraction(inp: ExternalResearchIngestInput) -> tuple[str | None, str | None, str | None, str | None]:
    rr = inp.research_result
    if rr is not None:
        return rr.topic, rr.species, rr.breed, rr.life_stage
    srcs = inp.extraction.research_evidence.sources
    if not srcs:
        return None, None, None, None
    s0 = srcs[0]
    return s0.topic, s0.species, s0.breed, s0.life_stage


def _maybe_ingest_document_chunks(_: Session, inp: ExternalResearchIngestInput) -> int:
    if inp.allow_document_chunk_ingest:
        logger.info(
            "allow_document_chunk_ingest is true but document_chunks bridge is not implemented; skipping.",
        )
    return 0


def _upsert_knowledge_source(session: Session, src: ExternalSource) -> KnowledgeSource:
    row = session.execute(select(KnowledgeSource).where(KnowledgeSource.source_key == src.source_key)).scalar_one_or_none()
    meta = {
        "external_source_id": src.id,
        "source_type": src.source_type.value,
    }
    if row is None:
        row = KnowledgeSource(
            source_key=src.source_key,
            base_url=str(src.base_url),
            category=src.topic,
            authority_score=float(src.authority_score),
            medical_sensitivity="none",
            auto_ingest_allowed=False,
            status="active",
            metadata_json=meta,
        )
        session.add(row)
        session.flush()
        return row
    row.base_url = str(src.base_url)
    if src.topic:
        row.category = src.topic
    row.authority_score = float(src.authority_score)
    row.metadata_json = {**row.metadata_json, **meta}
    session.flush()
    return row


def _snippet_ids_for_source_key(snippets: list[dict[str, Any]], sources: list[dict[str, Any]], source_key: str) -> list[str]:
    id_by_key = {s["source_key"]: s["id"] for s in sources}
    ext_id = id_by_key.get(source_key)
    if not ext_id:
        return []
    return [sn["id"] for sn in snippets if sn.get("external_source_id") == ext_id]


def persist_external_research_with_session(session: Session, inp: ExternalResearchIngestInput) -> dict[str, Any]:
    """
    Synchronous persistence using an open SQLAlchemy ``Session`` (caller manages transaction).

    Returns a summary dict suitable for APIs (IDs, status flags, chunk ingest count).
    """
    gate_ok, gate_reasons = evidence_bundle_eligible_for_persistence(inp.extraction)
    if not gate_ok:
        logger.warning(
            "persist_external_research_with_session skipped — evidence quality gate failed",
            extra={"reasons": gate_reasons},
        )
        return {
            "research_candidate_id": 0,
            "status": "skipped_evidence_quality_gate",
            "knowledge_source_ids": [],
            "research_candidate_source_ids": [],
            "document_chunks_ingested": 0,
            "provisional": True,
            "auto_promotion_blocked": True,
            "evidence_schema_version": 2,
            "gate_rejection_reasons": gate_reasons,
        }

    topic, species, breed, life_stage = _scope_from_extraction(inp)
    category = inp.research_result.topic if inp.research_result else topic
    status, auto_blocked = classify_external_candidate_state(
        inp.content_sensitivity,
        topic=topic,
        species=species,
        life_stage=life_stage,
        category=category,
    )
    assert_sensitive_medical_bundle_policies(
        ingest_initial_status=status,
        content_sensitivity=inp.content_sensitivity,
        topic=topic,
        species=species,
        life_stage=life_stage,
        category=category,
    )

    envelope = build_separated_evidence_json(inp)
    provider_id = inp.extraction.review_draft.provider_id
    authority = float(inp.research_result.authority_score) if inp.research_result else max(
        (float(s.authority_score) for s in inp.extraction.research_evidence.sources),
        default=0.0,
    )
    review_after = inp.research_result.review_after if inp.research_result else None

    knowledge_rows: list[KnowledgeSource] = []
    for src in inp.extraction.research_evidence.sources:
        knowledge_rows.append(_upsert_knowledge_source(session, src))

    cand = ResearchCandidate(
        user_id=inp.user_id,
        status=status,
        question_fingerprint=inp.question_fingerprint,
        provider_id=provider_id,
        evidence_json=envelope,
        synthesis_text=None,
        topic=topic,
        species=species,
        breed=breed,
        life_stage=life_stage,
        authority_score=authority,
        review_after=review_after,
    )
    session.add(cand)
    session.flush()

    re_block = envelope["evidence"]["research_evidence"]
    snippets_json = re_block["snippets"]
    sources_json = re_block["sources"]
    link_ids: list[int] = []
    for order, ks in enumerate(knowledge_rows):
        sids = _snippet_ids_for_source_key(snippets_json, sources_json, ks.source_key)
        link = ResearchCandidateSource(
            research_candidate_id=cand.id,
            knowledge_source_id=ks.id,
            role="citation",
            snippet_ids_json=sids,
            sort_order=order,
        )
        session.add(link)
        session.flush()
        link_ids.append(link.id)

    chunks = _maybe_ingest_document_chunks(session, inp)

    logger.info(
        "Persisted external research candidate",
        extra={
            "research_candidate_id": cand.id,
            "status": status,
            "knowledge_sources": len(knowledge_rows),
            "auto_promotion_blocked": auto_blocked,
        },
    )

    return {
        "research_candidate_id": cand.id,
        "status": status,
        "knowledge_source_ids": [k.id for k in knowledge_rows],
        "research_candidate_source_ids": link_ids,
        "document_chunks_ingested": chunks,
        "provisional": status == KnowledgeRecordStatus.provisional.value,
        "auto_promotion_blocked": auto_blocked,
        "evidence_schema_version": envelope.get("schema_version"),
    }


@runtime_checkable
class ExternalResearchPersistencePort(Protocol):
    """Port for persisting :class:`ExternalResearchIngestInput` rows."""

    async def persist_external_research(self, inp: ExternalResearchIngestInput) -> dict[str, Any]:
        """Persist and return a summary dict."""


@runtime_checkable
class ProvisionalCandidateStore(Protocol):
    """Persistence port for legacy :class:`CandidateKnowledgeRecord` rows."""

    async def save(self, record: CandidateKnowledgeRecord) -> str:
        """Persist a candidate and return its stable storage identifier."""


class SqlAlchemyResearchCandidateStore(ExternalResearchPersistencePort, ProvisionalCandidateStore):
    """PostgreSQL-backed store for research candidates and knowledge sources."""

    def __init__(self, *, session_factory: Any = get_db_session) -> None:
        """Solo guarda la factory; no abre conexión a la base hasta el primer ``persist``/``save``."""
        self._session_factory = session_factory

    def _run_with_session(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        with self._session_factory() as session:
            return fn(session, *args, **kwargs)

    async def persist_external_research(self, inp: ExternalResearchIngestInput) -> dict[str, Any]:
        return await asyncio.to_thread(self._run_with_session, persist_external_research_with_session, inp)

    async def save(self, record: CandidateKnowledgeRecord) -> str:
        now = record.retrieved_at or datetime.now(timezone.utc)
        fe = FrontendEvidenceBundle(citations=[], generated_at=now)
        rd = ReviewDraftEvidenceBundle(
            snippets=list(record.evidence.snippets),
            claims=[],
            sources=list(record.evidence.sources),
            provider_id="legacy_candidate_record",
            blocked_url_count=0,
            extraction_notes=["ingested_from_CandidateKnowledgeRecord"],
            generated_at=now,
        )
        ex = EvidenceExtractionResult(frontend=fe, review_draft=rd, research_evidence=record.evidence)
        sens: Literal["general", "medical", "behavioral", "nutrition", "other"] = (
            "medical" if record.status == KnowledgeRecordStatus.needs_review else "general"
        )
        inp = ExternalResearchIngestInput(
            extraction=ex,
            research_result=None,
            question_fingerprint=record.id,
            content_sensitivity=sens,
            synthesis_text=None,
        )
        summary = await self.persist_external_research(inp)
        return str(summary["research_candidate_id"])


class NullProvisionalCandidateStore(ProvisionalCandidateStore):
    """No-op legacy store."""

    async def save(self, record: CandidateKnowledgeRecord) -> str:
        logger.info(
            "Candidate knowledge ingest (no-op store)",
            extra={"status": record.status.value, "record_id": record.id},
        )
        return record.id


class NullExternalResearchPersistence(ExternalResearchPersistencePort):
    """No-op persistence for tests or offline mode."""

    async def persist_external_research(self, inp: ExternalResearchIngestInput) -> dict[str, Any]:
        _ = inp
        return {
            "research_candidate_id": 0,
            "status": "noop",
            "knowledge_source_ids": [],
            "research_candidate_source_ids": [],
            "document_chunks_ingested": 0,
            "provisional": True,
            "auto_promotion_blocked": False,
            "evidence_schema_version": 2,
        }


async def ingest_external_research_candidate(
    inp: ExternalResearchIngestInput,
    *,
    store: ExternalResearchPersistencePort | None = None,
) -> dict[str, Any]:
    """
    Persist one external research bundle as a provisional (or needs_review) candidate.

    Returns IDs and a small persistence summary; does not promote to approved knowledge.
    """
    gate_ok, gate_reasons = evidence_bundle_eligible_for_persistence(inp.extraction)
    if not gate_ok:
        logger.warning(
            "Skipping external research candidate persistence — evidence quality gate failed",
            extra={"reasons": gate_reasons},
        )
        return {
            "research_candidate_id": 0,
            "status": "skipped_evidence_quality_gate",
            "knowledge_source_ids": [],
            "research_candidate_source_ids": [],
            "document_chunks_ingested": 0,
            "provisional": True,
            "auto_promotion_blocked": True,
            "evidence_schema_version": 2,
            "gate_rejection_reasons": gate_reasons,
        }

    effective: ExternalResearchPersistencePort = store or NullExternalResearchPersistence()
    return await effective.persist_external_research(inp)


async def ingest_provisional_candidate(
    record: CandidateKnowledgeRecord,
    *,
    store: ProvisionalCandidateStore | None = None,
) -> dict[str, Any]:
    """
    Legacy entrypoint mapping :class:`CandidateKnowledgeRecord` to the new persistence path.

    When ``store`` is :class:`SqlAlchemyResearchCandidateStore`, returns numeric ``candidate_id`` as string.
    """
    effective_store: ProvisionalCandidateStore = store or NullProvisionalCandidateStore()
    record_id = await effective_store.save(record)
    return {"candidate_id": record_id, "status": record.status.value}
