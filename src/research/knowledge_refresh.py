"""
TTL and review_after helpers plus a refresh worker for provisional external knowledge (Layer 3).

Schedulers or cron jobs call :func:`run_refresh_batch` with a :class:`KnowledgeRefreshJob` and
injected dependencies; no FastAPI coupling.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from sqlalchemy import select

from src.db.models import ResearchCandidate
from src.db.session import get_db_session
from src.research.knowledge_promotion_policy import assert_refresh_status_never_auto_approved
from src.research.schemas import KnowledgeRecordStatus, KnowledgeRefreshJob
from src.utils.logger import get_logger

logger = get_logger(__name__)


def default_review_after(from_time: datetime | None = None, *, days: int = 30) -> datetime:
    """
    Compute a conservative default review_after instant (UTC).

    Callers should override for medical or sensitive content with shorter horizons.
    """
    from datetime import timedelta

    base = from_time or datetime.now(timezone.utc)
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)
    return base + timedelta(days=days)


def is_expired_ttl(*, as_of: datetime, ttl_expires_at: datetime | None) -> bool:
    """Return True if ttl_expires_at is set and strictly before as_of."""
    if ttl_expires_at is None:
        return False
    return ttl_expires_at < as_of


def build_refresh_job(
    *,
    as_of: datetime | None = None,
    limit: int = 100,
    target_status: KnowledgeRecordStatus = KnowledgeRecordStatus.provisional,
) -> KnowledgeRefreshJob:
    """Factory for ad-hoc or worker-driven refresh batches (stable for tests and schedulers)."""
    now = as_of or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return KnowledgeRefreshJob(
        id=str(uuid.uuid4()),
        as_of=now,
        limit=limit,
        target_status=target_status,
        created_at=now,
    )


def _norm_snippet(t: str) -> str:
    return " ".join(t.lower().strip().split())[:500]


def snippet_text_jaccard(old_texts: list[str], new_texts: list[str]) -> float:
    """Jaccard similarity over normalized snippet bodies (coarse drift signal)."""
    sa = {_norm_snippet(t) for t in old_texts if t and t.strip()}
    sb = {_norm_snippet(t) for t in new_texts if t and t.strip()}
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


@dataclass(frozen=True)
class RefreshEvidenceDecision:
    """Outcome of comparing stored vs freshly retrieved trusted evidence."""

    new_status: str
    new_authority_score: float
    reasons: tuple[str, ...] = ()


def decide_refresh_evidence(
    *,
    prior_status: str,
    old_snippet_texts: list[str],
    old_authority: float,
    new_snippet_texts: list[str],
    new_authority: float,
    jaccard: float,
) -> RefreshEvidenceDecision:
    """
    Decide lifecycle status after re-fetch. Never returns ``approved`` (no auto-promotion).

    Sensitive / medical rows that are already ``needs_review`` stay in that lane unless
    evidence is clearly unsupported (then ``expired``).

    Explicit provisional→* promotion rules for curator flows live in
    :mod:`src.research.knowledge_promotion_policy`.
    """
    old_n = len([t for t in old_snippet_texts if t and t.strip()])
    new_n = len([t for t in new_snippet_texts if t and t.strip()])
    medical_lock = prior_status == KnowledgeRecordStatus.needs_review.value
    reasons: list[str] = []

    decision: RefreshEvidenceDecision
    if new_n == 0 and old_n > 0:
        if old_n >= 2:
            reasons.append("refresh_returned_no_snippets_prior_bundle_had_multiple")
            decision = RefreshEvidenceDecision(
                new_status=KnowledgeRecordStatus.expired.value,
                new_authority_score=min(old_authority, new_authority),
                reasons=tuple(reasons),
            )
        else:
            reasons.append("refresh_returned_no_snippets")
            decision = RefreshEvidenceDecision(
                new_status=KnowledgeRecordStatus.needs_review.value,
                new_authority_score=min(old_authority, new_authority),
                reasons=tuple(reasons),
            )
    elif old_authority >= 0.55 and new_authority < old_authority * 0.6:
        reasons.append("authority_score_collapsed_vs_prior")
        decision = RefreshEvidenceDecision(
            new_status=KnowledgeRecordStatus.needs_review.value,
            new_authority_score=new_authority,
            reasons=tuple(reasons),
        )
    elif old_n >= 2 and new_n >= 2 and jaccard < 0.15:
        reasons.append("low_overlap_between_prior_and_refreshed_snippets")
        decision = RefreshEvidenceDecision(
            new_status=KnowledgeRecordStatus.needs_review.value,
            new_authority_score=new_authority,
            reasons=tuple(reasons),
        )
    elif medical_lock:
        reasons.append("medical_or_prior_needs_review_no_auto_promotion")
        decision = RefreshEvidenceDecision(
            new_status=KnowledgeRecordStatus.needs_review.value,
            new_authority_score=new_authority,
            reasons=tuple(reasons),
        )
    else:
        reasons.append("refresh_evidence_consistent_with_prior_bundle")
        decision = RefreshEvidenceDecision(
            new_status=KnowledgeRecordStatus.provisional.value,
            new_authority_score=new_authority,
            reasons=tuple(reasons),
        )

    assert_refresh_status_never_auto_approved(decision.new_status)
    return decision


def extract_snippet_texts_from_envelope(evidence_json: dict[str, Any]) -> list[str]:
    """Pull snippet bodies from persisted research candidate envelope (v1 legacy or v2)."""
    snippets: list[Any] = []
    if evidence_json.get("schema_version") == 2:
        ev = evidence_json.get("evidence") or {}
        recs = ev.get("records") or []
        if recs:
            snippets = recs
        else:
            re_block = ev.get("research_evidence") or {}
            snippets = re_block.get("snippets") or []
    else:
        snippets = evidence_json.get("snippets") or []
    out: list[str] = []
    for row in snippets:
        if isinstance(row, dict):
            t = row.get("snippet") or row.get("text") or ""
            if isinstance(t, str):
                out.append(t)
    return out


def derive_refresh_question(*, evidence_json: dict[str, Any], topic: str | None, species: str | None) -> str:
    """
    Reconstruct a retrieval query for trusted research.

    Prefer stored expanded queries from the original run; otherwise a conservative topic question.
    """
    meta = evidence_json.get("research_result_meta")
    if meta is None and evidence_json.get("schema_version") == 2:
        ev = evidence_json.get("evidence")
        if isinstance(ev, dict):
            meta = ev.get("research_result_meta")
    if isinstance(meta, dict):
        expanded = meta.get("expanded_queries") or []
        if isinstance(expanded, list) and expanded:
            first = expanded[0]
            if isinstance(first, str) and first.strip():
                return first.strip()
    parts = []
    if topic:
        parts.append(topic)
    if species:
        parts.append(species)
    if parts:
        return f"What should I know about {' '.join(parts)} for pet care guidance?"
    return "Trusted external knowledge refresh verification query."


def merge_envelope_after_refresh(
    prior: dict[str, Any],
    *,
    refresh_audit_entry: dict[str, Any],
    refreshed_research_dump: dict[str, Any] | None,
) -> dict[str, Any]:
    """Immutably merge audit + optional refreshed research snapshot into evidence_json."""
    merged = dict(prior)
    audit = list(merged.get("refresh_audit") or [])
    audit.append(refresh_audit_entry)
    merged["refresh_audit"] = audit
    if refreshed_research_dump is not None:
        merged["last_refresh_research_meta"] = refreshed_research_dump
    return merged


@runtime_checkable
class ProvisionalCandidateRecord(Protocol):
    """Minimal read model for refresh selection (DB row projection)."""

    @property
    def id(self) -> str: ...

    @property
    def review_after(self) -> datetime | None: ...

    @property
    def status(self) -> KnowledgeRecordStatus: ...


@dataclass
class DueResearchCandidate:
    """Concrete row for refresh selection (compatible with :class:`ProvisionalCandidateRecord`)."""

    db_id: int
    lifecycle_status: str
    review_after: datetime | None
    topic: str | None
    species: str | None
    breed: str | None
    life_stage: str | None
    authority_score: float
    provider_id: str
    evidence_json: dict[str, Any]
    synthesis_text: str | None = None

    @property
    def id(self) -> str:
        return str(self.db_id)

    @property
    def status(self) -> KnowledgeRecordStatus:
        try:
            return KnowledgeRecordStatus(self.lifecycle_status)
        except ValueError:
            return KnowledgeRecordStatus.provisional


@runtime_checkable
class ProvisionalCandidateRepository(Protocol):
    """Query and update provisional rows for refresh workflows."""

    async def list_due_for_review(self, job: KnowledgeRefreshJob) -> list[DueResearchCandidate]:
        """Return candidates that should be refreshed as of ``job.as_of``."""

    async def apply_refresh_update(
        self,
        *,
        db_id: int,
        last_verified_at: datetime,
        review_after: datetime | None,
        status: str,
        evidence_json: dict[str, Any],
        authority_score: float,
    ) -> None:
        """Persist verification outcome for one research candidate row."""


class NullProvisionalCandidateRepository:
    """Placeholder repository returning no rows."""

    async def list_due_for_review(self, job: KnowledgeRefreshJob) -> list[DueResearchCandidate]:
        _ = job
        return []

    async def apply_refresh_update(
        self,
        *,
        db_id: int,
        last_verified_at: datetime,
        review_after: datetime | None,
        status: str,
        evidence_json: dict[str, Any],
        authority_score: float,
    ) -> None:
        logger.debug(
            "apply_refresh_update noop",
            extra={
                "db_id": db_id,
                "status": status,
                "last_verified_at": last_verified_at.isoformat(),
            },
        )
        _ = (review_after, evidence_json, authority_score)


def _list_due_sync(session: Any, job: KnowledgeRefreshJob) -> list[DueResearchCandidate]:
    target = job.target_status.value
    stmt = (
        select(ResearchCandidate)
        .where(
            ResearchCandidate.review_after.is_not(None),
            ResearchCandidate.review_after <= job.as_of,
            ResearchCandidate.status == target,
        )
        .order_by(ResearchCandidate.review_after.asc())
        .limit(job.limit)
    )
    rows = session.execute(stmt).scalars().all()
    out: list[DueResearchCandidate] = []
    for r in rows:
        out.append(
            DueResearchCandidate(
                db_id=int(r.id),
                lifecycle_status=str(r.status),
                review_after=r.review_after,
                topic=r.topic,
                species=r.species,
                breed=r.breed,
                life_stage=r.life_stage,
                authority_score=float(r.authority_score or 0.0),
                provider_id=str(r.provider_id or "unknown"),
                evidence_json=dict(r.evidence_json or {}),
                synthesis_text=r.synthesis_text,
            )
        )
    return out


def _apply_refresh_sync(
    session: Any,
    *,
    db_id: int,
    last_verified_at: datetime,
    review_after: datetime | None,
    status: str,
    evidence_json: dict[str, Any],
    authority_score: float,
) -> None:
    row = session.get(ResearchCandidate, db_id)
    if row is None:
        raise ValueError(f"ResearchCandidate id={db_id} not found.")
    row.last_verified_at = last_verified_at
    row.review_after = review_after
    row.status = status
    row.authority_score = authority_score
    row.evidence_json = evidence_json


class SqlAlchemyKnowledgeRefreshRepository:
    """PostgreSQL-backed listing + updates for research candidate refresh."""

    def __init__(self, *, session_factory: Any = get_db_session) -> None:
        self._session_factory = session_factory

    def _run_with_session(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        with self._session_factory() as session:
            return fn(session, *args, **kwargs)

    async def list_due_for_review(self, job: KnowledgeRefreshJob) -> list[DueResearchCandidate]:
        return await asyncio.to_thread(self._run_with_session, _list_due_sync, job)

    async def apply_refresh_update(
        self,
        *,
        db_id: int,
        last_verified_at: datetime,
        review_after: datetime | None,
        status: str,
        evidence_json: dict[str, Any],
        authority_score: float,
    ) -> None:
        def _apply() -> None:
            self._run_with_session(
                _apply_refresh_sync,
                db_id=db_id,
                last_verified_at=last_verified_at,
                review_after=review_after,
                status=status,
                evidence_json=evidence_json,
                authority_score=authority_score,
            )

        await asyncio.to_thread(_apply)


@runtime_checkable
class TrustedResearchRunner(Protocol):
    """Pluggable trusted research (lazy-bound at runtime to avoid import cycles)."""

    async def run(self, request: Any) -> Any:
        """Execute trusted external research for ``request``."""


@dataclass
class KnowledgeRefreshBatchReport:
    """Summary for logging / job rows after a worker pass."""

    job_id: str
    candidates_scanned: int
    applied_updates: int
    marked_expired: int
    marked_needs_review: int
    marked_provisional: int
    errors: list[str] = field(default_factory=list)


async def select_candidates_for_refresh(
    job: KnowledgeRefreshJob,
    *,
    repository: ProvisionalCandidateRepository | None = None,
) -> list[DueResearchCandidate]:
    """
    Application-level helper: list candidates due for review using the injected repository.
    """
    repo: ProvisionalCandidateRepository = repository or NullProvisionalCandidateRepository()
    rows = await repo.list_due_for_review(job)
    logger.info(
        "Knowledge refresh selection",
        extra={"as_of": job.as_of.isoformat(), "count": len(rows), "job_id": job.id},
    )
    return rows


async def run_refresh_batch(
    job: KnowledgeRefreshJob,
    *,
    research: TrustedResearchRunner,
    repository: ProvisionalCandidateRepository | None = None,
    review_after_days: int = 30,
) -> KnowledgeRefreshBatchReport:
    """
    Revalidate provisional (or other targeted) external bundles whose ``review_after`` has passed.

    For each row: re-run trusted retrieval, compare evidence, update ``last_verified_at`` and
    ``review_after``, and adjust ``status`` to ``provisional``, ``needs_review``, or ``expired``.
    Does **not** auto-promote to ``approved``; medical / prior ``needs_review`` rows stay gated.
    Intended to be invoked from a scheduled worker or CLI.
    """
    from src.api.schemas import PetProfile, QueryFilters
    from src.research.trusted_research_service import TrustedResearchRequest

    repo = repository or SqlAlchemyKnowledgeRefreshRepository()
    candidates = await repo.list_due_for_review(job)
    report = KnowledgeRefreshBatchReport(
        job_id=job.id,
        candidates_scanned=len(candidates),
        applied_updates=0,
        marked_expired=0,
        marked_needs_review=0,
        marked_provisional=0,
        errors=[],
    )

    for cand in candidates:
        try:
            question = derive_refresh_question(
                evidence_json=cand.evidence_json,
                topic=cand.topic,
                species=cand.species,
            )
            pet_profile = None
            if cand.species:
                pet_profile = PetProfile(
                    species=cand.species or "unknown",
                    breed=cand.breed,
                    life_stage=cand.life_stage,
                )
            filters = QueryFilters(category=cand.topic) if cand.topic else None
            req = TrustedResearchRequest(
                question=question,
                pet_profile=pet_profile,
                filters=filters,
            )
            outcome = await research.run(req)
            research_result = outcome.research
            new_texts = [s.text for s in research_result.evidence.snippets]
            old_texts = extract_snippet_texts_from_envelope(cand.evidence_json)
            jaccard = snippet_text_jaccard(old_texts, new_texts)
            decision = decide_refresh_evidence(
                prior_status=cand.lifecycle_status,
                old_snippet_texts=old_texts,
                old_authority=cand.authority_score,
                new_snippet_texts=new_texts,
                new_authority=float(research_result.authority_score),
                jaccard=jaccard,
            )
            audit = {
                "at": job.as_of.isoformat(),
                "job_id": job.id,
                "prior_status": cand.lifecycle_status,
                "new_status": decision.new_status,
                "reasons": list(decision.reasons),
                "prior_snippet_count": len(old_texts),
                "new_snippet_count": len(new_texts),
                "jaccard": round(jaccard, 4),
                "prior_authority": cand.authority_score,
                "new_authority": float(research_result.authority_score),
            }
            refreshed_dump = research_result.model_dump(mode="json")
            merged_env = merge_envelope_after_refresh(
                cand.evidence_json,
                refresh_audit_entry=audit,
                refreshed_research_dump=refreshed_dump,
            )
            next_review = default_review_after(job.as_of, days=review_after_days)
            await repo.apply_refresh_update(
                db_id=cand.db_id,
                last_verified_at=job.as_of,
                review_after=next_review,
                status=decision.new_status,
                evidence_json=merged_env,
                authority_score=decision.new_authority_score,
            )
            report.applied_updates += 1
            if decision.new_status == KnowledgeRecordStatus.expired.value:
                report.marked_expired += 1
            elif decision.new_status == KnowledgeRecordStatus.needs_review.value:
                report.marked_needs_review += 1
            elif decision.new_status == KnowledgeRecordStatus.provisional.value:
                report.marked_provisional += 1
        except Exception as exc:  # noqa: BLE001 — batch worker isolates row failures
            msg = f"candidate_id={cand.db_id}: {exc!s}"
            logger.exception("Knowledge refresh row failed", extra={"error": msg})
            report.errors.append(msg)

    logger.info(
        "Knowledge refresh batch complete",
        extra={
            "job_id": job.id,
            "scanned": report.candidates_scanned,
            "applied": report.applied_updates,
            "errors": len(report.errors),
        },
    )
    return report
