"""Unit tests for knowledge refresh TTL, evidence comparison, and batch worker."""

from datetime import datetime, timezone

import pytest
from pydantic import HttpUrl

from src.research.knowledge_refresh import (
    DueResearchCandidate,
    NullProvisionalCandidateRepository,
    RefreshEvidenceDecision,
    SqlAlchemyKnowledgeRefreshRepository,
    build_refresh_job,
    decide_refresh_evidence,
    derive_refresh_question,
    extract_snippet_texts_from_envelope,
    merge_envelope_after_refresh,
    run_refresh_batch,
    select_candidates_for_refresh,
    snippet_text_jaccard,
)
from src.research.schemas import (
    ExternalSourceType,
    ExtractedSnippet,
    ExternalSource,
    KnowledgeRecordStatus,
    ResearchEvidence,
    ResearchResult,
)
from src.research.trusted_research_service import TrustedResearchOutcome


def _snippet(now: datetime, text: str, sid: str = "s1", ext: str = "00000000-0000-4000-8000-000000000001") -> ExtractedSnippet:
    return ExtractedSnippet(
        id=sid,
        external_source_id=ext,
        text=text,
        authority_score=0.8,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
    )


def _research_result(*, snippets: list[ExtractedSnippet], authority: float = 0.8) -> ResearchResult:
    now = datetime.now(timezone.utc)
    src = ExternalSource(
        id="00000000-0000-4000-8000-000000000001",
        source_key="k",
        base_url=HttpUrl("https://avma.org/"),
        authority_score=0.8,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
    )
    ev = ResearchEvidence(snippets=snippets, sources=[src] if snippets else [])
    return ResearchResult(
        evidence=ev,
        authority_score=authority,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
        evidence_summary="test",
        external_confidence="medium",
    )


@pytest.mark.asyncio
async def test_select_candidates_uses_null_repository_by_default() -> None:
    job = build_refresh_job(limit=5)
    rows = await select_candidates_for_refresh(job)
    assert rows == []


@pytest.mark.asyncio
async def test_null_repository_apply_refresh_is_noop() -> None:
    repo = NullProvisionalCandidateRepository()
    await repo.apply_refresh_update(
        db_id=1,
        last_verified_at=datetime.now(timezone.utc),
        review_after=None,
        status="provisional",
        evidence_json={},
        authority_score=0.5,
    )


def test_extract_snippet_texts_schema_v2_prefers_records() -> None:
    env = {
        "schema_version": 2,
        "evidence": {
            "records": [{"snippet": "from record", "source_domain": "example.com"}],
        },
    }
    assert extract_snippet_texts_from_envelope(env) == ["from record"]


def test_extract_snippet_texts_schema_v2_fallback_research_evidence() -> None:
    env = {
        "schema_version": 2,
        "evidence": {
            "research_evidence": {"snippets": [{"text": "body"}]},
        },
    }
    assert extract_snippet_texts_from_envelope(env) == ["body"]


def test_derive_refresh_question_reads_nested_research_meta_v2() -> None:
    env = {
        "schema_version": 2,
        "evidence": {
            "research_result_meta": {"expanded_queries": ["  q1  "]},
        },
    }
    assert derive_refresh_question(evidence_json=env, topic="t", species="dog") == "q1"


def test_decide_refresh_marks_expired_when_refresh_empty_and_prior_substantial() -> None:
    d = decide_refresh_evidence(
        prior_status=KnowledgeRecordStatus.provisional.value,
        old_snippet_texts=["alpha guidance here", "beta guidance here"],
        old_authority=0.82,
        new_snippet_texts=[],
        new_authority=0.0,
        jaccard=0.0,
    )
    assert isinstance(d, RefreshEvidenceDecision)
    assert d.new_status == KnowledgeRecordStatus.expired.value
    assert any("no_snippets" in r for r in d.reasons)


def test_decide_refresh_needs_review_when_single_old_snippet_and_refresh_empty() -> None:
    d = decide_refresh_evidence(
        prior_status=KnowledgeRecordStatus.provisional.value,
        old_snippet_texts=["only one"],
        old_authority=0.7,
        new_snippet_texts=[],
        new_authority=0.0,
        jaccard=0.0,
    )
    assert d.new_status == KnowledgeRecordStatus.needs_review.value


def test_decide_refresh_needs_review_on_authority_collapse() -> None:
    d = decide_refresh_evidence(
        prior_status=KnowledgeRecordStatus.provisional.value,
        old_snippet_texts=["x", "y"],
        old_authority=0.9,
        new_snippet_texts=["x refreshed", "y refreshed"],
        new_authority=0.35,
        jaccard=0.4,
    )
    assert d.new_status == KnowledgeRecordStatus.needs_review.value
    assert any("authority" in r for r in d.reasons)


def test_decide_refresh_needs_review_on_low_jaccard_drift() -> None:
    jac = snippet_text_jaccard(
        ["completely different a", "completely different b"],
        ["other topic one", "other topic two"],
    )
    assert jac < 0.15
    d = decide_refresh_evidence(
        prior_status=KnowledgeRecordStatus.provisional.value,
        old_snippet_texts=["completely different a", "completely different b"],
        old_authority=0.75,
        new_snippet_texts=["other topic one", "other topic two"],
        new_authority=0.76,
        jaccard=jac,
    )
    assert d.new_status == KnowledgeRecordStatus.needs_review.value


def test_medical_prior_needs_review_never_auto_provisional_even_when_evidence_ok() -> None:
    d = decide_refresh_evidence(
        prior_status=KnowledgeRecordStatus.needs_review.value,
        old_snippet_texts=["insulin discussion excerpt"],
        old_authority=0.85,
        new_snippet_texts=["insulin discussion excerpt"],
        new_authority=0.85,
        jaccard=1.0,
    )
    assert d.new_status == KnowledgeRecordStatus.needs_review.value
    assert any("medical" in r or "needs_review" in r for r in d.reasons)


def test_decide_never_emits_approved() -> None:
    for prior in (KnowledgeRecordStatus.provisional.value, KnowledgeRecordStatus.needs_review.value):
        d = decide_refresh_evidence(
            prior_status=prior,
            old_snippet_texts=["a"],
            old_authority=0.9,
            new_snippet_texts=["a"],
            new_authority=0.9,
            jaccard=1.0,
        )
        assert d.new_status != KnowledgeRecordStatus.approved.value


def test_derive_refresh_question_prefers_expanded_queries() -> None:
    env = {"research_result_meta": {"expanded_queries": ["dog hydration summer heat", "backup"]}}
    q = derive_refresh_question(evidence_json=env, topic="nutrition", species="dog")
    assert q == "dog hydration summer heat"


def test_derive_refresh_question_fallback_uses_topic_and_species() -> None:
    q = derive_refresh_question(evidence_json={}, topic="nutrition", species="dog")
    assert "nutrition" in q.lower() and "dog" in q.lower()


def test_extract_snippet_texts_from_envelope() -> None:
    env = {"snippets": [{"text": "first"}, {"text": "second"}]}
    assert extract_snippet_texts_from_envelope(env) == ["first", "second"]


def test_merge_envelope_appends_refresh_audit() -> None:
    prior = {"snippets": [{"id": "1", "text": "old"}], "refresh_audit": [{"at": "t0"}]}
    merged = merge_envelope_after_refresh(
        prior,
        refresh_audit_entry={"at": "t1", "new_status": "provisional"},
        refreshed_research_dump={"authority_score": 0.7},
    )
    assert len(merged["refresh_audit"]) == 2
    assert merged["last_refresh_research_meta"]["authority_score"] == 0.7


class _FakeResearch:
    def __init__(self, outcome: TrustedResearchOutcome) -> None:
        self.outcome = outcome
        self.requests: list[object] = []

    async def run(self, request: object) -> TrustedResearchOutcome:
        self.requests.append(request)
        return self.outcome


class _FakeRefreshRepo:
    def __init__(self, candidates: list[DueResearchCandidate]) -> None:
        self.candidates = candidates
        self.updates: list[dict[str, object]] = []

    async def list_due_for_review(self, job):  # noqa: ANN001
        _ = job
        return list(self.candidates)

    async def apply_refresh_update(self, **kwargs: object) -> None:
        self.updates.append(dict(kwargs))


@pytest.mark.asyncio
async def test_run_refresh_batch_runs_research_and_applies_updates() -> None:
    now = datetime.now(timezone.utc)
    long1 = (
        "Refreshed veterinary-aligned guidance recommends offering fresh water at all times while "
        "observing appetite changes during convalescence from mild gastrointestinal upset in dogs."
    )
    long2 = (
        "Follow up monitoring should include energy level, stool consistency, and willingness to "
        "exercise when reintroducing kibble after a prescriptive elimination diet trial ends safely."
    )
    rr = _research_result(
        snippets=[_snippet(now, long1), _snippet(now, long2, sid="s2")],
        authority=0.81,
    )
    from src.research.evidence_extractor import RetrievalEvidenceExtractor

    extraction = RetrievalEvidenceExtractor.from_research_evidence(
        rr.evidence,
        provider_id="stub",
        blocked_url_count=0,
    )
    outcome = TrustedResearchOutcome(research=rr, extraction=extraction)
    cand = DueResearchCandidate(
        db_id=42,
        lifecycle_status=KnowledgeRecordStatus.provisional.value,
        review_after=now,
        topic="nutrition",
        species="dog",
        breed=None,
        life_stage="adult",
        authority_score=0.8,
        provider_id="stub",
        evidence_json={
            "snippets": [
                {"id": "s1", "text": long1},
                {"id": "s2", "text": long2},
            ],
            "research_result_meta": {"expanded_queries": ["dog nutrition adult"]},
        },
    )
    repo = _FakeRefreshRepo([cand])
    research = _FakeResearch(outcome)
    job = build_refresh_job(as_of=now, limit=10)

    report = await run_refresh_batch(job, research=research, repository=repo, review_after_days=14)

    assert report.candidates_scanned == 1
    assert report.applied_updates == 1
    assert len(research.requests) == 1
    assert len(repo.updates) == 1
    u = repo.updates[0]
    assert u["db_id"] == 42
    assert u["status"] == KnowledgeRecordStatus.provisional.value
    assert u["last_verified_at"] == now
    assert u["review_after"] is not None
    assert u["authority_score"] == pytest.approx(0.81)
    env = u["evidence_json"]
    assert "refresh_audit" in env
    assert env["refresh_audit"][-1]["new_status"] == KnowledgeRecordStatus.provisional.value


@pytest.mark.asyncio
async def test_run_refresh_batch_records_error_and_continues() -> None:
    now = datetime.now(timezone.utc)
    rr = _research_result(snippets=[_snippet(now, "x")])
    from src.research.evidence_extractor import RetrievalEvidenceExtractor

    extraction = RetrievalEvidenceExtractor.from_research_evidence(rr.evidence, provider_id="stub")
    outcome = TrustedResearchOutcome(research=rr, extraction=extraction)

    class _BrokenRepo(_FakeRefreshRepo):
        async def apply_refresh_update(self, **kwargs: object) -> None:
            raise RuntimeError("disk full")

    cand = DueResearchCandidate(
        db_id=7,
        lifecycle_status=KnowledgeRecordStatus.provisional.value,
        review_after=now,
        topic=None,
        species=None,
        breed=None,
        life_stage=None,
        authority_score=0.5,
        provider_id="stub",
        evidence_json={"snippets": [{"text": "x"}]},
    )
    report = await run_refresh_batch(
        build_refresh_job(as_of=now),
        research=_FakeResearch(outcome),
        repository=_BrokenRepo([cand]),
    )
    assert report.applied_updates == 0
    assert len(report.errors) == 1
    assert "disk full" in report.errors[0]


def test_sqlalchemy_refresh_repository_is_constructible() -> None:
    """Smoke: class binds session factory without connecting until used."""
    SqlAlchemyKnowledgeRefreshRepository()
