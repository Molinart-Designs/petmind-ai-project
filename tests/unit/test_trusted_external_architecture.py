"""
Architecture-level tests for trusted external fallback (RAG → trusted research → provisional
persistence), dual LLM outputs, allowlist policy, and knowledge refresh TTL.

Each test maps to an explicit product requirement for traceability.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import HttpUrl

from src.api.schemas import PetProfile, QueryFilters, QueryRequest
from src.core.config import settings
from src.core.orchestrator import RAGOrchestrator
from src.research.evidence_extractor import RetrievalEvidenceExtractor
from src.research.ingest_candidates import (
    ExternalResearchIngestInput,
    NullExternalResearchPersistence,
    classify_external_candidate_state,
)
from src.research.knowledge_refresh import (
    DueResearchCandidate,
    build_refresh_job,
    decide_refresh_evidence,
    run_refresh_batch,
)
from src.research.schemas import (
    ExternalSourceType,
    ExtractedSnippet,
    ExternalSource,
    KnowledgeRecordStatus,
    ResearchEvidence,
    ResearchResult,
)
from src.research.source_registry import MedicalSensitivityLevel, TrustedSourceEntry, TrustedSourceRegistry
from src.research.trusted_research_service import TrustedResearchOutcome, TrustedResearchRequest
from src.research.web_retriever import TrustedExternalRetrievalService, TrustedRetrievalInput
from src.research.evidence_quality import snippet_text_meets_evidence_quality
from src.security.guardrails import EXTERNAL_PROVISIONAL_CONTEXT_DISCLAIMER

from tests.unit.test_orchestrator import FakeIngestionService, FakeLLMClient, FakeRetriever


class _ExplodingTrustedResearch:
    """Fails if invoked — proves the orchestrator skipped trusted external retrieval."""

    async def run(self, request: TrustedResearchRequest) -> TrustedResearchOutcome:  # noqa: ARG002
        raise AssertionError("trusted external retrieval must not run for this scenario")


class _CountingTrustedResearch:
    def __init__(self, outcome: TrustedResearchOutcome) -> None:
        self.run_count = 0
        self.last_request: TrustedResearchRequest | None = None
        self._outcome = outcome

    async def run(self, request: TrustedResearchRequest) -> TrustedResearchOutcome:
        self.run_count += 1
        self.last_request = request
        return self._outcome


def _snippet(now: datetime, text: str, sid: str = "sn-1") -> ExtractedSnippet:
    return ExtractedSnippet(
        id=sid,
        external_source_id="00000000-0000-4000-8000-000000000001",
        text=text,
        authority_score=0.82,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
    )


_SECOND_QUALITY_SNIPPET = (
    "Peer reviewed veterinary references recommend measuring portions, offering puzzle feeders, and "
    "scheduling consistent mealtimes to reduce scavenging behavior during canine weight management."
)


def _ensure_gate_ready_snippet_text(text: str) -> str:
    candidate = text.strip()
    if snippet_text_meets_evidence_quality(candidate):
        return candidate
    merged = f"{candidate} {_SECOND_QUALITY_SNIPPET}".strip()
    return merged if snippet_text_meets_evidence_quality(merged) else _SECOND_QUALITY_SNIPPET


def _outcome_with_snippets(texts: list[str]) -> TrustedResearchOutcome:
    now = datetime.now(timezone.utc)
    src = ExternalSource(
        id="00000000-0000-4000-8000-000000000001",
        source_key="doc_example",
        base_url=HttpUrl("https://avma.org/"),
        authority_score=0.82,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
    )
    normalized = [_ensure_gate_ready_snippet_text(t) for t in texts] if texts else [_SECOND_QUALITY_SNIPPET]
    while len(normalized) < 2:
        normalized.append(_SECOND_QUALITY_SNIPPET)
    snippets = [_snippet(now, t, sid=f"sn-{i}") for i, t in enumerate(normalized)]
    ev = ResearchEvidence(snippets=snippets, sources=[src])
    rr = ResearchResult(
        evidence=ev,
        authority_score=0.82,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
        evidence_summary="architecture test",
        external_confidence="medium",
    )
    ext = RetrievalEvidenceExtractor.from_research_evidence(ev, provider_id="stub", blocked_url_count=0)
    return TrustedResearchOutcome(research=rr, extraction=ext)


def _outcome_empty() -> TrustedResearchOutcome:
    now = datetime.now(timezone.utc)
    ev = ResearchEvidence(snippets=[], sources=[])
    rr = ResearchResult(
        evidence=ev,
        authority_score=0.0,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
        evidence_summary="empty",
        external_confidence="low",
    )
    ext = RetrievalEvidenceExtractor.from_research_evidence(ev, provider_id="stub", blocked_url_count=0)
    return TrustedResearchOutcome(research=rr, extraction=ext)


class _RecordingExternalStore(NullExternalResearchPersistence):
    """Captures ingest payloads while delegating summary shape to the null store."""

    def __init__(self) -> None:
        self.ingests: list[ExternalResearchIngestInput] = []

    async def persist_external_research(self, inp: ExternalResearchIngestInput) -> dict:  # type: ignore[override]
        self.ingests.append(inp)
        base = await super().persist_external_research(inp)
        topic = inp.research_result.topic if inp.research_result else None
        species = inp.research_result.species if inp.research_result else None
        life_stage = inp.research_result.life_stage if inp.research_result else None
        category = topic
        status, _ = classify_external_candidate_state(
            inp.content_sensitivity,
            topic=topic,
            species=species,
            life_stage=life_stage,
            category=category,
        )
        return {**base, "status": status, "provisional": status == KnowledgeRecordStatus.provisional.value}


@pytest.mark.asyncio
async def test_architecture_internal_retrieval_wins_when_context_is_sufficient(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    With strong internal matches (above similarity threshold and above external trigger),
    trusted external retrieval must not run even when ENABLE_TRUSTED_EXTERNAL_RETRIEVAL is on.
    """
    monkeypatch.setattr(settings, "enable_trusted_external_retrieval", True)
    monkeypatch.setattr(settings, "allow_provisional_in_query", True)
    monkeypatch.setattr(settings, "external_research_trigger_threshold", 0.75)

    llm = FakeLLMClient(response_text="Grounded internal-only reply.")
    retriever = FakeRetriever(
        results=[
            {
                "chunk_id": "c-strong",
                "document_id": "d-1",
                "title": "Curated nutrition",
                "source": "internal-kb",
                "similarity_score": 0.92,
                "snippet": "Adult dogs benefit from consistent feeding schedules.",
                "metadata": {},
            }
        ]
    )
    ingestion = FakeIngestionService()
    orchestrator = RAGOrchestrator(
        llm_client=llm,
        retriever=retriever,
        ingestion_service=ingestion,
        trusted_research=_ExplodingTrustedResearch(),
    )
    payload = QueryRequest(
        question="What is a good feeding routine for my adult dog?",
        pet_profile=PetProfile(species="dog", life_stage="adult"),
        filters=QueryFilters(category="nutrition"),
    )
    result = await orchestrator.answer(payload)
    assert result["confidence"] == "high"
    assert EXTERNAL_PROVISIONAL_CONTEXT_DISCLAIMER not in result.get("disclaimers", [])
    assert "Grounded internal-only reply" in result["answer"]


@pytest.mark.asyncio
async def test_architecture_allow_provisional_in_query_false_blocks_external_despite_master_enable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ALLOW_PROVISIONAL_IN_QUERY=false keeps /query internal-only even if ENABLE_TRUSTED_EXTERNAL_RETRIEVAL=true."""
    monkeypatch.setattr(settings, "enable_trusted_external_retrieval", True)
    monkeypatch.setattr(settings, "allow_provisional_in_query", False)
    monkeypatch.setattr(settings, "external_research_trigger_threshold", 0.75)

    counting = _CountingTrustedResearch(_outcome_with_snippets(["must not load"]))
    llm = FakeLLMClient()
    orchestrator = RAGOrchestrator(
        llm_client=llm,
        retriever=FakeRetriever(results=[]),
        ingestion_service=FakeIngestionService(),
        trusted_research=counting,
    )
    result = await orchestrator.answer(
        QueryRequest(
            question="What is the best routine for my adult dog?",
            pet_profile=PetProfile(species="dog", life_stage="adult"),
        )
    )
    assert counting.run_count == 0
    assert llm.called is False
    assert result.get("review_draft") is None
    assert EXTERNAL_PROVISIONAL_CONTEXT_DISCLAIMER not in result.get("disclaimers", [])


@pytest.mark.asyncio
async def test_architecture_external_fallback_triggers_when_internal_context_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No internal chunks → trusted research runs and augments the answer path."""
    monkeypatch.setattr(settings, "enable_trusted_external_retrieval", True)
    monkeypatch.setattr(settings, "allow_provisional_in_query", True)
    monkeypatch.setattr(settings, "external_research_trigger_threshold", 0.75)
    monkeypatch.setattr(settings, "enable_auto_save_provisional_knowledge", False)

    outcome = _outcome_with_snippets(["External stub evidence about hydration."])
    counting = _CountingTrustedResearch(outcome)
    llm = FakeLLMClient(
        responses=[
            "Owner-facing: encourage water access based on mixed context.",
            "## Internal synthesis\nCites external stub evidence.\n\n## Evidence index\n- sn-0",
        ]
    )
    orchestrator = RAGOrchestrator(
        llm_client=llm,
        retriever=FakeRetriever(results=[]),
        ingestion_service=FakeIngestionService(),
        trusted_research=counting,
    )
    result = await orchestrator.answer(
        QueryRequest(
            question="What is the best routine for my adult dog?",
            pet_profile=PetProfile(species="dog", life_stage="adult"),
            filters=QueryFilters(category="nutrition"),
        )
    )
    assert counting.run_count == 1
    assert counting.last_request is not None
    assert EXTERNAL_PROVISIONAL_CONTEXT_DISCLAIMER in result["disclaimers"]
    assert "water" in result["answer"].lower()


@pytest.mark.asyncio
async def test_architecture_external_fallback_triggers_when_internal_context_insufficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chunks below ``SIMILARITY_THRESHOLD`` count as weak internal grounding → trusted research runs."""
    monkeypatch.setattr(settings, "enable_trusted_external_retrieval", True)
    monkeypatch.setattr(settings, "allow_provisional_in_query", True)
    monkeypatch.setattr(settings, "similarity_threshold", 0.75)
    monkeypatch.setattr(settings, "external_research_trigger_threshold", 0.75)
    monkeypatch.setattr(settings, "enable_auto_save_provisional_knowledge", False)

    counting = _CountingTrustedResearch(_outcome_with_snippets(["Trusted layer fills the gap."]))
    llm = FakeLLMClient(
        responses=[
            "Owner guidance with provisional external support.",
            "## Evidence index\n- sn-0",
        ]
    )
    orchestrator = RAGOrchestrator(
        llm_client=llm,
        retriever=FakeRetriever(
            results=[
                {
                    "chunk_id": "c-weak",
                    "document_id": "d-1",
                    "title": "Weak match",
                    "source": "kb",
                    "similarity_score": 0.55,
                    "snippet": "Loosely related text.",
                    "metadata": {},
                }
            ]
        ),
        ingestion_service=FakeIngestionService(),
        trusted_research=counting,
    )
    await orchestrator.answer(
        QueryRequest(
            question="What is the best routine for my adult dog?",
            pet_profile=PetProfile(species="dog", life_stage="adult"),
        )
    )
    assert counting.run_count == 1


@pytest.mark.asyncio
async def test_architecture_external_fallback_triggers_when_top_score_below_trigger_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Internal context can be ``similarity_threshold``-sufficient while ``top_score`` is still
    below ``EXTERNAL_RESEARCH_TRIGGER_THRESHOLD`` → external augmentation still runs.
    """
    monkeypatch.setattr(settings, "enable_trusted_external_retrieval", True)
    monkeypatch.setattr(settings, "allow_provisional_in_query", True)
    monkeypatch.setattr(settings, "similarity_threshold", 0.70)
    monkeypatch.setattr(settings, "external_research_trigger_threshold", 0.95)

    counting = _CountingTrustedResearch(_outcome_with_snippets(["Augmenting excerpt from trusted layer."]))
    llm = FakeLLMClient(
        responses=[
            "Combined guidance referencing internal and supplementary context.",
            "## Evidence index\n- internal\n- sn-0",
        ]
    )
    orchestrator = RAGOrchestrator(
        llm_client=llm,
        retriever=FakeRetriever(
            results=[
                {
                    "chunk_id": "c-mid",
                    "document_id": "d-1",
                    "title": "Internal",
                    "source": "kb",
                    "similarity_score": 0.88,
                    "snippet": "Internal baseline text.",
                    "metadata": {},
                }
            ]
        ),
        ingestion_service=FakeIngestionService(),
        trusted_research=counting,
    )
    result = await orchestrator.answer(
        QueryRequest(
            question="What is a good feeding routine for my adult dog?",
            pet_profile=PetProfile(species="dog", life_stage="adult"),
        )
    )
    assert counting.run_count == 1
    assert EXTERNAL_PROVISIONAL_CONTEXT_DISCLAIMER in result["disclaimers"]


@pytest.mark.asyncio
async def test_architecture_allowlist_blocks_unknown_domains_in_trusted_retrieval_service() -> None:
    """Unknown hosts never become hits; policy rows record ``domain_not_allowlisted``."""
    reg = TrustedSourceRegistry(
        entries=(
            TrustedSourceEntry(
                source_key="doc_example",
                allowlisted_domains=("example.com",),
                category="documentation",
                authority_score=0.5,
                medical_sensitivity=MedicalSensitivityLevel.none,
                auto_ingest_allowed=True,
            ),
        )
    )
    svc = TrustedExternalRetrievalService(registry=reg)
    out = await svc.retrieve(
        TrustedRetrievalInput(
            candidate_urls=[
                HttpUrl("https://example.com/allowed"),
                HttpUrl("https://malicious.example/blocked"),
            ],
            context_queries=["q"],
        )
    )
    assert len(out.hits) == 1
    assert str(out.hits[0].url).startswith("https://example.com/")
    assert len(out.blocked) == 1
    assert out.blocked[0].reason == "domain_not_allowlisted"


@pytest.mark.asyncio
async def test_architecture_candidate_bundle_saved_as_provisional_for_general_topic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persisted external research for non-medical flows is classified ``provisional`` (not approved)."""
    monkeypatch.setattr(settings, "enable_trusted_external_retrieval", True)
    monkeypatch.setattr(settings, "allow_provisional_in_query", True)
    monkeypatch.setattr(settings, "external_research_trigger_threshold", 0.75)
    monkeypatch.setattr(settings, "enable_auto_save_provisional_knowledge", True)

    store = _RecordingExternalStore()
    llm = FakeLLMClient(
        responses=[
            "Routine feeding tips for adult dogs.",
            "## Evidence index\n- sn-0",
        ]
    )
    orchestrator = RAGOrchestrator(
        llm_client=llm,
        retriever=FakeRetriever(results=[]),
        ingestion_service=FakeIngestionService(),
        trusted_research=_CountingTrustedResearch(_outcome_with_snippets(["External feeding note."])),
        external_research_store=store,
    )
    result = await orchestrator.answer(
        QueryRequest(
            question="What is the best routine for my adult dog?",
            pet_profile=PetProfile(species="dog", life_stage="adult"),
            filters=QueryFilters(category="nutrition"),
        )
    )
    assert len(store.ingests) == 1
    inp = store.ingests[0]
    assert inp.content_sensitivity == "general"
    st, blocked = classify_external_candidate_state(
        inp.content_sensitivity,
        topic=inp.research_result.topic if inp.research_result else None,
        species=inp.research_result.species if inp.research_result else None,
        life_stage=inp.research_result.life_stage if inp.research_result else None,
        category=inp.research_result.topic if inp.research_result else None,
    )
    assert st == KnowledgeRecordStatus.provisional.value
    assert blocked is False
    assert "Routine feeding" in result["answer"]


@pytest.mark.asyncio
async def test_architecture_sensitive_medical_content_not_auto_promoted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    - Sensitive queries skip external retrieval (no provisional ingest path).
    - Medical-topic external saves use ``content_sensitivity=medical`` → ``needs_review``, never approved.
    """
    monkeypatch.setattr(settings, "enable_trusted_external_retrieval", True)
    monkeypatch.setattr(settings, "allow_provisional_in_query", True)
    monkeypatch.setattr(settings, "enable_auto_save_provisional_knowledge", True)

    store = _RecordingExternalStore()
    counting_sensitive = _CountingTrustedResearch(_outcome_with_snippets(["should not run"]))
    orch_sensitive = RAGOrchestrator(
        llm_client=FakeLLMClient(),
        retriever=FakeRetriever(results=[]),
        ingestion_service=FakeIngestionService(),
        trusted_research=counting_sensitive,
        external_research_store=store,
    )
    sens_result = await orch_sensitive.answer(
        QueryRequest(
            question="My dog is having a seizure emergency what do I do?",
            pet_profile=PetProfile(species="dog", life_stage="adult"),
        )
    )
    assert store.ingests == []
    assert counting_sensitive.run_count == 0
    assert sens_result.get("review_draft") is None
    assert EXTERNAL_PROVISIONAL_CONTEXT_DISCLAIMER not in sens_result.get("disclaimers", [])

    llm = FakeLLMClient(
        responses=[
            "Educational note: fever warrants veterinary evaluation.",
            "## Evidence index\n- sn-0",
        ]
    )
    counting = _CountingTrustedResearch(_outcome_with_snippets(["Fever may indicate infection in dogs."]))
    orch_med = RAGOrchestrator(
        llm_client=llm,
        retriever=FakeRetriever(results=[]),
        ingestion_service=FakeIngestionService(),
        trusted_research=counting,
        external_research_store=store,
    )
    store.ingests.clear()
    await orch_med.answer(
        QueryRequest(
            question="My dog has a fever and seems lethargic; what should I watch for?",
            pet_profile=PetProfile(species="dog", life_stage="adult"),
        )
    )
    assert counting.run_count == 1
    assert len(store.ingests) == 1
    med_inp = store.ingests[0]
    assert med_inp.content_sensitivity == "medical"
    st_med, blocked_med = classify_external_candidate_state(
        med_inp.content_sensitivity,
        topic=med_inp.research_result.topic if med_inp.research_result else None,
        species=med_inp.research_result.species if med_inp.research_result else None,
        life_stage=med_inp.research_result.life_stage if med_inp.research_result else None,
        category=med_inp.research_result.topic if med_inp.research_result else None,
    )
    assert st_med == KnowledgeRecordStatus.needs_review.value
    assert blocked_med is True
    assert KnowledgeRecordStatus.approved.value != st_med


@pytest.mark.asyncio
async def test_architecture_dual_output_generation_produces_distinct_frontend_and_review_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "enable_trusted_external_retrieval", False)
    monkeypatch.setattr(settings, "allow_provisional_in_query", False)

    front = "SHORT OWNER REPLY: feed twice daily."
    review = (
        "## Synthesis\nInternal article for reviewers.\n\n"
        "## Evidence index\n- chunk-strong\n"
    )
    llm = FakeLLMClient(responses=[front, review])
    orchestrator = RAGOrchestrator(
        llm_client=llm,
        retriever=FakeRetriever(
            results=[
                {
                    "chunk_id": "chunk-strong",
                    "document_id": "d-1",
                    "title": "Nutrition",
                    "source": "kb",
                    "similarity_score": 0.93,
                    "snippet": "Twice-daily meals work well for many adult dogs.",
                    "metadata": {"source_url": "https://kb.example/nutrition"},
                }
            ]
        ),
        ingestion_service=FakeIngestionService(),
    )
    result = await orchestrator.answer(
        QueryRequest(
            question="How often should I feed my adult dog?",
            pet_profile=PetProfile(species="dog", life_stage="adult"),
        )
    )
    assert llm.call_count == 2
    assert result["answer"].startswith("SHORT OWNER REPLY")
    assert result.get("review_draft")
    assert "Evidence index" in result["review_draft"]
    assert result["answer"] != result["review_draft"]
    assert "chunk-strong" in llm.user_prompts[1]


@pytest.mark.asyncio
async def test_architecture_refresh_job_expires_stale_provisional_when_new_evidence_empty() -> None:
    """TTL refresh: empty trusted re-fetch downgrades multi-snippet bundles to ``expired``."""
    now = datetime.now(timezone.utc)
    cand = DueResearchCandidate(
        db_id=501,
        lifecycle_status=KnowledgeRecordStatus.provisional.value,
        review_after=now,
        topic="nutrition",
        species="dog",
        breed=None,
        life_stage="adult",
        authority_score=0.9,
        provider_id="stub",
        evidence_json={
            "snippets": [{"text": "older one"}, {"text": "older two"}],
            "research_result_meta": {"expanded_queries": ["dog nutrition refresh"]},
        },
    )

    class _Repo:
        def __init__(self) -> None:
            self.updates: list[dict[str, object]] = []

        async def list_due_for_review(self, job):  # noqa: ANN001
            _ = job
            return [cand]

        async def apply_refresh_update(self, **kwargs: object) -> None:
            self.updates.append(dict(kwargs))

    repo = _Repo()
    report = await run_refresh_batch(
        build_refresh_job(as_of=now, limit=10),
        research=_CountingTrustedResearch(_outcome_empty()),
        repository=repo,
        review_after_days=30,
    )
    assert report.candidates_scanned == 1
    assert report.marked_expired == 1
    assert repo.updates[0]["status"] == KnowledgeRecordStatus.expired.value
    assert repo.updates[0]["last_verified_at"] == now
    assert repo.updates[0]["review_after"] is not None


def test_architecture_refresh_decision_never_auto_promotes_to_approved() -> None:
    """Refresh policy layer never emits ``approved`` (human gate only)."""
    d = decide_refresh_evidence(
        prior_status=KnowledgeRecordStatus.provisional.value,
        old_snippet_texts=["a", "b"],
        old_authority=0.9,
        new_snippet_texts=["a", "b"],
        new_authority=0.91,
        jaccard=1.0,
    )
    assert d.new_status != KnowledgeRecordStatus.approved.value
