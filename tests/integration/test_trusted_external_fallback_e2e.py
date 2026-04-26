"""
End-to-end style tests for the trusted external fallback path (RAGOrchestrator + mocks).

No real HTTP: retrieval and trusted research are faked or use in-memory registries + stub providers.
"""

from __future__ import annotations

import pytest
from src.api.schemas import PetProfile, QueryFilters, QueryRequest
from src.core.config import settings
from src.core.orchestrator import RAGOrchestrator
from src.research.ingest_candidates import classify_external_candidate_state
from src.research.schemas import KnowledgeRecordStatus
from src.research.trusted_research_service import TrustedResearchService
from src.research.web_retriever import BlockedUrl, TrustedRetrievalResult
from src.security.guardrails import EXTERNAL_PROVISIONAL_CONTEXT_DISCLAIMER

from tests.unit.test_orchestrator import FakeIngestionService, FakeLLMClient, FakeRetriever
from tests.unit.test_trusted_external_architecture import (
    _CountingTrustedResearch,
    _outcome_with_snippets,
    _RecordingExternalStore,
)
from tests.unit.test_trusted_research_service import _example_registry, _FakeRetrieval, _FixedQueryExpander


def _flags_external_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "enable_trusted_external_retrieval", True)
    monkeypatch.setattr(settings, "allow_provisional_in_query", True)
    monkeypatch.setattr(settings, "external_research_trigger_threshold", 0.75)
    monkeypatch.setattr(settings, "similarity_threshold", 0.75)


def _flags_external_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "enable_trusted_external_retrieval", False)
    monkeypatch.setattr(settings, "allow_provisional_in_query", False)


@pytest.mark.asyncio
async def test_e2e_internal_sufficient_external_fallback_not_called(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strong internal grounding above threshold → trusted research ``run`` is never invoked."""
    _flags_external_on(monkeypatch)

    llm = FakeLLMClient(response_text="Answer from internal KB only.")
    counting = _CountingTrustedResearch(_outcome_with_snippets(["must not appear"]))
    orch = RAGOrchestrator(
        llm_client=llm,
        retriever=FakeRetriever(
            results=[
                {
                    "chunk_id": "c-strong",
                    "document_id": "d-1",
                    "title": "Nutrition",
                    "source": "kb",
                    "similarity_score": 0.92,
                    "snippet": "Adult dogs benefit from consistent feeding schedules.",
                    "metadata": {},
                }
            ]
        ),
        ingestion_service=FakeIngestionService(),
        trusted_research=counting,
    )
    result = await orch.answer(
        QueryRequest(
            question="What is a good feeding routine for my adult dog?",
            pet_profile=PetProfile(species="dog", life_stage="adult"),
            filters=QueryFilters(category="nutrition"),
        )
    )
    assert counting.run_count == 0
    assert EXTERNAL_PROVISIONAL_CONTEXT_DISCLAIMER not in result.get("disclaimers", [])
    assert "internal KB" in result["answer"].lower() or "Answer from internal" in result["answer"]
    assert result["answer_source"] == "internal"
    assert result["knowledge_status"] == "approved"


@pytest.mark.asyncio
async def test_e2e_internal_weak_external_fallback_called(monkeypatch: pytest.MonkeyPatch) -> None:
    """Weak internal match triggers trusted external ``run`` exactly once."""
    _flags_external_on(monkeypatch)

    llm = FakeLLMClient(
        responses=[
            "Owner reply augmented with external context.",
            "## Evidence index\n- sn-0",
        ]
    )
    counting = _CountingTrustedResearch(_outcome_with_snippets(["Trusted hydration guidance from allowlisted source."]))
    orch = RAGOrchestrator(
        llm_client=llm,
        retriever=FakeRetriever(
            results=[
                {
                    "chunk_id": "c-weak",
                    "document_id": "d-1",
                    "title": "Vague",
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
    result = await orch.answer(
        QueryRequest(
            question="What is the best routine for my adult dog?",
            pet_profile=PetProfile(species="dog", life_stage="adult"),
            filters=QueryFilters(category="nutrition"),
        )
    )
    assert counting.run_count == 1
    assert counting.last_request is not None
    assert result["answer_source"] == "external_trusted"
    assert result["knowledge_status"] == "provisional"


@pytest.mark.asyncio
async def test_e2e_external_returns_trusted_evidence_frontend_answer_in_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """When external snippets exist, the user-facing answer comes from the frontend LLM pass (mocked)."""
    _flags_external_on(monkeypatch)

    front = "OWNER_VISIBLE: use fresh water daily per trusted excerpt."
    review = "## Evidence index\n- sn-0\n"
    llm = FakeLLMClient(responses=[front, review])
    orch = RAGOrchestrator(
        llm_client=llm,
        retriever=FakeRetriever(results=[]),
        ingestion_service=FakeIngestionService(),
        trusted_research=_CountingTrustedResearch(_outcome_with_snippets(["Offer fresh water and monitor intake."])),
    )
    result = await orch.answer(
        QueryRequest(
            question="What is the best routine for my adult dog?",
            pet_profile=PetProfile(species="dog", life_stage="adult"),
            filters=QueryFilters(category="nutrition"),
        )
    )
    assert EXTERNAL_PROVISIONAL_CONTEXT_DISCLAIMER in result.get("disclaimers", [])
    assert result["answer"].startswith("OWNER_VISIBLE:")
    assert "fresh water" in result["answer"].lower()
    assert result["answer_source"] == "external_trusted"
    assert result["knowledge_status"] == "provisional"


@pytest.mark.asyncio
async def test_e2e_candidate_knowledge_persisted_as_provisional(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto-save stores a general-topic bundle as ``provisional`` (not approved)."""
    _flags_external_on(monkeypatch)
    monkeypatch.setattr(settings, "enable_auto_save_provisional_knowledge", True)

    llm = FakeLLMClient(
        responses=[
            "Routine feeding tips for adult dogs.",
            "## Evidence index\n- sn-0",
        ]
    )
    store = _RecordingExternalStore()
    orch = RAGOrchestrator(
        llm_client=llm,
        retriever=FakeRetriever(results=[]),
        ingestion_service=FakeIngestionService(),
        trusted_research=_CountingTrustedResearch(_outcome_with_snippets(["External feeding note."])),
        external_research_store=store,
    )
    result = await orch.answer(
        QueryRequest(
            question="What is the best routine for my adult dog?",
            pet_profile=PetProfile(species="dog", life_stage="adult"),
            filters=QueryFilters(category="nutrition"),
        )
    )
    assert result["answer_source"] == "external_trusted"
    assert result["knowledge_status"] == "provisional"
    assert len(store.ingests) == 1
    inp = store.ingests[0]
    st, blocked = classify_external_candidate_state(
        inp.content_sensitivity,
        topic=inp.research_result.topic if inp.research_result else None,
        species=inp.research_result.species if inp.research_result else None,
        life_stage=inp.research_result.life_stage if inp.research_result else None,
        category=inp.research_result.topic if inp.research_result else None,
    )
    assert st == KnowledgeRecordStatus.provisional.value
    assert blocked is False
    assert inp.extraction.research_evidence.snippets


@pytest.mark.asyncio
async def test_e2e_medical_diagnosis_needs_vet_followup_and_auto_promotion_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Medical-style question (``diagnosis``) keeps ``needs_vet_followup`` true; persisted row is ``needs_review``.
    """
    _flags_external_on(monkeypatch)
    monkeypatch.setattr(settings, "enable_auto_save_provisional_knowledge", True)

    llm = FakeLLMClient(
        responses=[
            "Educational: seek veterinary evaluation for limping.",
            "## Evidence index\n- sn-0",
        ]
    )
    store = _RecordingExternalStore()
    orch = RAGOrchestrator(
        llm_client=llm,
        retriever=FakeRetriever(results=[]),
        ingestion_service=FakeIngestionService(),
        trusted_research=_CountingTrustedResearch(_outcome_with_snippets(["Orthopedic signs may need exam."])),
        external_research_store=store,
    )
    result = await orch.answer(
        QueryRequest(
            question="What could be the diagnosis if my dog is limping on the back leg?",
            pet_profile=PetProfile(species="dog", life_stage="adult"),
        )
    )
    assert result["needs_vet_followup"] is True
    assert result["answer_source"] == "external_trusted"
    assert result["knowledge_status"] == "provisional"
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


@pytest.mark.asyncio
async def test_e2e_non_allowlisted_domains_rejected_no_provisional_augmentation(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    TrustedResearchService + fake retrieval: blocked evil domain, no hits → no external context in RAG.
    """
    _flags_external_on(monkeypatch)

    fake = _FakeRetrieval(
        TrustedRetrievalResult(
            hits=[],
            blocked=[
                BlockedUrl(
                    url="https://evil-untrusted.example/page",
                    reason="domain_not_allowlisted",
                )
            ],
            provider_id="allowlist_stub",
        )
    )
    trusted = TrustedResearchService(
        registry=_example_registry(),
        query_expander=_FixedQueryExpander(),
        retrieval=fake,
        content_fetcher=None,
    )
    llm = FakeLLMClient(
        responses=[
            "Would have been owner answer if context existed.",
            "## Evidence index\n- n/a\n",
        ]
    )
    orch = RAGOrchestrator(
        llm_client=llm,
        retriever=FakeRetriever(results=[]),
        ingestion_service=FakeIngestionService(),
        trusted_research=trusted,
    )
    result = await orch.answer(
        QueryRequest(
            question="What is the best routine for my adult dog?",
            pet_profile=PetProfile(species="dog", life_stage="adult"),
            filters=QueryFilters(category="nutrition"),
        )
    )
    assert fake.last_input is not None
    assert fake.last_input.candidate_urls  # allowlisted targets attempted; evil URL blocked upstream
    assert len(fake.result.blocked) == 1
    assert "not_allowlisted" in fake.result.blocked[0].reason.lower()
    assert EXTERNAL_PROVISIONAL_CONTEXT_DISCLAIMER not in result.get("disclaimers", [])
    # No snippets after policy rejection → no chunks for LLM → safe fallback (still no internet).
    assert "grounded information" in result["answer"].lower()
    assert result["confidence"] == "low"
    assert result["answer_source"] == "fallback"
    assert result["knowledge_status"] == "none"


@pytest.mark.asyncio
async def test_e2e_feature_flags_off_matches_internal_only_no_external_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """With flags off, orchestrator uses disabled stack and never calls injected exploding research."""
    _flags_external_off(monkeypatch)

    llm = FakeLLMClient(response_text="Internal-only path answer.")
    counting = _CountingTrustedResearch(_outcome_with_snippets(["must not run"]))
    orch = RAGOrchestrator(
        llm_client=llm,
        retriever=FakeRetriever(
            results=[
                {
                    "chunk_id": "c-weak",
                    "document_id": "d-1",
                    "title": "Vague",
                    "source": "kb",
                    "similarity_score": 0.55,
                    "snippet": "Thin internal context only.",
                    "metadata": {},
                }
            ]
        ),
        ingestion_service=FakeIngestionService(),
        trusted_research=counting,
    )
    result = await orch.answer(
        QueryRequest(
            question="What is the best routine for my adult dog?",
            pet_profile=PetProfile(species="dog", life_stage="adult"),
            filters=QueryFilters(category="nutrition"),
        )
    )
    assert counting.run_count == 0
    assert EXTERNAL_PROVISIONAL_CONTEXT_DISCLAIMER not in result.get("disclaimers", [])
    assert "Internal-only path" in result["answer"]
    assert result["answer_source"] == "internal"
    assert result["knowledge_status"] == "approved"
