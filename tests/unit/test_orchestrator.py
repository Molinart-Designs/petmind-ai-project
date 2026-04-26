from datetime import datetime, timezone

import pytest
from pydantic import HttpUrl

from src.api.schemas import IngestRequest, PetProfile, QueryFilters, QueryRequest
from src.core.config import settings
from src.core import orchestrator as orchestrator_module
from src.core.orchestrator import RAGOrchestrator, get_orchestrator
from src.research.evidence_extractor import RetrievalEvidenceExtractor
from src.research.schemas import (
    ExternalSourceType,
    ExtractedSnippet,
    ExternalSource,
    ResearchEvidence,
    ResearchResult,
)
from src.research.trusted_research_service import TrustedResearchOutcome, TrustedResearchRequest
from src.security.guardrails import EXTERNAL_PROVISIONAL_CONTEXT_DISCLAIMER


class FakeLLMClient:
    def __init__(
        self,
        response_text: str = "Grounded answer from the LLM.",
        *,
        responses: list[str] | None = None,
    ) -> None:
        default_review = (
            "Internal review article for editors.\n\n"
            "## Evidence index\n"
            "- Referenced catalog rows above; cite chunk_id and URLs from the evidence catalog.\n"
        )
        self._responses = responses if responses is not None else [response_text, default_review]
        self.called = False
        self.call_count = 0
        self.last_system_prompt = None
        self.last_user_prompt = None
        self.last_max_output_tokens = None
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []

    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int = 600,
    ) -> str:
        self.called = True
        self.call_count += 1
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        self.last_max_output_tokens = max_output_tokens
        self.system_prompts.append(system_prompt)
        self.user_prompts.append(user_prompt)
        idx = min(self.call_count - 1, len(self._responses) - 1)
        return self._responses[idx]


class FakeRetriever:
    def __init__(self, results=None) -> None:
        self.results = results or []
        self.called = False
        self.last_question = None
        self.last_top_k = None
        self.last_filters = None

    async def retrieve(self, *, question: str, top_k: int, filters: dict):
        self.called = True
        self.last_question = question
        self.last_top_k = top_k
        self.last_filters = filters
        return self.results


class FakeIngestionService:
    def __init__(self, result=None) -> None:
        self.result = result or {
            "documents_processed": 1,
            "chunks_created": 3,
            "document_ids": ["doc-demo-1"],
            "message": "Documents ingested successfully into the knowledge base.",
        }
        self.called = False
        self.last_payload = None

    async def ingest_documents(self, payload: IngestRequest):
        self.called = True
        self.last_payload = payload
        return self.result


@pytest.mark.asyncio
async def test_answer_returns_safe_fallback_when_no_context_is_found(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "enable_trusted_external_retrieval", False)
    monkeypatch.setattr(settings, "allow_provisional_in_query", False)

    llm_client = FakeLLMClient()
    retriever = FakeRetriever(results=[])
    ingestion_service = FakeIngestionService()

    orchestrator = RAGOrchestrator(
        llm_client=llm_client,
        retriever=retriever,
        ingestion_service=ingestion_service,
    )

    payload = QueryRequest(
        question="What is the best routine for my adult dog?",
        pet_profile={
            "species": "dog",
            "life_stage": "adult",
        },
        filters={
            "category": "nutrition",
        },
        top_k=4,
    )

    result = await orchestrator.answer(payload)

    assert retriever.called is True
    assert retriever.last_question == "What is the best routine for my adult dog?"
    assert retriever.last_top_k == 4
    assert retriever.last_filters == {
        "category": "nutrition",
        "species": "dog",
        "life_stage": "adult",
    }

    assert llm_client.called is False
    assert result["confidence"] == "low"
    assert result["retrieval_count"] == 0
    assert result["used_filters"] == {
        "category": "nutrition",
        "species": "dog",
        "life_stage": "adult",
    }
    assert "grounded information" in result["answer"].lower()
    assert "licensed veterinarian" not in result["answer"].lower()
    assert result.get("needs_vet_followup") is False
    assert result.get("review_draft") is None
    assert "generated_at" in result
    assert result["answer_source"] == "fallback"
    assert result["knowledge_status"] == "none"


@pytest.mark.asyncio
async def test_answer_uses_external_trusted_when_internal_empty_and_allowlist_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "enable_trusted_external_retrieval", True)
    monkeypatch.setattr(settings, "allow_provisional_in_query", True)
    monkeypatch.setattr(settings, "external_research_trigger_threshold", 0.75)
    monkeypatch.setattr(settings, "similarity_threshold", 0.75)
    monkeypatch.setattr(settings, "trusted_external_allowlist_domains", "avma.org")
    monkeypatch.setattr(settings, "trusted_search_provider", "stub")

    orchestrator = RAGOrchestrator(
        llm_client=FakeLLMClient(response_text="Guidance grounded in the provided trusted external excerpts."),
        retriever=FakeRetriever(results=[]),
        ingestion_service=FakeIngestionService(),
        trusted_research=_FixedTrustedResearch(),
    )

    payload = QueryRequest(
        question="What is a good weekly routine for brushing a short-haired dog?",
        pet_profile=None,
        filters=None,
        top_k=4,
    )
    result = await orchestrator.answer(payload)

    assert result["answer_source"] == "external_trusted"
    assert result["knowledge_status"] == "provisional"
    assert EXTERNAL_PROVISIONAL_CONTEXT_DISCLAIMER in result["disclaimers"]


@pytest.mark.asyncio
async def test_answer_calls_llm_when_grounding_is_sufficient():
    llm_client = FakeLLMClient(
        response_text="Adult dogs usually benefit from consistent feeding schedules and portion control."
    )
    retriever = FakeRetriever(
        results=[
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "title": "Feeding basics for adult dogs",
                "source": "curated-demo-source",
                "category": "nutrition",
                "species": "dog",
                "life_stage": "adult",
                "similarity_score": 0.91,
                "snippet": "Adult dogs benefit from consistent feeding schedules.",
                "metadata": {"chunk_index": 1},
            }
        ]
    )
    ingestion_service = FakeIngestionService()

    orchestrator = RAGOrchestrator(
        llm_client=llm_client,
        retriever=retriever,
        ingestion_service=ingestion_service,
    )

    payload = QueryRequest(
        question="What is a good feeding routine for my adult dog?",
        pet_profile={
            "species": "dog",
            "breed": "Labrador",
            "life_stage": "adult",
            "age_years": 4,
        },
        filters={
            "category": "nutrition",
        },
        top_k=3,
    )

    result = await orchestrator.answer(payload)

    assert retriever.called is True
    assert retriever.last_top_k == 3
    assert retriever.last_filters == {
        "category": "nutrition",
        "species": "dog",
        "life_stage": "adult",
    }

    assert llm_client.called is True
    assert llm_client.call_count == 2
    assert "petmind ai" in llm_client.system_prompts[0].lower()
    assert "user question" in llm_client.user_prompts[0].lower()
    assert "pet profile" in llm_client.user_prompts[0].lower()
    assert "knowledge context" in llm_client.user_prompts[0].lower()
    assert "internal petmind analyst" in llm_client.system_prompts[1].lower()
    assert "evidence catalog" in llm_client.user_prompts[1].lower()
    assert "chunk_id=chunk-1" in llm_client.user_prompts[1]

    assert result["answer"].startswith("Adult dogs usually benefit")
    assert result.get("review_draft")
    assert "evidence index" in result["review_draft"].lower()
    assert result["confidence"] == "high"
    assert result["needs_vet_followup"] is False
    assert result["retrieval_count"] == 1
    assert result["answer_source"] == "internal"
    assert result["knowledge_status"] == "approved"
    assert result["used_filters"] == {
        "category": "nutrition",
        "species": "dog",
        "life_stage": "adult",
    }
    assert len(result["sources"]) == 1
    assert "generated_at" in result


@pytest.mark.asyncio
async def test_answer_prefers_explicit_filters_over_pet_profile_defaults():
    llm_client = FakeLLMClient()
    retriever = FakeRetriever(
        results=[
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "title": "Senior dog nutrition",
                "source": "curated-demo-source",
                "category": "nutrition",
                "species": "dog",
                "life_stage": "senior",
                "similarity_score": 0.88,
                "snippet": "Senior dogs may benefit from adjusted feeding routines.",
                "metadata": {},
            }
        ]
    )
    ingestion_service = FakeIngestionService()

    orchestrator = RAGOrchestrator(
        llm_client=llm_client,
        retriever=retriever,
        ingestion_service=ingestion_service,
    )

    payload = QueryRequest(
        question="How should I feed my senior dog?",
        pet_profile={
            "species": "dog",
            "life_stage": "adult",
        },
        filters={
            "category": "nutrition",
            "life_stage": "senior",
        },
    )

    result = await orchestrator.answer(payload)

    assert retriever.last_filters == {
        "category": "nutrition",
        "species": "dog",
        "life_stage": "senior",
    }
    assert result["used_filters"]["life_stage"] == "senior"
    assert llm_client.call_count == 2
    assert result.get("review_draft") is not None


@pytest.mark.asyncio
async def test_medical_question_frontend_prompt_is_extra_conservative():
    llm_client = FakeLLMClient(response_text="Contact your veterinarian if fever persists.")
    retriever = FakeRetriever(
        results=[
            {
                "chunk_id": "c-med",
                "document_id": "d-med",
                "title": "When to call the vet",
                "source": "kb",
                "similarity_score": 0.88,
                "snippet": "Fever can indicate infection; seek veterinary advice if concerned.",
                "metadata": {},
            }
        ]
    )
    orchestrator = RAGOrchestrator(
        llm_client=llm_client,
        retriever=retriever,
        ingestion_service=FakeIngestionService(),
    )
    payload = QueryRequest(
        question="My dog has a fever and seems lethargic; what should I watch for?",
        pet_profile=PetProfile(species="dog", life_stage="adult"),
    )
    await orchestrator.answer(payload)

    assert "stay conservative" in llm_client.system_prompts[0].lower()
    assert "not a source of truth" in llm_client.system_prompts[1].lower()
    assert "veterinary" in llm_client.system_prompts[1].lower()


@pytest.mark.asyncio
async def test_answer_dual_outputs_review_prompt_includes_urls_and_chunk_ids():
    llm_client = FakeLLMClient(
        responses=[
            "Keep meals regular and offer fresh water.",
            "## Synthesis\nGrounded in cited rows.\n\n## Evidence index\n- chunk-nut-1\n- https://kb.example/dog-nutrition",
        ]
    )
    retriever = FakeRetriever(
        results=[
            {
                "chunk_id": "chunk-nut-1",
                "document_id": "doc-nut",
                "title": "Adult dog nutrition",
                "source": "kb-internal",
                "category": "nutrition",
                "species": "dog",
                "life_stage": "adult",
                "similarity_score": 0.9,
                "snippet": "Consistent meal timing supports digestion in adult dogs.",
                "metadata": {"source_url": "https://kb.example/dog-nutrition"},
            }
        ]
    )
    ingestion_service = FakeIngestionService()
    orchestrator = RAGOrchestrator(
        llm_client=llm_client,
        retriever=retriever,
        ingestion_service=ingestion_service,
    )
    payload = QueryRequest(
        question="What feeding habits help an adult dog?",
        pet_profile=PetProfile(species="dog", life_stage="adult"),
        filters=QueryFilters(category="nutrition"),
    )
    result = await orchestrator.answer(payload)

    assert result["answer"].startswith("Keep meals regular")
    assert "chunk-nut-1" in result["review_draft"]
    catalog_prompt = llm_client.user_prompts[1]
    assert "chunk-nut-1" in catalog_prompt
    assert "https://kb.example/dog-nutrition" in catalog_prompt
    assert result["answer"] != result["review_draft"]


@pytest.mark.asyncio
async def test_ingest_returns_normalized_response():
    llm_client = FakeLLMClient()
    retriever = FakeRetriever()
    ingestion_service = FakeIngestionService(
        result={
            "documents_processed": 2,
            "chunks_created": 5,
            "document_ids": ["doc-1", "doc-2"],
            "message": "Ingested documents successfully.",
        }
    )

    orchestrator = RAGOrchestrator(
        llm_client=llm_client,
        retriever=retriever,
        ingestion_service=ingestion_service,
    )

    payload = IngestRequest(
        source="demo_batch",
        documents=[
            {
                "title": "Healthy feeding basics for adult dogs",
                "content": "Adult dogs benefit from routine feeding schedules and fresh water.",
                "category": "nutrition",
                "species": "dog",
                "life_stage": "adult",
                "tags": ["feeding"],
                "metadata": {"reviewed_by": "demo"},
            },
            {
                "title": "Hydration tips for senior dogs",
                "content": "Senior dogs may need closer hydration monitoring in warm weather.",
                "category": "hydration",
                "species": "dog",
                "life_stage": "senior",
                "tags": ["hydration"],
                "metadata": {"reviewed_by": "demo"},
            },
        ],
    )

    result = await orchestrator.ingest(payload)

    assert ingestion_service.called is True
    assert ingestion_service.last_payload.source == "demo_batch"
    assert len(ingestion_service.last_payload.documents) == 2

    assert result["status"] == "completed"
    assert result["source"] == "demo_batch"
    assert result["documents_received"] == 2
    assert result["documents_processed"] == 2
    assert result["chunks_created"] == 5
    assert result["document_ids"] == ["doc-1", "doc-2"]
    assert result["message"] == "Ingested documents successfully."
    assert "ingested_at" in result


@pytest.mark.asyncio
async def test_ingest_raises_when_no_documents_are_provided():
    llm_client = FakeLLMClient()
    retriever = FakeRetriever()
    ingestion_service = FakeIngestionService()

    orchestrator = RAGOrchestrator(
        llm_client=llm_client,
        retriever=retriever,
        ingestion_service=ingestion_service,
    )

    payload = IngestRequest.model_construct(source="demo_batch", documents=[])

    with pytest.raises(ValueError, match="At least one document is required for ingestion."):
        await orchestrator.ingest(payload)


class _FixedTrustedResearch:
    """Minimal stub: returns two gate-eligible snippets without running real retrieval."""

    async def run(self, request: TrustedResearchRequest) -> TrustedResearchOutcome:
        now = datetime.now(timezone.utc)
        src = ExternalSource(
            id="00000000-0000-4000-8000-000000000099",
            source_key="test_src",
            base_url=HttpUrl("https://avma.org/"),
            authority_score=0.8,
            source_type=ExternalSourceType.allowlisted_web,
            retrieved_at=now,
        )
        sn1 = ExtractedSnippet(
            id="sn-test-1",
            external_source_id=src.id,
            text=(
                "Provisional external excerpt recommends offering fresh water frequently while "
                "monitoring appetite and energy during recovery from mild digestive upset in adult dogs."
            ),
            authority_score=0.8,
            source_type=ExternalSourceType.allowlisted_web,
            retrieved_at=now,
        )
        sn2 = ExtractedSnippet(
            id="sn-test-2",
            external_source_id=src.id,
            text=(
                "The same provisional bundle suggests noting stool frequency, vomiting episodes, and "
                "willingness for light walking before returning to normal exercise intensity at home."
            ),
            authority_score=0.8,
            source_type=ExternalSourceType.allowlisted_web,
            retrieved_at=now,
        )
        ev = ResearchEvidence(snippets=[sn1, sn2], sources=[src])
        research = ResearchResult(
            evidence=ev,
            authority_score=0.8,
            source_type=ExternalSourceType.allowlisted_web,
            retrieved_at=now,
            evidence_summary="stub trusted retrieval",
            external_confidence="medium",
        )
        extraction = RetrievalEvidenceExtractor.from_research_evidence(
            ev,
            provider_id="test_provider",
            blocked_url_count=0,
        )
        _ = request
        return TrustedResearchOutcome(research=research, extraction=extraction)


class _CapturingExternalStore:
    def __init__(self) -> None:
        self.last_inp = None

    async def persist_external_research(self, inp):  # noqa: ANN001
        self.last_inp = inp
        return {
            "research_candidate_id": 99,
            "status": "provisional",
            "knowledge_source_ids": [1],
            "research_candidate_source_ids": [2],
            "document_chunks_ingested": 0,
            "provisional": True,
            "auto_promotion_blocked": False,
            "evidence_schema_version": 1,
        }


@pytest.mark.asyncio
async def test_sensitive_query_skips_external_research_even_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "enable_trusted_external_retrieval", True)
    monkeypatch.setattr(settings, "allow_provisional_in_query", True)
    monkeypatch.setattr(settings, "external_research_trigger_threshold", 0.75)

    llm_client = FakeLLMClient()
    retriever = FakeRetriever(results=[])
    ingestion_service = FakeIngestionService()
    orchestrator = RAGOrchestrator(
        llm_client=llm_client,
        retriever=retriever,
        ingestion_service=ingestion_service,
        trusted_research=_FixedTrustedResearch(),
    )

    payload = QueryRequest(
        question="My dog is having a seizure emergency what do I do?",
        pet_profile=PetProfile(species="dog", life_stage="adult"),
    )
    result = await orchestrator.answer(payload)

    assert llm_client.called is False
    assert result["confidence"] == "low"
    assert result.get("review_draft") is None
    assert EXTERNAL_PROVISIONAL_CONTEXT_DISCLAIMER not in result.get("disclaimers", [])


@pytest.mark.asyncio
async def test_external_fallback_generates_answer_and_persists_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "enable_trusted_external_retrieval", True)
    monkeypatch.setattr(settings, "allow_provisional_in_query", True)
    monkeypatch.setattr(settings, "external_research_trigger_threshold", 0.75)
    monkeypatch.setattr(settings, "enable_auto_save_provisional_knowledge", True)

    llm_client = FakeLLMClient(
        responses=[
            "Based on the context, encourage hydration and monitor appetite.",
            "- Limitations: single provisional excerpt\n- Human review: yes",
        ]
    )
    retriever = FakeRetriever(results=[])
    ingestion_service = FakeIngestionService()
    store = _CapturingExternalStore()
    orchestrator = RAGOrchestrator(
        llm_client=llm_client,
        retriever=retriever,
        ingestion_service=ingestion_service,
        trusted_research=_FixedTrustedResearch(),
        external_research_store=store,
    )

    payload = QueryRequest(
        question="What is the best routine for my adult dog?",
        pet_profile=PetProfile(species="dog", life_stage="adult"),
        filters=QueryFilters(category="nutrition"),
        top_k=4,
    )
    result = await orchestrator.answer(payload)

    assert llm_client.call_count == 2
    assert EXTERNAL_PROVISIONAL_CONTEXT_DISCLAIMER in result["disclaimers"]
    assert "hydration" in result["answer"].lower()
    assert result["answer_source"] == "external_trusted"
    assert result["knowledge_status"] == "provisional"
    assert result.get("review_draft")
    assert "limitations" in result["review_draft"].lower()
    assert "internal petmind analyst" in llm_client.system_prompts[1].lower()
    assert "sn-test-1" in llm_client.user_prompts[1]
    assert "https://avma.org/" in llm_client.user_prompts[1]
    assert store.last_inp is not None
    assert store.last_inp.frontend_answer_text == result["answer"]
    assert store.last_inp.synthesis_text is None
    assert store.last_inp.internal_review_llm_draft == result["review_draft"]


@pytest.mark.parametrize(
    "auto_save, trusted, provisional",
    [
        (False, True, True),
        (True, False, True),
        (True, True, False),
    ],
)
def test_get_orchestrator_never_builds_sqlalchemy_store_unless_all_flags_on(
    monkeypatch: pytest.MonkeyPatch,
    auto_save: bool,
    trusted: bool,
    provisional: bool,
) -> None:
    """Si falta cualquiera de los tres flags, no se debe instanciar el store SQL (sin efectos colaterales)."""
    get_orchestrator.cache_clear()
    try:
        monkeypatch.setattr(settings, "enable_auto_save_provisional_knowledge", auto_save)
        monkeypatch.setattr(settings, "enable_trusted_external_retrieval", trusted)
        monkeypatch.setattr(settings, "allow_provisional_in_query", provisional)

        def _must_not_construct(*args, **kwargs):  # noqa: ARG001
            raise AssertionError("SqlAlchemyResearchCandidateStore must not be constructed")

        monkeypatch.setattr(orchestrator_module, "SqlAlchemyResearchCandidateStore", _must_not_construct)
        orch = get_orchestrator()
        assert isinstance(orch._external_store, orchestrator_module.NullExternalResearchPersistence)
    finally:
        get_orchestrator.cache_clear()