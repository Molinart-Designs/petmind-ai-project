import pytest

from src.api.schemas import IngestRequest, QueryRequest
from src.core.orchestrator import RAGOrchestrator


class FakeLLMClient:
    def __init__(self, response_text: str = "Grounded answer from the LLM.") -> None:
        self.response_text = response_text
        self.called = False
        self.last_system_prompt = None
        self.last_user_prompt = None
        self.last_max_output_tokens = None

    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int = 600,
    ) -> str:
        self.called = True
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        self.last_max_output_tokens = max_output_tokens
        return self.response_text


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
async def test_answer_returns_safe_fallback_when_grounding_is_insufficient():
    llm_client = FakeLLMClient()
    retriever = FakeRetriever(
        results=[
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "title": "Weakly related content",
                "source": "demo-source",
                "category": "nutrition",
                "species": "dog",
                "life_stage": "adult",
                "similarity_score": 0.40,
                "snippet": "General pet wellness guidance.",
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
    assert result["needs_vet_followup"] is False
    assert result["retrieval_count"] == 1
    assert result["used_filters"] == {
        "category": "nutrition",
        "species": "dog",
        "life_stage": "adult",
    }
    assert "grounded information" in result["answer"].lower()
    assert "generated_at" in result


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
    assert "provided context" in llm_client.last_system_prompt.lower()
    assert "user question" in llm_client.last_user_prompt.lower()
    assert "pet profile" in llm_client.last_user_prompt.lower()
    assert "knowledge context" in llm_client.last_user_prompt.lower()

    assert result["answer"].startswith("Adult dogs usually benefit")
    assert result["confidence"] == "high"
    assert result["needs_vet_followup"] is False
    assert result["retrieval_count"] == 1
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