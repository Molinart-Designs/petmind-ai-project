import pytest
from sqlalchemy import text

from src.api.main import app
from src.core.config import settings
from src.core.orchestrator import RAGOrchestrator, get_orchestrator
from src.db.session import get_db_session
from src.rag.ingestion import get_ingestion_service
from src.rag.retriever import Retriever
from src.rag.vector_store import get_vector_store


TEST_EMBEDDING = [0.1, 0.2, 0.3]


class FakeLLMClient:
    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int = 600,
    ) -> str:
        return (
            "Adult dogs generally benefit from a consistent daily routine, "
            "including feeding, exercise, walks, hydration, and rest."
        )


class FakeEmbeddingService:
    async def embed_text(self, text: str) -> list[float]:
        return TEST_EMBEDDING

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [TEST_EMBEDDING for _ in texts]


@pytest.fixture
def seeded_rag_chunk():
    chunk_id = "test-rag-chunk-1"
    document_id = "test-rag-doc-1"

    with get_db_session() as session:
        session.execute(
            text(
                """
                DELETE FROM document_chunks
                WHERE chunk_id = :chunk_id
                   OR document_id = :document_id
                """
            ),
            {"chunk_id": chunk_id, "document_id": document_id},
        )

        session.execute(
            text(
                """
                INSERT INTO document_chunks (
                    chunk_id,
                    document_id,
                    title,
                    content,
                    source,
                    category,
                    species,
                    life_stage,
                    metadata,
                    embedding,
                    created_at
                )
                VALUES (
                    :chunk_id,
                    :document_id,
                    :title,
                    :content,
                    :source,
                    :category,
                    :species,
                    :life_stage,
                    CAST(:metadata AS jsonb),
                    CAST(:embedding AS vector),
                    NOW()
                )
                """
            ),
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "title": "Adult dog daily care basics",
                "content": (
                    "Adult dogs generally benefit from a consistent daily routine "
                    "including feeding, regular walks, exercise, hydration, and rest."
                ),
                "source": "integration-test-source",
                "category": "daily-care",
                "species": "dog",
                "life_stage": "adult",
                "metadata": "{}",
                "embedding": "[0.1,0.2,0.3]",
            },
        )

    yield

    with get_db_session() as session:
        session.execute(
            text(
                """
                DELETE FROM document_chunks
                WHERE chunk_id = :chunk_id
                   OR document_id = :document_id
                """
            ),
            {"chunk_id": chunk_id, "document_id": document_id},
        )


@pytest.fixture
def rag_test_client(client, monkeypatch):
    # Disable trusted external fallback for this test:
    # we want a deterministic internal-RAG e2e only.
    monkeypatch.setattr(settings, "enable_trusted_external_retrieval", False)
    monkeypatch.setattr(settings, "allow_provisional_in_query", False)
    monkeypatch.setattr(settings, "enable_auto_save_provisional_knowledge", False)

    fake_llm = FakeLLMClient()
    fake_embeddings = FakeEmbeddingService()

    real_vector_store = get_vector_store()
    real_retriever = Retriever(
        embedding_service=fake_embeddings,
        vector_store=real_vector_store,
    )

    orchestrator = RAGOrchestrator(
        llm_client=fake_llm,
        retriever=real_retriever,
        ingestion_service=get_ingestion_service(),
    )

    app.dependency_overrides[get_orchestrator] = lambda: orchestrator

    yield client

    app.dependency_overrides.pop(get_orchestrator, None)


def test_rag_e2e_query_returns_grounded_response_from_real_pipeline(
    rag_test_client,
    auth_headers,
    seeded_rag_chunk,
):
    payload = {
        "question": "What is a good daily routine for my adult dog?",
        "pet_profile": {
            "species": "dog",
            "life_stage": "adult",
        },
        "filters": {
            "category": "daily-care",
            "species": "dog",
            "life_stage": "adult",
        },
        "top_k": 4,
    }

    response = rag_test_client.post(
        "/api/v1/query",
        json=payload,
        headers=auth_headers,
    )

    assert response.status_code == 200

    body = response.json()
    assert body["answer"]
    assert body["retrieval_count"] >= 1
    assert len(body["sources"]) >= 1
    assert body["sources"][0]["document_id"] == "test-rag-doc-1"
    assert body["sources"][0]["chunk_id"] == "test-rag-chunk-1"
    assert body["sources"][0]["category"] == "daily-care"
    assert body["sources"][0]["species"] == "dog"
    assert body["used_filters"]["category"] == "daily-care"
    assert body["used_filters"]["species"] == "dog"
    assert body["used_filters"]["life_stage"] == "adult"
    assert "generated_at" in body