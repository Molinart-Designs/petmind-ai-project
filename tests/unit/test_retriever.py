import pytest

from src.rag.retriever import Retriever


class FakeEmbeddingService:
    def __init__(self, embedding=None) -> None:
        self.embedding = embedding or [0.1, 0.2, 0.3]
        self.called = False
        self.last_text = None

    async def embed_text(self, text: str) -> list[float]:
        self.called = True
        self.last_text = text
        return self.embedding


class FakeVectorStore:
    def __init__(self, results=None) -> None:
        self.results = results or []
        self.called = False
        self.last_query_embedding = None
        self.last_top_k = None
        self.last_filters = None
        self.last_similarity_threshold = None

    async def search_similar(
        self,
        *,
        query_embedding: list[float],
        top_k: int = 4,
        filters: dict | None = None,
        similarity_threshold: float | None = None,
    ):
        self.called = True
        self.last_query_embedding = query_embedding
        self.last_top_k = top_k
        self.last_filters = filters or {}
        self.last_similarity_threshold = similarity_threshold
        return self.results


@pytest.mark.asyncio
async def test_retrieve_returns_results_and_calls_dependencies():
    embedding_service = FakeEmbeddingService(embedding=[0.9, 0.8, 0.7])
    vector_store = FakeVectorStore(
        results=[
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "title": "Adult dog feeding basics",
                "source": "demo-source",
                "category": "nutrition",
                "species": "dog",
                "life_stage": "adult",
                "similarity_score": 0.91,
                "snippet": "Adult dogs generally benefit from routine feeding schedules.",
                "metadata": {"chunk_index": 1},
            }
        ]
    )

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    results = await retriever.retrieve(
        question="  What is a good feeding routine for my adult dog?  ",
        top_k=3,
        filters={
            "category": "nutrition",
            "species": "dog",
            "life_stage": "adult",
        },
    )

    assert embedding_service.called is True
    assert embedding_service.last_text == "What is a good feeding routine for my adult dog?"

    assert vector_store.called is True
    assert vector_store.last_query_embedding == [0.9, 0.8, 0.7]
    assert vector_store.last_top_k == 3
    assert vector_store.last_filters == {
        "category": "nutrition",
        "species": "dog",
        "life_stage": "adult",
    }
    assert vector_store.last_similarity_threshold == 0.75

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk-1"
    assert results[0]["similarity_score"] == 0.91


@pytest.mark.asyncio
async def test_retrieve_normalizes_filters_and_ignores_unknown_keys():
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore(results=[])

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    await retriever.retrieve(
        question="How should I feed my senior dog?",
        filters={
            "category": "nutrition",
            "species": "dog",
            "life_stage": "senior",
            "unknown_field": "should_be_ignored",
            "empty_field": "",
            "none_field": None,
        },
    )

    assert vector_store.called is True
    assert vector_store.last_filters == {
        "category": "nutrition",
        "species": "dog",
        "life_stage": "senior",
    }


@pytest.mark.asyncio
async def test_retrieve_uses_default_top_k_when_not_provided():
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore(results=[])

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    await retriever.retrieve(
        question="What should I know about hydration for adult dogs?",
        filters={"species": "dog"},
    )

    assert vector_store.called is True
    assert vector_store.last_top_k == 4


@pytest.mark.asyncio
async def test_retrieve_raises_value_error_when_question_is_empty():
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    with pytest.raises(ValueError, match="question must not be empty."):
        await retriever.retrieve(question="   ")


@pytest.mark.asyncio
async def test_retrieve_returns_empty_list_when_vector_store_finds_nothing():
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore(results=[])

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    results = await retriever.retrieve(
        question="What is a healthy daily routine for an adult dog?",
        filters={"species": "dog"},
    )

    assert embedding_service.called is True
    assert vector_store.called is True
    assert results == []