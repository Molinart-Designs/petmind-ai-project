from functools import lru_cache
from typing import Any

from src.core.config import settings
from src.rag.embeddings import EmbeddingService, get_embedding_service
from src.rag.vector_store import PgVectorStore, get_vector_store
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        vector_store: PgVectorStore,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    async def retrieve(
        self,
        *,
        question: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        cleaned_question = self._clean_question(question)
        effective_top_k = top_k or settings.retriever_top_k
        normalized_filters = self._normalize_filters(filters or {})

        logger.info(
            "Starting retrieval flow",
            extra={
                "question_preview": cleaned_question[:120],
                "top_k": effective_top_k,
                "filters": normalized_filters,
            },
        )

        query_embedding = await self._embedding_service.embed_text(cleaned_question)

        results = await self._vector_store.search_similar(
            query_embedding=query_embedding,
            top_k=effective_top_k,
            filters=normalized_filters,
            similarity_threshold=settings.similarity_threshold,
        )

        logger.info(
            "Completed retrieval flow",
            extra={
                "results_count": len(results),
                "top_k": effective_top_k,
                "filters": normalized_filters,
            },
        )

        return results

    def _clean_question(self, question: str) -> str:
        if not question or not question.strip():
            raise ValueError("question must not be empty.")
        return " ".join(question.strip().split())

    def _normalize_filters(self, filters: dict[str, Any]) -> dict[str, Any]:
        allowed_keys = {"category", "species", "life_stage"}
        normalized: dict[str, Any] = {}

        for key, value in filters.items():
            if key not in allowed_keys:
                continue
            if value is None:
                continue

            value_str = str(value).strip()
            if value_str:
                normalized[key] = value_str

        return normalized


@lru_cache
def get_retriever() -> Retriever:
    return Retriever(
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
    )