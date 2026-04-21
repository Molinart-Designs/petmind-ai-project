from functools import lru_cache

from src.core.llm_client import OpenAILLMClient, get_llm_client
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self, llm_client: OpenAILLMClient) -> None:
        self._llm_client = llm_client

    async def embed_text(self, text: str) -> list[float]:
        cleaned_text = self._clean_text(text)

        embeddings = await self._llm_client.embed_texts([cleaned_text])
        return embeddings[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        cleaned_texts = [self._clean_text(text) for text in texts if text and text.strip()]

        if not cleaned_texts:
            raise ValueError("texts must contain at least one non-empty item.")

        logger.info(
            "Embedding batch of texts",
            extra={"input_count": len(cleaned_texts)},
        )

        return await self._llm_client.embed_texts(cleaned_texts)

    def _clean_text(self, text: str) -> str:
        if not text or not text.strip():
            raise ValueError("text must not be empty.")
        return " ".join(text.strip().split())


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(llm_client=get_llm_client())