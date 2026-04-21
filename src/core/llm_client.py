from functools import lru_cache

from openai import AsyncOpenAI

from src.core.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMClientError(Exception):
    """Raised when the LLM client cannot complete a request."""


class OpenAILLMClient:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.llm_model
        self._embedding_model = settings.embedding_model

    @property
    def model(self) -> str:
        return self._model

    @property
    def embedding_model(self) -> str:
        return self._embedding_model

    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int = 600,
    ) -> str:
        """
        Generate a grounded text response using the OpenAI Responses API.
        """
        system_prompt = system_prompt.strip()
        user_prompt = user_prompt.strip()

        if not system_prompt:
            raise ValueError("system_prompt must not be empty.")
        if not user_prompt:
            raise ValueError("user_prompt must not be empty.")

        try:
            logger.info(
                "Generating LLM response",
                extra={
                    "provider": settings.llm_provider,
                    "model": self._model,
                    "max_output_tokens": max_output_tokens,
                },
            )

            response = await self._client.responses.create(
                model=self._model,
                instructions=system_prompt,
                input=user_prompt,
                max_output_tokens=max_output_tokens,
            )

            output_text = (response.output_text or "").strip()

            if not output_text:
                raise LLMClientError("OpenAI response did not contain output text.")

            return output_text

        except ValueError:
            raise
        except Exception as exc:
            logger.exception(
                "Failed to generate LLM response",
                extra={"provider": settings.llm_provider, "model": self._model},
            )
            raise LLMClientError("Failed to generate text from the LLM provider.") from exc

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.
        """
        cleaned_texts = [text.strip() for text in texts if text and text.strip()]

        if not cleaned_texts:
            raise ValueError("texts must contain at least one non-empty item.")

        try:
            logger.info(
                "Generating embeddings",
                extra={
                    "provider": settings.llm_provider,
                    "embedding_model": self._embedding_model,
                    "input_count": len(cleaned_texts),
                },
            )

            response = await self._client.embeddings.create(
                model=self._embedding_model,
                input=cleaned_texts,
            )

            embeddings = [item.embedding for item in response.data]

            if len(embeddings) != len(cleaned_texts):
                raise LLMClientError(
                    "Embedding response size does not match the number of inputs."
                )

            return embeddings

        except ValueError:
            raise
        except Exception as exc:
            logger.exception(
                "Failed to generate embeddings",
                extra={
                    "provider": settings.llm_provider,
                    "embedding_model": self._embedding_model,
                },
            )
            raise LLMClientError("Failed to generate embeddings.") from exc

    async def close(self) -> None:
        await self._client.close()


@lru_cache
def get_llm_client() -> OpenAILLMClient:
    if settings.llm_provider != "openai":
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
    return OpenAILLMClient()
