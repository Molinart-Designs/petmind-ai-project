from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from src.api.schemas import IngestRequest, QueryRequest
from src.core.config import settings
from src.core.llm_client import OpenAILLMClient, get_llm_client
from src.rag.ingestion import IngestionService, get_ingestion_service
from src.rag.retriever import Retriever, get_retriever
from src.security.guardrails import (
    assess_query_risk,
    assess_retrieval_grounding,
    build_safe_fallback_answer,
    postprocess_answer,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RAGOrchestrator:
    def __init__(
        self,
        *,
        llm_client: OpenAILLMClient,
        retriever: Retriever,
        ingestion_service: IngestionService,
    ) -> None:
        self._llm_client = llm_client
        self._retriever = retriever
        self._ingestion_service = ingestion_service

    async def answer(self, payload: QueryRequest) -> dict[str, Any]:
        question = payload.question.strip()
        if not question:
            raise ValueError("Question must not be empty.")

        risk = assess_query_risk(question)
        retrieval_filters = self._build_retrieval_filters(payload)
        top_k = payload.top_k or settings.retriever_top_k

        logger.info(
            "Starting RAG answer flow",
            extra={
                "top_k": top_k,
                "filters": retrieval_filters,
                "is_sensitive": risk.is_sensitive,
                "is_medical": risk.is_medical,
            },
        )

        retrieved_chunks = await self._retriever.retrieve(
            question=question,
            top_k=top_k,
            filters=retrieval_filters,
        )

        grounding = assess_retrieval_grounding(
            retrieved_chunks=retrieved_chunks,
            similarity_threshold=settings.similarity_threshold,
        )

        if not grounding["has_sufficient_context"]:
            logger.warning(
                "Insufficient retrieval grounding, returning fallback answer",
                extra={
                    "matched_count": grounding["matched_count"],
                    "top_score": grounding["top_score"],
                },
            )

            fallback = build_safe_fallback_answer(
                question=question,
                retrieved_chunks=retrieved_chunks,
            )
            fallback["used_filters"] = retrieval_filters
            fallback["generated_at"] = datetime.now(timezone.utc)
            return fallback

        system_prompt = self._build_system_prompt(risk.is_sensitive, risk.is_medical)
        user_prompt = self._build_user_prompt(
            question=question,
            pet_profile=payload.pet_profile.model_dump(exclude_none=True)
            if payload.pet_profile
            else {},
            retrieved_chunks=retrieved_chunks,
        )

        llm_answer = await self._llm_client.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=600,
        )

        result = postprocess_answer(
            answer=llm_answer,
            question=question,
            retrieved_chunks=retrieved_chunks,
            similarity_threshold=settings.similarity_threshold,
        )
        result["used_filters"] = retrieval_filters
        result["generated_at"] = datetime.now(timezone.utc)

        logger.info(
            "Completed RAG answer flow",
            extra={
                "retrieval_count": result["retrieval_count"],
                "confidence": result["confidence"],
                "needs_vet_followup": result["needs_vet_followup"],
            },
        )

        return result

    async def ingest(self, payload: IngestRequest) -> dict[str, Any]:
        if not payload.documents:
            raise ValueError("At least one document is required for ingestion.")

        logger.info(
            "Starting ingestion flow",
            extra={
                "source": payload.source,
                "documents_received": len(payload.documents),
            },
        )

        result = await self._ingestion_service.ingest_documents(payload)

        response = {
            "status": "completed",
            "source": payload.source,
            "documents_received": len(payload.documents),
            "documents_processed": result["documents_processed"],
            "chunks_created": result["chunks_created"],
            "document_ids": result["document_ids"],
            "message": result.get(
                "message",
                "Documents ingested successfully into the knowledge base.",
            ),
            "ingested_at": datetime.now(timezone.utc),
        }

        logger.info(
            "Completed ingestion flow",
            extra={
                "source": payload.source,
                "documents_processed": response["documents_processed"],
                "chunks_created": response["chunks_created"],
            },
        )

        return response

    def _build_retrieval_filters(self, payload: QueryRequest) -> dict[str, Any]:
        filters: dict[str, Any] = {}

        if payload.filters:
            filters.update(payload.filters.model_dump(exclude_none=True))

        if payload.pet_profile:
            pet_profile = payload.pet_profile.model_dump(exclude_none=True)

            if "species" not in filters and pet_profile.get("species"):
                filters["species"] = pet_profile["species"]

            if "life_stage" not in filters and pet_profile.get("life_stage"):
                filters["life_stage"] = pet_profile["life_stage"]

        return filters

    def _build_system_prompt(self, is_sensitive: bool, is_medical: bool) -> str:
        safety_block = (
            "You are PetMind AI, a pet care guidance assistant. "
            "You must answer only from the provided context. "
            "Do not invent facts. "
            "Do not provide definitive veterinary diagnoses. "
            "Do not prescribe medication dosages unless that exact information is explicitly present in the provided context, "
            "and even then, advise professional veterinary confirmation. "
            "If the context is insufficient, say so clearly. "
            "Prefer educational, cautious, practical guidance."
        )

        medical_block = (
            "If the question involves symptoms, treatment, diagnosis, medication, or health risk, "
            "make it clear that professional veterinary evaluation may be needed."
            if is_medical
            else "If relevant, encourage safe monitoring and good pet care practices."
        )

        sensitive_block = (
            "This appears potentially urgent or sensitive. "
            "Advise prompt veterinary attention if symptoms are severe, sudden, or worsening."
            if is_sensitive
            else "Keep the tone calm, clear, and grounded."
        )

        style_block = (
            "Respond in concise natural language. "
            "Use short paragraphs or bullets only when helpful. "
            "Base the answer strictly on the provided knowledge context and pet profile."
        )

        return "\n".join([safety_block, medical_block, sensitive_block, style_block])

    def _build_user_prompt(
        self,
        *,
        question: str,
        pet_profile: dict[str, Any],
        retrieved_chunks: list[dict[str, Any]],
    ) -> str:
        pet_profile_block = self._format_pet_profile(pet_profile)
        context_block = self._format_context(retrieved_chunks)

        return (
            f"User question:\n{question}\n\n"
            f"Pet profile:\n{pet_profile_block}\n\n"
            f"Knowledge context:\n{context_block}\n\n"
            "Instructions:\n"
            "- Answer using only the knowledge context above.\n"
            "- If the context does not fully support a claim, say that clearly.\n"
            "- Do not present speculative statements as facts.\n"
            "- Be helpful, practical, and safety-conscious.\n"
        )

    def _format_pet_profile(self, pet_profile: dict[str, Any]) -> str:
        if not pet_profile:
            return "No pet profile provided."

        lines = []
        for key, value in pet_profile.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def _format_context(self, retrieved_chunks: list[dict[str, Any]]) -> str:
        if not retrieved_chunks:
            return "No context retrieved."

        formatted_chunks: list[str] = []
        for index, chunk in enumerate(retrieved_chunks, start=1):
            title = chunk.get("title") or "Untitled"
            source = chunk.get("source") or "unknown"
            snippet = (
                chunk.get("snippet")
                or chunk.get("content")
                or chunk.get("text")
                or ""
            ).strip()
            similarity_score = chunk.get("similarity_score")

            formatted_chunks.append(
                "\n".join(
                    [
                        f"[Chunk {index}]",
                        f"Title: {title}",
                        f"Source: {source}",
                        f"Similarity: {similarity_score}",
                        f"Content: {snippet}",
                    ]
                )
            )

        return "\n\n".join(formatted_chunks)


@lru_cache
def get_orchestrator() -> RAGOrchestrator:
    return RAGOrchestrator(
        llm_client=get_llm_client(),
        retriever=get_retriever(),
        ingestion_service=get_ingestion_service(),
    )