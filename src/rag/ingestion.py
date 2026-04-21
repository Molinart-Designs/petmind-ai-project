import hashlib
from functools import lru_cache
from typing import Any

from src.api.schemas import IngestDocument, IngestRequest
from src.core.config import settings
from src.rag.embeddings import EmbeddingService, get_embedding_service
from src.rag.vector_store import PgVectorStore, get_vector_store
from src.utils.logger import get_logger

logger = get_logger(__name__)


class IngestionService:
    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        vector_store: PgVectorStore,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    async def ingest_documents(self, payload: IngestRequest) -> dict[str, Any]:
        if not payload.documents:
            raise ValueError("At least one document is required for ingestion.")

        all_chunks: list[dict[str, Any]] = []
        processed_document_ids: list[str] = []

        logger.info(
            "Starting document ingestion",
            extra={
                "source": payload.source,
                "documents_received": len(payload.documents),
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
            },
        )

        for document in payload.documents:
            document_id = self._build_document_id(payload.source, document)
            processed_document_ids.append(document_id)

            chunks = self._chunk_document(
                source=payload.source,
                document_id=document_id,
                document=document,
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("No chunks were generated from the provided documents.")

        embeddings = await self._embedding_service.embed_texts(
            [chunk["content"] for chunk in all_chunks]
        )

        for chunk, embedding in zip(all_chunks, embeddings, strict=True):
            chunk["embedding"] = embedding

        store_result = await self._vector_store.add_chunks(all_chunks)

        result = {
            "documents_processed": len(processed_document_ids),
            "chunks_created": store_result["chunks_created"],
            "document_ids": store_result["document_ids"],
            "message": (
                f"Ingested {len(processed_document_ids)} document(s) and "
                f"created {store_result['chunks_created']} chunk(s)."
            ),
        }

        logger.info(
            "Completed document ingestion",
            extra={
                "source": payload.source,
                "documents_processed": result["documents_processed"],
                "chunks_created": result["chunks_created"],
            },
        )

        return result

    def _chunk_document(
        self,
        *,
        source: str,
        document_id: str,
        document: IngestDocument,
    ) -> list[dict[str, Any]]:
        normalized_content = self._normalize_text(document.content)
        chunk_texts = self._chunk_text(
            normalized_content,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        chunks: list[dict[str, Any]] = []
        for index, chunk_text in enumerate(chunk_texts, start=1):
            chunk_id = f"{document_id}-chunk-{index}"

            metadata = {
                "external_id": document.external_id,
                "source_batch": source,
                "source_url": document.source_url,
                "tags": document.tags,
                "chunk_index": index,
                "title": document.title,
                **document.metadata,
            }

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "title": document.title,
                    "content": chunk_text,
                    "source": document.source_url or source,
                    "category": document.category,
                    "species": document.species,
                    "life_stage": document.life_stage,
                    "metadata": metadata,
                }
            )

        logger.info(
            "Chunked document for ingestion",
            extra={
                "document_id": document_id,
                "title": document.title,
                "chunks_count": len(chunks),
            },
        )

        return chunks

    def _chunk_text(
        self,
        text: str,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[str]:
        if not text:
            return []

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        chunks: list[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)

            if end < text_length:
                last_space = text.rfind(" ", start, end)
                if last_space > start + int(chunk_size * 0.6):
                    end = last_space

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            start = max(end - chunk_overlap, start + 1)

        return chunks

    def _normalize_text(self, value: str) -> str:
        return " ".join(value.strip().split())

    def _build_document_id(self, source: str, document: IngestDocument) -> str:
        base = "|".join(
            [
                source.strip(),
                (document.external_id or "").strip(),
                document.title.strip(),
                document.category.strip(),
                document.species.strip(),
                (document.life_stage or "").strip(),
            ]
        )
        digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:24]
        return f"doc-{digest}"


@lru_cache
def get_ingestion_service() -> IngestionService:
    return IngestionService(
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
    )