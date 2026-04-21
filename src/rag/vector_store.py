import json
from functools import lru_cache
from typing import Any
from uuid import uuid4

from sqlalchemy import text

from src.core.config import settings
from src.db.session import get_db_session
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStoreError(Exception):
    """Raised when a vector store operation fails."""


class PgVectorStore:
    """
    Minimal pgvector-backed store using raw SQL.

    Expected table (to be created later via models/migrations):
      - document_chunks
        * chunk_id TEXT PRIMARY KEY
        * document_id TEXT NOT NULL
        * title TEXT NULL
        * content TEXT NOT NULL
        * source TEXT NULL
        * category TEXT NULL
        * species TEXT NULL
        * life_stage TEXT NULL
        * metadata JSONB NOT NULL DEFAULT '{}'::jsonb
        * embedding VECTOR(<dim>) NOT NULL
        * created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    """

    async def add_chunks(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        if not chunks:
            raise ValueError("chunks must contain at least one item.")

        inserted_chunk_ids: list[str] = []
        document_ids: set[str] = set()

        try:
            with get_db_session() as session:
                for chunk in chunks:
                    embedding = chunk.get("embedding")
                    content = (chunk.get("content") or "").strip()

                    if not embedding:
                        raise ValueError("Each chunk must include an embedding.")
                    if not content:
                        raise ValueError("Each chunk must include non-empty content.")

                    chunk_id = chunk.get("chunk_id") or str(uuid4())
                    document_id = chunk.get("document_id") or str(uuid4())
                    title = chunk.get("title")
                    source = chunk.get("source")
                    category = chunk.get("category")
                    species = chunk.get("species")
                    life_stage = chunk.get("life_stage")
                    metadata = chunk.get("metadata") or {}

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
                            ON CONFLICT (chunk_id) DO UPDATE SET
                                document_id = EXCLUDED.document_id,
                                title = EXCLUDED.title,
                                content = EXCLUDED.content,
                                source = EXCLUDED.source,
                                category = EXCLUDED.category,
                                species = EXCLUDED.species,
                                life_stage = EXCLUDED.life_stage,
                                metadata = EXCLUDED.metadata,
                                embedding = EXCLUDED.embedding
                            """
                        ),
                        {
                            "chunk_id": chunk_id,
                            "document_id": document_id,
                            "title": title,
                            "content": content,
                            "source": source,
                            "category": category,
                            "species": species,
                            "life_stage": life_stage,
                            "metadata": json.dumps(metadata, ensure_ascii=False),
                            "embedding": self._to_vector_literal(embedding),
                        },
                    )

                    inserted_chunk_ids.append(chunk_id)
                    document_ids.add(document_id)

            logger.info(
                "Stored chunks in pgvector",
                extra={
                    "chunks_count": len(inserted_chunk_ids),
                    "documents_count": len(document_ids),
                },
            )

            return {
                "document_ids": sorted(document_ids),
                "chunk_ids": inserted_chunk_ids,
                "chunks_created": len(inserted_chunk_ids),
            }

        except ValueError:
            raise
        except Exception as exc:
            logger.exception("Failed to store chunks in pgvector", extra={"error": str(exc)})
            raise VectorStoreError("Failed to store chunks in pgvector.") from exc

    async def search_similar(
        self,
        *,
        query_embedding: list[float],
        top_k: int = 4,
        filters: dict[str, Any] | None = None,
        similarity_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        if not query_embedding:
            raise ValueError("query_embedding must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be greater than 0.")

        filters = self._normalize_filters(filters or {})
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.similarity_threshold
        )

        where_clauses = ["1=1"]
        params: dict[str, Any] = {
            "query_embedding": self._to_vector_literal(query_embedding),
            "top_k": top_k,
            "threshold": threshold,
        }

        if filters.get("category"):
            where_clauses.append("category = :category")
            params["category"] = filters["category"]

        if filters.get("species"):
            where_clauses.append("species = :species")
            params["species"] = filters["species"]

        if filters.get("life_stage"):
            where_clauses.append("life_stage = :life_stage")
            params["life_stage"] = filters["life_stage"]

        where_clauses.append(
            "(1 - (embedding <=> CAST(:query_embedding AS vector))) >= :threshold"
        )

        sql = f"""
            SELECT
                chunk_id,
                document_id,
                title,
                content,
                source,
                category,
                species,
                life_stage,
                metadata,
                (1 - (embedding <=> CAST(:query_embedding AS vector))) AS similarity_score
            FROM document_chunks
            WHERE {" AND ".join(where_clauses)}
            ORDER BY embedding <=> CAST(:query_embedding AS vector)
            LIMIT :top_k
        """

        try:
            with get_db_session() as session:
                rows = session.execute(text(sql), params).mappings().all()

            results: list[dict[str, Any]] = []
            for row in rows:
                metadata = row.get("metadata") or {}
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}

                content = (row.get("content") or "").strip()

                results.append(
                    {
                        "chunk_id": row["chunk_id"],
                        "document_id": row["document_id"],
                        "title": row.get("title"),
                        "source": row.get("source"),
                        "category": row.get("category"),
                        "species": row.get("species"),
                        "life_stage": row.get("life_stage"),
                        "similarity_score": float(row["similarity_score"]),
                        "snippet": self._build_snippet(content),
                        "content": content,
                        "metadata": metadata,
                    }
                )

            logger.info(
                "Retrieved similar chunks from pgvector",
                extra={
                    "returned_count": len(results),
                    "top_k": top_k,
                    "filters": filters,
                    "threshold": threshold,
                },
            )

            return results

        except ValueError:
            raise
        except Exception as exc:
            logger.exception(
                "Failed to search similar chunks in pgvector",
                extra={"error": str(exc)},
            )
            raise VectorStoreError("Failed to search similar chunks in pgvector.") from exc

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

    def _to_vector_literal(self, embedding: list[float]) -> str:
        if not embedding:
            raise ValueError("embedding must not be empty.")

        try:
            values = ",".join(str(float(value)) for value in embedding)
        except (TypeError, ValueError) as exc:
            raise ValueError("embedding must contain only numeric values.") from exc

        return f"[{values}]"

    def _build_snippet(self, content: str, max_length: int = 280) -> str:
        normalized = " ".join(content.split())
        if len(normalized) <= max_length:
            return normalized
        return normalized[: max_length - 3].rstrip() + "..."


@lru_cache
def get_vector_store() -> PgVectorStore:
    if settings.vector_store_provider != "pgvector":
        raise ValueError(
            f"Unsupported vector store provider: {settings.vector_store_provider}"
        )
    return PgVectorStore()