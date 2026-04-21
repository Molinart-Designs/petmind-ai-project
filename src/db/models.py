from datetime import datetime

from sqlalchemy import DateTime, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import UserDefinedType

from src.db.session import Base


class VectorType(UserDefinedType):
    """
    Minimal PostgreSQL pgvector column type.
    We keep it simple here because vector operations are handled in raw SQL
    inside src/rag/vector_store.py.
    """

    cache_ok = True

    def get_col_spec(self, **kwargs) -> str:
        return "vector"


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    chunk_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    title: Mapped[str | None] = mapped_column(String(300), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    source: Mapped[str | None] = mapped_column(String(500), nullable=True)
    category: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    species: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    life_stage: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )

    embedding = mapped_column(VectorType(), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        Index(
            "ix_document_chunks_filters",
            "category",
            "species",
            "life_stage",
        ),
    )