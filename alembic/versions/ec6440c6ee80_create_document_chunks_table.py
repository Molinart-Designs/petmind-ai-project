"""create document_chunks table

Revision ID: ec6440c6ee80
Revises:
Create Date: 2026-04-21 18:22:42.699645
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.types import UserDefinedType


class VectorType(UserDefinedType):
    cache_ok = True

    def get_col_spec(self, **kwargs) -> str:
        return "vector"


# revision identifiers, used by Alembic.
revision: str = "ec6440c6ee80"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "document_chunks",
        sa.Column("chunk_id", sa.String(length=128), nullable=False),
        sa.Column("document_id", sa.String(length=128), nullable=False),
        sa.Column("title", sa.String(length=300), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=500), nullable=True),
        sa.Column("category", sa.String(length=100), nullable=True),
        sa.Column("species", sa.String(length=50), nullable=True),
        sa.Column("life_stage", sa.String(length=50), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("embedding", VectorType(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("chunk_id"),
    )

    op.create_index(
        op.f("ix_document_chunks_category"),
        "document_chunks",
        ["category"],
        unique=False,
    )
    op.create_index(
        op.f("ix_document_chunks_document_id"),
        "document_chunks",
        ["document_id"],
        unique=False,
    )
    op.create_index(
        "ix_document_chunks_filters",
        "document_chunks",
        ["category", "species", "life_stage"],
        unique=False,
    )
    op.create_index(
        op.f("ix_document_chunks_life_stage"),
        "document_chunks",
        ["life_stage"],
        unique=False,
    )
    op.create_index(
        op.f("ix_document_chunks_species"),
        "document_chunks",
        ["species"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_document_chunks_species"), table_name="document_chunks")
    op.drop_index(op.f("ix_document_chunks_life_stage"), table_name="document_chunks")
    op.drop_index("ix_document_chunks_filters", table_name="document_chunks")
    op.drop_index(op.f("ix_document_chunks_document_id"), table_name="document_chunks")
    op.drop_index(op.f("ix_document_chunks_category"), table_name="document_chunks")
    op.drop_table("document_chunks")