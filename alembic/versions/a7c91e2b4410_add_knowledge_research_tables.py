"""add knowledge_sources research_candidates research_candidate_sources knowledge_refresh_jobs

Revision ID: a7c91e2b4410
Revises: f8a1c2d3e4b5
Create Date: 2026-04-25 22:55:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "a7c91e2b4410"
down_revision: Union[str, Sequence[str], None] = "f8a1c2d3e4b5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "knowledge_sources",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source_key", sa.String(length=64), nullable=False),
        sa.Column("base_url", sa.String(length=2000), nullable=False),
        sa.Column("category", sa.String(length=100), nullable=True),
        sa.Column("authority_score", sa.Float(), server_default=sa.text("0.5"), nullable=False),
        sa.Column(
            "medical_sensitivity",
            sa.String(length=20),
            server_default=sa.text("'none'"),
            nullable=False,
        ),
        sa.Column("auto_ingest_allowed", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("status", sa.String(length=32), server_default=sa.text("'active'"), nullable=False),
        sa.Column("last_verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("review_after", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_knowledge_sources_category"), "knowledge_sources", ["category"], unique=False)
    op.create_index(op.f("ix_knowledge_sources_source_key"), "knowledge_sources", ["source_key"], unique=True)
    op.create_index(op.f("ix_knowledge_sources_status"), "knowledge_sources", ["status"], unique=False)

    op.create_table(
        "research_candidates",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=32), server_default=sa.text("'provisional'"), nullable=False),
        sa.Column("question_fingerprint", sa.String(length=128), nullable=True),
        sa.Column("provider_id", sa.String(length=64), server_default=sa.text("'unknown'"), nullable=False),
        sa.Column(
            "evidence_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("synthesis_text", sa.Text(), nullable=True),
        sa.Column("topic", sa.String(length=200), nullable=True),
        sa.Column("species", sa.String(length=50), nullable=True),
        sa.Column("breed", sa.String(length=120), nullable=True),
        sa.Column("life_stage", sa.String(length=50), nullable=True),
        sa.Column("authority_score", sa.Float(), server_default=sa.text("0.0"), nullable=False),
        sa.Column("review_after", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_research_candidates_life_stage"), "research_candidates", ["life_stage"], unique=False)
    op.create_index(
        op.f("ix_research_candidates_question_fingerprint"),
        "research_candidates",
        ["question_fingerprint"],
        unique=False,
    )
    op.create_index(op.f("ix_research_candidates_review_after"), "research_candidates", ["review_after"], unique=False)
    op.create_index(op.f("ix_research_candidates_species"), "research_candidates", ["species"], unique=False)
    op.create_index(op.f("ix_research_candidates_status"), "research_candidates", ["status"], unique=False)
    op.create_index(op.f("ix_research_candidates_topic"), "research_candidates", ["topic"], unique=False)
    op.create_index(op.f("ix_research_candidates_user_id"), "research_candidates", ["user_id"], unique=False)

    op.create_table(
        "research_candidate_sources",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("research_candidate_id", sa.Integer(), nullable=False),
        sa.Column("knowledge_source_id", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=32), server_default=sa.text("'citation'"), nullable=False),
        sa.Column(
            "snippet_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("sort_order", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["knowledge_source_id"], ["knowledge_sources.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["research_candidate_id"], ["research_candidates.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_research_candidate_sources_knowledge_source_id"),
        "research_candidate_sources",
        ["knowledge_source_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_research_candidate_sources_research_candidate_id"),
        "research_candidate_sources",
        ["research_candidate_id"],
        unique=False,
    )
    op.create_index(
        "ix_research_candidate_sources_unique_pair",
        "research_candidate_sources",
        ["research_candidate_id", "knowledge_source_id"],
        unique=True,
    )

    op.create_table(
        "knowledge_refresh_jobs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("public_id", sa.String(length=36), nullable=False),
        sa.Column("target_status", sa.String(length=32), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("batch_limit", sa.Integer(), nullable=False),
        sa.Column("job_status", sa.String(length=32), server_default=sa.text("'pending'"), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_knowledge_refresh_jobs_as_of"), "knowledge_refresh_jobs", ["as_of"], unique=False)
    op.create_index(
        op.f("ix_knowledge_refresh_jobs_job_status"), "knowledge_refresh_jobs", ["job_status"], unique=False
    )
    op.create_index(
        op.f("ix_knowledge_refresh_jobs_public_id"), "knowledge_refresh_jobs", ["public_id"], unique=True
    )
    op.create_index(
        op.f("ix_knowledge_refresh_jobs_target_status"), "knowledge_refresh_jobs", ["target_status"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_knowledge_refresh_jobs_target_status"), table_name="knowledge_refresh_jobs")
    op.drop_index(op.f("ix_knowledge_refresh_jobs_public_id"), table_name="knowledge_refresh_jobs")
    op.drop_index(op.f("ix_knowledge_refresh_jobs_job_status"), table_name="knowledge_refresh_jobs")
    op.drop_index(op.f("ix_knowledge_refresh_jobs_as_of"), table_name="knowledge_refresh_jobs")
    op.drop_table("knowledge_refresh_jobs")

    op.drop_index("ix_research_candidate_sources_unique_pair", table_name="research_candidate_sources")
    op.drop_index(
        op.f("ix_research_candidate_sources_research_candidate_id"), table_name="research_candidate_sources"
    )
    op.drop_index(
        op.f("ix_research_candidate_sources_knowledge_source_id"), table_name="research_candidate_sources"
    )
    op.drop_table("research_candidate_sources")

    op.drop_index(op.f("ix_research_candidates_user_id"), table_name="research_candidates")
    op.drop_index(op.f("ix_research_candidates_topic"), table_name="research_candidates")
    op.drop_index(op.f("ix_research_candidates_status"), table_name="research_candidates")
    op.drop_index(op.f("ix_research_candidates_species"), table_name="research_candidates")
    op.drop_index(op.f("ix_research_candidates_review_after"), table_name="research_candidates")
    op.drop_index(op.f("ix_research_candidates_question_fingerprint"), table_name="research_candidates")
    op.drop_index(op.f("ix_research_candidates_life_stage"), table_name="research_candidates")
    op.drop_table("research_candidates")

    op.drop_index(op.f("ix_knowledge_sources_status"), table_name="knowledge_sources")
    op.drop_index(op.f("ix_knowledge_sources_source_key"), table_name="knowledge_sources")
    op.drop_index(op.f("ix_knowledge_sources_category"), table_name="knowledge_sources")
    op.drop_table("knowledge_sources")
