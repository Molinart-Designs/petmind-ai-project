"""add users pets and query_history tables

Revision ID: f8a1c2d3e4b5
Revises: ec6440c6ee80
Create Date: 2026-04-25 12:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "f8a1c2d3e4b5"
down_revision: Union[str, Sequence[str], None] = "ec6440c6ee80"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("auth0_sub", sa.String(length=255), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=True),
        sa.Column("full_name", sa.String(length=255), nullable=True),
        sa.Column("role_label", sa.String(length=50), nullable=True),
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
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_users_auth0_sub"), "users", ["auth0_sub"], unique=True)
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=False)

    op.create_table(
        "pets",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("species", sa.String(length=50), nullable=False),
        sa.Column("breed", sa.String(length=120), nullable=True),
        sa.Column("age_years", sa.Float(), nullable=True),
        sa.Column("life_stage", sa.String(length=50), nullable=True),
        sa.Column("weight_kg", sa.Float(), nullable=True),
        sa.Column("sex", sa.String(length=20), nullable=True),
        sa.Column("neutered", sa.Boolean(), nullable=True),
        sa.Column(
            "conditions_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("notes", sa.Text(), nullable=True),
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
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_pets_life_stage"), "pets", ["life_stage"], unique=False)
    op.create_index(op.f("ix_pets_species"), "pets", ["species"], unique=False)
    op.create_index(op.f("ix_pets_user_id"), "pets", ["user_id"], unique=False)

    op.create_table(
        "query_history",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("pet_id", sa.Integer(), nullable=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("confidence", sa.String(length=20), nullable=False),
        sa.Column("needs_vet_followup", sa.Boolean(), nullable=False),
        sa.Column(
            "sources_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "filters_json",
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
        sa.ForeignKeyConstraint(["pet_id"], ["pets.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_query_history_pet_id"), "query_history", ["pet_id"], unique=False)
    op.create_index(op.f("ix_query_history_user_id"), "query_history", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_query_history_user_id"), table_name="query_history")
    op.drop_index(op.f("ix_query_history_pet_id"), table_name="query_history")
    op.drop_table("query_history")

    op.drop_index(op.f("ix_pets_user_id"), table_name="pets")
    op.drop_index(op.f("ix_pets_species"), table_name="pets")
    op.drop_index(op.f("ix_pets_life_stage"), table_name="pets")
    op.drop_table("pets")

    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_index(op.f("ix_users_auth0_sub"), table_name="users")
    op.drop_table("users")
