"""Initial schema for episodes and memories tables.

Revision ID: 001
Revises:
Create Date: 2026-10-14 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # -------------------------------------------------------------------------
    # episodes table
    # -------------------------------------------------------------------------
    op.create_table(
        "episodes",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column(
            "session_id",
            sa.String(length=128),
            nullable=False,
            server_default="default",
        ),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_episodes_user_id"), "episodes", ["user_id"], unique=False)
    op.create_index(op.f("ix_episodes_session_id"), "episodes", ["session_id"], unique=False)
    op.create_index(op.f("ix_episodes_ts"), "episodes", ["ts"], unique=False)

    # -------------------------------------------------------------------------
    # memories table
    # -------------------------------------------------------------------------
    op.create_table(
        "memories",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("entity", sa.String(length=256), nullable=True),
        sa.Column("event_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("importance", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="active"),
        sa.Column("superseded_by", sa.String(length=36), nullable=True),
        sa.Column("source_episode_id", sa.String(length=36), nullable=True),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("access_count", sa.Integer(), nullable=False, server_default="0"),
        sa.ForeignKeyConstraint(["source_episode_id"], ["episodes.id"]),
        sa.ForeignKeyConstraint(["superseded_by"], ["memories.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_memories_user_id"), "memories", ["user_id"], unique=False)
    op.create_index(op.f("ix_memories_entity"), "memories", ["entity"], unique=False)
    op.create_index(op.f("ix_memories_status"), "memories", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_memories_status"), table_name="memories")
    op.drop_index(op.f("ix_memories_entity"), table_name="memories")
    op.drop_index(op.f("ix_memories_user_id"), table_name="memories")
    op.drop_table("memories")

    op.drop_index(op.f("ix_episodes_ts"), table_name="episodes")
    op.drop_index(op.f("ix_episodes_session_id"), table_name="episodes")
    op.drop_index(op.f("ix_episodes_user_id"), table_name="episodes")
    op.drop_table("episodes")
