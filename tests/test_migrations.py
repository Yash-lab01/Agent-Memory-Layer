"""Tests for Alembic migration configuration and revision lifecycle."""

import tempfile
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect

from alembic import command


@pytest.fixture
def alembic_config() -> Config:
    """Fixture providing Alembic Config object pointing to repo root."""
    repo_root = Path(__file__).resolve().parent.parent
    ini_path = repo_root / "alembic.ini"
    assert ini_path.exists(), "alembic.ini must exist in project root"
    cfg = Config(str(ini_path))
    cfg.set_main_option("script_location", str(repo_root / "alembic"))
    return cfg


def test_migration_revisions_discovered(alembic_config: Config) -> None:
    """Verify Alembic script directory discovers revision 001."""
    script = ScriptDirectory.from_config(alembic_config)
    head_rev = script.get_current_head()
    assert head_rev == "001"

    revision_001 = script.get_revision("001")
    assert revision_001 is not None
    assert revision_001.doc == "Initial schema for episodes and memories tables."


def test_migration_upgrade_and_downgrade_lifecycle(alembic_config: Config) -> None:
    """Verify upgrade to head and downgrade to base on temporary database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        tmp_db_path = tmp_db.name

    try:
        # Configure Alembic to use synchronous sqlite connection for test migration
        sync_url = f"sqlite:///{tmp_db_path}"
        alembic_config.set_main_option("sqlalchemy.url", sync_url)

        # 1. Upgrade to head
        command.upgrade(alembic_config, "head")

        # Inspect tables created
        engine = create_engine(sync_url)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert "episodes" in tables
        assert "memories" in tables
        assert "alembic_version" in tables

        # Verify columns in episodes
        episode_cols = [c["name"] for c in inspector.get_columns("episodes")]
        assert "id" in episode_cols
        assert "user_id" in episode_cols
        assert "session_id" in episode_cols
        assert "text" in episode_cols
        assert "ts" in episode_cols

        # Verify columns in memories
        memory_cols = [c["name"] for c in inspector.get_columns("memories")]
        assert "id" in memory_cols
        assert "user_id" in memory_cols
        assert "content" in memory_cols
        assert "status" in memory_cols
        assert "superseded_by" in memory_cols
        assert "source_episode_id" in memory_cols

        engine.dispose()

        # 2. Downgrade to base
        command.downgrade(alembic_config, "base")

        engine_post = create_engine(sync_url)
        inspector_post = inspect(engine_post)
        tables_post = inspector_post.get_table_names()
        assert "episodes" not in tables_post
        assert "memories" not in tables_post
        engine_post.dispose()

    finally:
        Path(tmp_db_path).unlink(missing_ok=True)
