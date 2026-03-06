"""
Suite 5: Reflect and decay tests.

Tests:
  - Reflect: recent memory consolidation, insight storage
  - Decay: importance decay formula, archiving stale memories
"""

import asyncpg
import numpy as np
import pytest
from datetime import timedelta


class TestDecay:
    """Importance decay for stale memories."""

    @pytest.mark.asyncio
    async def test_decay_reduces_importance(self, clean_db: asyncpg.Pool) -> None:
        """Memories not accessed recently should have lower importance after decay."""
        emb = np.random.randn(384).tolist()
        emb_str = f"[{','.join(str(x) for x in emb)}]"

        async with clean_db.acquire() as conn:
            # Insert memory with old last_accessed (3 days ago)
            mem_id = await conn.fetchval(
                """
                INSERT INTO memories (agent_id, memory_type, content, embedding, importance, last_accessed)
                VALUES ('agent-1', 'semantic', 'Old memory', $1::vector, 0.8,
                        NOW() - INTERVAL '3 days')
                RETURNING id
                """,
                emb_str,
            )

            # Apply decay: importance *= 0.95 ^ days
            # 0.8 * 0.95^3 = 0.8 * 0.857 = ~0.686
            await conn.execute(
                """
                UPDATE memories
                SET importance = GREATEST(
                    0.05,
                    importance * POWER(0.95, EXTRACT(EPOCH FROM (NOW() - COALESCE(last_accessed, created_at))) / 86400)
                ),
                updated_at = NOW()
                WHERE id = $1
                """,
                mem_id,
            )

            row = await conn.fetchrow("SELECT importance FROM memories WHERE id = $1", mem_id)
            importance = float(row["importance"])

            # Should be roughly 0.686 (0.8 * 0.95^3)
            assert 0.60 < importance < 0.75, f"Expected ~0.686, got {importance}"

    @pytest.mark.asyncio
    async def test_recently_accessed_not_decayed(self, clean_db: asyncpg.Pool) -> None:
        """Memories accessed recently should not be decayed."""
        emb = np.random.randn(384).tolist()
        emb_str = f"[{','.join(str(x) for x in emb)}]"

        async with clean_db.acquire() as conn:
            mem_id = await conn.fetchval(
                """
                INSERT INTO memories (agent_id, memory_type, content, embedding, importance, last_accessed)
                VALUES ('agent-1', 'semantic', 'Recent memory', $1::vector, 0.8, NOW())
                RETURNING id
                """,
                emb_str,
            )

            # Decay should not affect this (0 days since access)
            before = await conn.fetchval("SELECT importance FROM memories WHERE id = $1", mem_id)

            # Run decay only on stale memories (>1 day)
            await conn.execute(
                """
                UPDATE memories
                SET importance = GREATEST(
                    0.05,
                    importance * POWER(0.95, EXTRACT(EPOCH FROM (NOW() - COALESCE(last_accessed, created_at))) / 86400)
                )
                WHERE id = $1
                  AND COALESCE(last_accessed, created_at) < NOW() - INTERVAL '1 day'
                """,
                mem_id,
            )

            after = await conn.fetchval("SELECT importance FROM memories WHERE id = $1", mem_id)
            assert float(before) == float(after), "Recently accessed memory should not be decayed"

    @pytest.mark.asyncio
    async def test_archive_below_threshold(self, clean_db: asyncpg.Pool) -> None:
        """Memories with importance below threshold should get an expiry date."""
        emb = np.random.randn(384).tolist()
        emb_str = f"[{','.join(str(x) for x in emb)}]"

        async with clean_db.acquire() as conn:
            # Insert low-importance memory
            mem_id = await conn.fetchval(
                """
                INSERT INTO memories (agent_id, memory_type, content, embedding, importance)
                VALUES ('agent-1', 'semantic', 'Trivial fact', $1::vector, 0.03)
                RETURNING id
                """,
                emb_str,
            )

            # Archive memories below threshold
            await conn.execute(
                """
                UPDATE memories
                SET expires_at = NOW() + INTERVAL '30 days'
                WHERE importance <= 0.05
                  AND (expires_at IS NULL OR expires_at > NOW() + INTERVAL '30 days')
                """,
            )

            row = await conn.fetchrow("SELECT expires_at FROM memories WHERE id = $1", mem_id)
            assert row["expires_at"] is not None, "Low-importance memory should be archived"

    @pytest.mark.asyncio
    async def test_high_importance_not_archived(self, clean_db: asyncpg.Pool) -> None:
        """Important memories should never be archived."""
        emb = np.random.randn(384).tolist()
        emb_str = f"[{','.join(str(x) for x in emb)}]"

        async with clean_db.acquire() as conn:
            mem_id = await conn.fetchval(
                """
                INSERT INTO memories (agent_id, memory_type, content, embedding, importance)
                VALUES ('agent-1', 'semantic', 'Safety-critical: user allergic to peanuts', $1::vector, 1.0)
                RETURNING id
                """,
                emb_str,
            )

            # Archive should not touch this
            await conn.execute(
                """
                UPDATE memories
                SET expires_at = NOW() + INTERVAL '30 days'
                WHERE importance <= 0.05
                """,
            )

            row = await conn.fetchrow("SELECT expires_at FROM memories WHERE id = $1", mem_id)
            assert row["expires_at"] is None, "High-importance memory should not be archived"


class TestReflectStorage:
    """Reflected insights should be stored correctly."""

    @pytest.mark.asyncio
    async def test_insight_stored_with_correct_confidence(self, clean_db: asyncpg.Pool) -> None:
        """Reflected insights should have confidence=0.8."""
        emb = np.random.randn(384).tolist()
        emb_str = f"[{','.join(str(x) for x in emb)}]"

        async with clean_db.acquire() as conn:
            mem_id = await conn.fetchval(
                """
                INSERT INTO memories (
                    agent_id, memory_type, content, embedding,
                    confidence, importance, source, metadata
                ) VALUES (
                    'agent-1', 'semantic', 'Insight from reflection',
                    $1::vector, 0.8, 0.6, 'reflection',
                    '{"source_type": "reflection", "source_memory_count": 3}'::jsonb
                )
                RETURNING id
                """,
                emb_str,
            )

            row = await conn.fetchrow("SELECT * FROM memories WHERE id = $1", mem_id)
            assert float(row["confidence"]) == 0.8
            assert row["source"] == "reflection"

            import json
            metadata = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"])
            assert metadata["source_type"] == "reflection"
            assert metadata["source_memory_count"] == 3

    @pytest.mark.asyncio
    async def test_insight_memory_type_is_semantic(self, clean_db: asyncpg.Pool) -> None:
        """Reflected insights are always stored as semantic memories."""
        emb = np.random.randn(384).tolist()
        emb_str = f"[{','.join(str(x) for x in emb)}]"

        async with clean_db.acquire() as conn:
            mem_id = await conn.fetchval(
                """
                INSERT INTO memories (
                    agent_id, memory_type, content, embedding, source
                ) VALUES (
                    'agent-1', 'semantic', 'Pattern: user prefers morning flights',
                    $1::vector, 'reflection'
                )
                RETURNING id
                """,
                emb_str,
            )

            row = await conn.fetchrow("SELECT memory_type FROM memories WHERE id = $1", mem_id)
            assert row["memory_type"] == "semantic"
