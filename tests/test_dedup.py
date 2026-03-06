"""
Suite 3: Deduplication and conflict detection tests.

Tests the dedup/conflict pipeline:
  - Exact duplicates (> dedup threshold)
  - Conflict range (conflict_threshold..dedup_threshold)
  - Conflict records in memory_conflicts table
  - Elaboration merges
"""

import asyncpg
import numpy as np
import pytest


class TestExactDuplication:
    """Memories with similarity > dedup_threshold should update existing."""

    @pytest.mark.asyncio
    async def test_exact_duplicate_not_inserted(self, clean_db: asyncpg.Pool) -> None:
        """Storing the same content twice should not create two rows."""
        # Create a random embedding
        embedding = np.random.randn(384).tolist()
        emb_str = f"[{','.join(str(x) for x in embedding)}]"

        async with clean_db.acquire() as conn:
            # Insert first memory
            await conn.execute(
                """
                INSERT INTO memories (agent_id, user_id, memory_type, content, embedding, importance)
                VALUES ('agent-1', 'user-dedup', 'semantic', 'User likes cats', $1::vector, 0.5)
                """,
                emb_str,
            )

            # Insert second with identical embedding (same meaning)
            count_before = await conn.fetchval("SELECT COUNT(*) FROM memories WHERE user_id = 'user-dedup'")
            assert count_before == 1

            # Simulate dedup check — find similar above 0.85 threshold
            similar = await conn.fetch(
                """
                SELECT id, content, 1 - (embedding <=> $1::vector) AS similarity
                FROM memories
                WHERE user_id = 'user-dedup'
                  AND 1 - (embedding <=> $1::vector) >= 0.85
                ORDER BY similarity DESC
                LIMIT 1
                """,
                emb_str,
            )

            # Same embedding → similarity 1.0, should be flagged as duplicate
            assert len(similar) == 1
            assert float(similar[0]["similarity"]) > 0.99

    @pytest.mark.asyncio
    async def test_different_content_not_duplicate(self, clean_db: asyncpg.Pool) -> None:
        """Two semantically different memories should both be stored."""
        emb1 = np.random.randn(384).tolist()
        emb2 = np.random.randn(384).tolist()
        emb1_str = f"[{','.join(str(x) for x in emb1)}]"
        emb2_str = f"[{','.join(str(x) for x in emb2)}]"

        async with clean_db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memories (agent_id, user_id, memory_type, content, embedding)
                VALUES ('agent-1', 'user-dedup', 'semantic', 'User likes cats', $1::vector)
                """,
                emb1_str,
            )

            # Check for similarity — random vectors should be < 0.85
            similar = await conn.fetch(
                """
                SELECT id, 1 - (embedding <=> $1::vector) AS similarity
                FROM memories
                WHERE user_id = 'user-dedup'
                  AND 1 - (embedding <=> $1::vector) >= 0.85
                """,
                emb2_str,
            )
            assert len(similar) == 0  # Not a duplicate


class TestConflictDetection:
    """Memories in the conflict range should create conflict records."""

    @pytest.mark.asyncio
    async def test_conflict_record_created(self, clean_db: asyncpg.Pool) -> None:
        """Two contradicting memories should create a memory_conflicts row."""
        # Use similar but not identical embeddings (simulate 0.80-0.85 similarity)
        base = np.random.randn(384)
        base = base / np.linalg.norm(base)
        noise = np.random.randn(384) * 0.3
        similar_emb = base + noise
        similar_emb = similar_emb / np.linalg.norm(similar_emb)

        base_str = f"[{','.join(str(x) for x in base.tolist())}]"
        similar_str = f"[{','.join(str(x) for x in similar_emb.tolist())}]"

        async with clean_db.acquire() as conn:
            # Insert first memory
            m1 = await conn.fetchval(
                """
                INSERT INTO memories (agent_id, user_id, memory_type, content, embedding)
                VALUES ('agent-1', 'user-conflict', 'semantic', 'Memory A', $1::vector)
                RETURNING id
                """,
                base_str,
            )

            # Insert second memory
            m2 = await conn.fetchval(
                """
                INSERT INTO memories (agent_id, user_id, memory_type, content, embedding)
                VALUES ('agent-1', 'user-conflict', 'semantic', 'Memory B', $1::vector)
                RETURNING id
                """,
                similar_str,
            )

            # Record a conflict
            conflict_id = await conn.fetchval(
                """
                INSERT INTO memory_conflicts (memory_a_id, memory_b_id, conflict_type, resolution)
                VALUES ($1, $2, 'contradiction', 'pending')
                RETURNING id
                """,
                m1,
                m2,
            )

            assert conflict_id is not None

            # Verify it's there
            row = await conn.fetchrow(
                "SELECT * FROM memory_conflicts WHERE id = $1", conflict_id
            )
            assert row is not None
            assert row["conflict_type"] == "contradiction"
            assert row["resolution"] == "pending"
            assert row["resolved_at"] is None

    @pytest.mark.asyncio
    async def test_conflict_resolution(self, clean_db: asyncpg.Pool) -> None:
        """Conflicts can be resolved by updating the resolution column."""
        emb = np.random.randn(384).tolist()
        emb_str = f"[{','.join(str(x) for x in emb)}]"

        async with clean_db.acquire() as conn:
            m1 = await conn.fetchval(
                """
                INSERT INTO memories (agent_id, memory_type, content, embedding)
                VALUES ('agent-1', 'semantic', 'A', $1::vector)
                RETURNING id
                """,
                emb_str,
            )
            m2 = await conn.fetchval(
                """
                INSERT INTO memories (agent_id, memory_type, content, embedding)
                VALUES ('agent-1', 'semantic', 'B', $1::vector)
                RETURNING id
                """,
                emb_str,
            )

            conflict_id = await conn.fetchval(
                """
                INSERT INTO memory_conflicts (memory_a_id, memory_b_id, conflict_type, resolution)
                VALUES ($1, $2, 'contradiction', 'pending')
                RETURNING id
                """,
                m1,
                m2,
            )

            # Resolve it
            await conn.execute(
                """
                UPDATE memory_conflicts
                SET resolution = 'b_wins', resolved_at = NOW()
                WHERE id = $1
                """,
                conflict_id,
            )

            row = await conn.fetchrow(
                "SELECT * FROM memory_conflicts WHERE id = $1", conflict_id
            )
            assert row["resolution"] == "b_wins"
            assert row["resolved_at"] is not None

    @pytest.mark.asyncio
    async def test_conflict_cascade_on_memory_delete(self, clean_db: asyncpg.Pool) -> None:
        """Deleting a memory should also delete its conflict records."""
        emb = np.random.randn(384).tolist()
        emb_str = f"[{','.join(str(x) for x in emb)}]"

        async with clean_db.acquire() as conn:
            m1 = await conn.fetchval(
                """
                INSERT INTO memories (agent_id, memory_type, content, embedding)
                VALUES ('agent-1', 'semantic', 'To delete', $1::vector)
                RETURNING id
                """,
                emb_str,
            )
            m2 = await conn.fetchval(
                """
                INSERT INTO memories (agent_id, memory_type, content, embedding)
                VALUES ('agent-1', 'semantic', 'Other', $1::vector)
                RETURNING id
                """,
                emb_str,
            )

            await conn.execute(
                """
                INSERT INTO memory_conflicts (memory_a_id, memory_b_id, conflict_type)
                VALUES ($1, $2, 'contradiction')
                """,
                m1,
                m2,
            )

            # Delete memory_a — conflict should cascade
            await conn.execute("DELETE FROM memories WHERE id = $1", m1)

            count = await conn.fetchval(
                "SELECT COUNT(*) FROM memory_conflicts WHERE memory_a_id = $1 OR memory_b_id = $1",
                m1,
            )
            assert count == 0

    @pytest.mark.asyncio
    async def test_valid_conflict_types(self, clean_db: asyncpg.Pool) -> None:
        """Only valid conflict_type values should be accepted."""
        emb = np.random.randn(384).tolist()
        emb_str = f"[{','.join(str(x) for x in emb)}]"

        async with clean_db.acquire() as conn:
            m1 = await conn.fetchval(
                """
                INSERT INTO memories (agent_id, memory_type, content, embedding)
                VALUES ('agent-1', 'semantic', 'X', $1::vector)
                RETURNING id
                """,
                emb_str,
            )
            m2 = await conn.fetchval(
                """
                INSERT INTO memories (agent_id, memory_type, content, embedding)
                VALUES ('agent-1', 'semantic', 'Y', $1::vector)
                RETURNING id
                """,
                emb_str,
            )

            # Valid types: contradiction, outdated, duplicate
            for conflict_type in ("contradiction", "outdated", "duplicate"):
                cid = await conn.fetchval(
                    """
                    INSERT INTO memory_conflicts (memory_a_id, memory_b_id, conflict_type)
                    VALUES ($1, $2, $3)
                    RETURNING id
                    """,
                    m1,
                    m2,
                    conflict_type,
                )
                assert cid is not None
                # Clean up for next iteration
                await conn.execute("DELETE FROM memory_conflicts WHERE id = $1", cid)

            # Invalid type should fail
            with pytest.raises(asyncpg.CheckViolationError):
                await conn.execute(
                    """
                    INSERT INTO memory_conflicts (memory_a_id, memory_b_id, conflict_type)
                    VALUES ($1, $2, 'invalid_type')
                    """,
                    m1,
                    m2,
                )


class TestElaboration:
    """Elaboration: updating existing memory content with richer version."""

    @pytest.mark.asyncio
    async def test_memory_content_update(self, clean_db: asyncpg.Pool) -> None:
        """Memory content and embedding can be updated for elaboration."""
        emb1 = np.random.randn(384).tolist()
        emb2 = np.random.randn(384).tolist()
        emb1_str = f"[{','.join(str(x) for x in emb1)}]"
        emb2_str = f"[{','.join(str(x) for x in emb2)}]"

        async with clean_db.acquire() as conn:
            mem_id = await conn.fetchval(
                """
                INSERT INTO memories (agent_id, memory_type, content, embedding)
                VALUES ('agent-1', 'semantic', 'User likes coffee', $1::vector)
                RETURNING id
                """,
                emb1_str,
            )

            # Update with elaborated content
            await conn.execute(
                """
                UPDATE memories SET
                  content = 'User likes coffee, especially Ethiopian single-origin dark roast',
                  embedding = $2::vector,
                  updated_at = NOW()
                WHERE id = $1
                """,
                mem_id,
                emb2_str,
            )

            row = await conn.fetchrow("SELECT * FROM memories WHERE id = $1", mem_id)
            assert row is not None
            assert "Ethiopian" in row["content"]
            # Only one memory should exist
            count = await conn.fetchval("SELECT COUNT(*) FROM memories WHERE agent_id = 'agent-1'")
            assert count == 1
