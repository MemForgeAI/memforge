"""
Suite 1: Memory CRUD — remember and forget.

Tests the write path: classification, embedding, storage, and deletion.
These tests hit real Postgres via testcontainers.
"""

import asyncpg
import pytest


class TestRememberCreatesMemory:
    """Test 1.1: Create semantic memory via remember tool."""

    async def test_insert_and_retrieve(self, clean_db: asyncpg.Pool) -> None:
        """Verify a memory can be inserted and retrieved by ID."""
        async with clean_db.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO memories (agent_id, user_id, memory_type, content, importance)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id, agent_id, memory_type, content, importance
                """,
                "agent-1",
                "user-1",
                "semantic",
                "User prefers window seats on long flights",
                0.5,
            )

            assert row is not None
            assert row["id"] is not None
            assert row["memory_type"] == "semantic"
            assert row["content"] == "User prefers window seats on long flights"

    async def test_default_values(self, clean_db: asyncpg.Pool) -> None:
        """Verify default values are set correctly."""
        async with clean_db.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO memories (agent_id, memory_type, content)
                VALUES ($1, $2, $3)
                RETURNING confidence, importance, shared, access_count, metadata
                """,
                "agent-1",
                "semantic",
                "Test memory",
            )

            assert row is not None
            assert row["confidence"] == 1.0
            assert row["importance"] == 0.5
            assert row["shared"] is False
            assert row["access_count"] == 0


class TestConfidenceBounds:
    """Test 1.3: Reject invalid confidence/importance values."""

    async def test_importance_below_zero_rejected(
        self, clean_db: asyncpg.Pool
    ) -> None:
        async with clean_db.acquire() as conn:
            with pytest.raises(asyncpg.CheckViolationError):
                await conn.execute(
                    """
                    INSERT INTO memories (agent_id, memory_type, content, importance)
                    VALUES ($1, $2, $3, $4)
                    """,
                    "agent-1",
                    "semantic",
                    "test",
                    -0.1,
                )

    async def test_importance_above_one_rejected(
        self, clean_db: asyncpg.Pool
    ) -> None:
        async with clean_db.acquire() as conn:
            with pytest.raises(asyncpg.CheckViolationError):
                await conn.execute(
                    """
                    INSERT INTO memories (agent_id, memory_type, content, importance)
                    VALUES ($1, $2, $3, $4)
                    """,
                    "agent-1",
                    "semantic",
                    "test",
                    1.1,
                )

    async def test_confidence_below_zero_rejected(
        self, clean_db: asyncpg.Pool
    ) -> None:
        async with clean_db.acquire() as conn:
            with pytest.raises(asyncpg.CheckViolationError):
                await conn.execute(
                    """
                    INSERT INTO memories (agent_id, memory_type, content, confidence)
                    VALUES ($1, $2, $3, $4)
                    """,
                    "agent-1",
                    "semantic",
                    "test",
                    -0.1,
                )

    async def test_valid_boundary_values_accepted(
        self, clean_db: asyncpg.Pool
    ) -> None:
        async with clean_db.acquire() as conn:
            # 0.0 and 1.0 should both be accepted
            for val in [0.0, 1.0, 0.5]:
                await conn.execute(
                    """
                    INSERT INTO memories (agent_id, memory_type, content, importance, confidence)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    "agent-1",
                    "semantic",
                    f"test-{val}",
                    val,
                    val,
                )


class TestMemoryTypeConstraint:
    """Test that invalid memory_type values are rejected."""

    async def test_invalid_type_rejected(self, clean_db: asyncpg.Pool) -> None:
        async with clean_db.acquire() as conn:
            with pytest.raises(asyncpg.CheckViolationError):
                await conn.execute(
                    """
                    INSERT INTO memories (agent_id, memory_type, content)
                    VALUES ($1, $2, $3)
                    """,
                    "agent-1",
                    "invalid_type",
                    "test",
                )

    async def test_valid_types_accepted(self, clean_db: asyncpg.Pool) -> None:
        async with clean_db.acquire() as conn:
            for mem_type in ["semantic", "episodic", "procedural"]:
                await conn.execute(
                    """
                    INSERT INTO memories (agent_id, memory_type, content)
                    VALUES ($1, $2, $3)
                    """,
                    "agent-1",
                    mem_type,
                    f"test-{mem_type}",
                )


class TestForgetCleansUp:
    """Test 1.4: Forget deletes memory."""

    async def test_delete_removes_memory(self, clean_db: asyncpg.Pool) -> None:
        async with clean_db.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO memories (agent_id, memory_type, content)
                VALUES ($1, $2, $3)
                RETURNING id
                """,
                "agent-1",
                "semantic",
                "User loves sushi in Tokyo",
            )
            assert row is not None
            memory_id = row["id"]

            # Delete
            result = await conn.execute(
                "DELETE FROM memories WHERE id = $1", memory_id
            )
            assert result == "DELETE 1"

            # Verify gone
            check = await conn.fetchrow(
                "SELECT id FROM memories WHERE id = $1", memory_id
            )
            assert check is None

    async def test_conflict_cascade_on_delete(self, clean_db: asyncpg.Pool) -> None:
        """Deleting a memory should cascade to memory_conflicts."""
        async with clean_db.acquire() as conn:
            # Create two memories
            m1 = await conn.fetchrow(
                """
                INSERT INTO memories (agent_id, memory_type, content)
                VALUES ($1, $2, $3) RETURNING id
                """,
                "agent-1",
                "semantic",
                "Memory A",
            )
            m2 = await conn.fetchrow(
                """
                INSERT INTO memories (agent_id, memory_type, content)
                VALUES ($1, $2, $3) RETURNING id
                """,
                "agent-1",
                "semantic",
                "Memory B",
            )
            assert m1 is not None and m2 is not None

            # Create a conflict between them
            await conn.execute(
                """
                INSERT INTO memory_conflicts (memory_a_id, memory_b_id, conflict_type)
                VALUES ($1, $2, $3)
                """,
                m1["id"],
                m2["id"],
                "contradiction",
            )

            # Delete memory A — conflict should cascade
            await conn.execute("DELETE FROM memories WHERE id = $1", m1["id"])

            conflicts = await conn.fetch("SELECT * FROM memory_conflicts")
            assert len(conflicts) == 0
