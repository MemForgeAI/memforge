"""
Suite 6: Multi-agent visibility scope tests.

Tests the four visibility scopes:
  - Private: agent_id must match
  - User-shared: user_id match, any agent, shared=true
  - Task-shared: task_id match, any agent, shared=true
  - Global: shared=true, no filter
"""

import asyncpg
import numpy as np
import pytest


def make_embedding(seed: int) -> tuple[list[float], str]:
    """Generate a deterministic embedding for testing."""
    rng = np.random.RandomState(seed)
    emb = rng.randn(384).tolist()
    return emb, f"[{','.join(str(x) for x in emb)}]"


class TestPrivateScope:
    """Private memories should only be visible to the owning agent."""

    @pytest.mark.asyncio
    async def test_agent_sees_own_memories(self, clean_db: asyncpg.Pool) -> None:
        emb, emb_str = make_embedding(1)
        query_emb, query_str = make_embedding(1)  # Same embedding = high similarity

        async with clean_db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memories (agent_id, user_id, memory_type, content, embedding, shared)
                VALUES ('agent-A', 'user-1', 'semantic', 'Agent A private memory', $1::vector, FALSE)
                """,
                emb_str,
            )

            # Agent A should see its own memory
            rows = await conn.fetch(
                """
                SELECT content, 1 - (embedding <=> $1::vector) AS similarity
                FROM memories
                WHERE agent_id = 'agent-A' AND user_id = 'user-1'
                  AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY similarity DESC
                """,
                query_str,
            )
            assert len(rows) == 1
            assert rows[0]["content"] == "Agent A private memory"

    @pytest.mark.asyncio
    async def test_other_agent_cannot_see_private(self, clean_db: asyncpg.Pool) -> None:
        emb, emb_str = make_embedding(2)

        async with clean_db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memories (agent_id, user_id, memory_type, content, embedding, shared)
                VALUES ('agent-A', 'user-1', 'semantic', 'Agent A secret', $1::vector, FALSE)
                """,
                emb_str,
            )

            # Agent B should NOT see Agent A's private memory
            rows = await conn.fetch(
                """
                SELECT content FROM memories
                WHERE agent_id = 'agent-B' AND user_id = 'user-1'
                """,
            )
            assert len(rows) == 0


class TestUserSharedScope:
    """User-shared memories should be visible to any agent serving the same user."""

    @pytest.mark.asyncio
    async def test_shared_memory_visible_to_other_agents(self, clean_db: asyncpg.Pool) -> None:
        emb, emb_str = make_embedding(3)

        async with clean_db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memories (agent_id, user_id, memory_type, content, embedding, shared)
                VALUES ('agent-A', 'user-1', 'semantic', 'User prefers dark mode', $1::vector, TRUE)
                """,
                emb_str,
            )

            # Agent B should see shared memories for user-1
            rows = await conn.fetch(
                """
                SELECT content FROM memories
                WHERE user_id = 'user-1' AND shared = TRUE
                """,
            )
            assert len(rows) == 1
            assert rows[0]["content"] == "User prefers dark mode"

    @pytest.mark.asyncio
    async def test_shared_not_visible_to_different_user(self, clean_db: asyncpg.Pool) -> None:
        emb, emb_str = make_embedding(4)

        async with clean_db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memories (agent_id, user_id, memory_type, content, embedding, shared)
                VALUES ('agent-A', 'user-1', 'semantic', 'User 1 data', $1::vector, TRUE)
                """,
                emb_str,
            )

            # User-2 should NOT see user-1's shared memories
            rows = await conn.fetch(
                """
                SELECT content FROM memories
                WHERE user_id = 'user-2' AND shared = TRUE
                """,
            )
            assert len(rows) == 0


class TestTaskSharedScope:
    """Task-shared memories should be visible to any agent on the same task."""

    @pytest.mark.asyncio
    async def test_task_shared_visible_to_other_agents(self, clean_db: asyncpg.Pool) -> None:
        emb, emb_str = make_embedding(5)

        async with clean_db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memories (agent_id, user_id, memory_type, content, embedding, shared, task_id)
                VALUES ('agent-A', 'user-1', 'semantic', 'Task booking details', $1::vector, TRUE, 'task-123')
                """,
                emb_str,
            )

            # Agent B on the same task should see this
            rows = await conn.fetch(
                """
                SELECT content FROM memories
                WHERE task_id = 'task-123' AND shared = TRUE
                """,
            )
            assert len(rows) == 1
            assert rows[0]["content"] == "Task booking details"

    @pytest.mark.asyncio
    async def test_task_shared_not_visible_to_other_tasks(self, clean_db: asyncpg.Pool) -> None:
        emb, emb_str = make_embedding(6)

        async with clean_db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memories (agent_id, user_id, memory_type, content, embedding, shared, task_id)
                VALUES ('agent-A', 'user-1', 'semantic', 'Task 123 data', $1::vector, TRUE, 'task-123')
                """,
                emb_str,
            )

            # Different task should NOT see this
            rows = await conn.fetch(
                """
                SELECT content FROM memories
                WHERE task_id = 'task-456' AND shared = TRUE
                """,
            )
            assert len(rows) == 0


class TestGlobalScope:
    """Global memories (shared, no user/task restriction) should be visible to all."""

    @pytest.mark.asyncio
    async def test_global_shared_visible_to_all(self, clean_db: asyncpg.Pool) -> None:
        emb, emb_str = make_embedding(7)

        async with clean_db.acquire() as conn:
            # Global memory: shared, no specific user
            await conn.execute(
                """
                INSERT INTO memories (agent_id, memory_type, content, embedding, shared)
                VALUES ('system', 'semantic', 'Company policy: 30-day return policy', $1::vector, TRUE)
                """,
                emb_str,
            )

            # Any query for shared memories should find this
            rows = await conn.fetch(
                """
                SELECT content FROM memories
                WHERE shared = TRUE
                """,
            )
            assert len(rows) >= 1
            contents = [r["content"] for r in rows]
            assert "Company policy: 30-day return policy" in contents


class TestVisibilityCombination:
    """Test combined visibility scenarios."""

    @pytest.mark.asyncio
    async def test_recall_sees_private_and_shared(self, clean_db: asyncpg.Pool) -> None:
        """An agent should see its private memories AND shared memories for the user."""
        emb, emb_str = make_embedding(8)

        async with clean_db.acquire() as conn:
            # Agent A's private memory
            await conn.execute(
                """
                INSERT INTO memories (agent_id, user_id, memory_type, content, embedding, shared)
                VALUES ('agent-A', 'user-1', 'semantic', 'Private note from A', $1::vector, FALSE)
                """,
                emb_str,
            )

            # Agent B's shared memory for same user
            await conn.execute(
                """
                INSERT INTO memories (agent_id, user_id, memory_type, content, embedding, shared)
                VALUES ('agent-B', 'user-1', 'semantic', 'Shared note from B', $1::vector, TRUE)
                """,
                emb_str,
            )

            # Agent A with include_shared should see both
            rows = await conn.fetch(
                """
                SELECT content FROM memories
                WHERE (
                    (agent_id = 'agent-A' AND user_id = 'user-1')
                    OR (user_id = 'user-1' AND shared = TRUE)
                )
                AND (expires_at IS NULL OR expires_at > NOW())
                """,
            )
            assert len(rows) == 2
            contents = {r["content"] for r in rows}
            assert "Private note from A" in contents
            assert "Shared note from B" in contents

    @pytest.mark.asyncio
    async def test_recall_without_shared_sees_only_private(self, clean_db: asyncpg.Pool) -> None:
        """Without include_shared, an agent should only see its own memories."""
        emb, emb_str = make_embedding(9)

        async with clean_db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memories (agent_id, user_id, memory_type, content, embedding, shared)
                VALUES ('agent-A', 'user-1', 'semantic', 'A private', $1::vector, FALSE)
                """,
                emb_str,
            )
            await conn.execute(
                """
                INSERT INTO memories (agent_id, user_id, memory_type, content, embedding, shared)
                VALUES ('agent-B', 'user-1', 'semantic', 'B shared', $1::vector, TRUE)
                """,
                emb_str,
            )

            # Agent A WITHOUT include_shared
            rows = await conn.fetch(
                """
                SELECT content FROM memories
                WHERE agent_id = 'agent-A' AND user_id = 'user-1'
                AND (expires_at IS NULL OR expires_at > NOW())
                """,
            )
            assert len(rows) == 1
            assert rows[0]["content"] == "A private"
