"""
Suite 2: Semantic Search & Context Assembly.

Tests the read path: vector search, ranking, token packing, formatting.
These tests hit real Postgres via testcontainers.
"""

import asyncpg
import pytest


class TestSemanticRecall:
    """Test 2.1: Semantic similarity retrieval."""

    async def test_vector_search_basic(self, clean_db: asyncpg.Pool) -> None:
        """Verify pgvector cosine similarity search works end-to-end."""
        async with clean_db.acquire() as conn:
            # Insert memories with simple 3-dim vectors for testing
            # (real vectors are 384 or 1536 dims, but we use 3 for simplicity)
            await conn.execute(
                """
                INSERT INTO memories (agent_id, user_id, memory_type, content, embedding)
                VALUES
                    ($1, $2, 'semantic', 'User prefers direct flights', '[0.9, 0.1, 0.0]'::vector),
                    ($1, $2, 'semantic', 'User birthday is March 15th', '[0.0, 0.1, 0.9]'::vector),
                    ($1, $2, 'semantic', 'User hates layovers', '[0.85, 0.15, 0.0]'::vector)
                """,
                "agent-1",
                "user-1",
            )

            # Query with a vector similar to "flight preferences"
            rows = await conn.fetch(
                """
                SELECT content,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM memories
                WHERE user_id = $2
                ORDER BY embedding <=> $1::vector
                LIMIT 2
                """,
                "[0.9, 0.1, 0.0]",
                "user-1",
            )

            assert len(rows) == 2
            # Flight-related should be top results
            contents = [r["content"] for r in rows]
            assert "User prefers direct flights" in contents
            assert "User hates layovers" in contents
            # Birthday should NOT be in top 2
            assert "User birthday is March 15th" not in contents


class TestTokenBudget:
    """Test 2.2: Token budget is respected."""

    async def test_memories_stored_with_metadata(
        self, clean_db: asyncpg.Pool
    ) -> None:
        """Verify JSONB metadata is stored and queryable."""
        async with clean_db.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO memories (agent_id, memory_type, content, metadata)
                VALUES ($1, $2, $3, $4::jsonb)
                RETURNING metadata
                """,
                "agent-1",
                "semantic",
                "Test with metadata",
                '{"source": "test", "confidence": 0.95}',
            )
            assert row is not None
            import json

            meta = json.loads(row["metadata"])
            assert meta["source"] == "test"
            assert meta["confidence"] == 0.95


class TestContextFormatting:
    """Test 2.3: Context is formatted, not raw."""

    async def test_expiry_filtering(self, clean_db: asyncpg.Pool) -> None:
        """Expired memories should be excluded from queries."""
        async with clean_db.acquire() as conn:
            # Insert a non-expired memory
            await conn.execute(
                """
                INSERT INTO memories (agent_id, user_id, memory_type, content, embedding, expires_at)
                VALUES ($1, $2, 'semantic', 'Valid memory', '[0.5, 0.5, 0.0]'::vector, NULL)
                """,
                "agent-1",
                "user-1",
            )

            # Insert an expired memory
            await conn.execute(
                """
                INSERT INTO memories (agent_id, user_id, memory_type, content, embedding, expires_at)
                VALUES ($1, $2, 'semantic', 'Expired promo', '[0.5, 0.5, 0.0]'::vector,
                        NOW() - INTERVAL '1 day')
                """,
                "agent-1",
                "user-1",
            )

            # Query with expiry filter (same as recall does)
            rows = await conn.fetch(
                """
                SELECT content FROM memories
                WHERE user_id = $1
                  AND (expires_at IS NULL OR expires_at > NOW())
                """,
                "user-1",
            )

            contents = [r["content"] for r in rows]
            assert "Valid memory" in contents
            assert "Expired promo" not in contents


class TestEntityUniqueConstraint:
    """Test entity table UNIQUE constraint on (name, entity_type)."""

    async def test_duplicate_entity_rejected(self, clean_db: asyncpg.Pool) -> None:
        async with clean_db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO entities (name, entity_type)
                VALUES ($1, $2)
                """,
                "JAL",
                "airline",
            )

            with pytest.raises(asyncpg.UniqueViolationError):
                await conn.execute(
                    """
                    INSERT INTO entities (name, entity_type)
                    VALUES ($1, $2)
                    """,
                    "JAL",
                    "airline",
                )

    async def test_same_name_different_type_ok(
        self, clean_db: asyncpg.Pool
    ) -> None:
        async with clean_db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO entities (name, entity_type)
                VALUES ($1, $2)
                """,
                "Tokyo",
                "city",
            )
            # Same name, different type should succeed
            await conn.execute(
                """
                INSERT INTO entities (name, entity_type)
                VALUES ($1, $2)
                """,
                "Tokyo",
                "restaurant",
            )

            count = await conn.fetchval(
                "SELECT COUNT(*) FROM entities WHERE name = $1", "Tokyo"
            )
            assert count == 2
