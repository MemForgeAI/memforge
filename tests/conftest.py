"""
Shared test fixtures for MemForge integration tests.

Uses testcontainers to spin up a real Postgres instance with pgvector.
No mocking — what passes in tests works in production.
"""

from collections.abc import AsyncGenerator, Generator
from pathlib import Path

import asyncpg
import pytest
import pytest_asyncio
from testcontainers.postgres import PostgresContainer

# Path to the init SQL script
INIT_SQL = Path(__file__).parent.parent / "init.sql"


@pytest.fixture(scope="session")
def postgres_url() -> Generator[str, None, None]:
    """
    Start a Postgres container with pgvector for the test session.

    Note: This uses pgvector/pgvector:pg17 which includes pgvector
    but NOT Apache AGE. Graph tests (Suite 4) will need
    the custom MemForge image once Phase 0 is complete.
    """
    with PostgresContainer(
        image="pgvector/pgvector:pg17",
        username="memforge",
        password="memforge_test",
        dbname="memforge_test",
    ) as pg:
        url = pg.get_connection_url()
        # Convert to asyncpg format (postgresql+asyncpg://)
        async_url = url.replace("postgresql+psycopg2://", "postgresql://")
        yield async_url


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def db(postgres_url: str) -> AsyncGenerator[asyncpg.Pool, None]:
    """
    Create a connection pool and initialize the schema.
    Shared across all tests in the session.
    """
    pool = await asyncpg.create_pool(postgres_url)
    assert pool is not None

    # Read and execute init SQL (skip AGE-specific lines for now)
    init_sql = INIT_SQL.read_text()

    # Filter out AGE-specific statements (not available in base pgvector image)
    filtered_lines: list[str] = []
    skip_age = False
    for line in init_sql.split("\n"):
        stripped = line.strip().upper()
        if "CREATE EXTENSION" in stripped and "AGE" in stripped:
            skip_age = True
            continue
        if "LOAD" in stripped and "AGE" in stripped:
            continue
        if "AG_CATALOG" in stripped:
            continue
        if "CREATE_GRAPH" in stripped:
            continue
        if skip_age and stripped == "":
            skip_age = False
            continue
        if not skip_age:
            filtered_lines.append(line)

    schema_sql = "\n".join(filtered_lines)

    async with pool.acquire() as conn:
        await conn.execute(schema_sql)

    yield pool
    await pool.close()


@pytest_asyncio.fixture(loop_scope="session")
async def clean_db(db: asyncpg.Pool) -> AsyncGenerator[asyncpg.Pool, None]:
    """
    Provide a clean database for each test.
    Truncates all tables before yielding.
    """
    async with db.acquire() as conn:
        await conn.execute(
            """
            TRUNCATE memories, entities, sessions, memory_conflicts
            CASCADE
            """
        )
    yield db
