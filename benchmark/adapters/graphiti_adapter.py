"""Graphiti adapter — wraps the graphiti-core Python SDK."""

from __future__ import annotations

import time
from datetime import datetime

from benchmark.adapters.base import MemoryAdapter
from benchmark.models import RecallResult, StoreResult


class GraphitiAdapter(MemoryAdapter):
    name = "graphiti"

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "demodemo",
    ) -> None:
        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password
        self._client = None
        self._episode_counter = 0

    async def setup(self) -> float:
        start = time.perf_counter()
        from graphiti_core import Graphiti

        self._client = Graphiti(
            uri=self._neo4j_uri,
            user=self._neo4j_user,
            password=self._neo4j_password,
        )
        await self._client.build_indices_and_constraints()
        return time.perf_counter() - start

    async def store(
        self,
        content: str,
        *,
        user_id: str = "alex",
        agent_id: str = "coding-assistant",
        importance: float | None = None,
        shared: bool = False,
        task_id: str | None = None,
    ) -> StoreResult:
        from graphiti_core.nodes import EpisodeType

        self._episode_counter += 1
        start = time.perf_counter()
        await self._client.add_episode(
            name=f"memory-{self._episode_counter}",
            episode_body=content,
            source=EpisodeType.text,
            source_description=f"agent:{agent_id} user:{user_id}",
            reference_time=datetime.now(),
        )
        elapsed = (time.perf_counter() - start) * 1000
        return StoreResult(
            memory_id=None,
            api_calls=3,
            latency_ms=elapsed,
        )

    async def recall(
        self,
        query: str,
        *,
        user_id: str = "alex",
        agent_id: str = "coding-assistant",
        token_budget: int = 2000,
        include_shared: bool = False,
        task_id: str | None = None,
    ) -> RecallResult:
        start = time.perf_counter()
        edges = await self._client.search(query)
        elapsed = (time.perf_counter() - start) * 1000
        memories = [edge.fact for edge in edges if hasattr(edge, "fact") and edge.fact]
        total_text = " ".join(memories)
        total_tokens = int(len(total_text.split()) * 1.3)
        return RecallResult(
            memories=memories,
            formatted_context=None,
            total_tokens=total_tokens,
            latency_ms=elapsed,
            api_calls=1,
        )

    async def reset(self) -> None:
        if self._client:
            try:
                from neo4j import AsyncGraphDatabase

                driver = AsyncGraphDatabase.driver(
                    self._neo4j_uri,
                    auth=(self._neo4j_user, self._neo4j_password),
                )
                async with driver.session() as session:
                    await session.run("MATCH (n) DETACH DELETE n")
                await driver.close()
            except Exception:
                pass
        self._episode_counter = 0

    async def teardown(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
