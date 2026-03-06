"""Mem0 adapter — wraps the mem0ai Python SDK."""

from __future__ import annotations

import asyncio
import time
from functools import partial

from benchmark.adapters.base import MemoryAdapter
from benchmark.models import RecallResult, StoreResult


class Mem0Adapter(MemoryAdapter):
    name = "mem0"

    def __init__(self, config: dict | None = None) -> None:
        self._config = config
        self._memory = None
        self._api_call_count = 0

    async def setup(self) -> float:
        start = time.perf_counter()
        from mem0 import Memory

        if self._config:
            self._memory = Memory.from_config(self._config)
        else:
            self._memory = Memory()
        return time.perf_counter() - start

    def _run_sync(self, fn, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(None, partial(fn, *args, **kwargs))

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
        start = time.perf_counter()
        result = await self._run_sync(
            self._memory.add,
            content,
            user_id=user_id,
            agent_id=agent_id,
        )
        elapsed = (time.perf_counter() - start) * 1000
        results = result.get("results", [])
        memory_id = results[0]["id"] if results else None
        return StoreResult(
            memory_id=memory_id,
            api_calls=2,
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
        result = await self._run_sync(
            self._memory.search,
            query,
            user_id=user_id,
            limit=20,
        )
        elapsed = (time.perf_counter() - start) * 1000
        results = result.get("results", [])
        memories = [r["memory"] for r in results if "memory" in r]
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
        if self._memory:
            try:
                await self._run_sync(self._memory.delete_all, user_id="alex")
            except Exception:
                pass

    async def teardown(self) -> None:
        self._memory = None
