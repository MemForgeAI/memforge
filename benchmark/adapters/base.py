"""Abstract base class for memory system adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from benchmark.models import RecallResult, StoreResult


class MemoryAdapter(ABC):
    """Common interface for all memory systems under test."""

    name: str

    @abstractmethod
    async def setup(self) -> float:
        """Initialize the system. Returns setup time in seconds."""

    @abstractmethod
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
        """Store a single memory observation."""

    @abstractmethod
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
        """Retrieve memories for a query."""

    @abstractmethod
    async def reset(self) -> None:
        """Clear all stored data. Used between benchmark runs."""

    @abstractmethod
    async def teardown(self) -> None:
        """Release resources (connections, processes)."""
