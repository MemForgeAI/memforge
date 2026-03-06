"""Tracks API call counts and estimated costs per system."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostSummary:
    system: str
    total_store_api_calls: int = 0
    total_recall_api_calls: int = 0
    total_store_latency_ms: float = 0.0
    total_recall_latency_ms: float = 0.0
    store_count: int = 0
    recall_count: int = 0

    @property
    def avg_store_latency_ms(self) -> float:
        return self.total_store_latency_ms / self.store_count if self.store_count else 0.0

    @property
    def avg_recall_latency_ms(self) -> float:
        return self.total_recall_latency_ms / self.recall_count if self.recall_count else 0.0

    @property
    def cost_per_1000_ops(self) -> float:
        total_ops = self.store_count + self.recall_count
        if total_ops == 0:
            return 0.0
        total_api_calls = self.total_store_api_calls + self.total_recall_api_calls
        cost_per_api_call = 0.0005
        total_cost = total_api_calls * cost_per_api_call
        return (total_cost / total_ops) * 1000


class CostTracker:
    def __init__(self) -> None:
        self._summaries: dict[str, CostSummary] = {}

    def get_or_create(self, system: str) -> CostSummary:
        if system not in self._summaries:
            self._summaries[system] = CostSummary(system=system)
        return self._summaries[system]

    def record_store(self, system: str, api_calls: int, latency_ms: float) -> None:
        s = self.get_or_create(system)
        s.total_store_api_calls += api_calls
        s.total_store_latency_ms += latency_ms
        s.store_count += 1

    def record_recall(self, system: str, api_calls: int, latency_ms: float) -> None:
        s = self.get_or_create(system)
        s.total_recall_api_calls += api_calls
        s.total_recall_latency_ms += latency_ms
        s.recall_count += 1

    def get_summary(self, system: str) -> CostSummary:
        return self._summaries.get(system, CostSummary(system=system))

    def all_summaries(self) -> list[CostSummary]:
        return list(self._summaries.values())
