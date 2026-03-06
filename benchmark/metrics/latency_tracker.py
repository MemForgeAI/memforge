"""Tracks latency distributions (p50, p95) per system."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field


@dataclass
class LatencySummary:
    system: str
    recall_latencies: list[float] = field(default_factory=list)
    store_latencies: list[float] = field(default_factory=list)

    @property
    def recall_p50(self) -> float:
        return statistics.median(self.recall_latencies) if self.recall_latencies else 0.0

    @property
    def recall_p95(self) -> float:
        if not self.recall_latencies:
            return 0.0
        sorted_lat = sorted(self.recall_latencies)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def store_p50(self) -> float:
        return statistics.median(self.store_latencies) if self.store_latencies else 0.0

    @property
    def store_p95(self) -> float:
        if not self.store_latencies:
            return 0.0
        sorted_lat = sorted(self.store_latencies)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]


class LatencyTracker:
    def __init__(self) -> None:
        self._summaries: dict[str, LatencySummary] = {}

    def get_or_create(self, system: str) -> LatencySummary:
        if system not in self._summaries:
            self._summaries[system] = LatencySummary(system=system)
        return self._summaries[system]

    def record_store(self, system: str, latency_ms: float) -> None:
        self.get_or_create(system).store_latencies.append(latency_ms)

    def record_recall(self, system: str, latency_ms: float) -> None:
        self.get_or_create(system).recall_latencies.append(latency_ms)

    def get_summary(self, system: str) -> LatencySummary:
        return self._summaries.get(system, LatencySummary(system=system))
