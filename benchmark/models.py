"""Shared data models for the benchmark suite."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ScoringMethod(str, Enum):
    STRING_MATCH = "string_match"
    LLM_JUDGE = "llm_judge"


@dataclass
class ExpectedMemory:
    content_match: str
    required: bool = True
    rank: int | None = None
    should_be_superseded: bool = False


@dataclass
class GroundTruthQuery:
    query_id: int
    query: str
    category: str
    expected_memories: list[ExpectedMemory]
    expected_absent: list[str] = field(default_factory=list)
    scoring_method: ScoringMethod = ScoringMethod.STRING_MATCH
    tests: list[str] = field(default_factory=list)
    token_budget: int = 2000
    user_id: str | None = None
    agent_id: str | None = None
    task_id: str | None = None
    include_shared: bool | None = None


@dataclass
class Conversation:
    conversation_id: int
    simulated_date: str
    topic: str
    turns: list[dict[str, str]]
    memories_to_extract: list[str]
    entities: list[str]
    tags: list[str]
    agent_id: str = "coding-assistant"
    user_id: str = "alex"
    importance: float | None = None
    shared: bool = False
    task_id: str | None = None


@dataclass
class StoreResult:
    memory_id: str | None
    api_calls: int
    latency_ms: float


@dataclass
class RecallResult:
    memories: list[str]
    formatted_context: str | None
    total_tokens: int
    latency_ms: float
    api_calls: int


@dataclass
class QueryResult:
    query_id: int
    system: str
    recall_result: RecallResult
    precision_at_5: float | None = None
    precision_at_10: float | None = None
    recall_at_10: float | None = None
    mrr: float | None = None
    temporal_correct: bool | None = None
    noise_rejected: float | None = None
    token_efficiency: float | None = None
