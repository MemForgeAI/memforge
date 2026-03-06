"""Scoring engine for benchmark evaluation."""

from __future__ import annotations

from benchmark.models import ExpectedMemory


def score_string_match(
    memories: list[str],
    expected: list[ExpectedMemory],
) -> list[bool]:
    """Check if expected memories appear in returned results (case-insensitive substring)."""
    results = []
    memories_lower = [m.lower() for m in memories]
    for exp in expected:
        match_str = exp.content_match.lower()
        found = any(match_str in m for m in memories_lower)
        results.append(found)
    return results


def compute_precision_at_k(relevant_flags: list[bool], k: int) -> float:
    """Precision@k: fraction of top-k results that are relevant."""
    if k == 0:
        return 0.0
    padded = relevant_flags[:k] + [False] * max(0, k - len(relevant_flags))
    return sum(padded[:k]) / k


def compute_recall_at_k(
    relevant_flags: list[bool],
    total_relevant: int,
    k: int,
) -> float:
    """Recall@k: fraction of all relevant items found in top-k results."""
    if total_relevant == 0:
        return 0.0
    found = sum(relevant_flags[:k])
    return found / total_relevant


def compute_mrr(relevant_flags: list[bool]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant result."""
    for i, is_relevant in enumerate(relevant_flags):
        if is_relevant:
            return 1.0 / (i + 1)
    return 0.0


def compute_noise_rejection(memories: list[str], noise_terms: list[str]) -> float:
    """Fraction of noise terms NOT found in returned memories."""
    if not noise_terms:
        return 1.0
    memories_lower = " ".join(memories).lower()
    noise_found = sum(1 for term in noise_terms if term.lower() in memories_lower)
    return 1.0 - (noise_found / len(noise_terms))


def compute_token_efficiency(relevant_tokens: int, total_tokens: int) -> float:
    """Ratio of relevant tokens to total tokens returned."""
    if total_tokens == 0:
        return 0.0
    return relevant_tokens / total_tokens


async def llm_judge_relevance(
    query: str,
    memories: list[str],
    api_key: str | None = None,
) -> list[float]:
    """Use Claude Haiku to score relevance of each memory to the query."""
    if not memories:
        return []

    import anthropic

    client = anthropic.AsyncAnthropic(api_key=api_key) if api_key else anthropic.AsyncAnthropic()

    numbered = "\n".join(f"{i+1}. {m}" for i, m in enumerate(memories))
    prompt = f"""Rate the relevance of each memory to the query on a scale of 0.0 to 1.0.

Query: "{query}"

Memories:
{numbered}

Respond with ONLY a JSON array of numbers, one per memory. Example: [0.9, 0.3, 0.0]"""

    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )

    import json
    text = response.content[0].text.strip()
    scores = json.loads(text)
    return [float(s) for s in scores]
