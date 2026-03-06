# MemForge Benchmark Suite

Automated comparison of MemForge vs Mem0 vs Graphiti (Zep) on a 50-conversation coding assistant scenario.

## Quick Start

```bash
# 1. Install Python dependencies
cd memforge
pip install httpx mem0ai graphiti-core anthropic tiktoken pytest pytest-asyncio

# 2. Start infrastructure (MemForge + Neo4j)
docker compose -f benchmark/docker-compose.benchmark.yml up -d --wait

# 3. Set API keys
export OPENAI_API_KEY=sk-...          # Required for Mem0 and Graphiti
export ANTHROPIC_API_KEY=sk-ant-...   # Required for LLM judge (8 queries)

# 4. Run the benchmark
python benchmark/run_benchmark.py

# 5. Run MemForge only (no API keys needed)
python benchmark/run_benchmark.py --systems memforge
```

## What It Measures

| Metric | Description |
|--------|-------------|
| Precision@5/10 | Are returned memories relevant? |
| Recall@10 | Were all relevant memories found? |
| MRR | Does the best answer come first? |
| Temporal Correctness | Does it handle "switched from X to Y"? |
| Noise Rejection | Does it filter irrelevant memories? |
| Token Efficiency | Signal-to-noise in context output |
| Cost per 1000 ops | API calls and estimated cost |
| Setup Time | Time from zero to first query |
| Latency p50/p95 | Speed of recall operations |

## Scenario

50 conversations between developer "Alex" and an AI coding assistant over 3 months. Includes:

- Preference changes (VS Code -> Cursor, REST -> GraphQL)
- Team transitions (payments -> auth)
- Scattered preferences across conversations
- Noise conversations (weather, lunch)
- Multi-agent shared memories
- High-importance safety practices

## Results

After running, check:
- `benchmark/results/comparison_table.md` -- Summary comparison
- `benchmark/results/per_query_results.csv` -- Per-query breakdown

## Architecture

```
benchmark/
  seed/
    conversations.json          # 50 pre-written conversations
    ground_truth.json           # 37 queries with expected results
  adapters/
    base.py                     # Abstract adapter interface
    memforge_adapter.py          # MemForge MCP HTTP client
    mem0_adapter.py             # Mem0 Python SDK wrapper
    graphiti_adapter.py         # Graphiti Python SDK wrapper
  metrics/
    scorer.py                   # String match + LLM judge
    cost_tracker.py             # API call counter
    latency_tracker.py          # p50/p95 tracker
  run_benchmark.py              # Main orchestrator
  docker-compose.benchmark.yml  # Infrastructure for all systems
```

## Methodology

See `docs/plans/2026-03-02-benchmark-design.md` for full design rationale.
