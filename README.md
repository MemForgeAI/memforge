<div align="center">

# MemForge

### The open-source memory layer for AI agents.

One database. One MCP server. Four tools.

[![License: BSL 1.1](https://img.shields.io/badge/License-BSL_1.1-blue.svg)](LICENSE)
[![PostgreSQL 17](https://img.shields.io/badge/PostgreSQL-17-336791.svg)](https://www.postgresql.org/)
[![MCP](https://img.shields.io/badge/MCP-compatible-blue.svg)](https://modelcontextprotocol.io/)
[![CI](https://github.com/MemForgeAI/memforge/actions/workflows/ci.yml/badge.svg)](https://github.com/MemForgeAI/memforge/actions)

</div>

---

AI agents are stateless. Every LLM call starts from zero. **MemForge fixes that.**

Your agent talks to MemForge through MCP — the same protocol it uses for every other tool. It remembers what it learns, recalls what's relevant, reflects on patterns, and forgets what's outdated. No SDK. No integration code. One URL.

```
┌─────────────────────────────────────────────┐
│  Your AI Agent (Claude, Cursor, custom)     │
│  Just adds the MemForge MCP server URL.     │
├─────────────────────────────────────────────┤
│  MEMFORGE MCP SERVER                        │
│  Tools: remember | recall | reflect | forget│
├─────────────────────────────────────────────┤
│  CONTEXT ENGINE                             │
│  embed → extract → dedup → rank → pack      │
├─────────────────────────────────────────────┤
│  PostgreSQL 17                              │
│  pgvector (semantic) + AGE (graph) + JSONB  │
└─────────────────────────────────────────────┘
```

## Benchmarks

MemForge is **#1 on the LOCOMO benchmark** — the standard evaluation for long-term conversational memory systems.

| System | Overall Score |
|--------|:------------:|
| **MemForge** | **92.8%** |
| MemMachine v0.2 | 91.2% |
| Memobase v0.0.37 | 75.8% |
| Zep (updated) | 75.1% |
| Mem0 | 66.9% |

Mean of 5 runs: **92.9% +/- 1.0%** (min 91.4%, max 93.4%).

## Quick Start

### 1. Start MemForge

```bash
git clone https://github.com/MemForgeAI/memforge.git
cd memforge
docker compose up -d --build --wait
```

No API keys needed. MemForge uses a local embedding model by default (`all-MiniLM-L6-v2`, 384 dims, runs on CPU).

### 2. Connect Your Agent

**Claude Desktop / Cursor / Windsurf** — add to MCP config:

```json
{
  "mcpServers": {
    "memforge": {
      "url": "http://localhost:3100/mcp"
    }
  }
}
```

That's it. Your agent now has memory.

### 3. Demo

```bash
./demo.sh
```

Stores memories, recalls context, detects a duplicate, and cleans up — full lifecycle in one script.

## The Four Tools

| Tool | What It Does |
|------|-------------|
| `remember` | Store an observation. MemForge auto-classifies, embeds, extracts entities, detects duplicates, and builds graph relationships. |
| `recall` | Retrieve assembled context. Not a list of memories — a **compiled, token-budgeted context document** grouped by type and ranked by relevance. |
| `reflect` | Generate higher-order insights from recent observations. Turns raw experiences into durable knowledge. |
| `forget` | Remove a memory and clean up associated graph edges. For privacy or corrections. |

### Example: Remember

```json
{
  "method": "tools/call",
  "params": {
    "name": "remember",
    "arguments": {
      "content": "User is allergic to shellfish",
      "user_id": "user-123",
      "importance": 1.0
    }
  }
}
```

### Example: Recall

```json
{
  "method": "tools/call",
  "params": {
    "name": "recall",
    "arguments": {
      "query": "dietary restrictions for dinner reservation",
      "user_id": "user-123",
      "token_budget": 500
    }
  }
}
```

Returns a compiled context document:

```
## FACTS
- User is allergic to shellfish

## HISTORY
- User booked a restaurant last week, asked for seafood-free options
```

## How It Works

### Write Path (remember)

1. **Auto-classify** — Rule-based pattern matching (semantic / episodic / procedural)
2. **Embed** — Convert to vector via local model or OpenAI
3. **Dedup** — Cosine similarity check against existing memories
4. **Conflict detect** — Identify contradictions via similarity + arbitration
5. **Extract entities** — NER extracts people, locations, topics
6. **Build relationships** — Entity edges in Apache AGE knowledge graph
7. **Store** — Atomic write: memory row + embedding + graph edges in one transaction

### Read Path (recall)

1. **Embed** the query
2. **Hybrid search** — Vector (pgvector) + BM25 full-text + knowledge graph traversal
3. **Rerank** — Cross-encoder reranking for precision
4. **Score** — `similarity * 0.4 + recency * 0.3 + importance * 0.2 + frequency * 0.1`
5. **Token pack** — Greedy fill within the token budget
6. **Format** — Group by type: `FACTS | HISTORY | PROCEDURES`

## Features

- **MCP-native** — Agent decides when to remember/recall. Zero integration code.
- **Single database** — PostgreSQL with pgvector + Apache AGE. `docker compose up`.
- **Hybrid search** — Vector similarity + BM25 full-text + knowledge graph traversal
- **Agentic recall** — Multi-pass retrieval with LLM gap analysis for complex queries
- **Cross-encoder reranking** — Precision reranking after initial retrieval
- **Auto-classification** — Rule-based (80%) + LLM fallback (20%) memory type detection
- **Entity extraction** — Automatic NER with knowledge graph construction
- **Dedup and conflict detection** — Prevents redundant storage, flags contradictions
- **Importance decay** — Unused memories gradually fade (`importance *= 0.95^days`)
- **Multi-agent shared memory** — Private, user-shared, task-shared, and global scopes
- **Token-budgeted context** — Returns compiled documents, not raw memory lists
- **Local embeddings** — Zero cost with `all-MiniLM-L6-v2`. No API key required.

## Configuration

```bash
# .env — all have sensible defaults
DATABASE_URL=postgresql://memforge:memforge_dev@localhost:5432/memforge
EMBEDDING_PROVIDER=local       # "local" (default), "openai", or "azure"
MEMFORGE_PORT=3100
DEFAULT_TOKEN_BUDGET=4000
DECAY_INTERVAL_HOURS=24
REFLECT_INTERVAL_HOURS=12
```

See [.env.example](.env.example) for all options.

## Storage Architecture

MemForge runs on a single PostgreSQL 17 instance:

```
PostgreSQL 17
├── pgvector 0.8     → Semantic similarity search
├── Apache AGE 1.7   → Knowledge graph (Cypher queries)
└── JSONB            → Flexible metadata
```

One service. One backup. ACID transactions across vector + relational + graph.

## Testing

```bash
# Unit tests (vitest, no Docker needed)
npm test

# Integration tests (pytest + testcontainers, needs Docker)
pip install -e ".[test]"
pytest tests/ -v

# MCP roundtrip tests (needs running server)
npm run test:integration
```

All tests use real Postgres. No mocks.

## Development

```bash
git clone https://github.com/MemForgeAI/memforge.git
cd memforge
npm install
cp .env.example .env
docker compose up postgres -d
npm run dev
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

## How It Compares

| | Mem0 | Zep | MemForge |
|---|---|---|---|
| **Interface** | Python/JS SDK | REST API + SDK | **MCP (zero code)** |
| **Storage** | 3 stores | Neo4j + PostgreSQL | **1 Postgres** |
| **Graph** | Paywalled | Requires Neo4j | **Included (AGE)** |
| **Setup** | pip install + API key + code | Docker (PG + Neo4j) | **`docker compose up`** |
| **Output** | List of memories | Formatted text | **Token-budgeted context** |
| **Embeddings** | Requires API key | Requires API key | **Local (no key)** |
| **LOCOMO** | 66.9% | 75.1% | **92.8%** |

## License

Business Source License 1.1 (BSL 1.1). Free to use and self-host. You cannot offer MemForge as a competing hosted memory service.

Converts to Apache License 2.0 on 2029-03-06.

See [LICENSE](LICENSE) for details.

---

<div align="center">

**MemForge** — Built by [RoamX AI](https://memforge.io)

*The open-source memory layer for AI agents.*

</div>
