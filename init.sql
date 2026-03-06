-- MemForge: The Open-Source Memory Layer for AI Agents
-- Schema initialization script
-- Run once on database creation via docker-entrypoint-initdb.d

-- ============================================================
-- Extensions
-- ============================================================
CREATE EXTENSION IF NOT EXISTS vector;       -- pgvector for semantic search
CREATE EXTENSION IF NOT EXISTS age;          -- Apache AGE for knowledge graph
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";  -- UUID generation fallback

-- Load AGE into search path
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

-- ============================================================
-- Table: memories
-- Primary storage for all agent knowledge.
-- ============================================================
CREATE TABLE IF NOT EXISTS memories (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  agent_id        VARCHAR(64) NOT NULL,
  user_id         VARCHAR(64),
  memory_type     VARCHAR(20) NOT NULL
                  CHECK (memory_type IN ('semantic', 'episodic', 'procedural')),
  content         TEXT NOT NULL,
  embedding       vector,  -- dimension set by embedding provider config
  confidence      FLOAT DEFAULT 1.0
                  CHECK (confidence BETWEEN 0 AND 1),
  importance      FLOAT DEFAULT 0.5
                  CHECK (importance BETWEEN 0 AND 1),
  source          VARCHAR(50) DEFAULT 'agent_observation',
  shared          BOOLEAN DEFAULT FALSE,
  task_id         VARCHAR(64),
  metadata        JSONB DEFAULT '{}',
  expires_at      TIMESTAMPTZ,
  created_at      TIMESTAMPTZ DEFAULT NOW(),
  updated_at      TIMESTAMPTZ DEFAULT NOW(),
  access_count    INT DEFAULT 0,
  last_accessed   TIMESTAMPTZ,
  valid_at        TIMESTAMPTZ DEFAULT NOW(),
  invalid_at      TIMESTAMPTZ,
  content_tsv     tsvector GENERATED ALWAYS AS (
    to_tsvector('english', content) ||
    to_tsvector('english', COALESCE(metadata->>'query_hints', ''))
  ) STORED
);

-- ============================================================
-- Table: entities
-- Real-world things agents have learned about.
-- Become nodes in the knowledge graph.
-- ============================================================
CREATE TABLE IF NOT EXISTS entities (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name            VARCHAR(255) NOT NULL,
  entity_type     VARCHAR(50) NOT NULL,
  attributes      JSONB DEFAULT '{}',
  embedding       vector,  -- dimension set by embedding provider config
  created_at      TIMESTAMPTZ DEFAULT NOW(),
  updated_at      TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(name, entity_type)
);

-- ============================================================
-- Table: sessions
-- Active conversation state (working memory).
-- ============================================================
CREATE TABLE IF NOT EXISTS sessions (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  agent_id        VARCHAR(64) NOT NULL,
  user_id         VARCHAR(64),
  status          VARCHAR(20) DEFAULT 'active'
                  CHECK (status IN ('active', 'paused', 'completed')),
  context         JSONB DEFAULT '{}',
  tool_outputs    JSONB DEFAULT '[]',
  token_count     INT DEFAULT 0,
  created_at      TIMESTAMPTZ DEFAULT NOW(),
  updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- Table: memory_conflicts
-- Tracks contradictions and duplicates for resolution.
-- ============================================================
CREATE TABLE IF NOT EXISTS memory_conflicts (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  memory_a_id     UUID REFERENCES memories(id) ON DELETE CASCADE,
  memory_b_id     UUID REFERENCES memories(id) ON DELETE CASCADE,
  conflict_type   VARCHAR(30) NOT NULL
                  CHECK (conflict_type IN ('contradiction', 'outdated', 'duplicate')),
  resolution      VARCHAR(30) DEFAULT 'pending'
                  CHECK (resolution IN ('pending', 'a_wins', 'b_wins', 'merged', 'manual')),
  resolved_at     TIMESTAMPTZ,
  created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- Table: memory_facts
-- Atomic facts extracted from memories. Each fact has its own
-- embedding for focused retrieval (no dilution from multi-topic
-- memories). Linked to parent memory via FK with CASCADE delete.
-- ============================================================
CREATE TABLE IF NOT EXISTS memory_facts (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  memory_id     UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  fact          TEXT NOT NULL,
  embedding     vector,
  query_hints   TEXT DEFAULT '',
  fact_tsv      tsvector GENERATED ALWAYS AS (
    to_tsvector('english', fact) || to_tsvector('english', COALESCE(query_hints, ''))
  ) STORED,
  confidence    FLOAT DEFAULT 1.0 CHECK (confidence BETWEEN 0 AND 1),
  importance    FLOAT DEFAULT 0.5 CHECK (importance BETWEEN 0 AND 1),
  version       INT DEFAULT 1,
  superseded_by UUID REFERENCES memory_facts(id) ON DELETE SET NULL,
  active_from   TIMESTAMPTZ DEFAULT NOW(),
  active_until  TIMESTAMPTZ DEFAULT NULL,
  created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- Indexes
-- ============================================================

-- Composite index for most common query pattern
CREATE INDEX IF NOT EXISTS idx_memories_agent_user_type
  ON memories(agent_id, user_id, memory_type);

-- Multi-agent: find shared memories for a user
CREATE INDEX IF NOT EXISTS idx_memories_shared
  ON memories(user_id, shared) WHERE shared = TRUE;

-- Task-scoped: find memories for a specific task
CREATE INDEX IF NOT EXISTS idx_memories_task
  ON memories(task_id) WHERE task_id IS NOT NULL;

-- Ranking and decay
CREATE INDEX IF NOT EXISTS idx_memories_importance
  ON memories(importance DESC);

CREATE INDEX IF NOT EXISTS idx_memories_created
  ON memories(created_at DESC);

-- Expiry filtering
CREATE INDEX IF NOT EXISTS idx_memories_expires
  ON memories(expires_at) WHERE expires_at IS NOT NULL;

-- Entity lookup
CREATE INDEX IF NOT EXISTS idx_entities_type
  ON entities(entity_type);

-- Flexible metadata queries
CREATE INDEX IF NOT EXISTS idx_memories_metadata
  ON memories USING gin(metadata);

-- Temporal validity
CREATE INDEX IF NOT EXISTS idx_memories_valid_at
  ON memories(valid_at);

CREATE INDEX IF NOT EXISTS idx_memories_invalid_at
  ON memories(invalid_at) WHERE invalid_at IS NOT NULL;

-- BM25 full-text search
CREATE INDEX IF NOT EXISTS idx_memories_content_tsv
  ON memories USING gin(content_tsv);

-- Memory facts: full-text search
CREATE INDEX IF NOT EXISTS idx_facts_tsv
  ON memory_facts USING gin(fact_tsv);

-- Memory facts: parent memory lookup
CREATE INDEX IF NOT EXISTS idx_facts_memory_id
  ON memory_facts(memory_id);

-- Memory facts: active facts only (partial index)
CREATE INDEX IF NOT EXISTS idx_facts_active
  ON memory_facts(active_until) WHERE active_until IS NULL;

-- ============================================================
-- Apache AGE Graph
-- ============================================================
SELECT create_graph('memforge_kg');

-- ============================================================
-- Note: Vector indexes (HNSW/IVFFlat) are created AFTER first
-- data insert, since they require knowing the vector dimension.
-- See migrations/002_create_vector_indexes.sql
-- ============================================================
