-- Migration 005: Add memory_facts table for atomic fact storage.
-- Each atomic fact from a Memory Card becomes its own searchable row
-- with its own embedding, enabling focused retrieval without dilution.

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

CREATE INDEX IF NOT EXISTS idx_facts_tsv ON memory_facts USING gin(fact_tsv);
CREATE INDEX IF NOT EXISTS idx_facts_memory_id ON memory_facts(memory_id);
CREATE INDEX IF NOT EXISTS idx_facts_active ON memory_facts(active_until) WHERE active_until IS NULL;
