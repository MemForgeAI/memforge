-- BM25 full-text search support via tsvector.
-- Includes both memory content AND LLM-generated query_hints for better matching.
-- The GENERATED ALWAYS column auto-populates on INSERT/UPDATE.
-- GIN index enables fast @@ (full-text match) queries.

ALTER TABLE memories
  ADD COLUMN IF NOT EXISTS content_tsv tsvector
  GENERATED ALWAYS AS (
    to_tsvector('english', content) ||
    to_tsvector('english', COALESCE(metadata->>'query_hints', ''))
  ) STORED;

CREATE INDEX IF NOT EXISTS idx_memories_content_tsv
  ON memories USING gin(content_tsv);
