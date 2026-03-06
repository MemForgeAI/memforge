-- Temporal validity columns for time-aware fact retrieval.
-- valid_at: when this fact became true (defaults to creation time)
-- invalid_at: when this fact stopped being true (NULL = still valid)

ALTER TABLE memories ADD COLUMN IF NOT EXISTS valid_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE memories ADD COLUMN IF NOT EXISTS invalid_at TIMESTAMPTZ DEFAULT NULL;

-- Backfill existing memories: valid_at = created_at
UPDATE memories SET valid_at = created_at WHERE valid_at IS NULL;

-- Indexes for temporal filtering
CREATE INDEX IF NOT EXISTS idx_memories_valid_at ON memories (valid_at);
CREATE INDEX IF NOT EXISTS idx_memories_invalid_at ON memories (invalid_at) WHERE invalid_at IS NOT NULL;
