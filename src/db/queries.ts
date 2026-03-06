import { query } from './connection.js';

// ============================================================
// Types
// ============================================================

export interface MemoryRow {
  id: string;
  agent_id: string;
  user_id: string | null;
  memory_type: 'semantic' | 'episodic' | 'procedural';
  content: string;
  embedding: string | null;
  confidence: number;
  importance: number;
  source: string;
  shared: boolean;
  task_id: string | null;
  metadata: Record<string, unknown>;
  expires_at: string | null;
  created_at: string;
  updated_at: string;
  access_count: number;
  last_accessed: string | null;
  valid_at: string;
  invalid_at: string | null;
}

export interface MemoryInsert {
  agentId: string;
  userId?: string;
  memoryType: 'semantic' | 'episodic' | 'procedural';
  content: string;
  embedding: number[];
  confidence?: number;
  importance?: number;
  source?: string;
  shared?: boolean;
  taskId?: string;
  metadata?: Record<string, unknown>;
  expiresAt?: Date;
  validAt?: Date;
  invalidAt?: Date;
}

export interface RecallOptions {
  query: string;
  queryEmbedding: number[];
  userId?: string;
  agentId?: string;
  taskId?: string;
  tokenBudget: number;
  recencyWeight: number;
  includeShared?: boolean;
  limit?: number;
}

export interface ScoredMemory extends MemoryRow {
  similarity: number;
  score: number;
  // Hybrid search fields (populated when BM25 is used)
  bm25_rank?: number;
  vector_rank?: number;
  rrf_score?: number;
  match_sources?: ('vector' | 'bm25')[];
  cross_encoder_score?: number;
}

// ============================================================
// Fact Types (atomic facts extracted from memories)
// ============================================================

export interface FactRow {
  id: string;
  memory_id: string;
  fact: string;
  embedding: string | null;
  query_hints: string;
  confidence: number;
  importance: number;
  version: number;
  superseded_by: string | null;
  active_from: string;
  active_until: string | null;
  created_at: string;
}

export interface ScoredFact extends FactRow {
  similarity: number;
  score: number;
  bm25_rank?: number;
  vector_rank?: number;
  rrf_score?: number;
  match_sources?: ('vector' | 'bm25')[];
}

export interface FactInsert {
  memoryId: string;
  fact: string;
  embedding: number[];
  queryHints: string;
  confidence?: number;
  importance?: number;
}

export interface FactRecallOptions {
  query: string;
  queryEmbedding: number[];
  userId?: string;
  agentId?: string;
  taskId?: string;
  includeShared?: boolean;
  limit?: number;
}

// ============================================================
// Write Operations
// ============================================================

/**
 * Insert a new memory with its embedding.
 * Single atomic operation.
 */
export async function insertMemory(mem: MemoryInsert): Promise<MemoryRow> {
  const embeddingStr = `[${mem.embedding.join(',')}]`;

  const result = await query<MemoryRow>(
    `INSERT INTO memories (
      agent_id, user_id, memory_type, content, embedding,
      confidence, importance, source, shared, task_id,
      metadata, expires_at, valid_at
    ) VALUES ($1, $2, $3, $4, $5::vector, $6, $7, $8, $9, $10, $11, $12, $13)
    RETURNING *`,
    [
      mem.agentId,
      mem.userId ?? null,
      mem.memoryType,
      mem.content,
      embeddingStr,
      mem.confidence ?? 1.0,
      mem.importance ?? 0.5,
      mem.source ?? 'agent_observation',
      mem.shared ?? false,
      mem.taskId ?? null,
      JSON.stringify(mem.metadata ?? {}),
      mem.expiresAt ?? null,
      mem.validAt ?? new Date(),
    ],
  );

  return result.rows[0]!;
}

/**
 * Delete a memory by ID.
 * Returns true if a row was actually deleted.
 */
export async function deleteMemory(memoryId: string): Promise<boolean> {
  const result = await query(
    'DELETE FROM memories WHERE id = $1',
    [memoryId],
  );
  return (result.rowCount ?? 0) > 0;
}

/**
 * Update a memory's timestamp and confidence (for dedup hits).
 */
export async function touchMemory(
  memoryId: string,
  confidence?: number,
): Promise<void> {
  await query(
    `UPDATE memories SET
      updated_at = NOW(),
      last_accessed = NOW(),
      access_count = access_count + 1
      ${confidence !== undefined ? ', confidence = $2' : ''}
    WHERE id = $1`,
    confidence !== undefined ? [memoryId, confidence] : [memoryId],
  );
}

// ============================================================
// Read Operations
// ============================================================

interface VisibilityResult {
  conditions: string[];
  params: unknown[];
  nextParamIdx: number;
}

/**
 * Build WHERE conditions for multi-agent memory visibility.
 * Shared between vector search and keyword search.
 */
function buildVisibilityScope(
  opts: Pick<RecallOptions, 'agentId' | 'userId' | 'taskId' | 'includeShared'>,
  startParamIdx: number,
  startParams: unknown[],
): VisibilityResult {
  const conditions: string[] = [
    '(expires_at IS NULL OR expires_at > NOW())',
    '(invalid_at IS NULL OR invalid_at > NOW())',
  ];
  const params = [...startParams];
  let paramIdx = startParamIdx;

  const scopeFilters: string[] = [];

  if (opts.agentId) {
    const agentParam = paramIdx++;
    params.push(opts.agentId);

    if (opts.userId) {
      const userParam = paramIdx++;
      params.push(opts.userId);
      scopeFilters.push(`(agent_id = $${agentParam} AND user_id = $${userParam})`);

      if (opts.includeShared) {
        scopeFilters.push(`(user_id = $${userParam} AND shared = TRUE)`);
      }
    } else {
      scopeFilters.push(`(agent_id = $${agentParam})`);
    }
  } else if (opts.userId) {
    const userParam = paramIdx++;
    params.push(opts.userId);
    scopeFilters.push(`(user_id = $${userParam})`);

    if (opts.includeShared) {
      scopeFilters.push(`(user_id = $${userParam} AND shared = TRUE)`);
    }
  }

  if (opts.taskId) {
    const taskParam = paramIdx++;
    params.push(opts.taskId);
    scopeFilters.push(`(task_id = $${taskParam} AND shared = TRUE)`);
  }

  if (opts.includeShared && !opts.userId && !opts.taskId) {
    scopeFilters.push('(shared = TRUE)');
  }

  if (scopeFilters.length > 0) {
    conditions.push(`(${scopeFilters.join(' OR ')})`);
  }

  return { conditions, params, nextParamIdx: paramIdx };
}

/**
 * Semantic search: find memories similar to a query embedding.
 * Uses pgvector cosine distance operator (<=>).
 * Returns top N results with similarity scores.
 */
export async function searchByEmbedding(
  opts: RecallOptions,
): Promise<ScoredMemory[]> {
  const embeddingStr = `[${opts.queryEmbedding.join(',')}]`;
  const limit = opts.limit ?? 30;

  // Build visibility-scoped WHERE clause
  const { conditions, params } = buildVisibilityScope(
    opts,
    3,  // $1=embedding, $2=limit, so visibility starts at $3
    [embeddingStr, limit],
  );

  const whereClause = conditions.length > 0
    ? `WHERE ${conditions.join(' AND ')}`
    : '';

  const sql = `
    SELECT *,
      1 - (embedding <=> $1::vector) AS similarity
    FROM memories
    ${whereClause}
    ORDER BY embedding <=> $1::vector
    LIMIT $2
  `;

  const result = await query<ScoredMemory>(sql, params);

  // Update access counts
  if (result.rows.length > 0) {
    const ids = result.rows.map((r) => r.id);
    await query(
      `UPDATE memories SET
        access_count = access_count + 1,
        last_accessed = NOW()
      WHERE id = ANY($1)`,
      [ids],
    );
  }

  return result.rows;
}

/**
 * BM25 keyword search using PostgreSQL full-text search.
 * Returns memories ranked by keyword relevance (ts_rank).
 * Used alongside vector search for hybrid retrieval.
 */
export async function searchByKeyword(
  opts: RecallOptions,
): Promise<ScoredMemory[]> {
  const limit = opts.limit ?? 30;

  // Build visibility-scoped WHERE clause
  // $1 = query text, $2 = limit, visibility starts at $3
  const { conditions, params } = buildVisibilityScope(
    opts,
    3,
    [opts.query, limit],
  );

  // Add the full-text match condition
  conditions.push("content_tsv @@ plainto_tsquery('english', $1)");

  const whereClause = `WHERE ${conditions.join(' AND ')}`;

  const sql = `
    SELECT *,
      ts_rank(content_tsv, plainto_tsquery('english', $1)) AS similarity,
      0 AS score
    FROM memories
    ${whereClause}
    ORDER BY ts_rank(content_tsv, plainto_tsquery('english', $1)) DESC
    LIMIT $2
  `;

  const result = await query<ScoredMemory>(sql, params);
  return result.rows;
}

/**
 * Find memories with high cosine similarity to a given embedding.
 * Used for dedup and conflict detection.
 */
export async function findSimilar(
  embedding: number[],
  userId: string | undefined,
  threshold: number,
  limit: number = 5,
): Promise<ScoredMemory[]> {
  const embeddingStr = `[${embedding.join(',')}]`;

  const conditions: string[] = [
    '(expires_at IS NULL OR expires_at > NOW())',
  ];
  const params: unknown[] = [embeddingStr, threshold, limit];
  let paramIdx = 4;

  if (userId) {
    conditions.push(`user_id = $${paramIdx}`);
    params.push(userId);
    paramIdx += 1;
  }

  const whereClause = conditions.length > 0
    ? `AND ${conditions.join(' AND ')}`
    : '';

  const sql = `
    SELECT *,
      1 - (embedding <=> $1::vector) AS similarity
    FROM memories
    WHERE 1 - (embedding <=> $1::vector) >= $2
    ${whereClause}
    ORDER BY similarity DESC
    LIMIT $3
  `;

  const result = await query<ScoredMemory>(sql, params);
  return result.rows;
}

/**
 * Get a single memory by ID.
 */
export async function getMemory(memoryId: string): Promise<MemoryRow | null> {
  const result = await query<MemoryRow>(
    'SELECT * FROM memories WHERE id = $1',
    [memoryId],
  );
  return result.rows[0] ?? null;
}

// ============================================================
// Conflict Operations
// ============================================================

export interface ConflictRow {
  id: string;
  memory_a_id: string;
  memory_b_id: string;
  conflict_type: 'contradiction' | 'outdated' | 'duplicate';
  resolution: 'pending' | 'a_wins' | 'b_wins' | 'merged' | 'manual';
  created_at: string;
  resolved_at: string | null;
}

/**
 * Record a conflict between two memories.
 */
export async function insertConflict(
  memoryAId: string,
  memoryBId: string,
  conflictType: ConflictRow['conflict_type'],
  resolution: ConflictRow['resolution'] = 'pending',
): Promise<ConflictRow> {
  const result = await query<ConflictRow>(
    `INSERT INTO memory_conflicts (memory_a_id, memory_b_id, conflict_type, resolution)
     VALUES ($1, $2, $3, $4)
     RETURNING *`,
    [memoryAId, memoryBId, conflictType, resolution],
  );
  return result.rows[0]!;
}

/**
 * Resolve a conflict.
 */
export async function resolveConflict(
  conflictId: string,
  resolution: ConflictRow['resolution'],
): Promise<void> {
  await query(
    `UPDATE memory_conflicts SET resolution = $2, resolved_at = NOW() WHERE id = $1`,
    [conflictId, resolution],
  );
}

/**
 * Find pending conflicts for a memory.
 */
export async function findPendingConflicts(
  memoryId: string,
): Promise<ConflictRow[]> {
  const result = await query<ConflictRow>(
    `SELECT * FROM memory_conflicts
     WHERE (memory_a_id = $1 OR memory_b_id = $1)
       AND resolution = 'pending'
     ORDER BY created_at DESC`,
    [memoryId],
  );
  return result.rows;
}

/**
 * Invalidate a memory — mark it as no longer true.
 * The memory still exists for audit but won't appear in recall.
 */
export async function invalidateMemory(
  memoryId: string,
  invalidAt: Date = new Date(),
): Promise<void> {
  await query(
    `UPDATE memories SET
      invalid_at = $2,
      updated_at = NOW()
    WHERE id = $1 AND invalid_at IS NULL`,
    [memoryId, invalidAt],
  );
}

/**
 * Update a memory's content (for elaboration merges).
 */
export async function updateMemoryContent(
  memoryId: string,
  content: string,
  embedding: number[],
): Promise<void> {
  const embeddingStr = `[${embedding.join(',')}]`;
  await query(
    `UPDATE memories SET
      content = $2,
      embedding = $3::vector,
      updated_at = NOW()
    WHERE id = $1`,
    [memoryId, content, embeddingStr],
  );
}

// ============================================================
// Reflect Operations
// ============================================================

/**
 * Get recent episodic memories for reflection.
 * Used by the reflect tool to gather raw experiences to consolidate.
 */
export async function getRecentEpisodicMemories(
  userId: string | undefined,
  lookbackHours: number,
  limit: number = 50,
): Promise<MemoryRow[]> {
  const conditions: string[] = [
    "memory_type = 'episodic'",
    `created_at > NOW() - INTERVAL '${lookbackHours} hours'`,
    '(expires_at IS NULL OR expires_at > NOW())',
  ];
  const params: unknown[] = [limit];
  let paramIdx = 2;

  if (userId) {
    conditions.push(`user_id = $${paramIdx}`);
    params.push(userId);
    paramIdx += 1;
  }

  const result = await query<MemoryRow>(
    `SELECT * FROM memories
     WHERE ${conditions.join(' AND ')}
     ORDER BY created_at DESC
     LIMIT $1`,
    params,
  );

  return result.rows;
}

/**
 * Get all memories (any type) for a user within a lookback window.
 * Used by reflect to consolidate across all memory types.
 */
export async function getRecentMemories(
  userId: string | undefined,
  lookbackHours: number,
  limit: number = 100,
): Promise<MemoryRow[]> {
  const conditions: string[] = [
    `created_at > NOW() - INTERVAL '${lookbackHours} hours'`,
    '(expires_at IS NULL OR expires_at > NOW())',
  ];
  const params: unknown[] = [limit];
  let paramIdx = 2;

  if (userId) {
    conditions.push(`user_id = $${paramIdx}`);
    params.push(userId);
    paramIdx += 1;
  }

  const result = await query<MemoryRow>(
    `SELECT * FROM memories
     WHERE ${conditions.join(' AND ')}
     ORDER BY created_at DESC
     LIMIT $1`,
    params,
  );

  return result.rows;
}

// ============================================================
// Decay Operations
// ============================================================

/**
 * Apply importance decay to stale memories.
 *
 * Formula: importance *= decay_factor ^ days_since_last_access
 * Default: importance *= 0.95 ^ days
 *
 * Returns count of updated memories.
 */
export async function applyDecay(
  decayFactor: number = 0.95,
  minImportance: number = 0.05,
): Promise<number> {
  // Calculate days since last access (or creation if never accessed)
  const result = await query(
    `UPDATE memories
     SET importance = GREATEST(
       $2,
       importance * POWER($1, EXTRACT(EPOCH FROM (NOW() - COALESCE(last_accessed, created_at))) / 86400)
     ),
     updated_at = NOW()
     WHERE importance > $2
       AND (expires_at IS NULL OR expires_at > NOW())
       AND COALESCE(last_accessed, created_at) < NOW() - INTERVAL '1 day'`,
    [decayFactor, minImportance],
  );

  return result.rowCount ?? 0;
}

/**
 * Archive (soft-delete via expiry) memories below minimum importance threshold.
 * Returns count of archived memories.
 */
export async function archiveStaleMemories(
  threshold: number = 0.05,
): Promise<number> {
  const result = await query(
    `UPDATE memories
     SET expires_at = NOW() + INTERVAL '30 days'
     WHERE importance <= $1
       AND (expires_at IS NULL OR expires_at > NOW() + INTERVAL '30 days')`,
    [threshold],
  );

  return result.rowCount ?? 0;
}

/**
 * Get low-importance memories that could be consolidated.
 */
export async function getLowImportanceMemories(
  userId: string | undefined,
  threshold: number = 0.2,
  limit: number = 50,
): Promise<MemoryRow[]> {
  const conditions: string[] = [
    `importance <= $2`,
    '(expires_at IS NULL OR expires_at > NOW())',
  ];
  const params: unknown[] = [limit, threshold];
  let paramIdx = 3;

  if (userId) {
    conditions.push(`user_id = $${paramIdx}`);
    params.push(userId);
    paramIdx += 1;
  }

  const result = await query<MemoryRow>(
    `SELECT * FROM memories
     WHERE ${conditions.join(' AND ')}
     ORDER BY importance ASC, last_accessed ASC NULLS FIRST
     LIMIT $1`,
    params,
  );

  return result.rows;
}

// ============================================================
// Fact Operations
// ============================================================

/**
 * Build WHERE conditions for fact visibility.
 * JOINs memory_facts to memories for agent/user/shared/expiry filtering.
 * Uses 'm.' prefix for parent memory columns and 'f.' for fact columns.
 */
function buildFactVisibilityScope(
  opts: Pick<FactRecallOptions, 'agentId' | 'userId' | 'taskId' | 'includeShared'>,
  startParamIdx: number,
  startParams: unknown[],
): VisibilityResult {
  const conditions: string[] = [
    '(m.expires_at IS NULL OR m.expires_at > NOW())',
    '(m.invalid_at IS NULL OR m.invalid_at > NOW())',
    'f.active_until IS NULL',
  ];
  const params = [...startParams];
  let paramIdx = startParamIdx;

  const scopeFilters: string[] = [];

  if (opts.agentId) {
    const agentParam = paramIdx++;
    params.push(opts.agentId);

    if (opts.userId) {
      const userParam = paramIdx++;
      params.push(opts.userId);
      scopeFilters.push(`(m.agent_id = $${agentParam} AND m.user_id = $${userParam})`);

      if (opts.includeShared) {
        scopeFilters.push(`(m.user_id = $${userParam} AND m.shared = TRUE)`);
      }
    } else {
      scopeFilters.push(`(m.agent_id = $${agentParam})`);
    }
  } else if (opts.userId) {
    const userParam = paramIdx++;
    params.push(opts.userId);
    scopeFilters.push(`(m.user_id = $${userParam})`);

    if (opts.includeShared) {
      scopeFilters.push(`(m.user_id = $${userParam} AND m.shared = TRUE)`);
    }
  }

  if (opts.taskId) {
    const taskParam = paramIdx++;
    params.push(opts.taskId);
    scopeFilters.push(`(m.task_id = $${taskParam} AND m.shared = TRUE)`);
  }

  if (opts.includeShared && !opts.userId && !opts.taskId) {
    scopeFilters.push('(m.shared = TRUE)');
  }

  if (scopeFilters.length > 0) {
    conditions.push(`(${scopeFilters.join(' OR ')})`);
  }

  return { conditions, params, nextParamIdx: paramIdx };
}

/**
 * Insert a new atomic fact linked to a parent memory.
 */
export async function insertFact(fact: FactInsert): Promise<FactRow> {
  const embeddingStr = `[${fact.embedding.join(',')}]`;

  const result = await query<FactRow>(
    `INSERT INTO memory_facts (
      memory_id, fact, embedding, query_hints, confidence, importance
    ) VALUES ($1, $2, $3::vector, $4, $5, $6)
    RETURNING *`,
    [
      fact.memoryId,
      fact.fact,
      embeddingStr,
      fact.queryHints,
      fact.confidence ?? 1.0,
      fact.importance ?? 0.5,
    ],
  );

  return result.rows[0]!;
}

/**
 * Vector search on memory_facts with visibility filtering via parent memory.
 */
export async function searchFactsByEmbedding(
  opts: FactRecallOptions,
): Promise<ScoredFact[]> {
  const embeddingStr = `[${opts.queryEmbedding.join(',')}]`;
  const limit = opts.limit ?? 30;

  const { conditions, params } = buildFactVisibilityScope(
    opts,
    3, // $1=embedding, $2=limit, visibility starts at $3
    [embeddingStr, limit],
  );

  const whereClause = conditions.length > 0
    ? `WHERE ${conditions.join(' AND ')}`
    : '';

  const sql = `
    SELECT f.*,
      1 - (f.embedding <=> $1::vector) AS similarity
    FROM memory_facts f
    JOIN memories m ON f.memory_id = m.id
    ${whereClause}
    ORDER BY f.embedding <=> $1::vector
    LIMIT $2
  `;

  const result = await query<ScoredFact>(sql, params);
  return result.rows;
}

/**
 * BM25 keyword search on memory_facts with visibility filtering.
 */
export async function searchFactsByKeyword(
  opts: FactRecallOptions,
): Promise<ScoredFact[]> {
  const limit = opts.limit ?? 30;

  const { conditions, params } = buildFactVisibilityScope(
    opts,
    3, // $1=query, $2=limit, visibility starts at $3
    [opts.query, limit],
  );

  conditions.push("f.fact_tsv @@ plainto_tsquery('english', $1)");

  const whereClause = `WHERE ${conditions.join(' AND ')}`;

  const sql = `
    SELECT f.*,
      ts_rank(f.fact_tsv, plainto_tsquery('english', $1)) AS similarity,
      0 AS score
    FROM memory_facts f
    JOIN memories m ON f.memory_id = m.id
    ${whereClause}
    ORDER BY ts_rank(f.fact_tsv, plainto_tsquery('english', $1)) DESC
    LIMIT $2
  `;

  const result = await query<ScoredFact>(sql, params);
  return result.rows;
}

/**
 * Find facts with high cosine similarity to a given embedding.
 * Used for fact-level dedup.
 */
export async function findSimilarFacts(
  embedding: number[],
  userId: string | undefined,
  threshold: number,
  limit: number = 5,
): Promise<ScoredFact[]> {
  const embeddingStr = `[${embedding.join(',')}]`;

  const conditions: string[] = [
    'f.active_until IS NULL',
  ];
  const params: unknown[] = [embeddingStr, threshold, limit];
  let paramIdx = 4;

  if (userId) {
    conditions.push(`m.user_id = $${paramIdx}`);
    params.push(userId);
    paramIdx += 1;
  }

  const whereClause = conditions.length > 0
    ? `AND ${conditions.join(' AND ')}`
    : '';

  const sql = `
    SELECT f.*,
      1 - (f.embedding <=> $1::vector) AS similarity
    FROM memory_facts f
    JOIN memories m ON f.memory_id = m.id
    WHERE 1 - (f.embedding <=> $1::vector) >= $2
    ${whereClause}
    ORDER BY similarity DESC
    LIMIT $3
  `;

  const result = await query<ScoredFact>(sql, params);
  return result.rows;
}

/**
 * Supersede an old fact with a new one.
 * Sets active_until and superseded_by on the old fact.
 */
export async function supersedeFact(
  oldFactId: string,
  newFactId: string,
): Promise<void> {
  await query(
    `UPDATE memory_facts SET
      active_until = NOW(),
      superseded_by = $2
    WHERE id = $1 AND active_until IS NULL`,
    [oldFactId, newFactId],
  );
}

/**
 * Check if any active facts exist for a user/agent scope.
 * Used to decide whether to use fact-first or memory-fallback recall.
 */
export async function hasAnyFacts(
  userId?: string,
  agentId?: string,
): Promise<boolean> {
  const conditions: string[] = ['f.active_until IS NULL'];
  const params: unknown[] = [];
  let paramIdx = 1;

  if (agentId) {
    conditions.push(`m.agent_id = $${paramIdx}`);
    params.push(agentId);
    paramIdx += 1;
  }
  if (userId) {
    conditions.push(`m.user_id = $${paramIdx}`);
    params.push(userId);
    paramIdx += 1;
  }

  const whereClause = conditions.join(' AND ');

  const result = await query<{ exists: boolean }>(
    `SELECT EXISTS(
      SELECT 1 FROM memory_facts f
      JOIN memories m ON f.memory_id = m.id
      WHERE ${whereClause}
    ) AS exists`,
    params,
  );

  return result.rows[0]?.exists ?? false;
}

/**
 * Get all fact IDs for a given memory.
 * Used by forget tool for graph edge cleanup before CASCADE delete.
 */
export async function getFactIdsForMemory(
  memoryId: string,
): Promise<string[]> {
  const result = await query<{ id: string }>(
    'SELECT id FROM memory_facts WHERE memory_id = $1',
    [memoryId],
  );
  return result.rows.map((r) => r.id);
}

/**
 * Vector search on facts filtered to a specific set of entity-linked fact IDs.
 * Replaces the N+1 pattern (findFactsByEntity → loop → getFactById) with a
 * single ranked query. Returns facts ordered by similarity to the query.
 */
export async function searchFactsByEntityEmbedding(
  entityFactIds: string[],
  queryEmbedding: number[],
  limit: number,
): Promise<ScoredFact[]> {
  if (entityFactIds.length === 0) return [];

  const embeddingStr = `[${queryEmbedding.join(',')}]`;

  const sql = `
    SELECT f.*, m.created_at as memory_created_at,
      1 - (f.embedding <=> $1::vector) AS similarity
    FROM memory_facts f
    JOIN memories m ON f.memory_id = m.id
    WHERE f.id = ANY($3)
      AND f.active_until IS NULL
    ORDER BY f.embedding <=> $1::vector
    LIMIT $2
  `;

  const result = await query<ScoredFact>(sql, [embeddingStr, limit, entityFactIds]);
  return result.rows;
}

/**
 * Get a single fact by ID.
 * Used by entity-aware fact retrieval to hydrate graph-discovered fact IDs.
 */
export async function getFactById(
  factId: string,
): Promise<ScoredFact | null> {
  const result = await query<FactRow>(
    `SELECT f.*, m.created_at as memory_created_at
     FROM memory_facts f
     JOIN memories m ON f.memory_id = m.id
     WHERE f.id = $1 AND f.active_until IS NULL`,
    [factId],
  );
  if (result.rows.length === 0) return null;
  const row = result.rows[0]!;
  return {
    ...row,
    similarity: 0,
    score: 0,
  };
}
