/**
 * Deduplication and conflict detection types.
 *
 * Similarity thresholds (configurable via env):
 *   > DEDUP_THRESHOLD (0.95)  → exact duplicate, update existing
 *   CONFLICT_THRESHOLD (0.80) - DEDUP_THRESHOLD (0.95) → potential conflict, needs arbitration
 *   < CONFLICT_THRESHOLD (0.80) → new memory, no overlap
 */

export type ConflictType = 'contradiction' | 'outdated' | 'duplicate';
export type Resolution = 'pending' | 'a_wins' | 'b_wins' | 'merged' | 'manual';

export interface ConflictRecord {
  memoryAId: string;
  memoryBId: string;
  conflictType: ConflictType;
  resolution: Resolution;
}

export type ArbitrationVerdict =
  | { action: 'new'; reason: string }
  | { action: 'update'; existingId: string; reason: string }
  | { action: 'contradiction'; existingId: string; reason: string }
  | { action: 'elaboration'; existingId: string; reason: string };

export interface SimilarityCandidate {
  id: string;
  content: string;
  similarity: number;
  memoryType: string;
}
