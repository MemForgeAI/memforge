/**
 * Memory decay — reduce importance of stale, unaccessed memories.
 *
 * Formula: importance *= 0.95 ^ days_since_last_access
 *
 * Runs as a background cron job (default: every DECAY_INTERVAL_HOURS).
 * Memories below 0.05 importance are archived (given a 30-day expiry).
 *
 * This ensures the memory store stays clean over time:
 *   - Frequently accessed memories maintain their importance
 *   - Stale, unimportant memories gradually fade
 *   - Safety-critical memories (importance=1.0) decay very slowly
 */

import { applyDecay, archiveStaleMemories } from "../db/queries.js";

export interface DecayResult {
  memories_decayed: number;
  memories_archived: number;
}

/**
 * Run one decay cycle.
 *
 * @param decayFactor - Base decay rate per day (default: 0.95)
 * @param minImportance - Threshold below which memories get archived (default: 0.05)
 */
export async function runDecay(
  decayFactor: number = 0.95,
  minImportance: number = 0.05,
): Promise<DecayResult> {
  // 1. Apply decay to stale memories
  const memoriesDecayed = await applyDecay(decayFactor, minImportance);

  // 2. Archive memories that fell below threshold
  const memoriesArchived = await archiveStaleMemories(minImportance);

  return {
    memories_decayed: memoriesDecayed,
    memories_archived: memoriesArchived,
  };
}
