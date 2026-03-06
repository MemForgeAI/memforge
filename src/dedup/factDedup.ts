/**
 * Fact-level tiered deduplication.
 *
 * Three tiers, zero LLM calls:
 *   Tier 1: cosine > 0.92  → exact duplicate  → skip
 *   Tier 2: 0.70 - 0.92    → check temporal.supersedes
 *            → if set: version the old fact (supersede)
 *            → if not: store as new (ambiguous, no LLM)
 *   Tier 3: < 0.70         → new fact → store
 */

import { findSimilarFacts } from '../db/queries.js';

const EXACT_DUPLICATE_THRESHOLD = 0.92;
const AMBIGUOUS_THRESHOLD = 0.70;

export interface DedupResult {
  action: 'store' | 'skip' | 'supersede';
  oldFactId?: string;
}

export interface DedupTemporal {
  supersedes?: string | null;
}

/**
 * Determine whether a new fact should be stored, skipped, or supersede an existing fact.
 */
export async function deduplicateFact(
  embedding: number[],
  _factText: string,
  userId: string | undefined,
  temporal: DedupTemporal | undefined,
): Promise<DedupResult> {
  const similar = await findSimilarFacts(
    embedding,
    userId,
    AMBIGUOUS_THRESHOLD,
    1,
  );

  if (similar.length === 0) {
    // Tier 3: no similar facts → store as new
    return { action: 'store' };
  }

  const closest = similar[0]!;

  if (closest.similarity > EXACT_DUPLICATE_THRESHOLD) {
    // Tier 1: exact duplicate → skip
    return { action: 'skip' };
  }

  // Tier 2: ambiguous range (0.70 - 0.92)
  if (temporal?.supersedes) {
    // Memory Card says this fact supersedes something → version the old fact
    return { action: 'supersede', oldFactId: closest.id };
  }

  // Ambiguous but no supersession info → store as new (safe default)
  return { action: 'store' };
}
