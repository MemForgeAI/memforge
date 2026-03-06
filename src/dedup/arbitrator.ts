/**
 * Dedup arbitration — decides what to do when a new memory is
 * similar (but not identical) to an existing one.
 *
 * Similarity ranges:
 *   > dedupThreshold (0.95) → duplicate, handled in remember.ts
 *   conflictThreshold (0.80) - dedupThreshold (0.95) → THIS module
 *   < conflictThreshold (0.80) → new memory, no overlap
 *
 * Rule-based heuristics for the 0.80-0.95 range:
 *   1. Negation detection: if one memory negates the other → contradiction
 *   2. Temporal supersession: if new memory has temporal markers and old one doesn't → outdated
 *   3. Elaboration: if new memory is strictly longer and contains the core of the old → elaboration
 *   4. Default: treat as new memory with a conflict record for later review
 */

import type { ArbitrationVerdict, SimilarityCandidate } from './types.js';

// ============================================================
// Negation detection
// ============================================================

const NEGATION_PAIRS = [
  [/\b(is|are|was|were)\s+(?:a\s+)?(\w+)/i, /\b(is|are|was|were)\s+not\s+(?:a\s+)?(\w+)/i],
  [/\b(likes?|loves?|enjoys?|prefers?)\b/i, /\b(hates?|dislikes?|avoids?|doesn'?t\s+like)\b/i],
  [/\b(always)\b/i, /\b(never)\b/i],
  [/\b(vegetarian|vegan)\b/i, /\b(meat|steak|burger|chicken|pork|bacon)\b.*\b(loves?|enjoys?|prefers?|eats?)\b/i],
  [/\b(can)\b/i, /\b(cannot|can'?t)\b/i],
  [/\b(true)\b/i, /\b(false)\b/i],
  [/\b(yes)\b/i, /\b(no)\b/i],
];

function detectNegation(contentA: string, contentB: string): boolean {
  for (const [positive, negative] of NEGATION_PAIRS) {
    if (
      (positive!.test(contentA) && negative!.test(contentB)) ||
      (positive!.test(contentB) && negative!.test(contentA))
    ) {
      return true;
    }
  }
  return false;
}

// ============================================================
// Temporal markers
// ============================================================

const TEMPORAL_MARKERS = [
  /\b(now|currently|as of|since|recently|today|this (week|month|year))\b/i,
  /\b(changed|updated|switched|moved|no longer|stopped|started)\b/i,
  /\b(used to|formerly|previously|was|were)\b/i,
  /\b(20\d{2})\b/,
  /\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d/i,
];

function hasTemporalMarkers(content: string): number {
  let count = 0;
  for (const pattern of TEMPORAL_MARKERS) {
    if (pattern.test(content)) count++;
  }
  return count;
}

// ============================================================
// Elaboration detection
// ============================================================

/**
 * Check if newContent is an elaboration of existingContent.
 * An elaboration is strictly longer and contains the core words of the existing.
 */
function isElaboration(existingContent: string, newContent: string): boolean {
  if (newContent.length <= existingContent.length) return false;

  // Extract significant words (4+ chars, not stopwords)
  const stopwords = new Set([
    'the', 'that', 'this', 'with', 'from', 'they', 'have', 'been',
    'were', 'will', 'would', 'could', 'should', 'does', 'about',
    'into', 'than', 'then', 'them', 'when', 'what', 'which', 'their',
    'there', 'these', 'those', 'some', 'also', 'just', 'very', 'much',
  ]);

  const existingWords = existingContent
    .toLowerCase()
    .split(/\W+/)
    .filter((w) => w.length >= 4 && !stopwords.has(w));

  const newLower = newContent.toLowerCase();
  const matchCount = existingWords.filter((w) => newLower.includes(w)).length;

  // If 70%+ of the existing's significant words appear in the new, it's an elaboration
  return existingWords.length > 0 && matchCount / existingWords.length >= 0.7;
}

// ============================================================
// Main arbitration function
// ============================================================

/**
 * Arbitrate between a new memory and an existing similar memory.
 *
 * Called when similarity is in the conflict range (0.80-0.95).
 * Uses rule-based heuristics to decide the action.
 */
export function arbitrate(
  newContent: string,
  candidate: SimilarityCandidate,
): ArbitrationVerdict {
  const existing = candidate;

  // 1. Check for negation/contradiction
  if (detectNegation(newContent, existing.content)) {
    return {
      action: 'contradiction',
      existingId: existing.id,
      reason: `New memory contradicts existing memory (negation detected). Similarity: ${existing.similarity.toFixed(3)}`,
    };
  }

  // 2. Check for temporal supersession (new memory updates old one)
  const newTemporal = hasTemporalMarkers(newContent);
  const existingTemporal = hasTemporalMarkers(existing.content);

  if (newTemporal > existingTemporal && newTemporal >= 2) {
    return {
      action: 'contradiction',
      existingId: existing.id,
      reason: `New memory appears to supersede existing (temporal markers: new=${newTemporal}, existing=${existingTemporal}). Similarity: ${existing.similarity.toFixed(3)}`,
    };
  }

  // 3. Check for elaboration
  if (isElaboration(existing.content, newContent)) {
    return {
      action: 'elaboration',
      existingId: existing.id,
      reason: `New memory elaborates on existing (longer with shared core concepts). Similarity: ${existing.similarity.toFixed(3)}`,
    };
  }

  // 4. Check reverse elaboration (existing is more detailed)
  if (isElaboration(newContent, existing.content)) {
    return {
      action: 'update',
      existingId: existing.id,
      reason: `Existing memory already contains this information. Similarity: ${existing.similarity.toFixed(3)}`,
    };
  }

  // 5. Default: store as new with no conflict (similar topic but different info)
  return {
    action: 'new',
    reason: `Similar but distinct memory (no contradiction or elaboration detected). Similarity: ${existing.similarity.toFixed(3)}`,
  };
}
