import type { ScoredMemory, ScoredFact } from '../db/queries.js';
import type { MemoryType } from '../classification/classifier.js';

/**
 * Token counting: approximate for v0.1.
 * tokens ≈ words × 1.3
 */
export function countTokens(text: string): number {
  const words = text.split(/\s+/).filter((w) => w.length > 0).length;
  return Math.ceil(words * 1.3);
}

export interface PackedContext {
  /** Formatted context document, grouped by type */
  context: string;
  /** Total tokens used */
  totalTokens: number;
  /** Number of memories included */
  memoriesIncluded: number;
}

/** Score threshold for "definitely relevant" in pass 1 */
const HIGH_CONFIDENCE_SCORE = 0.4;
/** Score threshold for "maybe relevant" in pass 2 */
const LOW_CONFIDENCE_SCORE = 0.2;

/**
 * Deduplicate items by Jaccard similarity on word sets.
 * When two items share >70% of their words, keep the one with the higher score.
 */
function deduplicateItems<T extends { score: number }>(
  items: T[],
  getText: (item: T) => string,
  threshold: number = 0.70,
): T[] {
  if (items.length <= 1) return items;

  const wordSets = items.map((item) => {
    const words = getText(item).toLowerCase().split(/\s+/).filter((w) => w.length > 0);
    return new Set(words);
  });

  const removed = new Set<number>();

  for (let i = 0; i < items.length; i++) {
    if (removed.has(i)) continue;
    for (let j = i + 1; j < items.length; j++) {
      if (removed.has(j)) continue;

      const setA = wordSets[i]!;
      const setB = wordSets[j]!;

      // Jaccard similarity = |intersection| / |union|
      let intersection = 0;
      for (const w of setA) {
        if (setB.has(w)) intersection++;
      }
      const union = setA.size + setB.size - intersection;
      const jaccard = union > 0 ? intersection / union : 0;

      if (jaccard >= threshold) {
        // Keep the one with the higher score, remove the other
        if (items[i]!.score >= items[j]!.score) {
          removed.add(j);
        } else {
          removed.add(i);
          break; // i is removed, stop comparing it
        }
      }
    }
  }

  return items.filter((_, idx) => !removed.has(idx));
}

/**
 * Two-pass token packing.
 *
 * Pass 1: Pack high-confidence memories (score > 0.4) — definitely relevant.
 * Pass 2: Fill remaining budget with medium-confidence (score 0.2-0.4) — maybe relevant.
 *
 * This guarantees high-relevance memories fill the budget first, even if
 * a recency-boosted medium-relevance memory ranked higher in the composite score.
 */
export function packContext(
  memories: ScoredMemory[],
  tokenBudget: number,
): PackedContext {
  if (memories.length === 0) {
    return { context: '', totalTokens: 0, memoriesIncluded: 0 };
  }

  // Deduplicate near-identical memories before packing
  memories = deduplicateItems(memories, (m) => m.content);

  const buckets: Record<MemoryType, string[]> = {
    semantic: [],
    episodic: [],
    procedural: [],
  };

  let tokensUsed = 0;
  let memoriesIncluded = 0;

  // Reserve tokens for section headers
  const headerTokens = 15;
  const effectiveBudget = tokenBudget - headerTokens;

  // Helper to add a memory to buckets if budget allows
  const tryPack = (mem: ScoredMemory): boolean => {
    const memTokens = countTokens(mem.content);

    if (tokensUsed + memTokens > effectiveBudget) {
      // Can we fit a truncated version?
      const remaining = effectiveBudget - tokensUsed;
      if (remaining > 20) {
        const truncated = truncateToTokens(mem.content, remaining);
        buckets[mem.memory_type].push(truncated);
        tokensUsed += countTokens(truncated);
        memoriesIncluded++;
      }
      return false; // Budget exhausted
    }

    buckets[mem.memory_type].push(mem.content);
    tokensUsed += memTokens;
    memoriesIncluded++;
    return true;
  };

  // Pass 1: Pack high-confidence memories (definitely relevant)
  const highConfidence = memories.filter((m) => m.score >= HIGH_CONFIDENCE_SCORE);
  const packed = new Set<string>();

  for (const mem of highConfidence) {
    if (!tryPack(mem)) break;
    packed.add(mem.id);
  }

  // Pass 2: Fill remaining budget with medium-confidence memories
  if (tokensUsed < effectiveBudget) {
    const mediumConfidence = memories.filter(
      (m) => m.score >= LOW_CONFIDENCE_SCORE && m.score < HIGH_CONFIDENCE_SCORE && !packed.has(m.id),
    );

    for (const mem of mediumConfidence) {
      if (!tryPack(mem)) break;
    }
  }

  // Format into sections
  const sections: string[] = [];

  if (buckets.semantic.length > 0) {
    sections.push(
      '## ADDITIONAL CONTEXT\n' +
      buckets.semantic.map((m) => `- ${m}`).join('\n'),
    );
  }

  if (buckets.episodic.length > 0) {
    sections.push(
      '## HISTORY\n' +
      buckets.episodic.map((m) => `- ${m}`).join('\n'),
    );
  }

  if (buckets.procedural.length > 0) {
    sections.push(
      '## PROCEDURES\n' +
      buckets.procedural.map((m) => `- ${m}`).join('\n'),
    );
  }

  const context = sections.join('\n\n');
  const totalTokens = countTokens(context);

  return { context, totalTokens, memoriesIncluded };
}

/**
 * Truncate text to approximately the given number of tokens.
 */
function truncateToTokens(text: string, maxTokens: number): string {
  const words = text.split(/\s+/);
  const targetWords = Math.floor(maxTokens / 1.3);
  if (words.length <= targetWords) return text;
  return words.slice(0, targetWords).join(' ') + '...';
}

/**
 * Two-pass token packing for facts.
 *
 * All facts go under a single "## FACTS" header as bullet points.
 * Pass 1: facts with score >= 0.4 (high confidence)
 * Pass 2: facts with 0.2 <= score < 0.4 (medium confidence)
 */
export function packFactContext(
  facts: ScoredFact[],
  tokenBudget: number,
): PackedContext {
  if (facts.length === 0) {
    return { context: '', totalTokens: 0, memoriesIncluded: 0 };
  }

  // Deduplicate near-identical facts before packing
  facts = deduplicateItems(facts, (f) => f.fact);

  const items: string[] = [];
  let tokensUsed = 0;
  let memoriesIncluded = 0;

  // Reserve tokens for header
  const headerTokens = 15;
  const effectiveBudget = tokenBudget - headerTokens;

  // Helper to add a fact if budget allows
  const tryPack = (fact: ScoredFact): boolean => {
    const line = `- ${fact.fact}`;
    const lineTokens = countTokens(line);

    if (tokensUsed + lineTokens > effectiveBudget) {
      return false; // Budget exhausted
    }

    items.push(line);
    tokensUsed += lineTokens;
    memoriesIncluded++;
    return true;
  };

  // Pass 1: Pack high-confidence facts (definitely relevant)
  const highConfidence = facts.filter((f) => f.score >= HIGH_CONFIDENCE_SCORE);
  const packed = new Set<string>();

  for (const fact of highConfidence) {
    if (!tryPack(fact)) break;
    packed.add(fact.id);
  }

  // Pass 2: Fill remaining budget with medium-confidence facts
  if (tokensUsed < effectiveBudget) {
    const mediumConfidence = facts.filter(
      (f) => f.score >= LOW_CONFIDENCE_SCORE && f.score < HIGH_CONFIDENCE_SCORE && !packed.has(f.id),
    );

    for (const fact of mediumConfidence) {
      if (!tryPack(fact)) break;
    }
  }

  if (items.length === 0) {
    return { context: '', totalTokens: 0, memoriesIncluded: 0 };
  }

  const context = '## FACTS\n' + items.join('\n');
  const totalTokens = countTokens(context);

  return { context, totalTokens, memoriesIncluded };
}
