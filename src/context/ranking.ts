import type { ScoredMemory, ScoredFact } from "../db/queries.js";

/**
 * Ranking formula (v0.1.1 — tuned via benchmark):
 *   adjustedSim = similarity²  (power curve for separation)
 *   score = adjustedSim × 0.65 + recency × 0.15 + importance × 0.15 + frequency × 0.05
 *
 * Similarity dominates because the whole point of recall is relevance to the query.
 * Recency and importance are tiebreakers, not co-equal factors.
 */

const WEIGHT_SIMILARITY = 0.65;
const WEIGHT_RECENCY = 0.15;
const WEIGHT_IMPORTANCE = 0.15;
const WEIGHT_FREQUENCY = 0.05;

/** Memories below this cosine similarity are noise with local embeddings */
const MIN_SIMILARITY = 0.35;

const DEFAULT_RECENCY_LAMBDA = 0.01;

/**
 * Filter out low-similarity noise, then rank by composite score.
 * Mutates the `score` field on each ScoredMemory.
 */
export function rankMemories(
  memories: ScoredMemory[],
  recencyWeight: number = 0.5,
): ScoredMemory[] {
  if (memories.length === 0) return [];

  // Fix 3: Filter below minimum similarity threshold
  const filtered = memories.filter((m) => m.similarity >= MIN_SIMILARITY);
  if (filtered.length === 0) return [];

  const maxAccessCount = Math.max(1, ...filtered.map((m) => m.access_count));

  const now = Date.now();
  const lambda = DEFAULT_RECENCY_LAMBDA * (0.5 + recencyWeight);

  for (const mem of filtered) {
    const rawSimilarity = Math.max(0, mem.similarity);

    // Fix 2: Power curve for better separation
    // 0.9 → 0.81, 0.5 → 0.25, 0.35 → 0.12
    const similarity = rawSimilarity * rawSimilarity;

    const createdAt = new Date(mem.created_at).getTime();
    const hoursSince = Math.max(0, (now - createdAt) / (1000 * 60 * 60));
    const recency = Math.exp(-lambda * hoursSince);

    const importance = mem.importance;

    const frequency = mem.access_count / maxAccessCount;

    // Fix 1: Similarity-dominant weights
    mem.score =
      similarity * WEIGHT_SIMILARITY +
      recency * WEIGHT_RECENCY +
      importance * WEIGHT_IMPORTANCE +
      frequency * WEIGHT_FREQUENCY;
  }

  filtered.sort((a, b) => b.score - a.score);

  return filtered;
}

/**
 * Ranking formula for facts (v0.1.1 — no frequency component):
 *   adjustedSim = similarity²  (power curve for separation)
 *   score = adjustedSim × 0.70 + recency × 0.15 + importance × 0.15
 *
 * Same approach as rankMemories but without frequency, so similarity
 * gets the extra 5% weight (0.65 + 0.05 → 0.70).
 */
const FACT_WEIGHT_SIMILARITY = 0.7;
const FACT_WEIGHT_RECENCY = 0.15;
const FACT_WEIGHT_IMPORTANCE = 0.15;

/** Detect if a query is asking about time/dates */
const TEMPORAL_QUERY_PATTERNS = [
  /\bwhen\b/i,
  /\bwhat\s+(date|year|month|day|time)\b/i,
  /\bhow\s+long\b/i,
  /\bsince\s+when\b/i,
  /\b(recently|lately|latest|most\s+recent)\b/i,
];

/** Detect if a fact contains a date or time expression */
const DATE_PATTERNS = [
  /\b\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b/i,
  /\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b/i,
  /\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b/i,
  /\bin\s+(early|mid|late)\s+(january|february|march|april|may|june|july|august|september|october|november|december)/i,
  /\b\d{4}\b/,
];

/**
 * Filter out low-similarity noise, then rank facts by composite score.
 * Mutates the `score` field on each ScoredFact.
 * When the query is temporal, facts with dates get a 1.3x boost.
 */
export function rankFacts(
  facts: ScoredFact[],
  recencyWeight: number = 0.5,
  query?: string,
): ScoredFact[] {
  if (facts.length === 0) return [];

  const filtered = facts.filter((f) => f.similarity >= MIN_SIMILARITY);
  if (filtered.length === 0) return [];

  const now = Date.now();
  const lambda = DEFAULT_RECENCY_LAMBDA * (0.5 + recencyWeight);
  const isTemporalQuery = query
    ? TEMPORAL_QUERY_PATTERNS.some((p) => p.test(query))
    : false;

  for (const fact of filtered) {
    const rawSimilarity = Math.max(0, fact.similarity);

    // Power curve for better separation
    const similarity = rawSimilarity * rawSimilarity;

    const createdAt = new Date(fact.created_at).getTime();
    const hoursSince = Math.max(0, (now - createdAt) / (1000 * 60 * 60));
    const recency = Math.exp(-lambda * hoursSince);

    const importance = fact.importance;

    // No frequency component for facts
    let score =
      similarity * FACT_WEIGHT_SIMILARITY +
      recency * FACT_WEIGHT_RECENCY +
      importance * FACT_WEIGHT_IMPORTANCE;

    // Temporal boost: when query asks "when", boost facts with dates
    if (isTemporalQuery && DATE_PATTERNS.some((p) => p.test(fact.fact))) {
      score *= 1.3;
    }

    fact.score = score;
  }

  // For "recently" queries, apply graduated recency boost to date-containing facts
  const isRecencyQuery = query
    ? /\b(recently|lately|latest|most\s+recent)\b/i.test(query)
    : false;
  if (isRecencyQuery) {
    // Extract dates from facts and sort by date descending
    const dateRegex =
      /\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b/i;
    const MONTHS: Record<string, number> = {
      january: 0,
      february: 1,
      march: 2,
      april: 3,
      may: 4,
      june: 5,
      july: 6,
      august: 7,
      september: 8,
      october: 9,
      november: 10,
      december: 11,
    };
    const factsWithDates = filtered
      .map((f) => {
        const match = dateRegex.exec(f.fact);
        if (!match) return null;
        const ts = new Date(
          +match[3]!,
          MONTHS[match[2]!.toLowerCase()]!,
          +match[1]!,
        ).getTime();
        return { fact: f, ts };
      })
      .filter((x): x is { fact: ScoredFact; ts: number } => x !== null)
      .sort((a, b) => b.ts - a.ts);

    // Graduated boost: newest=1.5x, next=1.3x, then 1.1x
    const boosts = [1.5, 1.3, 1.1];
    for (let i = 0; i < Math.min(factsWithDates.length, boosts.length); i++) {
      factsWithDates[i]!.fact.score *= boosts[i]!;
    }
  }

  filtered.sort((a, b) => b.score - a.score);

  return filtered;
}
