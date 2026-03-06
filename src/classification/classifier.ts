/**
 * Memory type classifier.
 *
 * Hybrid approach:
 * - Rule-based for ~80% of cases (keyword matching)
 * - LLM fallback for ambiguous 20% (Phase 3+)
 *
 * Three memory types:
 * - semantic: facts, preferences, knowledge ("User prefers window seats")
 * - episodic: events, experiences, time-stamped ("User booked Tokyo trip Jan 15")
 * - procedural: workflows, instructions ("When booking, check loyalty first")
 */

export type MemoryType = "semantic" | "episodic" | "procedural";

export interface ClassificationResult {
  memoryType: MemoryType;
  confidence: number;
  method: "rule" | "llm";
}

// Keyword patterns for rule-based classification
const SEMANTIC_PATTERNS = [
  /\b(prefers?|likes?|loves?|hates?|dislikes?|enjoys?|favou?rites?)\b/i,
  /\b(always|never|usually|typically|generally)\b/i,
  /\b(is a|is an|are a|are an|works? (as|at|for))\b/i,
  /\b(allergic|intolerant|sensitive)\b/i,
  /\b(vegetarian|vegan|kosher|halal)\b/i,
  /\b(name is|called|known as|goes by)\b/i,
  /\b(lives? in|based in|from|located)\b/i,
  /\b(speaks?|fluent|language)\b/i,
  /\b(birthday|born on|age is)\b/i,
  /\b(member(ship)?|loyal(ty)?|status)\b/i,
];

const EPISODIC_PATTERNS = [
  /\b(booked|cancelled|purchased|ordered|visited|traveled|flew|arrived|went|came|met|saw|heard)\b/i,
  /\b(on (january|february|march|april|may|june|july|august|september|october|november|december))\b/i,
  /\b(on \d{1,2}(st|nd|rd|th)?|on \d{4}-\d{2}-\d{2})\b/i,
  /\b(yesterday|today|last (week|month|year|time|night|monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b/i,
  /\b(just (now|did|said|asked|mentioned))\b/i,
  /\b(this (morning|afternoon|evening|session|week|weekend))\b/i,
  /\b(recently|earlier|previously|ago)\b/i,
  /\b(complained|requested|asked (about|for)|mentioned|said|told)\b/i,
  /\b(searched|looked|browsed|viewed|clicked)\b/i,
  /\b(every (morning|evening|night|day|week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b/i,
  /\b(happened|occurred|took place|experience[ds]?)\b/i,
  /\b(at \d{1,2}(:\d{2})?\s*(am|pm))\b/i,
  /\b(during (the|my|our|a))\b/i,
];

const PROCEDURAL_PATTERNS = [
  /\b(when (booking|searching|looking|planning|ordering|processing|handling|deploying|building|running|testing|creating|updating|deleting|configuring))\b/i,
  /\b(always (check|verify|confirm|look|search|use|apply|include))\b/i,
  /\b(first[,]?\s+(do|run|check|verify|confirm|look|create|build|install|set up|configure|ensure|open|start|log in|connect))\b/i,
  /\b(step \d|steps?:)\b/i,
  /\b(before (booking|purchasing|confirming|sending|submitting|deploying|merging|pushing|releasing))\b/i,
  /\b(after (booking|purchasing|confirming|completing|deploying|merging|building))\b/i,
  /\b(make sure (to|that))\b/i,
  /\b(don'?t forget (to|that))\b/i,
  /\b(workflow|process|procedure|routine|pipeline)\b/i,
  /\b(if .+ then)\b/i,
  /\b(instructions?|guidelines?|rules?)\b/i,
  /\bfor this (user|customer|client|account)\b/i,
  /\b(to deploy|to build|to release|to set up|to configure|to install)\b/i,
  /\b(then\s+(run|build|push|deploy|tag|create|send|check|verify|execute|restart|update))\b/i,
  /\b(how to)\b/i,
];

/**
 * Classify a memory's type using rule-based keyword matching.
 * Returns the highest-scoring type with confidence.
 */
export function classify(content: string): ClassificationResult {
  const scores = {
    semantic: 0,
    episodic: 0,
    procedural: 0,
  };

  for (const pattern of SEMANTIC_PATTERNS) {
    if (pattern.test(content)) scores.semantic++;
  }
  for (const pattern of EPISODIC_PATTERNS) {
    if (pattern.test(content)) scores.episodic++;
  }
  for (const pattern of PROCEDURAL_PATTERNS) {
    if (pattern.test(content)) scores.procedural++;
  }

  const total = scores.semantic + scores.episodic + scores.procedural;

  // No patterns matched — default to semantic (most common type)
  if (total === 0) {
    return { memoryType: "semantic", confidence: 0.5, method: "rule" };
  }

  // Find the winner
  const entries = Object.entries(scores) as [MemoryType, number][];
  entries.sort((a, b) => b[1] - a[1]);

  const [winner, winnerScore] = entries[0]!;
  const [, runnerUpScore] = entries[1]!;

  // Clear winner
  if (winnerScore > runnerUpScore) {
    const confidence = Math.min(
      0.95,
      0.6 + (winnerScore - runnerUpScore) * 0.1,
    );
    return { memoryType: winner, confidence, method: "rule" };
  }

  // Tie — default to semantic with low confidence
  // In Phase 3, this is where we'd invoke the LLM
  return { memoryType: "semantic", confidence: 0.4, method: "rule" };
}
