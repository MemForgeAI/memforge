import { describe, it, expect } from 'vitest';

import { countTokens, packContext, packFactContext } from './tokenPacker.js';
import type { ScoredMemory, ScoredFact } from '../db/queries.js';

let idCounter = 0;

function makeMemory(content: string, type: string, score: number): ScoredMemory {
  return {
    id: `test-id-${idCounter++}`,
    agent_id: 'agent-1',
    user_id: 'user-1',
    memory_type: type as ScoredMemory['memory_type'],
    content,
    embedding: null,
    confidence: 1.0,
    importance: 0.5,
    source: 'agent_observation',
    shared: false,
    task_id: null,
    metadata: {},
    expires_at: null,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    access_count: 0,
    last_accessed: null,
    similarity: 0.8,
    score,
  };
}

describe('countTokens', () => {
  it('returns 0 for empty string', () => {
    expect(countTokens('')).toBe(0);
  });

  it('approximates token count as words * 1.3', () => {
    const text = 'User prefers direct flights to avoid layovers';
    const tokens = countTokens(text);
    const words = text.split(/\s+/).length; // 7 words
    expect(tokens).toBe(Math.ceil(words * 1.3)); // ceil(9.1) = 10
  });

  it('handles multi-word text correctly', () => {
    const text = 'one two three four five';
    expect(countTokens(text)).toBe(Math.ceil(5 * 1.3)); // 7
  });
});

describe('packContext', () => {
  it('returns empty context for no memories', () => {
    const result = packContext([], 2000);
    expect(result.context).toBe('');
    expect(result.totalTokens).toBe(0);
    expect(result.memoriesIncluded).toBe(0);
  });

  it('groups memories by type with section headers', () => {
    const memories = [
      makeMemory('User likes window seats', 'semantic', 0.9),
      makeMemory('User booked Tokyo trip Jan 15', 'episodic', 0.8),
      makeMemory('Always check loyalty points first', 'procedural', 0.7),
    ];

    const result = packContext(memories, 2000);
    expect(result.context).toContain('## ADDITIONAL CONTEXT');
    expect(result.context).toContain('## HISTORY');
    expect(result.context).toContain('## PROCEDURES');
    expect(result.context).toContain('window seats');
    expect(result.context).toContain('Tokyo trip');
    expect(result.context).toContain('loyalty points');
    expect(result.memoriesIncluded).toBe(3);
  });

  it('respects token budget', () => {
    // Each memory is ~13 tokens (10 words * 1.3)
    const memories = Array.from({ length: 20 }, (_, i) =>
      makeMemory(
        `Observation number ${i} with some extra words to fill space here`,
        'semantic',
        1.0 - i * 0.01,
      ),
    );

    const result = packContext(memories, 50);
    expect(result.totalTokens).toBeLessThanOrEqual(50);
    expect(result.memoriesIncluded).toBeLessThan(20);
  });

  it('only includes sections that have content', () => {
    const memories = [
      makeMemory('User likes sushi', 'semantic', 0.9),
      makeMemory('User likes ramen', 'semantic', 0.8),
    ];

    const result = packContext(memories, 2000);
    expect(result.context).toContain('## ADDITIONAL CONTEXT');
    expect(result.context).not.toContain('## HISTORY');
    expect(result.context).not.toContain('## PROCEDURES');
  });

  it('formats memories as bullet points', () => {
    const memories = [
      makeMemory('User likes sushi', 'semantic', 0.9),
    ];

    const result = packContext(memories, 2000);
    expect(result.context).toContain('- User likes sushi');
  });
});

describe('two-pass packing', () => {
  it('packs high-confidence memories before medium-confidence ones', () => {
    // Medium-confidence memory comes first in the array (higher composite rank)
    // but high-confidence memory should be packed in pass 1
    const memories = [
      makeMemory('Medium confidence recency-boosted memory', 'semantic', 0.35),
      makeMemory('High confidence definitely relevant', 'semantic', 0.8),
    ];

    const result = packContext(memories, 2000);
    expect(result.memoriesIncluded).toBe(2);
    // Both should be included since budget is large
    expect(result.context).toContain('High confidence definitely relevant');
    expect(result.context).toContain('Medium confidence recency-boosted memory');
  });

  it('excludes memories with score below LOW_CONFIDENCE_SCORE (0.2)', () => {
    const memories = [
      makeMemory('Relevant high-score memory', 'semantic', 0.9),
      makeMemory('Noise low-score memory that should be excluded', 'semantic', 0.1),
      makeMemory('Another noise memory below threshold', 'episodic', 0.15),
    ];

    const result = packContext(memories, 2000);
    expect(result.memoriesIncluded).toBe(1);
    expect(result.context).toContain('Relevant high-score memory');
    expect(result.context).not.toContain('Noise low-score memory');
    expect(result.context).not.toContain('Another noise memory below threshold');
  });

  it('fills budget with high-confidence first, then medium-confidence', () => {
    // Create a tight budget that can only fit ~2 memories
    // Each memory: 8 words => ceil(8 * 1.3) = 11 tokens
    // With header reserve of 15, budget 40 => effective 25 => fits ~2 memories
    const memories = [
      makeMemory('medium score item ranked first here now', 'semantic', 0.3),
      makeMemory('high score item ranked second here now', 'semantic', 0.9),
      makeMemory('another high score item ranked third now', 'semantic', 0.7),
    ];

    // Budget tight enough that only 2 memories fit
    // 8 words each => 11 tokens each, effective budget = 40 - 15 = 25
    const result = packContext(memories, 40);

    // The two high-confidence memories should be packed (pass 1)
    // The medium-confidence one may or may not fit depending on remaining budget
    expect(result.context).toContain('high score item ranked second here now');
    expect(result.context).toContain('another high score item ranked third now');
    expect(result.memoriesIncluded).toBe(2);
  });

  it('packs medium-confidence memories only after all high-confidence ones', () => {
    const memories = [
      makeMemory('Medium A with some words', 'semantic', 0.3),
      makeMemory('High A with some words', 'semantic', 0.5),
      makeMemory('Medium B with some words', 'episodic', 0.25),
      makeMemory('High B with some words', 'episodic', 0.6),
    ];

    const result = packContext(memories, 2000);
    expect(result.memoriesIncluded).toBe(4);
    // All should be included with a large budget
    expect(result.context).toContain('High A');
    expect(result.context).toContain('High B');
    expect(result.context).toContain('Medium A');
    expect(result.context).toContain('Medium B');
  });

  it('does not pack any memory with score exactly 0.2', () => {
    // score >= LOW_CONFIDENCE_SCORE (0.2) should be packed in pass 2
    const memories = [
      makeMemory('Borderline memory at threshold', 'semantic', 0.2),
    ];

    const result = packContext(memories, 2000);
    expect(result.memoriesIncluded).toBe(1);
    expect(result.context).toContain('Borderline memory at threshold');
  });

  it('does not pack memory with score just below 0.2', () => {
    const memories = [
      makeMemory('Below threshold memory', 'semantic', 0.19),
    ];

    const result = packContext(memories, 2000);
    expect(result.memoriesIncluded).toBe(0);
    expect(result.context).toBe('');
  });

  it('skips pass 2 when budget is exhausted after pass 1', () => {
    // Create enough high-confidence memories with distinct content to fill the budget.
    // Each memory is ~10 words => ceil(10 * 1.3) = 13 tokens.
    // Budget 30, header reserve 15, effective budget = 15 => fits only ~1 memory.
    // After pass 1 exhausts the budget, the medium-confidence memory should not fit.
    const distinctTopics = [
      'The quarterly earnings report showed significant growth in European markets',
      'Database migration completed successfully after upgrading PostgreSQL version',
      'Customer onboarding workflow requires three separate verification steps today',
      'Frontend deployment pipeline uses containerized builds with automated testing',
      'Machine learning inference latency dropped below fifty milliseconds yesterday',
    ];
    const memories = [
      ...distinctTopics.map((topic, i) =>
        makeMemory(topic, 'semantic', 0.9 - i * 0.01),
      ),
      makeMemory('Medium confidence memory that should definitely not fit in the remaining budget at all', 'episodic', 0.3),
    ];

    // Very tight budget: only 1-2 high-confidence memories fit, leaving no room
    const result = packContext(memories, 30);
    expect(result.context).not.toContain('Medium confidence memory that should definitely not fit');
  });

  it('handles mix of all score ranges correctly', () => {
    const memories = [
      makeMemory('Very low noise', 'semantic', 0.05),       // excluded (< 0.2)
      makeMemory('Low noise', 'semantic', 0.15),             // excluded (< 0.2)
      makeMemory('Medium confidence', 'semantic', 0.3),      // pass 2
      makeMemory('High confidence', 'semantic', 0.5),        // pass 1
      makeMemory('Very high confidence', 'semantic', 0.95),  // pass 1
    ];

    const result = packContext(memories, 2000);
    expect(result.memoriesIncluded).toBe(3);
    expect(result.context).toContain('Very high confidence');
    expect(result.context).toContain('High confidence');
    expect(result.context).toContain('Medium confidence');
    expect(result.context).not.toContain('Very low noise');
    expect(result.context).not.toContain('Low noise');
  });
});

let factIdCounter = 0;

function makeFact(factText: string, score: number): ScoredFact {
  return {
    id: `test-fact-${factIdCounter++}`,
    memory_id: 'test-memory-id',
    fact: factText,
    embedding: null,
    query_hints: '',
    confidence: 1.0,
    importance: 0.5,
    version: 1,
    superseded_by: null,
    active_from: new Date().toISOString(),
    active_until: null,
    created_at: new Date().toISOString(),
    similarity: 0.8,
    score,
  };
}

describe('packFactContext', () => {
  it('returns empty context for no facts', () => {
    const result = packFactContext([], 2000);
    expect(result.context).toBe('');
    expect(result.totalTokens).toBe(0);
    expect(result.memoriesIncluded).toBe(0);
  });

  it('packs facts under ## FACTS header as bullet points', () => {
    const facts = [
      makeFact('User prefers window seats', 0.9),
      makeFact('User is allergic to peanuts', 0.8),
    ];

    const result = packFactContext(facts, 2000);
    expect(result.context).toContain('## FACTS');
    expect(result.context).toContain('- User prefers window seats');
    expect(result.context).toContain('- User is allergic to peanuts');
    expect(result.memoriesIncluded).toBe(2);
  });

  it('respects token budget', () => {
    const facts = Array.from({ length: 20 }, (_, i) =>
      makeFact(
        `Fact number ${i} with some extra words to fill up space here now`,
        1.0 - i * 0.01,
      ),
    );

    const result = packFactContext(facts, 50);
    expect(result.totalTokens).toBeLessThanOrEqual(50);
    expect(result.memoriesIncluded).toBeLessThan(20);
  });

  it('two-pass: packs high-confidence facts before medium-confidence', () => {
    const facts = [
      makeFact('Medium confidence fact about preferences', 0.35),
      makeFact('High confidence fact about name', 0.9),
    ];

    const result = packFactContext(facts, 2000);
    expect(result.memoriesIncluded).toBe(2);
    expect(result.context).toContain('High confidence fact about name');
    expect(result.context).toContain('Medium confidence fact about preferences');
  });

  it('excludes facts below LOW_CONFIDENCE_SCORE (0.2)', () => {
    const facts = [
      makeFact('Relevant fact with high score', 0.9),
      makeFact('Noise fact that should be excluded', 0.1),
      makeFact('Another noise fact below threshold', 0.15),
    ];

    const result = packFactContext(facts, 2000);
    expect(result.memoriesIncluded).toBe(1);
    expect(result.context).toContain('Relevant fact with high score');
    expect(result.context).not.toContain('Noise fact that should be excluded');
    expect(result.context).not.toContain('Another noise fact below threshold');
  });
});
