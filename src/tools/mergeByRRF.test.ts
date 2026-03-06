import { describe, it, expect } from 'vitest';
import { mergeByRRF } from './recall.js';
import type { ScoredMemory, ScoredFact } from '../db/queries.js';

function makeMem(id: string, overrides?: Partial<ScoredMemory>): ScoredMemory {
  return {
    id,
    agent_id: 'test',
    user_id: 'test',
    memory_type: 'semantic',
    content: `Memory ${id}`,
    embedding: null,
    confidence: 1.0,
    importance: 0.5,
    source: 'test',
    shared: false,
    task_id: null,
    metadata: {},
    expires_at: null,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    access_count: 0,
    last_accessed: null,
    valid_at: new Date().toISOString(),
    invalid_at: null,
    similarity: 0.5,
    score: 0,
    ...overrides,
  };
}

describe('mergeByRRF', () => {
  it('returns empty array for empty inputs', () => {
    expect(mergeByRRF([], [])).toEqual([]);
  });

  it('returns vector-only results when no keyword matches', () => {
    const vector = [makeMem('a'), makeMem('b')];
    const result = mergeByRRF(vector, []);
    expect(result).toHaveLength(2);
    expect(result[0]!.match_sources).toEqual(['vector']);
  });

  it('returns keyword-only results when no vector matches', () => {
    const keyword = [makeMem('a'), makeMem('b')];
    const result = mergeByRRF([], keyword);
    expect(result).toHaveLength(2);
    expect(result[0]!.match_sources).toEqual(['bm25']);
  });

  it('marks dual-source memories and boosts their score', () => {
    const shared = makeMem('shared');
    const vector = [shared, makeMem('v-only')];
    const keyword = [makeMem('shared', { id: 'shared' }), makeMem('k-only')];

    const result = mergeByRRF(vector, keyword);
    const sharedResult = result.find((m) => m.id === 'shared');
    expect(sharedResult!.match_sources).toEqual(['vector', 'bm25']);

    // Dual-source should rank higher
    expect(result[0]!.id).toBe('shared');
  });

  it('normalizes similarity to 0-1 range', () => {
    const vector = [makeMem('a'), makeMem('b')];
    const result = mergeByRRF(vector, []);

    // First result should have similarity = 1.0 (max normalized)
    expect(result[0]!.similarity).toBeCloseTo(1.0, 2);
    // Second should be less
    expect(result[1]!.similarity).toBeLessThan(1.0);
  });

  it('deduplicates memories found by both methods', () => {
    const mem = makeMem('same');
    const vector = [mem];
    const keyword = [makeMem('same', { id: 'same' })];

    const result = mergeByRRF(vector, keyword);
    expect(result).toHaveLength(1);
    expect(result[0]!.id).toBe('same');
  });

  it('preserves vector similarity for vector-only results', () => {
    const vector = [makeMem('a', { similarity: 0.9 }), makeMem('b', { similarity: 0.3 })];
    const result = mergeByRRF(vector, []);

    // After normalization, order should be preserved
    expect(result[0]!.id).toBe('a');
    expect(result[1]!.id).toBe('b');
  });
});

// ============================================================
// Generic RRF with fact-shaped objects
// ============================================================

function makeFact(id: string, overrides?: Partial<ScoredFact>): ScoredFact {
  return {
    id,
    memory_id: 'mem-1',
    fact: `Fact ${id}`,
    embedding: null,
    query_hints: '',
    confidence: 1.0,
    importance: 0.5,
    version: 1,
    superseded_by: null,
    active_from: new Date().toISOString(),
    active_until: null,
    created_at: new Date().toISOString(),
    similarity: 0.5,
    score: 0,
    ...overrides,
  };
}

describe('mergeByRRF with ScoredFact', () => {
  it('works with fact-shaped objects', () => {
    const vector = [makeFact('f1'), makeFact('f2')];
    const keyword = [makeFact('f1', { id: 'f1' }), makeFact('f3')];

    const result = mergeByRRF(vector, keyword);
    expect(result).toHaveLength(3);
    expect(result[0]!.id).toBe('f1'); // dual-source should rank first
    expect(result[0]!.match_sources).toEqual(['vector', 'bm25']);
  });

  it('normalizes fact similarity to 0-1 range', () => {
    const vector = [makeFact('f1')];
    const result = mergeByRRF(vector, []);
    expect(result[0]!.similarity).toBeCloseTo(1.0, 2);
  });
});
