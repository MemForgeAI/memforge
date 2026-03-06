import { describe, it, expect, vi } from 'vitest';

import { deduplicateFact } from './factDedup.js';
import type { ScoredFact } from '../db/queries.js';

// Mock findSimilarFacts
vi.mock('../db/queries.js', () => ({
  findSimilarFacts: vi.fn(),
}));

import { findSimilarFacts } from '../db/queries.js';
const mockFindSimilarFacts = vi.mocked(findSimilarFacts);

function makeFact(overrides: Partial<ScoredFact>): ScoredFact {
  return {
    id: 'existing-fact-id',
    memory_id: 'mem-1',
    fact: 'existing fact',
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
    score: 0,
    ...overrides,
  };
}

describe('deduplicateFact', () => {
  const dummyEmbedding = [0.1, 0.2, 0.3];

  it('Tier 3: stores new fact when no similar facts exist', async () => {
    mockFindSimilarFacts.mockResolvedValue([]);

    const result = await deduplicateFact(dummyEmbedding, 'new fact', undefined, undefined);
    expect(result).toEqual({ action: 'store' });
  });

  it('Tier 1: skips exact duplicate (cosine > 0.92)', async () => {
    mockFindSimilarFacts.mockResolvedValue([makeFact({ similarity: 0.95 })]);

    const result = await deduplicateFact(dummyEmbedding, 'duplicate fact', undefined, undefined);
    expect(result).toEqual({ action: 'skip' });
  });

  it('Tier 2: supersedes when temporal.supersedes is set', async () => {
    mockFindSimilarFacts.mockResolvedValue([makeFact({ id: 'old-fact', similarity: 0.85 })]);

    const result = await deduplicateFact(
      dummyEmbedding,
      'updated fact',
      undefined,
      { supersedes: 'Alex used VS Code' },
    );
    expect(result).toEqual({ action: 'supersede', oldFactId: 'old-fact' });
  });

  it('Tier 2: stores as new when no temporal.supersedes', async () => {
    mockFindSimilarFacts.mockResolvedValue([makeFact({ similarity: 0.80 })]);

    const result = await deduplicateFact(dummyEmbedding, 'ambiguous fact', undefined, undefined);
    expect(result).toEqual({ action: 'store' });
  });

  it('Tier 2: stores as new when temporal.supersedes is null', async () => {
    mockFindSimilarFacts.mockResolvedValue([makeFact({ similarity: 0.80 })]);

    const result = await deduplicateFact(
      dummyEmbedding,
      'ambiguous fact',
      undefined,
      { supersedes: null },
    );
    expect(result).toEqual({ action: 'store' });
  });

  it('passes userId to findSimilarFacts', async () => {
    mockFindSimilarFacts.mockResolvedValue([]);

    await deduplicateFact(dummyEmbedding, 'fact', 'user-123', undefined);
    expect(mockFindSimilarFacts).toHaveBeenCalledWith(
      dummyEmbedding,
      'user-123',
      0.70,
      1,
    );
  });

  it('boundary: similarity exactly at 0.92 is NOT a duplicate (uses >)', async () => {
    mockFindSimilarFacts.mockResolvedValue([makeFact({ similarity: 0.92 })]);

    const result = await deduplicateFact(dummyEmbedding, 'borderline fact', undefined, undefined);
    // 0.92 is NOT > 0.92, so it falls into Tier 2 (ambiguous), not Tier 1 (skip)
    expect(result.action).not.toBe('skip');
  });
});
