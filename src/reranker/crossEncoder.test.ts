import { describe, it, expect, beforeAll } from 'vitest';

import { initCrossEncoder, scorePairs, isReady } from './crossEncoder.js';

describe('crossEncoder', () => {
  beforeAll(async () => {
    await initCrossEncoder();
  }, 30_000);

  it('reports ready after initialization', () => {
    expect(isReady()).toBe(true);
  });

  it('returns empty array for empty documents', async () => {
    const results = await scorePairs('test query', []);
    expect(results).toEqual([]);
  });

  it('scores a relevant document higher than an irrelevant one', async () => {
    const results = await scorePairs(
      'What is the capital of France?',
      [
        'Paris is the capital and most populous city of France.',
        'The quick brown fox jumps over the lazy dog.',
      ],
    );
    expect(results).toHaveLength(2);
    expect(results[0]!.score).toBeGreaterThan(results[1]!.score);
  }, 10_000);

  it('produces scores in [0, 1] range', async () => {
    const results = await scorePairs('travel preferences', [
      'User prefers window seats on long flights',
      'The weather today is sunny',
    ]);
    for (const r of results) {
      expect(r.score).toBeGreaterThanOrEqual(0);
      expect(r.score).toBeLessThanOrEqual(1);
    }
  }, 10_000);

  it('preserves input order via index field', async () => {
    const results = await scorePairs('test', ['doc A', 'doc B', 'doc C']);
    expect(results.map((r) => r.index)).toEqual([0, 1, 2]);
  }, 10_000);
});
