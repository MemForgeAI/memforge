import { describe, it, expect } from "vitest";

import { rankMemories, rankFacts } from "./ranking.js";
import type { ScoredMemory, ScoredFact } from "../db/queries.js";

function makeMemory(overrides: Partial<ScoredMemory>): ScoredMemory {
  return {
    id: "test-id",
    agent_id: "agent-1",
    user_id: "user-1",
    memory_type: "semantic",
    content: "test content",
    embedding: null,
    confidence: 1.0,
    importance: 0.5,
    source: "agent_observation",
    shared: false,
    task_id: null,
    metadata: {},
    expires_at: null,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    access_count: 0,
    last_accessed: null,
    similarity: 0.8,
    score: 0,
    ...overrides,
  };
}

describe("rankMemories", () => {
  it("returns empty array for empty input", () => {
    expect(rankMemories([])).toEqual([]);
  });

  it("filters out memories below minimum similarity threshold", () => {
    const memories = [
      makeMemory({ id: "low", similarity: 0.2 }),
      makeMemory({ id: "high", similarity: 0.8 }),
      makeMemory({ id: "noise", similarity: 0.1 }),
    ];

    const ranked = rankMemories(memories);
    expect(ranked).toHaveLength(1);
    expect(ranked[0]!.id).toBe("high");
  });

  it("returns empty when all memories are below threshold", () => {
    const memories = [
      makeMemory({ id: "noise1", similarity: 0.2 }),
      makeMemory({ id: "noise2", similarity: 0.1 }),
    ];

    const ranked = rankMemories(memories);
    expect(ranked).toHaveLength(0);
  });

  it("scores higher for more similar memories", () => {
    const memories = [
      makeMemory({ id: "low", similarity: 0.4, importance: 0.5 }),
      makeMemory({ id: "high", similarity: 0.95, importance: 0.5 }),
    ];

    const ranked = rankMemories(memories);
    expect(ranked[0]!.id).toBe("high");
    expect(ranked[0]!.score).toBeGreaterThan(ranked[1]!.score);
  });

  it("scores higher for more important memories", () => {
    const now = new Date().toISOString();
    const memories = [
      makeMemory({
        id: "low-imp",
        similarity: 0.7,
        importance: 0.1,
        created_at: now,
      }),
      makeMemory({
        id: "high-imp",
        similarity: 0.7,
        importance: 1.0,
        created_at: now,
      }),
    ];

    const ranked = rankMemories(memories);
    expect(ranked[0]!.id).toBe("high-imp");
  });

  it("scores higher for more recent memories", () => {
    const now = new Date();
    const oldDate = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);

    const memories = [
      makeMemory({
        id: "old",
        similarity: 0.7,
        importance: 0.5,
        created_at: oldDate.toISOString(),
      }),
      makeMemory({
        id: "new",
        similarity: 0.7,
        importance: 0.5,
        created_at: now.toISOString(),
      }),
    ];

    const ranked = rankMemories(memories);
    expect(ranked[0]!.id).toBe("new");
  });

  it("scores higher for frequently accessed memories", () => {
    const now = new Date().toISOString();
    const memories = [
      makeMemory({
        id: "low-freq",
        similarity: 0.7,
        importance: 0.5,
        access_count: 1,
        created_at: now,
      }),
      makeMemory({
        id: "high-freq",
        similarity: 0.7,
        importance: 0.5,
        access_count: 100,
        created_at: now,
      }),
    ];

    const ranked = rankMemories(memories);
    expect(ranked[0]!.id).toBe("high-freq");
  });

  it("similarity dominates over other factors (65% weight)", () => {
    const now = new Date().toISOString();
    const memories = [
      makeMemory({
        id: "high-sim",
        similarity: 0.9,
        importance: 0.1,
        access_count: 0,
        created_at: now,
      }),
      makeMemory({
        id: "low-sim",
        similarity: 0.4,
        importance: 1.0,
        access_count: 100,
        created_at: now,
      }),
    ];

    const ranked = rankMemories(memories);
    expect(ranked[0]!.id).toBe("high-sim");
  });

  it("relevant old memory beats irrelevant recent memory", () => {
    const now = new Date();
    const twoWeeksAgo = new Date(now.getTime() - 14 * 24 * 60 * 60 * 1000);

    const memories = [
      makeMemory({
        id: "recent-irrelevant",
        similarity: 0.4,
        importance: 0.5,
        created_at: now.toISOString(),
      }),
      makeMemory({
        id: "old-relevant",
        similarity: 0.9,
        importance: 0.5,
        created_at: twoWeeksAgo.toISOString(),
      }),
    ];

    const ranked = rankMemories(memories);
    expect(ranked[0]!.id).toBe("old-relevant");
  });

  it("produces scores between 0 and 1", () => {
    const memories = [
      makeMemory({ similarity: 1.0, importance: 1.0, access_count: 10 }),
      makeMemory({ similarity: 0.5, importance: 0.0, access_count: 0 }),
    ];

    const ranked = rankMemories(memories);
    for (const mem of ranked) {
      expect(mem.score).toBeGreaterThanOrEqual(0);
      expect(mem.score).toBeLessThanOrEqual(1);
    }
  });
});

function makeFact(overrides: Partial<ScoredFact>): ScoredFact {
  return {
    id: "test-fact-id",
    memory_id: "test-memory-id",
    fact: "test fact content",
    embedding: null,
    query_hints: "",
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

describe("rankFacts", () => {
  it("returns empty array for empty input", () => {
    expect(rankFacts([])).toEqual([]);
  });

  it("filters below MIN_SIMILARITY threshold (0.35)", () => {
    const facts = [
      makeFact({ id: "low", similarity: 0.2 }),
      makeFact({ id: "high", similarity: 0.8 }),
      makeFact({ id: "noise", similarity: 0.1 }),
    ];

    const ranked = rankFacts(facts);
    expect(ranked).toHaveLength(1);
    expect(ranked[0]!.id).toBe("high");
  });

  it("scores higher for more similar facts", () => {
    const facts = [
      makeFact({ id: "low", similarity: 0.4, importance: 0.5 }),
      makeFact({ id: "high", similarity: 0.95, importance: 0.5 }),
    ];

    const ranked = rankFacts(facts);
    expect(ranked[0]!.id).toBe("high");
    expect(ranked[0]!.score).toBeGreaterThan(ranked[1]!.score);
  });

  it("scores higher for more important facts", () => {
    const now = new Date().toISOString();
    const facts = [
      makeFact({
        id: "low-imp",
        similarity: 0.7,
        importance: 0.1,
        created_at: now,
      }),
      makeFact({
        id: "high-imp",
        similarity: 0.7,
        importance: 1.0,
        created_at: now,
      }),
    ];

    const ranked = rankFacts(facts);
    expect(ranked[0]!.id).toBe("high-imp");
  });

  it("scores higher for more recent facts", () => {
    const now = new Date();
    const oldDate = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);

    const facts = [
      makeFact({
        id: "old",
        similarity: 0.7,
        importance: 0.5,
        created_at: oldDate.toISOString(),
      }),
      makeFact({
        id: "new",
        similarity: 0.7,
        importance: 0.5,
        created_at: now.toISOString(),
      }),
    ];

    const ranked = rankFacts(facts);
    expect(ranked[0]!.id).toBe("new");
  });

  it("similarity dominates (70% weight)", () => {
    const now = new Date().toISOString();
    const facts = [
      makeFact({
        id: "high-sim",
        similarity: 0.9,
        importance: 0.1,
        created_at: now,
      }),
      makeFact({
        id: "low-sim",
        similarity: 0.4,
        importance: 1.0,
        created_at: now,
      }),
    ];

    const ranked = rankFacts(facts);
    expect(ranked[0]!.id).toBe("high-sim");
  });
});
