import { describe, it, expect } from 'vitest';

import { classify } from './classifier.js';

describe('classify', () => {
  describe('semantic memories (facts, preferences)', () => {
    it('classifies preferences as semantic', () => {
      const result = classify('User prefers direct flights');
      expect(result.memoryType).toBe('semantic');
      expect(result.confidence).toBeGreaterThan(0.5);
    });

    it('classifies likes/dislikes as semantic', () => {
      expect(classify('User likes window seats').memoryType).toBe('semantic');
      expect(classify('User hates layovers').memoryType).toBe('semantic');
      expect(classify('User loves sushi').memoryType).toBe('semantic');
    });

    it('classifies facts about a person as semantic', () => {
      expect(classify('User is a vegetarian').memoryType).toBe('semantic');
      expect(classify('User is allergic to shellfish').memoryType).toBe('semantic');
      expect(classify('User lives in San Francisco').memoryType).toBe('semantic');
    });

    it('classifies habitual statements as semantic', () => {
      expect(classify('User always flies JAL to Tokyo').memoryType).toBe('semantic');
      expect(classify('User never takes connecting flights').memoryType).toBe('semantic');
      expect(classify('User typically prefers morning flights').memoryType).toBe('semantic');
    });
  });

  describe('episodic memories (events, time-stamped)', () => {
    it('classifies past events as episodic', () => {
      const result = classify('User booked flight to Tokyo on January 15th');
      expect(result.memoryType).toBe('episodic');
      expect(result.confidence).toBeGreaterThan(0.5);
    });

    it('classifies recent actions as episodic', () => {
      expect(classify('User cancelled their Tokyo trip yesterday').memoryType).toBe('episodic');
      expect(classify('User searched for flights to London last week').memoryType).toBe('episodic');
      expect(classify('User just mentioned they want to go to Paris').memoryType).toBe('episodic');
    });

    it('classifies session events as episodic', () => {
      expect(classify('User asked about direct flights this morning').memoryType).toBe('episodic');
      expect(classify('User complained about the price earlier').memoryType).toBe('episodic');
    });

    it('classifies recurring routines as episodic', () => {
      expect(
        classify('Every morning I start with a 3-mile run around the lake, then have oatmeal with blueberries').memoryType,
      ).toBe('episodic');
      expect(classify('Every Monday we have a team standup at 9am').memoryType).toBe('episodic');
    });
  });

  describe('procedural memories (workflows, instructions)', () => {
    it('classifies booking instructions as procedural', () => {
      const result = classify('When booking for this user, always check loyalty points first');
      expect(result.memoryType).toBe('procedural');
      expect(result.confidence).toBeGreaterThan(0.5);
    });

    it('classifies step-by-step instructions as procedural', () => {
      expect(classify('Step 1: Check availability. Step 2: Compare prices.').memoryType).toBe('procedural');
      expect(classify('First check JAL, then check ANA for Tokyo flights').memoryType).toBe('procedural');
    });

    it('classifies deployment/build instructions as procedural', () => {
      expect(
        classify('To deploy to production, first run the test suite, then build the Docker image, tag it with the git SHA, and push to ECR').memoryType,
      ).toBe('procedural');
      expect(classify('How to set up the local dev environment').memoryType).toBe('procedural');
    });

    it('classifies rules and reminders as procedural', () => {
      expect(classify("Make sure to apply the corporate discount code").memoryType).toBe('procedural');
      expect(classify("Don't forget to check baggage allowance before booking").memoryType).toBe('procedural');
    });
  });

  describe('ambiguous cases', () => {
    it('defaults to semantic with low confidence for ambiguous text', () => {
      const result = classify('Tokyo');
      expect(result.memoryType).toBe('semantic');
      expect(result.confidence).toBeLessThanOrEqual(0.5);
    });

    it('uses rule method, not llm', () => {
      const result = classify('User prefers direct flights');
      expect(result.method).toBe('rule');
    });
  });
});
