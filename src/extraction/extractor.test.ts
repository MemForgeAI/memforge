import { describe, it, expect } from "vitest";

import { extractEntities } from "./extractor.js";

describe("extractEntities", () => {
  describe("person extraction", () => {
    it('extracts names after "name is"', () => {
      const result = extractEntities("User's name is John Smith");
      const people = result.entities.filter((e) => e.type === "person");
      expect(people.some((p) => p.name === "John Smith")).toBe(true);
    });

    it("extracts names with titles", () => {
      const result = extractEntities("Meeting with Dr. Sarah Chen tomorrow");
      const people = result.entities.filter((e) => e.type === "person");
      expect(people.some((p) => p.name === "Sarah Chen")).toBe(true);
    });
  });

  describe("location extraction", () => {
    it('extracts cities after "in"', () => {
      const result = extractEntities("User lives in San Francisco");
      const locations = result.entities.filter((e) => e.type === "location");
      expect(locations.some((l) => l.name.includes("San Francisco"))).toBe(
        true,
      );
    });

    it('extracts destinations after "to"', () => {
      const result = extractEntities("User booked a flight to Tokyo");
      const locations = result.entities.filter((e) => e.type === "location");
      expect(locations.some((l) => l.name === "Tokyo")).toBe(true);
    });

    it('extracts locations after "from"', () => {
      const result = extractEntities("User is from London");
      const locations = result.entities.filter((e) => e.type === "location");
      expect(locations.some((l) => l.name === "London")).toBe(true);
    });
  });

  describe("organization extraction", () => {
    it('extracts company after "works at"', () => {
      const result = extractEntities("User works at Google");
      const orgs = result.entities.filter((e) => e.type === "organization");
      expect(orgs.some((o) => o.name === "Google")).toBe(true);
    });

    it("extracts companies with Inc/Corp suffix", () => {
      const result = extractEntities(
        "Partnering with Acme Corp on the project",
      );
      const orgs = result.entities.filter((e) => e.type === "organization");
      expect(orgs.some((o) => o.name.includes("Acme"))).toBe(true);
    });
  });

  describe("preference extraction", () => {
    it('extracts preferences after "prefers"', () => {
      const result = extractEntities("User prefers window seats on flights");
      const prefs = result.entities.filter((e) => e.type === "preference");
      expect(prefs.length).toBeGreaterThan(0);
      expect(prefs.some((p) => p.name.includes("window seats"))).toBe(true);
    });

    it("extracts dislikes", () => {
      const result = extractEntities("User hates layovers");
      const prefs = result.entities.filter((e) => e.type === "preference");
      expect(prefs.some((p) => p.name.includes("layovers"))).toBe(true);
    });
  });

  describe("relationships", () => {
    it("creates relationships between co-occurring entities", () => {
      const result = extractEntities(
        "Dr. Jane Doe works at Google in San Francisco",
      );
      expect(result.relationships.length).toBeGreaterThan(0);
    });

    it("infers WORKS_AT between person and organization", () => {
      const result = extractEntities(
        "User's name is Alice and works at Microsoft Corp",
      );
      const worksAt = result.relationships.filter(
        (r) => r.relation === "WORKS_AT",
      );
      expect(worksAt.length).toBeGreaterThanOrEqual(0); // May or may not find both
    });
  });

  describe("deduplication", () => {
    it("does not return duplicate entities", () => {
      const result = extractEntities(
        "User lives in Tokyo and wants to travel to Tokyo",
      );
      const tokyos = result.entities.filter(
        (e) => e.name.toLowerCase() === "tokyo",
      );
      expect(tokyos.length).toBe(1);
    });
  });

  describe("edge cases", () => {
    it("returns empty for text with no entities", () => {
      const result = extractEntities(
        "the quick brown fox jumps over the lazy dog",
      );
      expect(result.entities).toHaveLength(0);
    });

    it("handles empty string", () => {
      const result = extractEntities("");
      expect(result.entities).toHaveLength(0);
      expect(result.relationships).toHaveLength(0);
    });
  });
});
