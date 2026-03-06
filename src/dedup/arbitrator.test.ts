import { describe, it, expect } from "vitest";

import { arbitrate } from "./arbitrator.js";
import type { SimilarityCandidate } from "./types.js";

function candidate(
  overrides: Partial<SimilarityCandidate> & { content: string },
): SimilarityCandidate {
  return {
    id: "existing-1",
    similarity: 0.88,
    memoryType: "semantic",
    ...overrides,
  };
}

describe("arbitrate", () => {
  describe("contradiction detection", () => {
    it("detects like/dislike contradiction", () => {
      const result = arbitrate(
        "User hates spicy food",
        candidate({ content: "User likes spicy food" }),
      );
      expect(result.action).toBe("contradiction");
    });

    it("detects always/never contradiction", () => {
      const result = arbitrate(
        "User never flies economy class",
        candidate({ content: "User always flies economy class" }),
      );
      expect(result.action).toBe("contradiction");
    });

    it("detects is/is not contradiction", () => {
      const result = arbitrate(
        "User is not a vegetarian",
        candidate({ content: "User is a vegetarian" }),
      );
      expect(result.action).toBe("contradiction");
    });

    it("detects can/cannot contradiction", () => {
      const result = arbitrate(
        "User can't swim",
        candidate({ content: "User can swim well" }),
      );
      expect(result.action).toBe("contradiction");
    });
  });

  describe("temporal supersession", () => {
    it("detects when new memory has stronger temporal markers", () => {
      const result = arbitrate(
        "User recently changed to a vegan diet as of 2026",
        candidate({ content: "User follows a vegetarian diet" }),
      );
      expect(result.action).toBe("contradiction");
      expect((result as { reason: string }).reason).toContain("temporal");
    });
  });

  describe("elaboration detection", () => {
    it("detects when new memory elaborates on existing", () => {
      const result = arbitrate(
        "User prefers window seats on long flights, especially on the left side for the sunrise view",
        candidate({ content: "User prefers window seats" }),
      );
      expect(result.action).toBe("elaboration");
    });

    it("detects when existing is more detailed (reverse elaboration)", () => {
      const result = arbitrate(
        "User likes coffee",
        candidate({
          content:
            "User likes coffee, particularly Ethiopian single-origin dark roast from Blue Bottle",
        }),
      );
      expect(result.action).toBe("update");
    });
  });

  describe("new memory (no conflict)", () => {
    it("returns new for similar but distinct memories", () => {
      const result = arbitrate(
        "User enjoys hiking in the mountains on weekends",
        candidate({ content: "User prefers mountain views for hotel rooms" }),
      );
      expect(result.action).toBe("new");
    });

    it("returns new when neither is elaboration nor contradiction", () => {
      const result = arbitrate(
        "User works as a software engineer at a startup",
        candidate({
          content: "User has experience with Python and TypeScript",
        }),
      );
      expect(result.action).toBe("new");
    });
  });

  describe("edge cases", () => {
    it("handles empty-ish content gracefully", () => {
      const result = arbitrate("yes", candidate({ content: "no" }));
      // yes/no are in the negation pairs
      expect(result.action).toBe("contradiction");
    });

    it("handles very long content without crashing", () => {
      const longContent = "User ".padEnd(5000, "a");
      const result = arbitrate(
        longContent,
        candidate({ content: "User prefers something" }),
      );
      expect(["new", "elaboration", "update", "contradiction"]).toContain(
        result.action,
      );
    });
  });
});
