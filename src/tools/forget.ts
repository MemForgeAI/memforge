import { z } from "zod";

import { deleteMemory, getMemory, getFactIdsForMemory } from "../db/queries.js";
import { removeMemoryEdges, removeFactEdges } from "../graph/age.js";

// ============================================================
// Input validation
// ============================================================

export const ForgetInputSchema = z.object({
  memory_id: z.string().uuid("Invalid memory ID format"),
});

export type ForgetInput = z.infer<typeof ForgetInputSchema>;

// ============================================================
// Output
// ============================================================

export interface ForgetResult {
  deleted: boolean;
  edges_removed: number;
}

// ============================================================
// Forget tool implementation
// ============================================================

export async function forget(input: ForgetInput): Promise<ForgetResult> {
  // Verify memory exists
  const memory = await getMemory(input.memory_id);
  if (!memory) {
    return { deleted: false, edges_removed: 0 };
  }

  // 1. Remove graph edges for this memory (best-effort)
  let edgesRemoved = 0;
  try {
    edgesRemoved = await removeMemoryEdges(input.memory_id);
  } catch (err) {
    console.warn(
      `[memforge] Graph cleanup failed for memory ${input.memory_id}:`,
      (err as Error).message,
    );
  }

  // 1.5. Remove graph edges for associated facts (best-effort)
  //      DB rows auto-cascade via FK, but graph edges need manual cleanup.
  try {
    const factIds = await getFactIdsForMemory(input.memory_id);
    for (const factId of factIds) {
      try {
        const factEdges = await removeFactEdges(factId);
        edgesRemoved += factEdges;
      } catch {
        // Individual fact edge cleanup failures are non-fatal
      }
    }
  } catch (err) {
    console.warn(
      `[memforge] Fact graph cleanup failed for memory ${input.memory_id}:`,
      (err as Error).message,
    );
  }

  // 2. Delete memory row (cascades to memory_conflicts + memory_facts via FK)
  const deleted = await deleteMemory(input.memory_id);

  return { deleted, edges_removed: edgesRemoved };
}
