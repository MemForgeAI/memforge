import { z } from "zod";

import type { EmbeddingProvider } from "../embeddings/types.js";
import { classify } from "../classification/classifier.js";
import {
  insertMemory,
  findSimilar,
  touchMemory,
  insertConflict,
  updateMemoryContent,
  invalidateMemory,
  insertFact,
  supersedeFact,
} from "../db/queries.js";
import { arbitrate } from "../dedup/arbitrator.js";
import { deduplicateFact } from "../dedup/factDedup.js";
import { extractEntities } from "../extraction/extractor.js";
import {
  upsertEntity,
  upsertRelationship,
  linkMemoryToEntity,
  linkFactToEntity,
} from "../graph/age.js";
import type { Config } from "../config.js";
import { createMemoryCard } from "../extraction/memoryCard.js";
import type { MemoryCard } from "../extraction/memoryCard.js";

// ============================================================
// Input validation
// ============================================================

export const RememberInputSchema = z.object({
  content: z.string().min(1, "Content cannot be empty"),
  user_id: z.string().optional(),
  agent_id: z.string().default("default"),
  importance: z.number().min(0).max(1).optional(),
  shared: z.boolean().optional(),
  task_id: z.string().optional(),
  metadata: z.record(z.unknown()).optional(),
  expires_at: z.string().datetime().optional(),
});

export type RememberInput = z.infer<typeof RememberInputSchema>;

// ============================================================
// Output
// ============================================================

export interface RememberResult {
  memory_id: string;
  duplicate: boolean;
  memory_type: "semantic" | "episodic" | "procedural";
  entities_created: string[]; // Phase 4: populated by entity extraction
  conflicts_detected: string[]; // Conflict IDs from memory_conflicts table
  facts_created: string[]; // Atomic fact IDs from memory_facts table
}

// ============================================================
// Remember tool implementation
// ============================================================

export async function remember(
  input: RememberInput,
  embedder: EmbeddingProvider,
  config: Config,
): Promise<RememberResult> {
  // 1. Classify memory type
  const classification = classify(input.content);

  // 2. Memory Card — transform raw observation into standardized, self-indexing knowledge.
  //
  //    Like Pantone assigns a code to every color, the Memory Card is MemForge's
  //    standardized representation. The LLM extracts atomic facts, categories,
  //    entities, and "query signatures" — the questions this memory answers.
  //
  //    The embedding includes both the fact AND its query signatures, so vector
  //    search can match "What programming languages?" to a memory about "TypeScript"
  //    because the signature "What programming languages does Alex use?" is embedded
  //    alongside the content.
  let card: MemoryCard | null = null;
  let llmMetadata: Record<string, unknown> = {};
  let textToEmbed = input.content;

  // Skip LLM extraction for very short content (greetings, filler)
  const contentWords = input.content.split(/\s+/).length;
  if (
    config.llmExtractionEnabled &&
    config.azureOpenaiEndpoint &&
    config.azureOpenaiKey &&
    contentWords >= 5
  ) {
    try {
      card = await createMemoryCard(input.content, {
        endpoint: config.azureOpenaiEndpoint,
        apiKey: config.azureOpenaiKey,
        model: config.azureOpenaiModel,
      });

      llmMetadata = {
        atomic_facts: card.atomic_facts,
        categories: card.categories,
        llm_entities: card.entities.map((e) => `${e.name}:${e.type}`),
        query_signatures: card.query_signatures,
        temporal: card.temporal,
        memory_card_version: 1,
      };

      // Enrich the embedding with query signatures — the "Pantone codes"
      if (card.query_signatures.length > 0) {
        textToEmbed = `${input.content} ${card.query_signatures.join(" ")}`;
      }

      // Use LLM-determined importance if not explicitly set
      if (input.importance === undefined && card.importance) {
        input.importance = card.importance;
      }
    } catch (err) {
      console.warn(
        "[memforge] Memory Card extraction failed:",
        (err as Error).message,
      );
    }
  }

  // Embed both raw content (for dedup comparison) and enriched text (for storage).
  // Query signatures inflate similarity scores between topically similar memories,
  // causing false dedup. Raw content gives honest similarity.
  const rawEmbedding = await embedder.embed(input.content);
  const embedding =
    textToEmbed === input.content
      ? rawEmbedding
      : await embedder.embed(textToEmbed);

  // 3. Dedup check: find exact duplicates (> dedupThreshold)
  //    Uses raw embedding to avoid query-signature similarity inflation
  const exactDupes = await findSimilar(
    rawEmbedding,
    input.user_id,
    config.dedupThreshold,
    1,
  );

  if (exactDupes.length > 0) {
    // Near-duplicate found — update existing instead of creating new
    const existing = exactDupes[0]!;
    await touchMemory(
      existing.id,
      Math.max(existing.confidence, input.importance ?? 0.5),
    );

    return {
      memory_id: existing.id,
      duplicate: true,
      memory_type: existing.memory_type as RememberResult["memory_type"],
      entities_created: [],
      conflicts_detected: [],
      facts_created: [],
    };
  }

  // 4. Conflict detection: find memories in the ambiguous range
  //    (conflictThreshold..dedupThreshold, e.g. 0.80..0.95)
  //    Uses raw embedding for honest similarity comparison
  const conflictCandidates = await findSimilar(
    rawEmbedding,
    input.user_id,
    config.conflictThreshold,
    5,
  );

  const conflicts: string[] = [];

  for (const candidate of conflictCandidates) {
    const verdict = arbitrate(input.content, {
      id: candidate.id,
      content: candidate.content,
      similarity: candidate.similarity,
      memoryType: candidate.memory_type,
    });

    switch (verdict.action) {
      case "update": {
        // Existing memory already contains this info — just touch it
        await touchMemory(
          verdict.existingId,
          Math.max(candidate.confidence, input.importance ?? 0.5),
        );
        return {
          memory_id: verdict.existingId,
          duplicate: true,
          memory_type: candidate.memory_type as RememberResult["memory_type"],
          entities_created: [],
          conflicts_detected: [],
          facts_created: [],
        };
      }

      case "elaboration": {
        // New memory is a richer version — update existing with new content
        await updateMemoryContent(verdict.existingId, input.content, embedding);
        await touchMemory(verdict.existingId, input.importance ?? 0.5);
        return {
          memory_id: verdict.existingId,
          duplicate: false,
          memory_type: candidate.memory_type as RememberResult["memory_type"],
          entities_created: [],
          conflicts_detected: [],
          facts_created: [],
        };
      }

      case "contradiction": {
        // Store the new memory AND record a conflict for later resolution.
        // The new memory (memory_b) is considered to potentially supersede
        // the existing (memory_a). Resolution is 'pending' until reviewed.
        // We'll record the conflict AFTER storing the new memory (below).
        conflicts.push(verdict.existingId);
        break;
      }

      case "new": {
        // Similar but distinct — no action needed, store normally
        break;
      }
    }
  }

  // 5. Entity extraction + graph construction
  //    Merge Memory Card entities (LLM-extracted, higher quality) with
  //    rule-based extraction (fallback when LLM unavailable)
  const extraction = extractEntities(input.content);
  const entityNames: string[] = [];

  // Add Memory Card entities to the extraction if available
  if (card?.entities) {
    for (const e of card.entities) {
      const exists = extraction.entities.some(
        (ex) => ex.name.toLowerCase() === e.name.toLowerCase(),
      );
      if (!exists) {
        extraction.entities.push({
          name: e.name,
          type: e.type,
          confidence: 0.9, // LLM extraction is higher confidence
        });
      }
    }
  }

  // 6. Store memory
  const memory = await insertMemory({
    agentId: input.agent_id,
    userId: input.user_id,
    memoryType: card?.memory_type ?? classification.memoryType,
    content: input.content,
    embedding,
    confidence: 1.0,
    importance: input.importance ?? 0.5,
    shared: input.shared ?? false,
    taskId: input.task_id,
    metadata: {
      ...input.metadata,
      ...llmMetadata,
      classification_confidence: classification.confidence,
      classification_method: classification.method,
    },
    expiresAt: input.expires_at ? new Date(input.expires_at) : undefined,
  });

  // 7. Record conflicts and mark old memories as superseded
  const conflictIds: string[] = [];
  for (const existingId of conflicts) {
    const conflict = await insertConflict(
      existingId, // memory_a = older memory
      memory.id, // memory_b = newer memory
      "contradiction",
    );
    conflictIds.push(conflict.id);

    // Invalidate the old memory so it won't appear in recall results.
    await invalidateMemory(existingId);
  }

  // 8. Build graph: create entity nodes, link to memory, create relationships
  for (const entity of extraction.entities) {
    try {
      await upsertEntity(entity.name, entity.type);
      await linkMemoryToEntity(memory.id, entity.name, entity.type);
      entityNames.push(entity.name);
    } catch (err) {
      // Graph ops are best-effort — don't fail the memory write
      console.warn(
        `[memforge] Graph entity upsert failed for "${entity.name}":`,
        (err as Error).message,
      );
    }
  }

  for (const rel of extraction.relationships) {
    try {
      await upsertRelationship(
        rel.from.name,
        rel.from.type,
        rel.to.name,
        rel.to.type,
        rel.relation,
      );
    } catch (err) {
      console.warn(
        `[memforge] Graph relationship failed:`,
        (err as Error).message,
      );
    }
  }

  // 9. Store atomic facts from Memory Card (each with its own embedding)
  //    Optimized: batch-embed all facts, then process sequentially for dedup/insert.
  const factsCreated: string[] = [];

  if (card?.atomic_facts && card.atomic_facts.length > 0) {
    const sigMap = distributeSignatures(
      card.atomic_facts,
      card.query_signatures,
    );

    // Build texts to embed and batch-embed all facts at once
    const factTexts = card.atomic_facts.map((fact) => {
      const sigs = sigMap.get(fact) ?? [];
      return sigs.length > 0 ? `${fact} ${sigs.join(" ")}` : fact;
    });
    const factEmbeddings = await embedder.embedBatch(factTexts);

    for (let i = 0; i < card.atomic_facts.length; i++) {
      const atomicFact = card.atomic_facts[i]!;
      const factEmbedding = factEmbeddings[i]!;
      const factSignatures = sigMap.get(atomicFact) ?? [];

      try {
        // Dedup check
        const dedupResult = await deduplicateFact(
          factEmbedding,
          atomicFact,
          input.user_id,
          card.temporal,
        );

        if (dedupResult.action === "skip") {
          continue;
        }

        const storedFact = await insertFact({
          memoryId: memory.id,
          fact: atomicFact,
          embedding: factEmbedding,
          queryHints: factSignatures.join(" "),
          confidence: 1.0,
          importance: input.importance ?? 0.5,
        });

        factsCreated.push(storedFact.id);

        // Link fact to entities in graph (best-effort)
        for (const entity of extraction.entities) {
          try {
            await linkFactToEntity(storedFact.id, entity.name, entity.type);
          } catch {
            // Graph linking is best-effort
          }
        }

        if (dedupResult.action === "supersede" && dedupResult.oldFactId) {
          await supersedeFact(dedupResult.oldFactId, storedFact.id);
        }
      } catch (err) {
        console.warn(
          `[memforge] Fact storage failed for "${atomicFact}":`,
          (err as Error).message,
        );
      }
    }
  }

  return {
    memory_id: memory.id,
    duplicate: false,
    memory_type: classification.memoryType,
    entities_created: entityNames,
    conflicts_detected: conflictIds,
    facts_created: factsCreated,
  };
}

// ============================================================
// Helpers
// ============================================================

/**
 * Distribute query signatures to atomic facts using a two-pass algorithm.
 * Pass 1: Ensure every fact gets at least one signature (best unassigned match).
 * Pass 2: Distribute remaining signatures by best keyword overlap.
 * This guarantees no fact has zero query_hints.
 */
export function distributeSignatures(
  atomicFacts: string[],
  querySignatures: string[],
): Map<string, string[]> {
  const result = new Map<string, string[]>();
  for (const fact of atomicFacts) {
    result.set(fact, []);
  }

  if (atomicFacts.length === 0 || querySignatures.length === 0) {
    return result;
  }

  const factKeywords = atomicFacts.map((f) => extractWords(f));

  // Compute overlap scores: signatures × facts
  const scores: number[][] = querySignatures.map((sig) => {
    const sigWords = extractWords(sig);
    return factKeywords.map((fkw) => {
      let overlap = 0;
      for (const w of sigWords) {
        if (fkw.has(w)) overlap++;
      }
      return overlap;
    });
  });

  const assigned = new Set<number>(); // indices of assigned signatures

  // Pass 1: Ensure every fact gets at least one signature
  for (let fi = 0; fi < atomicFacts.length; fi++) {
    let bestSigIdx = -1;
    let bestOverlap = -1;

    for (let si = 0; si < querySignatures.length; si++) {
      if (assigned.has(si)) continue;
      const overlap = scores[si]![fi]!;
      if (overlap > bestOverlap) {
        bestOverlap = overlap;
        bestSigIdx = si;
      }
    }

    if (bestSigIdx >= 0) {
      result.get(atomicFacts[fi]!)!.push(querySignatures[bestSigIdx]!);
      assigned.add(bestSigIdx);
    }
  }

  // Pass 2: Distribute remaining signatures by best keyword overlap
  for (let si = 0; si < querySignatures.length; si++) {
    if (assigned.has(si)) continue;

    let bestIdx = 0;
    let bestOverlap = 0;

    for (let fi = 0; fi < factKeywords.length; fi++) {
      const overlap = scores[si]![fi]!;
      if (overlap > bestOverlap) {
        bestOverlap = overlap;
        bestIdx = fi;
      }
    }

    result.get(atomicFacts[bestIdx]!)!.push(querySignatures[si]!);
  }

  return result;
}

function extractWords(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .split(/\W+/)
      .filter((w) => w.length >= 3),
  );
}
