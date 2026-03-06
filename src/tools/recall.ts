import { z } from "zod";

import type { EmbeddingProvider } from "../embeddings/types.js";
import {
  searchByEmbedding,
  searchByKeyword,
  getMemory,
  searchFactsByEmbedding,
  searchFactsByKeyword,
  hasAnyFacts,
  searchFactsByEntityEmbedding,
} from "../db/queries.js";
import type { ScoredMemory, ScoredFact } from "../db/queries.js";
import { rankMemories } from "../context/ranking.js";
import { rankFacts } from "../context/ranking.js";
import { packContext } from "../context/tokenPacker.js";
import { packFactContext } from "../context/tokenPacker.js";
import {
  extractEntities,
  extractQueryEntities,
} from "../extraction/extractor.js";
import {
  traverseFromEntity,
  findEntitiesForMemory,
  findMemoriesByEntity,
  findFactsByEntity,
  findEntitiesForFact,
} from "../graph/age.js";
import type { Config } from "../config.js";
import {
  scorePairs,
  isReady as isRerankerReady,
} from "../reranker/crossEncoder.js";
import { classifyQuery, decomposeQuery, callChatLLM } from "./queryRouter.js";

// ============================================================
// Input validation
// ============================================================

export const RecallInputSchema = z.object({
  query: z.string().min(1, "Query cannot be empty"),
  user_id: z.string().optional(),
  agent_id: z.string().default("default"),
  task_id: z.string().optional(),
  token_budget: z.number().int().positive().optional(),
  recency_weight: z.number().min(0).max(1).optional(),
  include_shared: z.boolean().optional(),
});

export type RecallInput = z.infer<typeof RecallInputSchema>;

// ============================================================
// Output
// ============================================================

export interface RecallResult {
  /** Formatted context document grouped by type, NOT a list */
  context: string;
  /** Total tokens in the context */
  total_tokens: number;
  /** Number of memories included */
  memories_used: number;
  /** Internal: parent memory IDs of returned facts (for dedup) */
  _factMemoryIds?: string[];
}

// ============================================================
// Generic Reciprocal Rank Fusion
// ============================================================

/** Minimal interface required by mergeByRRF */
interface RRFItem {
  id: string;
  similarity: number;
  score: number;
  bm25_rank?: number;
  vector_rank?: number;
  rrf_score?: number;
  match_sources?: ("vector" | "bm25")[];
}

/**
 * Merge vector and keyword search results using Reciprocal Rank Fusion.
 * Generic — works with any type extending RRFItem (ScoredMemory, ScoredFact).
 * Normalizes the RRF score into the `similarity` field so ranking works unchanged.
 */
export function mergeByRRF<T extends RRFItem>(
  vectorResults: T[],
  keywordResults: T[],
  k: number = 60,
  dualSourceBonus: number = 0.05,
): T[] {
  const map = new Map<string, T>();

  // Assign vector ranks
  for (let i = 0; i < vectorResults.length; i++) {
    const item = vectorResults[i]!;
    map.set(item.id, {
      ...item,
      vector_rank: i + 1,
      match_sources: ["vector"] as ("vector" | "bm25")[],
    });
  }

  // Merge keyword results
  for (let i = 0; i < keywordResults.length; i++) {
    const item = keywordResults[i]!;
    const existing = map.get(item.id);
    if (existing) {
      // Found by both methods
      existing.bm25_rank = i + 1;
      existing.match_sources = ["vector", "bm25"];
    } else {
      // Found by keyword only — use its data but note it has no vector score
      map.set(item.id, {
        ...item,
        bm25_rank: i + 1,
        similarity: 0, // no vector similarity
        match_sources: ["bm25"] as ("vector" | "bm25")[],
      });
    }
  }

  // Compute RRF scores
  for (const item of map.values()) {
    const vectorScore = item.vector_rank ? 1 / (k + item.vector_rank) : 0;
    const bm25Score = item.bm25_rank ? 1 / (k + item.bm25_rank) : 0;
    const bonus = (item.match_sources?.length ?? 0) >= 2 ? dualSourceBonus : 0;
    item.rrf_score = vectorScore + bm25Score + bonus;
  }

  // Sort by RRF score descending
  const results = Array.from(map.values()).sort(
    (a, b) => (b.rrf_score ?? 0) - (a.rrf_score ?? 0),
  );

  // Normalize RRF score into the similarity field (0-1 range)
  const maxRRF = Math.max(...results.map((m) => m.rrf_score ?? 0), 0.001);
  for (const item of results) {
    item.similarity = (item.rrf_score ?? 0) / maxRRF;
  }

  return results;
}

// ============================================================
// Recall from Facts (primary path when facts exist)
// ============================================================

async function recallFromFacts(
  input: RecallInput,
  queryEmbedding: number[],
  config: Config,
  tokenBudget: number,
  recencyWeight: number,
): Promise<RecallResult> {
  // 1. Hybrid search on facts
  let allFacts: ScoredFact[];

  const factOpts = {
    query: input.query,
    queryEmbedding,
    userId: input.user_id,
    agentId: input.agent_id,
    taskId: input.task_id,
    includeShared: input.include_shared,
    limit: 20,
  };

  if (config.hybridSearchEnabled) {
    const [vectorFacts, keywordFacts] = await Promise.all([
      searchFactsByEmbedding(factOpts),
      searchFactsByKeyword(factOpts),
    ]);
    allFacts = mergeByRRF(vectorFacts, keywordFacts);
  } else {
    allFacts = [...(await searchFactsByEmbedding(factOpts))];
  }

  if (allFacts.length === 0) {
    return {
      context: "",
      total_tokens: 0,
      memories_used: 0,
      _factMemoryIds: [],
    };
  }

  // Collect parent memory IDs for dedup against memory search
  const factMemoryIds = [...new Set(allFacts.map((f) => f.memory_id))];

  // 1.5. Person-entity direct search — ranked by query similarity
  //   Get all fact IDs linked to person entity, then vector-search within them.
  try {
    const personEntities = extractQueryEntities(input.query);
    if (personEntities.length > 0) {
      const existingIds = new Set(allFacts.map((f) => f.id));
      const MAX_PERSON_FACTS = 10;

      for (const pe of personEntities) {
        try {
          const factNodes = await findFactsByEntity(pe.name, pe.type);
          const candidateIds = factNodes
            .map((n) => n.properties["id"] as string)
            .filter((id) => !existingIds.has(id));

          if (candidateIds.length > 0) {
            const ranked = await searchFactsByEntityEmbedding(
              candidateIds,
              queryEmbedding,
              MAX_PERSON_FACTS,
            );
            for (const fact of ranked) {
              allFacts.push(fact);
              existingIds.add(fact.id);
            }
          }
        } catch {
          // Graph failures are non-fatal
        }
      }
    }
  } catch {
    // Person-entity search is best-effort
  }

  // 2. Entity-aware graph enrichment for facts
  try {
    const existingIds = new Set(allFacts.map((f) => f.id));
    const MAX_GRAPH_ADDITIONS = 10;
    let graphAdded = 0;

    // Extract entities from top-5 facts
    const entitySet = new Map<string, { name: string; type: string }>();
    for (const fact of allFacts.slice(0, 5)) {
      try {
        const entities = await findEntitiesForFact(fact.id);
        for (const e of entities) {
          const name = e.properties["name"] as string;
          const type = e.properties["type"] as string;
          const key = `${name.toLowerCase()}:${type}`;
          if (!entitySet.has(key)) {
            entitySet.set(key, { name, type });
          }
        }
      } catch {
        // Graph lookup failures are non-fatal
      }
    }

    // Also extract entities from the query text
    const queryEntities = extractEntities(input.query);
    for (const e of queryEntities.entities) {
      const key = `${e.name.toLowerCase()}:${e.type}`;
      if (!entitySet.has(key)) {
        entitySet.set(key, { name: e.name, type: e.type });
      }
    }

    // Collect all entity fact IDs, then rank by query relevance in one SQL query
    const allEntityFactIds: string[] = [];
    for (const entity of entitySet.values()) {
      try {
        const factNodes = await findFactsByEntity(entity.name, entity.type);
        for (const node of factNodes) {
          const factId = node.properties["id"] as string;
          if (!existingIds.has(factId)) {
            allEntityFactIds.push(factId);
          }
        }
      } catch {
        // Graph failures are non-fatal
      }
    }

    const uniqueEntityFactIds = [...new Set(allEntityFactIds)];
    if (uniqueEntityFactIds.length > 0) {
      const rankedGraphFacts = await searchFactsByEntityEmbedding(
        uniqueEntityFactIds,
        queryEmbedding,
        MAX_GRAPH_ADDITIONS,
      );
      for (const fact of rankedGraphFacts) {
        allFacts.push(fact);
        existingIds.add(fact.id);
        graphAdded++;
      }
    }

    if (graphAdded > 0) {
      console.log(
        `[memforge] Graph enrichment added ${graphAdded} ranked facts`,
      );
    }
  } catch (err) {
    console.warn(
      "[memforge] Fact graph enrichment failed:",
      (err as Error).message,
    );
  }

  // 3. Rank facts (pass query for temporal boost)
  const ranked = rankFacts(allFacts, recencyWeight, input.query);

  // 4. Token packing
  const packed = packFactContext(ranked, tokenBudget);

  return {
    context: packed.context,
    total_tokens: packed.totalTokens,
    memories_used: packed.memoriesIncluded,
    _factMemoryIds: factMemoryIds,
  };
}

// ============================================================
// Recall from Memories (fallback path)
// ============================================================

async function recallFromMemories(
  input: RecallInput,
  queryEmbedding: number[],
  config: Config,
  tokenBudget: number,
  recencyWeight: number,
  excludeMemoryIds?: Set<string>,
): Promise<RecallResult> {
  // 1. Search: vector + BM25 hybrid (or vector-only if disabled)
  let allCandidates: ScoredMemory[];

  if (config.hybridSearchEnabled) {
    const [vectorCandidates, keywordCandidates] = await Promise.all([
      searchByEmbedding({
        query: input.query,
        queryEmbedding,
        userId: input.user_id,
        agentId: input.agent_id,
        taskId: input.task_id,
        tokenBudget,
        recencyWeight,
        includeShared: input.include_shared,
        limit: 25,
      }),
      searchByKeyword({
        query: input.query,
        queryEmbedding, // not used by keyword search, but RecallOptions requires it
        userId: input.user_id,
        agentId: input.agent_id,
        taskId: input.task_id,
        tokenBudget,
        recencyWeight,
        includeShared: input.include_shared,
        limit: 25,
      }),
    ]);

    allCandidates = mergeByRRF(vectorCandidates, keywordCandidates);
  } else {
    const candidates = await searchByEmbedding({
      query: input.query,
      queryEmbedding,
      userId: input.user_id,
      agentId: input.agent_id,
      taskId: input.task_id,
      tokenBudget,
      recencyWeight,
      includeShared: input.include_shared,
      limit: 15,
    });
    allCandidates = [...candidates];
  }

  // 1.1. Exclude memories whose facts were already returned
  if (excludeMemoryIds && excludeMemoryIds.size > 0) {
    allCandidates = allCandidates.filter((c) => !excludeMemoryIds.has(c.id));
  }

  // 1.5. Cross-encoder refinement (top-10, blend with RRF, filter noise)
  if (config.rerankerEnabled && isRerankerReady() && allCandidates.length > 0) {
    const TOP_K = Math.min(10, allCandidates.length);
    const topCandidates = allCandidates.slice(0, TOP_K);
    const rest = allCandidates.slice(TOP_K);

    const documents = topCandidates.map((c) => c.content);
    const scores = await scorePairs(input.query, documents);

    for (const s of scores) {
      const mem = topCandidates[s.index]!;
      const rrfScore = mem.similarity;
      mem.similarity = 0.6 * rrfScore + 0.4 * s.score;
      mem.cross_encoder_score = s.score;
    }

    const filtered = topCandidates.filter(
      (m) => (m.cross_encoder_score ?? 1) >= 0.2,
    );

    allCandidates = [...filtered, ...rest];
  }

  if (allCandidates.length === 0) {
    return { context: "", total_tokens: 0, memories_used: 0 };
  }

  // 2. Graph enrichment
  try {
    const existingIds = new Set(allCandidates.map((c) => c.id));

    const queryEntities = extractEntities(input.query);

    const topCandidateEntities: Array<{ name: string; type: string }> = [];
    for (const c of allCandidates.slice(0, 5)) {
      try {
        const linked = await findEntitiesForMemory(c.id);
        for (const e of linked) {
          topCandidateEntities.push({
            name: e.properties["name"] as string,
            type: e.properties["type"] as string,
          });
        }
      } catch {
        // Graph might not have this memory — skip
      }
    }

    const allEntities = [
      ...queryEntities.entities.map((e) => ({ name: e.name, type: e.type })),
      ...topCandidateEntities,
    ];
    const seen = new Set<string>();
    const uniqueEntities = allEntities.filter((e) => {
      const key = `${e.name.toLowerCase()}:${e.type}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });

    // Cap graph-added candidates to prevent context flooding
    const MAX_GRAPH_ADDITIONS = 5;
    let graphAdded = 0;

    for (const entity of uniqueEntities.slice(0, 3)) {
      if (graphAdded >= MAX_GRAPH_ADDITIONS) break;
      try {
        const related = await traverseFromEntity(entity.name, entity.type, 1);
        for (const relEntity of related.slice(0, 2)) {
          if (graphAdded >= MAX_GRAPH_ADDITIONS) break;
          try {
            const relName = relEntity.properties["name"] as string;
            const relType = relEntity.properties["type"] as string;
            const memNodes = await findMemoriesByEntity(relName, relType);
            for (const memNode of memNodes.slice(0, 2)) {
              if (graphAdded >= MAX_GRAPH_ADDITIONS) break;
              const memId = memNode.properties["id"] as string;
              if (!existingIds.has(memId)) {
                const memRow = await getMemory(memId);
                if (memRow) {
                  existingIds.add(memId);
                  allCandidates.push({
                    ...memRow,
                    similarity: 0.45,
                    score: 0,
                  } as ScoredMemory);
                  graphAdded++;
                }
              }
            }
          } catch {
            // Individual entity lookup failures are non-fatal
          }
        }
      } catch {
        // Graph traversal failures are non-fatal
      }
    }
  } catch (err) {
    console.warn("[memforge] Graph enrichment failed:", (err as Error).message);
  }

  // 3. Rank by composite score
  const ranked = rankMemories(allCandidates, recencyWeight);

  // 3.5. Recall-time temporal filtering
  const deduped = filterTemporalDuplicates(ranked);

  // 4. Token packing: two-pass fill within budget
  const packed = packContext(deduped, tokenBudget);

  return {
    context: packed.context,
    total_tokens: packed.totalTokens,
    memories_used: packed.memoriesIncluded,
  };
}

// ============================================================
// Aggregation candidate collection (search + graph, no ranking)
// ============================================================

async function collectAggregationCandidates(
  input: RecallInput,
  queryEmbedding: number[],
  config: Config,
): Promise<{ facts: ScoredFact[]; factMemoryIds: string[] }> {
  // 1. Standard hybrid search on facts
  let allFacts: ScoredFact[];

  const factOpts = {
    query: input.query,
    queryEmbedding,
    userId: input.user_id,
    agentId: input.agent_id,
    taskId: input.task_id,
    includeShared: input.include_shared,
    limit: 25,
  };

  if (config.hybridSearchEnabled) {
    const [vectorFacts, keywordFacts] = await Promise.all([
      searchFactsByEmbedding(factOpts),
      searchFactsByKeyword(factOpts),
    ]);
    allFacts = mergeByRRF(vectorFacts, keywordFacts);
  } else {
    allFacts = [...(await searchFactsByEmbedding(factOpts))];
  }

  // Collect parent memory IDs for dedup against memory search
  const factMemoryIds = [...new Set(allFacts.map((f) => f.memory_id))];

  // 2. Person-entity direct search — ranked by query similarity (cap 30 for aggregation)
  try {
    const personEntities = extractQueryEntities(input.query);
    if (personEntities.length > 0) {
      const existingIds = new Set(allFacts.map((f) => f.id));
      const MAX_PERSON_FACTS = 30;

      for (const pe of personEntities) {
        try {
          const factNodes = await findFactsByEntity(pe.name, pe.type);
          const candidateIds = factNodes
            .map((n) => n.properties["id"] as string)
            .filter((id) => !existingIds.has(id));

          if (candidateIds.length > 0) {
            const ranked = await searchFactsByEntityEmbedding(
              candidateIds,
              queryEmbedding,
              MAX_PERSON_FACTS,
            );
            for (const fact of ranked) {
              allFacts.push(fact);
              existingIds.add(fact.id);
            }
            if (ranked.length > 0) {
              console.log(
                `[memforge] Aggregation: added ${ranked.length} ranked person-entity facts (cap 30)`,
              );
            }
          }
        } catch {
          // Graph failures are non-fatal
        }
      }
    }
  } catch {
    // Person-entity search is best-effort
  }

  // 3. Entity-aware graph enrichment
  try {
    const existingIds = new Set(allFacts.map((f) => f.id));
    const MAX_GRAPH_ADDITIONS = 10;

    const entitySet = new Map<string, { name: string; type: string }>();
    for (const fact of allFacts.slice(0, 5)) {
      try {
        const entities = await findEntitiesForFact(fact.id);
        for (const e of entities) {
          const name = e.properties["name"] as string;
          const type = e.properties["type"] as string;
          const key = `${name.toLowerCase()}:${type}`;
          if (!entitySet.has(key)) {
            entitySet.set(key, { name, type });
          }
        }
      } catch {
        // Non-fatal
      }
    }

    const queryEntities = extractEntities(input.query);
    for (const e of queryEntities.entities) {
      const key = `${e.name.toLowerCase()}:${e.type}`;
      if (!entitySet.has(key)) {
        entitySet.set(key, { name: e.name, type: e.type });
      }
    }

    const allEntityFactIds: string[] = [];
    for (const entity of entitySet.values()) {
      try {
        const factNodes = await findFactsByEntity(entity.name, entity.type);
        for (const node of factNodes) {
          const factId = node.properties["id"] as string;
          if (!existingIds.has(factId)) {
            allEntityFactIds.push(factId);
          }
        }
      } catch {
        // Non-fatal
      }
    }

    const uniqueEntityFactIds = [...new Set(allEntityFactIds)];
    if (uniqueEntityFactIds.length > 0) {
      const rankedGraphFacts = await searchFactsByEntityEmbedding(
        uniqueEntityFactIds,
        queryEmbedding,
        MAX_GRAPH_ADDITIONS,
      );
      for (const fact of rankedGraphFacts) {
        allFacts.push(fact);
        existingIds.add(fact.id);
      }
    }
  } catch {
    // Graph enrichment is best-effort
  }

  return { facts: allFacts, factMemoryIds };
}

// ============================================================
// Aggregation-aware fact retrieval (raised candidate pool)
// ============================================================

async function recallAggregationFacts(
  input: RecallInput,
  queryEmbedding: number[],
  config: Config,
  tokenBudget: number,
  recencyWeight: number,
): Promise<RecallResult> {
  const { facts: allFacts, factMemoryIds } = await collectAggregationCandidates(
    input,
    queryEmbedding,
    config,
  );

  if (allFacts.length === 0) {
    return {
      context: "",
      total_tokens: 0,
      memories_used: 0,
      _factMemoryIds: [],
    };
  }

  const ranked = rankFacts(allFacts, recencyWeight, input.query);
  const packed = packFactContext(ranked, tokenBudget);

  return {
    context: packed.context,
    total_tokens: packed.totalTokens,
    memories_used: packed.memoriesIncluded,
    _factMemoryIds: factMemoryIds,
  };
}

// ============================================================
// Agentic aggregation recall (multi-pass with LLM gap analysis)
// ============================================================

const GAP_SYSTEM_PROMPT = `You analyze memory retrieval results for completeness.
For queries asking for lists/collections: if results seem partial (e.g., 1 item when the query implies multiple), generate 2-3 follow-up search queries using DIFFERENT vocabulary to find missing items.
Each follow-up query should be a short, focused search phrase (not a question), using synonyms and alternative phrasings.
Return valid JSON only, no markdown fences: {"complete": true/false, "follow_up_queries": ["query1", "query2"]}`;

interface GapAnalysisResult {
  complete: boolean;
  followUpQueries: string[];
}

async function analyzeGaps(
  query: string,
  facts: ScoredFact[],
  config: Config,
): Promise<GapAnalysisResult | null> {
  const factList = facts
    .slice(0, 15)
    .map((f, i) => `${i + 1}. ${f.fact}`)
    .join("\n");

  const userPrompt = `Query: "${query}"

Retrieved facts (${facts.length} total):
${factList}

Are these results complete for the query? If the query asks for a list/collection and results seem partial, generate 2-3 follow-up search queries with different vocabulary to find missing items.

Return JSON: {"complete": true/false, "follow_up_queries": ["query1", "query2"]}`;

  const response = await callChatLLM(
    GAP_SYSTEM_PROMPT,
    userPrompt,
    config,
    15000,
  );
  if (!response) return null;

  try {
    const jsonMatch = response.match(/\{[\s\S]*\}/);
    if (!jsonMatch) return null;

    const parsed = JSON.parse(jsonMatch[0]) as {
      complete?: boolean;
      follow_up_queries?: string[];
    };

    if (typeof parsed.complete !== "boolean") return null;

    return {
      complete: parsed.complete,
      followUpQueries: Array.isArray(parsed.follow_up_queries)
        ? parsed.follow_up_queries
            .slice(0, 3)
            .filter((q): q is string => typeof q === "string" && q.length > 0)
        : [],
    };
  } catch {
    console.warn("[memforge] Gap analysis JSON parse failed");
    return null;
  }
}

async function recallAggregationWithRefinement(
  input: RecallInput,
  queryEmbedding: number[],
  embedder: EmbeddingProvider,
  config: Config,
  tokenBudget: number,
  recencyWeight: number,
): Promise<RecallResult> {
  // Phase 1: Initial candidates
  const { facts: allFacts, factMemoryIds } = await collectAggregationCandidates(
    input,
    queryEmbedding,
    config,
  );

  if (allFacts.length === 0) {
    return {
      context: "",
      total_tokens: 0,
      memories_used: 0,
      _factMemoryIds: [],
    };
  }

  // Phase 2: LLM gap analysis
  const gaps = await analyzeGaps(input.query, allFacts, config);

  if (!gaps || gaps.complete || gaps.followUpQueries.length === 0) {
    if (gaps?.complete) {
      console.log("[memforge] Agentic recall: LLM says results are complete");
    } else if (!gaps) {
      console.log(
        "[memforge] Agentic recall: gap analysis failed, using pass-1 results",
      );
    }

    const ranked = rankFacts(allFacts, recencyWeight, input.query);
    const packed = packFactContext(ranked, tokenBudget);
    return {
      context: packed.context,
      total_tokens: packed.totalTokens,
      memories_used: packed.memoriesIncluded,
      _factMemoryIds: factMemoryIds,
    };
  }

  // Phase 3: Follow-up retrieval (parallel)
  console.log(
    `[memforge] Agentic recall: ${gaps.followUpQueries.length} follow-up queries: ` +
      JSON.stringify(gaps.followUpQueries),
  );

  const followUpEmbeddings = await embedder.embedBatch(gaps.followUpQueries);
  const existingIds = new Set(allFacts.map((f) => f.id));
  let pass2Added = 0;

  const followUpResults = await Promise.all(
    gaps.followUpQueries.map(async (fq, i) => {
      const opts = {
        query: fq,
        queryEmbedding: followUpEmbeddings[i]!,
        userId: input.user_id,
        agentId: input.agent_id,
        taskId: input.task_id,
        includeShared: input.include_shared,
        limit: 15,
      };

      try {
        if (config.hybridSearchEnabled) {
          const [vec, kw] = await Promise.all([
            searchFactsByEmbedding(opts),
            searchFactsByKeyword(opts),
          ]);
          return mergeByRRF(vec, kw);
        }
        return await searchFactsByEmbedding(opts);
      } catch {
        return [] as ScoredFact[];
      }
    }),
  );

  for (const results of followUpResults) {
    for (const fact of results) {
      if (!existingIds.has(fact.id)) {
        allFacts.push(fact);
        existingIds.add(fact.id);
        pass2Added++;
      }
    }
  }

  console.log(
    `[memforge] Agentic recall: pass-2 added ${pass2Added} new facts`,
  );

  // Phase 4: Global rank + pack
  const ranked = rankFacts(allFacts, recencyWeight, input.query);
  const packed = packFactContext(ranked, tokenBudget);

  return {
    context: packed.context,
    total_tokens: packed.totalTokens,
    memories_used: packed.memoriesIncluded,
    _factMemoryIds: factMemoryIds,
  };
}

// ============================================================
// Multi-hop retrieval (decompose + retrieve per sub-query)
// ============================================================

async function recallMultiHop(
  input: RecallInput,
  originalEmbedding: number[],
  subQueries: string[],
  embedder: EmbeddingProvider,
  config: Config,
  tokenBudget: number,
  recencyWeight: number,
): Promise<RecallResult> {
  // 1. Embed all sub-queries in parallel
  const subEmbeddings = await embedder.embedBatch(subQueries);

  // 2. Run fact search per sub-query in parallel, each with proportional budget
  const perQueryBudget = Math.floor((tokenBudget * 0.75) / subQueries.length);
  const factPromises = subQueries.map((sq, i) =>
    recallFromFacts(
      { ...input, query: sq },
      subEmbeddings[i]!,
      config,
      perQueryBudget,
      recencyWeight,
    ).catch(() => null),
  );

  // 3. Run memory search on original query (25% budget)
  const memoryBudget = Math.floor(tokenBudget * 0.25);
  const memoryPromise = recallFromMemories(
    input,
    originalEmbedding,
    config,
    memoryBudget,
    recencyWeight,
  ).catch(() => null);

  const [memoryResult, ...factResults] = await Promise.all([
    memoryPromise,
    ...factPromises,
  ]);

  // 4. Merge contexts (deduplicate lines)
  const seenLines = new Set<string>();
  const factContextParts: string[] = [];
  let totalFactTokens = 0;
  let totalFactMemories = 0;

  for (const result of factResults) {
    if (!result || !result.context) continue;
    // Split context into lines and deduplicate
    const lines = result.context.split("\n");
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("##")) continue;
      if (!seenLines.has(trimmed)) {
        seenLines.add(trimmed);
        factContextParts.push(line);
      }
    }
    totalFactTokens += result.total_tokens;
    totalFactMemories += result.memories_used;
  }

  const factContext =
    factContextParts.length > 0
      ? "## FACTS\n" + factContextParts.join("\n")
      : "";

  // 5. Combine
  const parts: string[] = [];
  if (factContext) parts.push(factContext);
  if (memoryResult?.context) parts.push(memoryResult.context);

  const context = parts.join("\n\n");
  const total_tokens = totalFactTokens + (memoryResult?.total_tokens ?? 0);
  const memories_used = totalFactMemories + (memoryResult?.memories_used ?? 0);

  if (memories_used === 0) {
    return { context: "", total_tokens: 0, memories_used: 0 };
  }

  return { context, total_tokens, memories_used };
}

// ============================================================
// Recall tool implementation (fact-first with memory fallback)
// ============================================================

export async function recall(
  input: RecallInput,
  embedder: EmbeddingProvider,
  config: Config,
): Promise<RecallResult> {
  const tokenBudget = input.token_budget ?? config.defaultTokenBudget;
  const recencyWeight = input.recency_weight ?? 0.5;

  // 1. Embed the query
  const queryEmbedding = await embedder.embed(input.query);

  // 2. Check if facts exist
  const factsExist = await hasAnyFacts(input.user_id, input.agent_id);

  if (factsExist) {
    // 2a. Classify query for routing
    const classification = await classifyQuery(input.query, config);

    switch (classification.type) {
      case "AGGREGATION": {
        const factBudget = Math.floor(tokenBudget * 0.6);
        const memoryBudget = Math.floor(tokenBudget * 0.4);

        // Agentic path: multi-pass with LLM gap analysis
        const factPromise = config.agenticRecallEnabled
          ? recallAggregationWithRefinement(
              input,
              queryEmbedding,
              embedder,
              config,
              factBudget,
              recencyWeight,
            )
          : recallAggregationFacts(
              input,
              queryEmbedding,
              config,
              factBudget,
              recencyWeight,
            );

        // Memory search runs in parallel (not blocked by LLM gap analysis)
        const [factResult, memoryResult] = await Promise.all([
          factPromise,
          recallFromMemories(
            input,
            queryEmbedding,
            config,
            memoryBudget,
            recencyWeight,
          ),
        ]);

        return combineResults(factResult, memoryResult);
      }

      case "MULTI_HOP": {
        // Decompose into sub-queries
        const subQueries = await decomposeQuery(input.query, config);
        if (subQueries.length >= 2) {
          console.log(
            `[memforge] Multi-hop decomposed into ${subQueries.length} sub-queries: ${JSON.stringify(subQueries)}`,
          );
          return recallMultiHop(
            input,
            queryEmbedding,
            subQueries,
            embedder,
            config,
            tokenBudget,
            recencyWeight,
          );
        }
        // Decomposition failed — fall through to DIRECT
        console.log(
          "[memforge] Multi-hop decomposition failed, falling back to DIRECT",
        );
        break;
      }

      case "DIRECT":
      default:
        break;
    }

    // DIRECT path (unchanged from original)
    const factBudget = Math.floor(tokenBudget * 0.6);
    const memoryBudget = Math.floor(tokenBudget * 0.4);

    const [factResult, memoryResult] = await Promise.all([
      recallFromFacts(input, queryEmbedding, config, factBudget, recencyWeight),
      recallFromMemories(
        input,
        queryEmbedding,
        config,
        memoryBudget,
        recencyWeight,
      ),
    ]);

    return combineResults(factResult, memoryResult);
  }

  // 3. No facts: search raw memories with full budget
  return recallFromMemories(
    input,
    queryEmbedding,
    config,
    tokenBudget,
    recencyWeight,
  );
}

/** Combine fact and memory results, handling empty cases */
function combineResults(
  factResult: RecallResult,
  memoryResult: RecallResult,
): RecallResult {
  if (factResult.memories_used > 0 && memoryResult.memories_used > 0) {
    return {
      context: factResult.context + "\n\n" + memoryResult.context,
      total_tokens: factResult.total_tokens + memoryResult.total_tokens,
      memories_used: factResult.memories_used + memoryResult.memories_used,
    };
  }
  if (factResult.memories_used > 0) return factResult;
  if (memoryResult.memories_used > 0) return memoryResult;
  return { context: "", total_tokens: 0, memories_used: 0 };
}

// ============================================================
// Recall-time temporal filtering
// ============================================================

/** Explicit temporal supersession keywords in memory content */
const SUPERSESSION_PATTERNS = [
  /\b(switched|moved|changed|migrated)\s+(from|to)\b/i,
  /\b(no longer|stopped|dropped|replaced|now\s+uses?)\b/i,
  /\b(instead of|rather than|over)\b/i,
  /\b(used to|formerly|previously)\b/i,
];

/**
 * Extract significant keywords from memory content for subject comparison.
 * Returns lowercased words of 4+ chars, excluding common stopwords.
 */
function extractKeywords(content: string): Set<string> {
  const stopwords = new Set([
    "the",
    "that",
    "this",
    "with",
    "from",
    "they",
    "have",
    "been",
    "were",
    "will",
    "would",
    "could",
    "should",
    "does",
    "about",
    "into",
    "than",
    "then",
    "them",
    "when",
    "what",
    "which",
    "their",
    "there",
    "these",
    "those",
    "some",
    "also",
    "just",
    "very",
    "much",
    "uses",
    "using",
    "used",
    "alex",
    "team",
    "project",
    "prefers",
  ]);
  const words = content
    .toLowerCase()
    .split(/\W+/)
    .filter((w) => w.length >= 4 && !stopwords.has(w));
  return new Set(words);
}

/**
 * Check if two memories share enough keywords to be about the same subject.
 */
function sameSubject(a: string, b: string): boolean {
  const kwA = extractKeywords(a);
  const kwB = extractKeywords(b);
  if (kwA.size === 0 || kwB.size === 0) return false;

  let overlap = 0;
  for (const w of kwA) {
    if (kwB.has(w)) overlap++;
  }

  const minSize = Math.min(kwA.size, kwB.size);
  // 40% keyword overlap suggests same subject
  return minSize > 0 && overlap / minSize >= 0.4;
}

/**
 * Filter temporal duplicates from ranked memories.
 * When two memories are about the same subject and one has supersession
 * language ("switched to", "moved to", "no longer"), keep only the newer one.
 */
function filterTemporalDuplicates(memories: ScoredMemory[]): ScoredMemory[] {
  const suppressed = new Set<string>();

  for (let i = 0; i < memories.length; i++) {
    if (suppressed.has(memories[i]!.id)) continue;

    const memI = memories[i]!;
    const hasSupersessionLanguage = SUPERSESSION_PATTERNS.some((p) =>
      p.test(memI.content),
    );

    if (!hasSupersessionLanguage) continue;

    // This memory has "switched to" / "moved to" language.
    // Find older memories about the same subject and suppress them.
    const memIDate = new Date(memI.valid_at ?? memI.created_at).getTime();

    for (let j = 0; j < memories.length; j++) {
      if (i === j || suppressed.has(memories[j]!.id)) continue;

      const memJ = memories[j]!;
      const memJDate = new Date(memJ.valid_at ?? memJ.created_at).getTime();

      // Only suppress older memories
      if (memJDate >= memIDate) continue;

      if (sameSubject(memI.content, memJ.content)) {
        suppressed.add(memJ.id);
      }
    }
  }

  return memories.filter((m) => !suppressed.has(m.id));
}
