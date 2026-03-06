/**
 * Reflect tool — Consolidate recent memories into higher-level insights.
 *
 * Dual-mode:
 *   1. On-demand MCP tool (called by agent)
 *   2. Background cron job (automatic, every REFLECT_INTERVAL_HOURS)
 *
 * Process:
 *   1. Gather recent memories (default: last 48 hours)
 *   2. Group by theme/topic using entity co-occurrence
 *   3. Extract patterns and insights using rule-based consolidation
 *   4. Store insights as semantic memories with confidence: 0.8
 *
 * In later phases, LLM-based reflection can be added for richer insights.
 */

import { z } from 'zod';

import type { EmbeddingProvider } from '../embeddings/types.js';
import { getRecentMemories, insertMemory } from '../db/queries.js';
import type { MemoryRow } from '../db/queries.js';
import { extractEntities } from '../extraction/extractor.js';
import type { Config } from '../config.js';

// ============================================================
// Input validation
// ============================================================

export const ReflectInputSchema = z.object({
  user_id: z.string().optional(),
  agent_id: z.string().default('default'),
  lookback_hours: z.number().positive().default(48),
});

export type ReflectInput = z.infer<typeof ReflectInputSchema>;

// ============================================================
// Output
// ============================================================

export interface ReflectResult {
  insights_created: Array<{
    memory_id: string;
    content: string;
    confidence: number;
  }>;
  memories_analyzed: number;
}

// ============================================================
// Pattern consolidation (rule-based)
// ============================================================

interface ThemeGroup {
  theme: string;
  memories: MemoryRow[];
  entities: string[];
}

/**
 * Group memories by shared entities/themes.
 */
function groupByTheme(memories: MemoryRow[]): ThemeGroup[] {
  // Extract entities from each memory and build entity → memories map
  const entityToMemories = new Map<string, MemoryRow[]>();

  for (const mem of memories) {
    const extraction = extractEntities(mem.content);
    for (const entity of extraction.entities) {
      const key = entity.name.toLowerCase();
      if (!entityToMemories.has(key)) {
        entityToMemories.set(key, []);
      }
      entityToMemories.get(key)!.push(mem);
    }
  }

  // Group by entities that appear in 2+ memories
  const groups: ThemeGroup[] = [];
  const usedMemories = new Set<string>();

  for (const [entity, mems] of entityToMemories) {
    if (mems.length < 2) continue;

    // Only include memories not already grouped
    const ungrouped = mems.filter((m) => !usedMemories.has(m.id));
    if (ungrouped.length < 2) continue;

    groups.push({
      theme: entity,
      memories: ungrouped,
      entities: [entity],
    });

    for (const m of ungrouped) {
      usedMemories.add(m.id);
    }
  }

  return groups;
}

/**
 * Generate an insight from a group of related memories.
 *
 * Rule-based for now — uses simple pattern-based consolidation.
 * LLM-based reflection can be swapped in later.
 */
function generateInsight(group: ThemeGroup): string | null {
  if (group.memories.length < 2) return null;

  const contents = group.memories.map((m) => m.content);
  const memoryTypes = new Set(group.memories.map((m) => m.memory_type));

  // Episodic → semantic pattern: repeated actions become preferences
  if (memoryTypes.has('episodic') && group.memories.length >= 2) {
    return `Based on ${group.memories.length} recent interactions related to "${group.theme}": ${contents.slice(0, 3).join('. ')}. This appears to be a recurring pattern.`;
  }

  // Multiple semantic memories about same entity → consolidated fact
  if (memoryTypes.has('semantic') && group.memories.length >= 2) {
    return `Consolidated knowledge about "${group.theme}": ${contents.slice(0, 3).join('. ')}.`;
  }

  // Multiple procedural → refined workflow
  if (memoryTypes.has('procedural') && group.memories.length >= 2) {
    return `Refined procedure related to "${group.theme}": ${contents.slice(0, 3).join(' → ')}.`;
  }

  // Mixed types — general consolidation
  return `Pattern detected for "${group.theme}" across ${group.memories.length} memories: ${contents.slice(0, 2).join('. ')}.`;
}

// ============================================================
// Reflect tool implementation
// ============================================================

/**
 * Analyze recent memories and extract higher-level insights.
 * Insights are stored as semantic memories with confidence: 0.8.
 */
export async function reflect(
  input: ReflectInput,
  embedder: EmbeddingProvider,
  _config: Config,
): Promise<ReflectResult> {
  // 1. Gather recent memories
  const recentMemories = await getRecentMemories(
    input.user_id,
    input.lookback_hours,
  );

  if (recentMemories.length < 2) {
    return { insights_created: [], memories_analyzed: recentMemories.length };
  }

  // 2. Group by theme
  const groups = groupByTheme(recentMemories);

  // 3. Generate insights
  const insightsCreated: ReflectResult['insights_created'] = [];

  for (const group of groups) {
    const insightContent = generateInsight(group);
    if (!insightContent) continue;

    // 4. Embed and store the insight as a semantic memory
    const embedding = await embedder.embed(insightContent);

    const memory = await insertMemory({
      agentId: input.agent_id,
      userId: input.user_id,
      memoryType: 'semantic',
      content: insightContent,
      embedding,
      confidence: 0.8, // Reflected insights have lower confidence than direct observations
      importance: 0.6,
      source: 'reflection',
      metadata: {
        source_type: 'reflection',
        source_memory_count: group.memories.length,
        theme: group.theme,
        source_memory_ids: group.memories.map((m) => m.id),
      },
    });

    insightsCreated.push({
      memory_id: memory.id,
      content: insightContent,
      confidence: 0.8,
    });
  }

  return {
    insights_created: insightsCreated,
    memories_analyzed: recentMemories.length,
  };
}
