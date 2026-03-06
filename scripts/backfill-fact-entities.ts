/**
 * Backfill script: Link existing facts to entities in the knowledge graph.
 *
 * For each fact in memory_facts:
 *   1. Find its parent memory
 *   2. Look up entities linked to that memory in the graph
 *   3. Create Fact→Entity edges for each entity
 *
 * Usage: npx tsx scripts/backfill-fact-entities.ts
 */

import { initPool, closePool, getPool } from '../src/db/connection.js';
import { linkFactToEntity, findEntitiesForMemory } from '../src/graph/age.js';

const DATABASE_URL = process.env.DATABASE_URL ?? 'postgresql://memforge:memforge_dev@localhost:5433/memforge';

async function main(): Promise<void> {
  console.log('[backfill] Connecting to database...');
  initPool({ connectionString: DATABASE_URL });
  const pool = getPool();

  // Get all active facts
  const factsResult = await pool.query<{ id: string; memory_id: string; fact: string }>(
    'SELECT id, memory_id, fact FROM memory_facts WHERE active_until IS NULL',
  );
  const facts = factsResult.rows;
  console.log(`[backfill] Found ${facts.length} active facts`);

  // Cache entity lookups per memory to avoid repeated graph queries
  const entityCache = new Map<string, Array<{ name: string; type: string }>>();

  let linked = 0;
  let skipped = 0;
  let errors = 0;

  for (let i = 0; i < facts.length; i++) {
    const fact = facts[i]!;

    // Get entities for parent memory (cached)
    let entities = entityCache.get(fact.memory_id);
    if (!entities) {
      try {
        const vertices = await findEntitiesForMemory(fact.memory_id);
        entities = vertices.map((v) => ({
          name: v.properties['name'] as string,
          type: v.properties['type'] as string,
        }));
        entityCache.set(fact.memory_id, entities);
      } catch {
        entities = [];
        entityCache.set(fact.memory_id, entities);
      }
    }

    if (entities.length === 0) {
      skipped++;
      continue;
    }

    // Link fact to each entity
    for (const entity of entities) {
      try {
        await linkFactToEntity(fact.id, entity.name, entity.type);
        linked++;
      } catch (err) {
        errors++;
        if (errors <= 5) {
          console.warn(`[backfill] Error linking fact ${fact.id} to ${entity.name}: ${(err as Error).message}`);
        }
      }
    }

    if ((i + 1) % 100 === 0) {
      console.log(`[backfill] Progress: ${i + 1}/${facts.length} facts processed, ${linked} links created`);
    }
  }

  console.log(`[backfill] Done: ${linked} links created, ${skipped} facts skipped (no entities), ${errors} errors`);
  await closePool();
}

main().catch((err) => {
  console.error('[backfill] Fatal error:', err);
  process.exit(1);
});
