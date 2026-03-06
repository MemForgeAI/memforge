/**
 * Background scheduler for reflect and decay operations.
 *
 * Uses setInterval (lightweight, no external dependency like BullMQ).
 * In production, this can be upgraded to node-cron or BullMQ.
 *
 * Runs:
 *   - Decay: every DECAY_INTERVAL_HOURS (default: 24)
 *   - Reflect: every REFLECT_INTERVAL_HOURS (default: 12)
 */

import type { Config } from "./config.js";
import type { EmbeddingProvider } from "./embeddings/types.js";
import { runDecay } from "./decay/decay.js";
import { reflect } from "./tools/reflect.js";

interface SchedulerHandles {
  decayInterval: ReturnType<typeof setInterval> | null;
  reflectInterval: ReturnType<typeof setInterval> | null;
}

const handles: SchedulerHandles = {
  decayInterval: null,
  reflectInterval: null,
};

/**
 * Start background jobs for decay and reflection.
 */
export function startScheduler(
  config: Config,
  embedder: EmbeddingProvider,
): void {
  const decayMs = config.decayIntervalHours * 60 * 60 * 1000;
  const reflectMs = config.reflectIntervalHours * 60 * 60 * 1000;

  // Decay job
  handles.decayInterval = setInterval(async () => {
    try {
      const result = await runDecay();
      if (result.memories_decayed > 0 || result.memories_archived > 0) {
        console.log(
          `[memforge] Decay: ${result.memories_decayed} decayed, ${result.memories_archived} archived`,
        );
      }
    } catch (err) {
      console.error("[memforge] Decay job error:", (err as Error).message);
    }
  }, decayMs);

  // Reflect job (no user_id = global reflection across all memories)
  handles.reflectInterval = setInterval(async () => {
    try {
      const result = await reflect(
        { agent_id: "system", lookback_hours: config.reflectIntervalHours * 2 },
        embedder,
        config,
      );
      if (result.insights_created.length > 0) {
        console.log(
          `[memforge] Reflect: ${result.insights_created.length} insights from ${result.memories_analyzed} memories`,
        );
      }
    } catch (err) {
      console.error("[memforge] Reflect job error:", (err as Error).message);
    }
  }, reflectMs);

  console.log(
    `[memforge] Scheduler started: decay every ${config.decayIntervalHours}h, reflect every ${config.reflectIntervalHours}h`,
  );
}

/**
 * Stop all background jobs.
 */
export function stopScheduler(): void {
  if (handles.decayInterval) {
    clearInterval(handles.decayInterval);
    handles.decayInterval = null;
  }
  if (handles.reflectInterval) {
    clearInterval(handles.reflectInterval);
    handles.reflectInterval = null;
  }
}
