import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import express from 'express';
import { z } from 'zod';

import { loadConfig } from './config.js';
import { initPool, closePool } from './db/connection.js';
import { createEmbeddingProvider } from './embeddings/index.js';
import type { EmbeddingProvider } from './embeddings/types.js';
import type { Config } from './config.js';
import { RememberInputSchema, remember } from './tools/remember.js';
import { RecallInputSchema, recall } from './tools/recall.js';
import { ForgetInputSchema, forget } from './tools/forget.js';
import { ReflectInputSchema, reflect } from './tools/reflect.js';
import { startScheduler, stopScheduler } from './scheduler.js';

// ============================================================
// MCP Server Setup
// ============================================================

function createMcpServer(embedder: EmbeddingProvider, config: Config): McpServer {
  const server = new McpServer({
    name: 'memforge',
    version: '0.1.0',
  });

  // --- remember tool ---
  server.tool(
    'remember',
    `Store something you've learned about the user or task. Call this when you observe a preference, fact, event, or procedure that might be useful in future conversations. MemForge auto-classifies, embeds, deduplicates, and stores it. Examples: "User prefers window seats", "User booked flight to Tokyo on Jan 15", "When booking for this user, always check loyalty points first".`,
    {
      content: z.string().describe('What the agent learned'),
      user_id: z.string().optional().describe('Whose memory this is (optional)'),
      agent_id: z.string().optional().describe('Which agent is storing this (optional, defaults to "default")'),
      importance: z.number().min(0).max(1).optional().describe('How important is this? 0-1, defaults to 0.5. Use 1.0 for safety-critical info like allergies.'),
      shared: z.boolean().optional().describe('Make visible to other agents serving this user? Defaults to false.'),
      task_id: z.string().optional().describe('Associate with a specific task for task-scoped sharing (optional)'),
    },
    async (params) => {
      const input = RememberInputSchema.parse(params);
      const result = await remember(input, embedder, config);

      return {
        content: [
          {
            type: 'text' as const,
            text: JSON.stringify(result, null, 2),
          },
        ],
      };
    },
  );

  // --- recall tool ---
  server.tool(
    'recall',
    `Retrieve relevant context for your current task. Call this BEFORE responding when you need to remember something about the user, their preferences, past interactions, or procedures. Returns a compiled, token-budgeted context document grouped by type (FACTS / HISTORY / PROCEDURES), not a raw list. Examples: "What do I know about this user's travel preferences?", "Any procedures for booking flights for this user?", "What happened in the last session?"`,
    {
      query: z.string().describe('What you need to know — describe your current task or question in natural language'),
      user_id: z.string().optional().describe('Whose memories to search (optional)'),
      agent_id: z.string().optional().describe('Which agent is recalling (optional, defaults to "default")'),
      task_id: z.string().optional().describe('Include task-shared memories for this task ID. Useful for multi-agent task coordination.'),
      token_budget: z.number().optional().describe('Max tokens to return. Defaults to 2000. Use less for focused context, more for comprehensive.'),
      recency_weight: z.number().min(0).max(1).optional().describe('0-1, how much to favor recent memories. High (0.8+) for "what just happened?", low (0.2) for long-term preferences. Defaults to 0.5.'),
      include_shared: z.boolean().optional().describe('Include memories shared by other agents? Useful for multi-agent coordination.'),
    },
    async (params) => {
      const input = RecallInputSchema.parse(params);
      const result = await recall(input, embedder, config);

      if (!result.context) {
        return {
          content: [
            {
              type: 'text' as const,
              text: 'No relevant memories found.',
            },
          ],
        };
      }

      return {
        content: [
          {
            type: 'text' as const,
            text: result.context,
          },
        ],
      };
    },
  );

  // --- forget tool ---
  server.tool(
    'forget',
    `Remove a specific memory and clean up associated data. Use for corrections, privacy requests, GDPR compliance, or when information is no longer accurate. Requires the memory_id returned by the remember tool.`,
    {
      memory_id: z.string().describe('The UUID of the memory to delete'),
    },
    async (params) => {
      const input = ForgetInputSchema.parse(params);
      const result = await forget(input);

      return {
        content: [
          {
            type: 'text' as const,
            text: JSON.stringify(result, null, 2),
          },
        ],
      };
    },
  );

  // --- reflect tool ---
  server.tool(
    'reflect',
    `Analyze recent memories and extract higher-level insights. Call this periodically to consolidate episodic experiences into semantic knowledge. For example, if you've had multiple conversations about travel, reflect might produce "User frequently discusses travel to Japan — likely a strong interest." Returns a list of newly created insight memories.`,
    {
      user_id: z.string().optional().describe('Whose memories to reflect on (optional)'),
      agent_id: z.string().optional().describe('Which agent is reflecting (optional, defaults to "default")'),
      lookback_hours: z.number().optional().describe('How far back to look for memories to consolidate. Defaults to 48 hours.'),
    },
    async (params) => {
      const input = ReflectInputSchema.parse(params);
      const result = await reflect(input, embedder, config);

      return {
        content: [
          {
            type: 'text' as const,
            text: JSON.stringify(result, null, 2),
          },
        ],
      };
    },
  );

  return server;
}

// ============================================================
// HTTP Server + MCP Transport
// ============================================================

async function main(): Promise<void> {
  console.log('[memforge] Starting MemForge MCP Server...');

  // Load config
  const config = loadConfig();
  console.log(`[memforge] Config loaded: embedding=${config.embeddingProvider}, port=${config.port}`);

  // Initialize database
  initPool({ connectionString: config.databaseUrl });
  console.log('[memforge] Database pool initialized');

  // Initialize embedding provider
  const embedder = await createEmbeddingProvider(config);
  console.log(`[memforge] Embedding provider ready: ${embedder.name} (${embedder.dimension} dims)`);

  // Pre-load cross-encoder for re-ranking (if enabled)
  if (config.rerankerEnabled) {
    try {
      const { initCrossEncoder } = await import('./reranker/crossEncoder.js');
      await initCrossEncoder();
    } catch (err) {
      console.warn('[memforge] Cross-encoder failed to load, re-ranking disabled:', (err as Error).message);
    }
  }

  // Create MCP server
  const mcpServer = createMcpServer(embedder, config);

  // Express app for HTTP transport
  const app = express();
  app.use(express.json());

  // MCP endpoint — Streamable HTTP transport
  app.post('/mcp', async (req, res) => {
    const transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: undefined,
    });

    res.on('close', () => {
      transport.close().catch(console.error);
    });

    await mcpServer.connect(transport);
    await transport.handleRequest(req, res, req.body);
  });

  // Health check
  app.get('/health', (_req, res) => {
    res.json({ status: 'ok', version: '0.1.0' });
  });

  // Batch remember endpoint — processes multiple memories with concurrent LLM extraction.
  // Used by benchmarks/tools that need fast bulk ingestion.
  app.post('/api/batch-remember', async (req, res) => {
    try {
      const items = req.body?.items as Array<Record<string, unknown>>;
      if (!Array.isArray(items) || items.length === 0) {
        res.status(400).json({ error: 'items array required' });
        return;
      }

      const concurrency = Math.min(Number(req.body?.concurrency) || 10, 20);
      const results: Array<{ index: number; memory_id?: string; error?: string }> = [];
      let active = 0;
      let nextIdx = 0;

      await new Promise<void>((resolve) => {
        function processNext(): void {
          while (active < concurrency && nextIdx < items.length) {
            const idx = nextIdx++;
            const item = items[idx]!;
            active++;

            const input = RememberInputSchema.parse(item);
            remember(input, embedder, config)
              .then((result) => {
                results.push({ index: idx, memory_id: result.memory_id });
              })
              .catch((err: Error) => {
                results.push({ index: idx, error: err.message });
              })
              .finally(() => {
                active--;
                if (nextIdx >= items.length && active === 0) {
                  resolve();
                } else {
                  processNext();
                }
              });
          }
        }
        processNext();
      });

      results.sort((a, b) => a.index - b.index);
      res.json({ stored: results.length, results });
    } catch (err) {
      res.status(500).json({ error: (err as Error).message });
    }
  });

  // Start background scheduler (decay + reflect)
  startScheduler(config, embedder);

  // Start server
  app.listen(config.port, () => {
    console.log(`[memforge] MCP server listening on http://localhost:${config.port}/mcp`);
    console.log(`[memforge] Health check: http://localhost:${config.port}/health`);
  });

  // Graceful shutdown
  const shutdown = async () => {
    console.log('\n[memforge] Shutting down...');
    stopScheduler();
    await closePool();
    process.exit(0);
  };

  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

main().catch((err) => {
  console.error('[memforge] Fatal error:', err);
  process.exit(1);
});
