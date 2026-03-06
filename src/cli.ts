#!/usr/bin/env node

/**
 * memforge-cli — Developer CLI for manual testing of MemForge memory operations.
 *
 * Usage:
 *   npx tsx src/cli.ts remember "User prefers window seats"
 *   npx tsx src/cli.ts remember "User prefers window seats" --user user-1 --importance 0.8
 *   npx tsx src/cli.ts recall "What are this user's preferences?" --user user-1
 *   npx tsx src/cli.ts recall "travel preferences" --user user-1 --budget 1000
 *   npx tsx src/cli.ts forget <memory-id>
 *   npx tsx src/cli.ts status
 *
 * Requires DATABASE_URL to be set (reads from .env or environment).
 */

import { parseArgs } from "node:util";

import { loadConfig, type Config } from "./config.js";
import { initPool, closePool, query } from "./db/connection.js";
import {
  createEmbeddingProvider,
  type EmbeddingProvider,
} from "./embeddings/index.js";
import { remember, RememberInputSchema } from "./tools/remember.js";
import { recall, RecallInputSchema } from "./tools/recall.js";
import { forget, ForgetInputSchema } from "./tools/forget.js";
import { reflect, ReflectInputSchema } from "./tools/reflect.js";

// ============================================================
// Helpers
// ============================================================

function printUsage(): void {
  console.log(`
memforge-cli — Developer CLI for MemForge memory server

Commands:
  remember <content>   Store a memory
    --user <id>        User ID (optional)
    --agent <id>       Agent ID (default: "default")
    --importance <n>   0-1 importance score (default: 0.5)
    --shared           Make visible to other agents

  recall <query>       Retrieve relevant memories
    --user <id>        User ID (optional)
    --agent <id>       Agent ID (default: "default")
    --budget <n>       Token budget (default: 2000)
    --recency <n>      0-1 recency weight (default: 0.5)
    --shared           Include shared memories

  reflect              Consolidate recent memories into insights
    --user <id>        User ID (optional)
    --agent <id>       Agent ID (default: "default")
    --lookback <n>     Hours to look back (default: 48)

  forget <memory-id>   Delete a specific memory

  status               Show database health and memory counts

Environment:
  DATABASE_URL         PostgreSQL connection string (required)
  EMBEDDING_PROVIDER   "local" (default) or "openai"
  OPENAI_API_KEY       Required if EMBEDDING_PROVIDER=openai

Examples:
  npx tsx src/cli.ts remember "User prefers window seats" --user user-1
  npx tsx src/cli.ts recall "travel preferences" --user user-1 --budget 500
  npx tsx src/cli.ts forget 550e8400-e29b-41d4-a716-446655440000
  npx tsx src/cli.ts status
`);
}

function die(msg: string): never {
  console.error(`Error: ${msg}`);
  process.exit(1);
}

function formatJson(data: unknown): string {
  return JSON.stringify(data, null, 2);
}

// ============================================================
// Commands
// ============================================================

async function cmdRemember(
  content: string,
  opts: Record<string, string | boolean | undefined>,
  embedder: EmbeddingProvider,
  config: Config,
): Promise<void> {
  const importance = opts["importance"]
    ? parseFloat(opts["importance"] as string)
    : undefined;

  const input = RememberInputSchema.parse({
    content,
    user_id: opts["user"] as string | undefined,
    agent_id: (opts["agent"] as string) ?? "default",
    importance,
    shared: opts["shared"] === true ? true : undefined,
  });

  console.log(
    `Storing memory: "${content.slice(0, 60)}${content.length > 60 ? "..." : ""}"`,
  );
  console.log(`  provider: ${embedder.name} (${embedder.dimension} dims)`);

  const start = Date.now();
  const result = await remember(input, embedder, config);
  const elapsed = Date.now() - start;

  console.log();
  if (result.duplicate) {
    console.log(`Duplicate detected — updated existing memory`);
  } else {
    console.log(`Memory stored successfully`);
  }
  console.log(formatJson(result));
  console.log(`\n(${elapsed}ms)`);
}

async function cmdRecall(
  queryText: string,
  opts: Record<string, string | boolean | undefined>,
  embedder: EmbeddingProvider,
  config: Config,
): Promise<void> {
  const budget = opts["budget"]
    ? parseInt(opts["budget"] as string, 10)
    : undefined;

  const recency = opts["recency"]
    ? parseFloat(opts["recency"] as string)
    : undefined;

  const input = RecallInputSchema.parse({
    query: queryText,
    user_id: opts["user"] as string | undefined,
    agent_id: (opts["agent"] as string) ?? "default",
    token_budget: budget,
    recency_weight: recency,
    include_shared: opts["shared"] === true ? true : undefined,
  });

  console.log(
    `Recalling: "${queryText.slice(0, 60)}${queryText.length > 60 ? "..." : ""}"`,
  );
  console.log(
    `  budget: ${input.token_budget ?? config.defaultTokenBudget} tokens`,
  );

  const start = Date.now();
  const result = await recall(input, embedder, config);
  const elapsed = Date.now() - start;

  console.log();
  if (!result.context) {
    console.log("No relevant memories found.");
  } else {
    console.log("--- Context Document ---");
    console.log(result.context);
    console.log("--- End ---");
    console.log(
      `\n${result.memories_used} memories, ${result.total_tokens} tokens`,
    );
  }
  console.log(`(${elapsed}ms)`);
}

async function cmdForget(memoryId: string): Promise<void> {
  const input = ForgetInputSchema.parse({ memory_id: memoryId });

  console.log(`Deleting memory: ${memoryId}`);

  const start = Date.now();
  const result = await forget(input);
  const elapsed = Date.now() - start;

  console.log();
  if (result.deleted) {
    console.log(`Memory deleted.`);
    if (result.edges_removed > 0) {
      console.log(`  ${result.edges_removed} graph edges removed.`);
    }
  } else {
    console.log(`Memory not found.`);
  }
  console.log(`(${elapsed}ms)`);
}

async function cmdStatus(): Promise<void> {
  console.log("MemForge Status");
  console.log("==============\n");

  // Check database connection
  try {
    const versionResult = await query<{ version: string }>("SELECT version()");
    console.log(`Database: connected`);
    console.log(
      `  ${versionResult.rows[0]?.version.split(" ").slice(0, 2).join(" ")}`,
    );
  } catch (err) {
    die(`Database connection failed: ${(err as Error).message}`);
  }

  // Check extensions
  const extResult = await query<{ extname: string; extversion: string }>(
    `SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'age') ORDER BY extname`,
  );
  console.log(`\nExtensions:`);
  for (const ext of extResult.rows) {
    console.log(`  ${ext.extname} ${ext.extversion}`);
  }

  // Memory counts
  const countResult = await query<{ memory_type: string; count: string }>(
    `SELECT memory_type, COUNT(*)::text AS count
     FROM memories
     GROUP BY memory_type
     ORDER BY memory_type`,
  );

  const totalResult = await query<{ total: string }>(
    `SELECT COUNT(*)::text AS total FROM memories`,
  );

  console.log(`\nMemories: ${totalResult.rows[0]?.total ?? 0} total`);
  if (countResult.rows.length > 0) {
    for (const row of countResult.rows) {
      console.log(`  ${row.memory_type}: ${row.count}`);
    }
  }

  // Entity count
  const entityResult = await query<{ count: string }>(
    `SELECT COUNT(*)::text AS count FROM entities`,
  );
  console.log(`\nEntities: ${entityResult.rows[0]?.count ?? 0}`);

  // Conflict count
  const conflictResult = await query<{ count: string; pending: string }>(
    `SELECT
       COUNT(*)::text AS count,
       COUNT(*) FILTER (WHERE resolution = 'pending')::text AS pending
     FROM memory_conflicts`,
  );
  console.log(
    `Conflicts: ${conflictResult.rows[0]?.count ?? 0} (${conflictResult.rows[0]?.pending ?? 0} pending)`,
  );

  // Recent memory
  const recentResult = await query<{
    content: string;
    memory_type: string;
    created_at: string;
  }>(
    `SELECT content, memory_type, created_at
     FROM memories
     ORDER BY created_at DESC
     LIMIT 3`,
  );
  if (recentResult.rows.length > 0) {
    console.log(`\nRecent memories:`);
    for (const row of recentResult.rows) {
      const excerpt =
        row.content.length > 60
          ? row.content.slice(0, 60) + "..."
          : row.content;
      const time = new Date(row.created_at).toLocaleString();
      console.log(`  [${row.memory_type}] "${excerpt}" (${time})`);
    }
  }
}

// ============================================================
// Main
// ============================================================

async function main(): Promise<void> {
  // Parse args — command is the first positional, content/id is second
  const args = process.argv.slice(2);

  if (args.length === 0 || args[0] === "--help" || args[0] === "-h") {
    printUsage();
    process.exit(0);
  }

  const command = args[0]!;
  const validCommands = ["remember", "recall", "reflect", "forget", "status"];
  if (!validCommands.includes(command)) {
    die(`Unknown command: "${command}". Valid: ${validCommands.join(", ")}`);
  }

  // Parse options from remaining args
  const { values, positionals } = parseArgs({
    args: args.slice(1),
    options: {
      user: { type: "string", short: "u" },
      agent: { type: "string", short: "a" },
      importance: { type: "string", short: "i" },
      budget: { type: "string", short: "b" },
      recency: { type: "string", short: "r" },
      lookback: { type: "string", short: "l" },
      shared: { type: "boolean", short: "s" },
      help: { type: "boolean", short: "h" },
    },
    allowPositionals: true,
    strict: false,
  });

  if (values["help"]) {
    printUsage();
    process.exit(0);
  }

  // Load config (reads .env)
  let config: Config;
  try {
    config = loadConfig();
  } catch (err) {
    die(
      `Config error: ${(err as Error).message}\nMake sure DATABASE_URL is set.`,
    );
  }

  // Initialize database
  initPool({ connectionString: config.databaseUrl });

  // Status doesn't need embeddings
  if (command === "status") {
    await cmdStatus();
    await closePool();
    return;
  }

  // Reflect doesn't need a positional argument
  if (command === "reflect") {
    console.log("Loading embedding provider...");
    const embedder = await createEmbeddingProvider(config);
    console.log(`  Ready: ${embedder.name} (${embedder.dimension} dims)\n`);

    const lookback = values["lookback"]
      ? parseInt(values["lookback"] as string, 10)
      : undefined;

    const input = ReflectInputSchema.parse({
      user_id: values["user"] as string | undefined,
      agent_id: (values["agent"] as string) ?? "default",
      lookback_hours: lookback,
    });

    console.log(
      `Reflecting on memories from last ${input.lookback_hours} hours...`,
    );
    const start = Date.now();
    const result = await reflect(input, embedder, config);
    const elapsed = Date.now() - start;

    console.log();
    if (result.insights_created.length === 0) {
      console.log(
        `No new insights. Analyzed ${result.memories_analyzed} memories.`,
      );
    } else {
      console.log(
        `Created ${result.insights_created.length} insights from ${result.memories_analyzed} memories:`,
      );
      for (const insight of result.insights_created) {
        const excerpt =
          insight.content.length > 80
            ? insight.content.slice(0, 80) + "..."
            : insight.content;
        console.log(`  [confidence: ${insight.confidence}] ${excerpt}`);
      }
    }
    console.log(`(${elapsed}ms)`);

    await closePool();
    return;
  }

  // All other commands need a content/id argument
  const positionalArg = positionals[0];
  if (!positionalArg) {
    die(
      `Missing argument. Usage: memforge-cli ${command} <${command === "forget" ? "memory-id" : "text"}>`,
    );
  }

  // Initialize embedding provider (only for remember/recall)
  let embedder: EmbeddingProvider | undefined;
  if (command === "remember" || command === "recall") {
    console.log(`Loading embedding provider (${config.embeddingProvider})...`);
    embedder = await createEmbeddingProvider(config);
    console.log(`  Ready: ${embedder.name} (${embedder.dimension} dims)\n`);
  }

  // Dispatch
  switch (command) {
    case "remember":
      await cmdRemember(positionalArg, values, embedder!, config);
      break;
    case "recall":
      await cmdRecall(positionalArg, values, embedder!, config);
      break;
    case "forget":
      await cmdForget(positionalArg);
      break;
  }

  await closePool();
}

main().catch((err) => {
  console.error("Fatal error:", err.message ?? err);
  process.exit(1);
});
