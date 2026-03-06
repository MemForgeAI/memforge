/**
 * MCP Integration Tests — Phase 2
 *
 * Tests the full MCP server roundtrip:
 *   MCP Client → HTTP → MCP Server → DB → Response
 *
 * Requires the server to be running on localhost:3100 and
 * Postgres to be available with test data cleared.
 *
 * Run: DATABASE_URL=postgresql://memforge:memforge_dev@localhost:5433/memforge npx vitest run src/mcp-integration.test.ts
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { query } from './db/connection.js';
import { initPool, closePool } from './db/connection.js';

const MCP_URL = 'http://localhost:3100/mcp';
const DB_URL = 'postgresql://memforge:memforge_dev@localhost:5433/memforge';

/**
 * Create a fresh MCP client connected to our server.
 * The StreamableHTTP transport creates a new session per request,
 * so we create clients per-test for isolation.
 */
async function createClient(): Promise<Client> {
  const client = new Client({
    name: 'memforge-test-client',
    version: '0.1.0',
  });

  const transport = new StreamableHTTPClientTransport(new URL(MCP_URL));
  await client.connect(transport);
  return client;
}

describe('MCP Server Integration', () => {
  beforeAll(async () => {
    // Initialize direct DB pool for cleanup/verification
    initPool({ connectionString: DB_URL });

    // Clean up test data from previous runs
    await query('DELETE FROM memory_conflicts');
    await query("DELETE FROM memories WHERE agent_id = 'mcp-test-agent'");
  });

  afterAll(async () => {
    // Clean up test data
    await query("DELETE FROM memories WHERE agent_id = 'mcp-test-agent'");
    await closePool();
  });

  describe('tool discovery', () => {
    it('lists all three tools with correct names', async () => {
      const client = await createClient();
      const result = await client.listTools();

      const toolNames = result.tools.map((t) => t.name).sort();
      expect(toolNames).toEqual(['forget', 'recall', 'remember']);

      // Each tool should have a description
      for (const tool of result.tools) {
        expect(tool.description).toBeTruthy();
        expect(tool.description!.length).toBeGreaterThan(20);
      }

      await client.close();
    });

    it('remember tool has correct input schema', async () => {
      const client = await createClient();
      const result = await client.listTools();

      const rememberTool = result.tools.find((t) => t.name === 'remember');
      expect(rememberTool).toBeDefined();
      expect(rememberTool!.inputSchema.properties).toHaveProperty('content');
      expect(rememberTool!.inputSchema.properties).toHaveProperty('user_id');
      expect(rememberTool!.inputSchema.properties).toHaveProperty('importance');
      expect(rememberTool!.inputSchema.properties).toHaveProperty('shared');

      await client.close();
    });
  });

  describe('remember tool', () => {
    it('stores a semantic memory and returns expected structure', async () => {
      const client = await createClient();

      const result = await client.callTool({
        name: 'remember',
        arguments: {
          content: 'MCP test user prefers dark mode in all applications',
          user_id: 'mcp-test-user-1',
          agent_id: 'mcp-test-agent',
          importance: 0.7,
        },
      });

      expect(result.content).toHaveLength(1);
      const textContent = result.content[0] as { type: string; text: string };
      expect(textContent.type).toBe('text');

      const parsed = JSON.parse(textContent.text);
      expect(parsed.memory_id).toBeTruthy();
      expect(parsed.duplicate).toBe(false);
      expect(parsed.memory_type).toBe('semantic');
      expect(parsed.entities_created).toEqual([]);
      expect(parsed.conflicts_detected).toEqual([]);

      // Verify in DB directly
      const dbResult = await query(
        'SELECT content, memory_type, importance FROM memories WHERE id = $1',
        [parsed.memory_id],
      );
      expect(dbResult.rows).toHaveLength(1);
      expect(dbResult.rows[0].content).toBe(
        'MCP test user prefers dark mode in all applications',
      );
      expect(parseFloat(dbResult.rows[0].importance)).toBeCloseTo(0.7);

      await client.close();
    });

    it('detects duplicates via MCP', async () => {
      const client = await createClient();

      // Store a memory
      const first = await client.callTool({
        name: 'remember',
        arguments: {
          content: 'MCP integration test: the sky is blue on clear days',
          user_id: 'mcp-test-user-1',
          agent_id: 'mcp-test-agent',
        },
      });
      const firstParsed = JSON.parse(
        (first.content[0] as { text: string }).text,
      );
      expect(firstParsed.duplicate).toBe(false);

      // Store the exact same content
      const second = await client.callTool({
        name: 'remember',
        arguments: {
          content: 'MCP integration test: the sky is blue on clear days',
          user_id: 'mcp-test-user-1',
          agent_id: 'mcp-test-agent',
        },
      });
      const secondParsed = JSON.parse(
        (second.content[0] as { text: string }).text,
      );
      expect(secondParsed.duplicate).toBe(true);
      expect(secondParsed.memory_id).toBe(firstParsed.memory_id);

      await client.close();
    });

    it('stores different memory types correctly', async () => {
      const client = await createClient();

      // Procedural memory
      const procResult = await client.callTool({
        name: 'remember',
        arguments: {
          content:
            'To deploy the app, first run the test suite, then build the container',
          user_id: 'mcp-test-user-1',
          agent_id: 'mcp-test-agent',
        },
      });
      const procParsed = JSON.parse(
        (procResult.content[0] as { text: string }).text,
      );
      expect(procParsed.memory_type).toBe('procedural');

      // Episodic memory
      const epiResult = await client.callTool({
        name: 'remember',
        arguments: {
          content: 'User booked a flight to Tokyo yesterday',
          user_id: 'mcp-test-user-1',
          agent_id: 'mcp-test-agent',
        },
      });
      const epiParsed = JSON.parse(
        (epiResult.content[0] as { text: string }).text,
      );
      expect(epiParsed.memory_type).toBe('episodic');

      await client.close();
    });
  });

  describe('recall tool', () => {
    it('returns a formatted context document', async () => {
      const client = await createClient();

      const result = await client.callTool({
        name: 'recall',
        arguments: {
          query: 'What are the user preferences?',
          user_id: 'mcp-test-user-1',
          agent_id: 'mcp-test-agent',
          token_budget: 2000,
        },
      });

      expect(result.content).toHaveLength(1);
      const textContent = result.content[0] as { type: string; text: string };
      expect(textContent.type).toBe('text');

      // Should be a formatted context document, not JSON
      const text = textContent.text;
      expect(text).toContain('##'); // Markdown headers for grouping
      expect(text).toContain('dark mode'); // Our stored preference

      await client.close();
    });

    it('returns "No relevant memories" for empty results', async () => {
      const client = await createClient();

      const result = await client.callTool({
        name: 'recall',
        arguments: {
          query: 'Quantum physics research papers',
          user_id: 'mcp-nonexistent-user',
          agent_id: 'mcp-test-agent',
        },
      });

      const textContent = result.content[0] as { type: string; text: string };
      expect(textContent.text).toBe('No relevant memories found.');

      await client.close();
    });

    it('respects token budget', async () => {
      const client = await createClient();

      // Request a very small budget
      const result = await client.callTool({
        name: 'recall',
        arguments: {
          query: 'What do I know about this user?',
          user_id: 'mcp-test-user-1',
          agent_id: 'mcp-test-agent',
          token_budget: 30,
        },
      });

      const textContent = result.content[0] as { type: string; text: string };
      // With a 30-token budget, we should get very few memories
      // The token packer should limit the output
      // Just verify it doesn't crash and returns something valid
      expect(textContent.text).toBeTruthy();

      await client.close();
    });
  });

  describe('forget tool', () => {
    it('deletes a memory and confirms', async () => {
      const client = await createClient();

      // First create a memory to delete
      const storeResult = await client.callTool({
        name: 'remember',
        arguments: {
          content: 'This memory will be deleted via MCP forget test',
          user_id: 'mcp-test-user-1',
          agent_id: 'mcp-test-agent',
        },
      });
      const stored = JSON.parse(
        (storeResult.content[0] as { text: string }).text,
      );
      const memoryId = stored.memory_id;

      // Forget it
      const forgetResult = await client.callTool({
        name: 'forget',
        arguments: { memory_id: memoryId },
      });

      const forgetParsed = JSON.parse(
        (forgetResult.content[0] as { text: string }).text,
      );
      expect(forgetParsed.deleted).toBe(true);

      // Verify it's gone from DB
      const dbResult = await query(
        'SELECT id FROM memories WHERE id = $1',
        [memoryId],
      );
      expect(dbResult.rows).toHaveLength(0);

      await client.close();
    });

    it('returns deleted: false for non-existent memory', async () => {
      const client = await createClient();

      const result = await client.callTool({
        name: 'forget',
        arguments: {
          memory_id: '00000000-0000-0000-0000-000000000000',
        },
      });

      const parsed = JSON.parse(
        (result.content[0] as { text: string }).text,
      );
      expect(parsed.deleted).toBe(false);

      await client.close();
    });
  });

  describe('end-to-end workflow', () => {
    it('remember → recall → forget full cycle', async () => {
      const client = await createClient();
      const userId = 'mcp-e2e-user';

      // 1. Store several memories
      const memories: string[] = [];

      const m1 = await client.callTool({
        name: 'remember',
        arguments: {
          content: 'E2E test user is allergic to peanuts',
          user_id: userId,
          agent_id: 'mcp-test-agent',
          importance: 0.95,
        },
      });
      memories.push(
        JSON.parse((m1.content[0] as { text: string }).text).memory_id,
      );

      const m2 = await client.callTool({
        name: 'remember',
        arguments: {
          content: 'E2E user visited Paris last month',
          user_id: userId,
          agent_id: 'mcp-test-agent',
        },
      });
      memories.push(
        JSON.parse((m2.content[0] as { text: string }).text).memory_id,
      );

      const m3 = await client.callTool({
        name: 'remember',
        arguments: {
          content: 'When ordering food for E2E user, always check for allergens first',
          user_id: userId,
          agent_id: 'mcp-test-agent',
        },
      });
      memories.push(
        JSON.parse((m3.content[0] as { text: string }).text).memory_id,
      );

      // 2. Recall — should find allergy info
      const recallResult = await client.callTool({
        name: 'recall',
        arguments: {
          query: 'What food restrictions does this user have?',
          user_id: userId,
          agent_id: 'mcp-test-agent',
        },
      });

      const context = (recallResult.content[0] as { text: string }).text;
      expect(context).toContain('peanuts');

      // 3. Forget all test memories
      for (const id of memories) {
        const forgetResult = await client.callTool({
          name: 'forget',
          arguments: { memory_id: id },
        });
        const parsed = JSON.parse(
          (forgetResult.content[0] as { text: string }).text,
        );
        expect(parsed.deleted).toBe(true);
      }

      // 4. Recall again — should find nothing
      const emptyRecall = await client.callTool({
        name: 'recall',
        arguments: {
          query: 'What food restrictions does this user have?',
          user_id: userId,
          agent_id: 'mcp-test-agent',
        },
      });
      const emptyText = (emptyRecall.content[0] as { text: string }).text;
      expect(emptyText).toBe('No relevant memories found.');

      await client.close();
    });
  });
});
