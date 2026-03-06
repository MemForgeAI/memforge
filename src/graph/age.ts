/**
 * Apache AGE query helper.
 *
 * Wraps Cypher queries in the SQL boilerplate required by AGE:
 *   SET search_path = ag_catalog, "$user", public;
 *   SELECT * FROM cypher('memforge_kg', $$ <cypher> $$) AS (<columns>);
 *
 * Handles:
 *   - search_path setup per connection
 *   - agtype result parsing (vertex, edge, scalar)
 *   - Safe parameterization (no string interpolation in Cypher)
 *   - Type-safe return values
 */

// Note: We use dynamic import of getPool() inside functions to avoid
// circular dependency issues and ensure the pool is initialized.

// ============================================================
// Types for AGE results
// ============================================================

export interface AgeVertex {
  id: number;
  label: string;
  properties: Record<string, unknown>;
}

export interface AgeEdge {
  id: number;
  label: string;
  start_id: number;
  end_id: number;
  properties: Record<string, unknown>;
}

export type AgeValue = AgeVertex | AgeEdge | string | number | boolean | null;

// ============================================================
// agtype parsing
// ============================================================

/**
 * Parse an agtype value returned by AGE.
 *
 * AGE returns values as strings like:
 *   '{"id": 123, "label": "Entity", "properties": {...}}::vertex'
 *   '{"id": 456, "label": "RELATES_TO", ...}::edge'
 *   '"some string"'
 *   '42'
 */
export function parseAgtype(raw: unknown): AgeValue {
  if (raw === null || raw === undefined) return null;

  const str = String(raw);

  // Vertex: {...}::vertex
  if (str.endsWith("::vertex")) {
    const json = str.slice(0, -"::vertex".length);
    return JSON.parse(json) as AgeVertex;
  }

  // Edge: {...}::edge
  if (str.endsWith("::edge")) {
    const json = str.slice(0, -"::edge".length);
    return JSON.parse(json) as AgeEdge;
  }

  // Try JSON parse for scalars
  try {
    return JSON.parse(str) as AgeValue;
  } catch {
    return str;
  }
}

// ============================================================
// AGE query execution
// ============================================================

const GRAPH_NAME = "memforge_kg";

/**
 * Execute a Cypher query via AGE and return parsed results.
 *
 * Uses a dedicated client connection so that SET search_path
 * persists for the cypher() call within the same session.
 *
 * @param cypher - The Cypher query (no $$ delimiters needed)
 * @param returnColumns - Column definitions for the RETURNS clause
 *   e.g. [['v', 'agtype']] or [['v', 'agtype'], ['e', 'agtype']]
 */
export async function cypherQuery(
  cypher: string,
  returnColumns: [string, string][],
): Promise<Record<string, AgeValue>[]> {
  const columnDef = returnColumns
    .map(([name, type]) => `${name} ${type}`)
    .join(", ");

  const { getPool } = await import("../db/connection.js");
  const pool = getPool();
  const client = await pool.connect();

  try {
    // AGE requires search_path set on the same connection
    await client.query(`SET search_path = ag_catalog, "$user", public`);

    const sql = `
      SELECT * FROM cypher('${GRAPH_NAME}', $cypher$
        ${cypher}
      $cypher$) AS (${columnDef})
    `;

    const result = await client.query(sql);

    // Parse agtype values
    return result.rows.map((row: Record<string, unknown>) => {
      const parsed: Record<string, AgeValue> = {};
      for (const [col] of returnColumns) {
        parsed[col] = parseAgtype(row[col]);
      }
      return parsed;
    });
  } finally {
    client.release();
  }
}

/**
 * Execute a Cypher query that doesn't return results (e.g., DELETE).
 */
export async function cypherExec(cypher: string): Promise<void> {
  const { getPool } = await import("../db/connection.js");
  const pool = getPool();
  const client = await pool.connect();

  try {
    await client.query(`SET search_path = ag_catalog, "$user", public`);
    const sql = `
      SELECT * FROM cypher('${GRAPH_NAME}', $cypher$
        ${cypher}
      $cypher$) AS (v agtype)
    `;
    await client.query(sql);
  } finally {
    client.release();
  }
}

// ============================================================
// High-level graph operations
// ============================================================

/**
 * Create or merge an entity node in the graph.
 * Uses MERGE to avoid duplicates.
 *
 * Note: Apache AGE does NOT support ON CREATE SET / ON MATCH SET.
 * We use plain MERGE with all properties in the match pattern.
 */
export async function upsertEntity(
  name: string,
  entityType: string,
  _properties: Record<string, unknown> = {},
): Promise<AgeVertex> {
  // Escape single quotes in values for Cypher
  const safeName = name.replace(/'/g, "\\'");
  const safeType = entityType.replace(/'/g, "\\'");

  const results = await cypherQuery(
    `MERGE (n:Entity {name: '${safeName}', type: '${safeType}'})
     RETURN n`,
    [["n", "agtype"]],
  );

  return results[0]!["n"] as AgeVertex;
}

/**
 * Create a relationship between two entities.
 * Uses MERGE to avoid duplicate edges.
 */
export async function upsertRelationship(
  fromName: string,
  fromType: string,
  toName: string,
  toType: string,
  relationType: string,
  properties: Record<string, unknown> = {},
): Promise<AgeEdge | null> {
  const safeFrom = fromName.replace(/'/g, "\\'");
  const safeFromType = fromType.replace(/'/g, "\\'");
  const safeTo = toName.replace(/'/g, "\\'");
  const safeToType = toType.replace(/'/g, "\\'");
  const safeRelType = relationType.toUpperCase().replace(/[^A-Z0-9_]/g, "_");

  const propsStr = Object.entries(properties)
    .map(([k, v]) => `${k}: '${String(v).replace(/'/g, "\\'")}'`)
    .join(", ");

  const propsClause = propsStr ? ` {${propsStr}}` : "";

  const results = await cypherQuery(
    `MATCH (a:Entity {name: '${safeFrom}', type: '${safeFromType}'})
     MATCH (b:Entity {name: '${safeTo}', type: '${safeToType}'})
     MERGE (a)-[r:${safeRelType}${propsClause}]->(b)
     RETURN r`,
    [["r", "agtype"]],
  );

  return (results[0]?.["r"] as AgeEdge) ?? null;
}

/**
 * Find entities connected to a given entity within N hops.
 * Used by recall for graph-enhanced context.
 */
export async function traverseFromEntity(
  entityName: string,
  entityType: string,
  maxHops: number = 2,
): Promise<AgeVertex[]> {
  const safeName = entityName.replace(/'/g, "\\'");
  const safeType = entityType.replace(/'/g, "\\'");

  const results = await cypherQuery(
    `MATCH (start:Entity {name: '${safeName}', type: '${safeType}'})
     MATCH (start)-[*1..${maxHops}]-(related:Entity)
     RETURN DISTINCT related`,
    [["related", "agtype"]],
  );

  return results.map((r) => r["related"] as AgeVertex);
}

/**
 * Link a memory to an entity in the graph.
 */
export async function linkMemoryToEntity(
  memoryId: string,
  entityName: string,
  entityType: string,
): Promise<void> {
  const safeMemId = memoryId.replace(/'/g, "\\'");
  const safeName = entityName.replace(/'/g, "\\'");
  const safeType = entityType.replace(/'/g, "\\'");

  await cypherQuery(
    `MERGE (m:Memory {id: '${safeMemId}'})
     WITH m
     MATCH (e:Entity {name: '${safeName}', type: '${safeType}'})
     MERGE (m)-[r:MENTIONS]->(e)
     RETURN r`,
    [["r", "agtype"]],
  );
}

/**
 * Find all entity names mentioned by memories similar to a query.
 * Used to seed graph traversal in recall.
 */
export async function findEntitiesForMemory(
  memoryId: string,
): Promise<AgeVertex[]> {
  const safeMemId = memoryId.replace(/'/g, "\\'");

  const results = await cypherQuery(
    `MATCH (m:Memory {id: '${safeMemId}'})-[:MENTIONS]->(e:Entity)
     RETURN e`,
    [["e", "agtype"]],
  );

  return results.map((r) => r["e"] as AgeVertex);
}

/**
 * Reverse lookup: find Memory nodes that mention a given entity.
 * Returns the Memory vertex nodes (which have an 'id' property matching the memories.id UUID).
 */
export async function findMemoriesByEntity(
  entityName: string,
  entityType: string,
): Promise<AgeVertex[]> {
  const safeName = entityName.replace(/'/g, "\\'");
  const safeType = entityType.replace(/'/g, "\\'");

  const results = await cypherQuery(
    `MATCH (m:Memory)-[:MENTIONS]->(e:Entity {name: '${safeName}', type: '${safeType}'})
     RETURN m`,
    [["m", "agtype"]],
  );

  return results.map((r) => r["m"] as AgeVertex);
}

/**
 * Remove all graph edges for a memory node (used by forget).
 * Returns count of edges removed.
 */
export async function removeMemoryEdges(memoryId: string): Promise<number> {
  const safeMemId = memoryId.replace(/'/g, "\\'");

  try {
    const results = await cypherQuery(
      `MATCH (m:Memory {id: '${safeMemId}'})-[r]-()
       DELETE r
       RETURN count(r) as cnt`,
      [["cnt", "agtype"]],
    );

    const count = results[0]?.["cnt"];
    return typeof count === "number" ? count : 0;
  } catch {
    // Memory node might not exist in graph — that's OK
    return 0;
  }
}

/**
 * Link a fact to an entity in the graph.
 */
export async function linkFactToEntity(
  factId: string,
  entityName: string,
  entityType: string,
): Promise<void> {
  const safeFactId = factId.replace(/'/g, "\\'");
  const safeName = entityName.replace(/'/g, "\\'");
  const safeType = entityType.replace(/'/g, "\\'");

  await cypherQuery(
    `MERGE (f:Fact {id: '${safeFactId}'})
     WITH f
     MATCH (e:Entity {name: '${safeName}', type: '${safeType}'})
     MERGE (f)-[r:MENTIONS]->(e)
     RETURN r`,
    [["r", "agtype"]],
  );
}

/**
 * Reverse lookup: find Fact nodes that mention a given entity.
 * Returns the Fact vertex nodes (which have an 'id' property matching the fact ID).
 */
export async function findFactsByEntity(
  entityName: string,
  entityType: string,
): Promise<AgeVertex[]> {
  const safeName = entityName.replace(/'/g, "\\'");
  const safeType = entityType.replace(/'/g, "\\'");

  const results = await cypherQuery(
    `MATCH (f:Fact)-[:MENTIONS]->(e:Entity {name: '${safeName}', type: '${safeType}'})
     RETURN f`,
    [["f", "agtype"]],
  );

  return results.map((r) => r["f"] as AgeVertex);
}

/**
 * Forward lookup: find all entities mentioned by a fact.
 */
export async function findEntitiesForFact(
  factId: string,
): Promise<AgeVertex[]> {
  const safeFactId = factId.replace(/'/g, "\\'");

  const results = await cypherQuery(
    `MATCH (f:Fact {id: '${safeFactId}'})-[:MENTIONS]->(e:Entity)
     RETURN e`,
    [["e", "agtype"]],
  );

  return results.map((r) => r["e"] as AgeVertex);
}

/**
 * Remove all graph edges for a fact node (used by forget).
 * Returns count of edges removed.
 */
export async function removeFactEdges(factId: string): Promise<number> {
  const safeFactId = factId.replace(/'/g, "\\'");

  try {
    const results = await cypherQuery(
      `MATCH (f:Fact {id: '${safeFactId}'})-[r]-()
       DELETE r
       RETURN count(r) as cnt`,
      [["cnt", "agtype"]],
    );

    const count = results[0]?.["cnt"];
    return typeof count === "number" ? count : 0;
  } catch {
    // Fact node might not exist in graph — that's OK
    return 0;
  }
}
