import pg from "pg";

const { Pool } = pg;

let pool: pg.Pool | null = null;

export interface DatabaseConfig {
  connectionString: string;
}

/**
 * Initialize the connection pool.
 * Call once at startup.
 */
export function initPool(config: DatabaseConfig): pg.Pool {
  pool = new Pool({
    connectionString: config.connectionString,
    max: 20,
    idleTimeoutMillis: 30_000,
    connectionTimeoutMillis: 5_000,
  });

  pool.on("error", (err) => {
    console.error("[memforge] Unexpected pool error:", err.message);
  });

  return pool;
}

/**
 * Get the active connection pool.
 * Throws if initPool() hasn't been called.
 */
export function getPool(): pg.Pool {
  if (!pool) {
    throw new Error("Database pool not initialized. Call initPool() first.");
  }
  return pool;
}

/**
 * Run a query with parameterized values.
 * Never use string interpolation for user input.
 */
export async function query<T extends pg.QueryResultRow = pg.QueryResultRow>(
  text: string,
  values?: unknown[],
): Promise<pg.QueryResult<T>> {
  const p = getPool();
  const start = Date.now();
  const result = await p.query<T>(text, values);
  const duration = Date.now() - start;

  if (duration > 100) {
    console.warn(`[memforge] Slow query (${duration}ms):`, text.slice(0, 80));
  }

  return result;
}

/**
 * Execute multiple statements in a transaction.
 * Automatically commits on success, rolls back on error.
 */
export async function transaction<T>(
  fn: (client: pg.PoolClient) => Promise<T>,
): Promise<T> {
  const p = getPool();
  const client = await p.connect();

  try {
    await client.query("BEGIN");
    const result = await fn(client);
    await client.query("COMMIT");
    return result;
  } catch (err) {
    await client.query("ROLLBACK");
    throw err;
  } finally {
    client.release();
  }
}

/**
 * Gracefully close the pool.
 */
export async function closePool(): Promise<void> {
  if (pool) {
    await pool.end();
    pool = null;
  }
}
