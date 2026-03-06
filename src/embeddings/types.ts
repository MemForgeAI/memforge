/**
 * Embedding provider interface.
 * All providers must implement this contract.
 * Dimension is determined by the provider, not hardcoded.
 */
export interface EmbeddingProvider {
  /** Human-readable provider name */
  readonly name: string;

  /** Vector dimension this provider produces */
  readonly dimension: number;

  /**
   * Initialize the provider (load model, warm up, etc).
   * Called once at startup. May take a few seconds for local models.
   */
  init(): Promise<void>;

  /**
   * Generate an embedding vector for a single text string.
   * Returns a float array of length `dimension`.
   */
  embed(text: string): Promise<number[]>;

  /**
   * Generate embeddings for multiple texts in a single batch.
   * Default implementation calls embed() sequentially.
   * Providers can override for batch API calls.
   */
  embedBatch(texts: string[]): Promise<number[][]>;
}
