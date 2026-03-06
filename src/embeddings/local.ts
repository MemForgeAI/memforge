import type { EmbeddingProvider } from './types.js';

/**
 * Local embedding provider using all-MiniLM-L6-v2 via @xenova/transformers.
 *
 * - 384 dimensions
 * - Runs on CPU, no API key needed
 * - First load: ~2-3 seconds (downloads + compiles ONNX model)
 * - Subsequent embeds: ~15-30ms per text
 * - Good enough for memory retrieval at the scale MemForge targets
 */
export class LocalEmbeddingProvider implements EmbeddingProvider {
  readonly name = 'local';
  readonly dimension = 384;

  private pipeline: unknown = null;

  async init(): Promise<void> {
    console.log('[memforge] Loading local embedding model (all-MiniLM-L6-v2)...');
    const start = Date.now();

    // Dynamic import — @xenova/transformers is ESM-only
    const { pipeline } = await import('@xenova/transformers');
    this.pipeline = await pipeline(
      'feature-extraction',
      'Xenova/all-MiniLM-L6-v2',
    );

    const elapsed = Date.now() - start;
    console.log(`[memforge] Local embedding model loaded in ${elapsed}ms`);
  }

  async embed(text: string): Promise<number[]> {
    if (!this.pipeline) {
      throw new Error('Local embedding provider not initialized. Call init() first.');
    }

    const pipe = this.pipeline as (
      text: string,
      options: { pooling: string; normalize: boolean },
    ) => Promise<{ data: Float32Array }>;

    const output = await pipe(text, {
      pooling: 'mean',
      normalize: true,
    });

    return Array.from(output.data);
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    // Local model doesn't benefit from batching via API,
    // but we can process sequentially. For v0.1 this is fine.
    const results: number[][] = [];
    for (const text of texts) {
      results.push(await this.embed(text));
    }
    return results;
  }
}
