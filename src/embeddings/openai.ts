import type { EmbeddingProvider } from './types.js';

/**
 * OpenAI embedding provider using text-embedding-3-small.
 *
 * - 1536 dimensions (default), reducible via `dimensions` param
 * - Better quality than local, especially for longer/nuanced text
 * - Requires OPENAI_API_KEY
 * - ~$0.02 per 1M tokens
 */
export class OpenAIEmbeddingProvider implements EmbeddingProvider {
  readonly name = 'openai';
  readonly dimension = 1536;

  private apiKey: string;
  private model = 'text-embedding-3-small';

  constructor(apiKey: string) {
    if (!apiKey) {
      throw new Error(
        'OpenAI API key required. Set OPENAI_API_KEY in .env or use EMBEDDING_PROVIDER=local.',
      );
    }
    this.apiKey = apiKey;
  }

  async init(): Promise<void> {
    console.log('[memforge] OpenAI embedding provider initialized (text-embedding-3-small, 1536 dims)');
    // No warmup needed for API-based provider
  }

  async embed(text: string): Promise<number[]> {
    const response = await fetch('https://api.openai.com/v1/embeddings', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: this.model,
        input: text,
        encoding_format: 'float',
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`OpenAI embedding API error (${response.status}): ${error}`);
    }

    const data = (await response.json()) as {
      data: Array<{ embedding: number[] }>;
    };

    const embedding = data.data[0]?.embedding;
    if (!embedding) {
      throw new Error('OpenAI returned empty embedding');
    }

    return embedding;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    // OpenAI supports batch embedding in a single API call
    const response = await fetch('https://api.openai.com/v1/embeddings', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: this.model,
        input: texts,
        encoding_format: 'float',
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`OpenAI embedding API error (${response.status}): ${error}`);
    }

    const data = (await response.json()) as {
      data: Array<{ embedding: number[]; index: number }>;
    };

    // Sort by index to match input order
    return data.data
      .sort((a, b) => a.index - b.index)
      .map((d) => d.embedding);
  }
}
