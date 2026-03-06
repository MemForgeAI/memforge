import type { EmbeddingProvider } from "./types.js";

/**
 * Azure OpenAI embedding provider using text-embedding-3-large.
 *
 * - 3072 dimensions (default for large model)
 * - Higher quality than MiniLM for nuanced text and longer passages
 * - Requires AZURE_EMBEDDING_ENDPOINT and AZURE_OPENAI_KEY
 * - Uses Azure OpenAI REST API
 */
export class AzureEmbeddingProvider implements EmbeddingProvider {
  readonly name = "azure";
  readonly dimension = 3072;

  private endpoint: string;
  private apiKey: string;

  constructor(endpoint: string, apiKey: string) {
    if (!endpoint) {
      throw new Error(
        "Azure embedding endpoint required. Set AZURE_EMBEDDING_ENDPOINT.",
      );
    }
    if (!apiKey) {
      throw new Error("Azure API key required. Set AZURE_OPENAI_KEY.");
    }
    this.endpoint = endpoint;
    this.apiKey = apiKey;
  }

  async init(): Promise<void> {
    console.log(
      "[memforge] Azure embedding provider initialized (text-embedding-3-large, 3072 dims)",
    );
  }

  async embed(text: string): Promise<number[]> {
    const response = await fetch(this.endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "api-key": this.apiKey,
      },
      body: JSON.stringify({
        input: text,
        encoding_format: "float",
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(
        `Azure embedding API error (${response.status}): ${error}`,
      );
    }

    const data = (await response.json()) as {
      data: Array<{ embedding: number[] }>;
    };

    const embedding = data.data[0]?.embedding;
    if (!embedding) {
      throw new Error("Azure returned empty embedding");
    }

    return embedding;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    const response = await fetch(this.endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "api-key": this.apiKey,
      },
      body: JSON.stringify({
        input: texts,
        encoding_format: "float",
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(
        `Azure embedding API error (${response.status}): ${error}`,
      );
    }

    const data = (await response.json()) as {
      data: Array<{ embedding: number[]; index: number }>;
    };

    return data.data.sort((a, b) => a.index - b.index).map((d) => d.embedding);
  }
}
