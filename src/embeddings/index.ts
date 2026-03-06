import type { Config } from '../config.js';
import type { EmbeddingProvider } from './types.js';
import { LocalEmbeddingProvider } from './local.js';
import { OpenAIEmbeddingProvider } from './openai.js';
import { AzureEmbeddingProvider } from './azure.js';

export type { EmbeddingProvider } from './types.js';

/**
 * Create and initialize an embedding provider based on config.
 * Local (all-MiniLM-L6-v2) is the default — zero API keys needed.
 * OpenAI and Azure are opt-in for better quality.
 */
export async function createEmbeddingProvider(
  config: Config,
): Promise<EmbeddingProvider> {
  let provider: EmbeddingProvider;

  switch (config.embeddingProvider) {
    case 'local':
      provider = new LocalEmbeddingProvider();
      break;

    case 'openai':
      if (!config.openaiApiKey) {
        throw new Error(
          'EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY. ' +
          'Set it in .env or switch to EMBEDDING_PROVIDER=local.',
        );
      }
      provider = new OpenAIEmbeddingProvider(config.openaiApiKey);
      break;

    case 'azure': {
      const endpoint = config.azureEmbeddingEndpoint;
      const key = config.azureOpenaiKey;
      if (!endpoint || !key) {
        throw new Error(
          'EMBEDDING_PROVIDER=azure requires AZURE_EMBEDDING_ENDPOINT and AZURE_OPENAI_KEY.',
        );
      }
      provider = new AzureEmbeddingProvider(endpoint, key);
      break;
    }

    default:
      throw new Error(`Unknown embedding provider: ${config.embeddingProvider}`);
  }

  await provider.init();
  return provider;
}
