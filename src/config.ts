import { z } from 'zod';

/**
 * Default similarity thresholds per embedding provider.
 *
 * Local models (all-MiniLM-L6-v2, 384 dims) produce lower similarity
 * scores than OpenAI (text-embedding-3-small, 1536 dims). These defaults
 * were calibrated empirically:
 *   - local: exact paraphrase ~0.85, contradiction ~0.78, different topic ~0.20
 *   - openai: exact paraphrase ~0.95, contradiction ~0.88, different topic ~0.30
 */
const PROVIDER_DEFAULTS: Record<string, { dedup: number; conflict: number }> = {
  local: { dedup: 0.85, conflict: 0.65 },
  openai: { dedup: 0.95, conflict: 0.80 },
  azure: { dedup: 0.95, conflict: 0.80 },
};

const ConfigSchema = z.object({
  databaseUrl: z.string().url(),
  embeddingProvider: z.enum(['local', 'openai', 'azure']).default('local'),
  openaiApiKey: z.string().optional(),
  anthropicApiKey: z.string().optional(),
  extractionModel: z.enum(['claude-haiku', 'gpt-4o-mini']).default('claude-haiku'),
  port: z.coerce.number().int().positive().default(3100),
  decayIntervalHours: z.coerce.number().positive().default(24),
  reflectIntervalHours: z.coerce.number().positive().default(12),
  defaultTokenBudget: z.coerce.number().int().positive().default(4000),
  dedupThreshold: z.coerce.number().min(0).max(1).optional(),
  conflictThreshold: z.coerce.number().min(0).max(1).optional(),
  hybridSearchEnabled: z.string().transform((v) => v !== 'false').default('true'),
  rerankerEnabled: z.string().transform((v) => v !== 'false').default('true'),
  llmExtractionEnabled: z.string().transform((v) => v !== 'false').default('false'),
  azureOpenaiEndpoint: z.string().optional(),
  azureOpenaiKey: z.string().optional(),
  azureOpenaiModel: z.string().default('gpt-5.2-chat'),
  azureEmbeddingEndpoint: z.string().optional(),
  queryDecompositionEnabled: z.string().transform((v) => v !== 'false').default('true'),
  agenticRecallEnabled: z.string().transform((v) => v !== 'false').default('false'),
  azureChatEndpoint: z.string().optional(),
});

export type Config = z.infer<typeof ConfigSchema> & {
  dedupThreshold: number;
  conflictThreshold: number;
};

export function loadConfig(): Config {
  const raw = ConfigSchema.parse({
    databaseUrl: process.env['DATABASE_URL'],
    embeddingProvider: process.env['EMBEDDING_PROVIDER'],
    openaiApiKey: process.env['OPENAI_API_KEY'],
    anthropicApiKey: process.env['ANTHROPIC_API_KEY'],
    extractionModel: process.env['EXTRACTION_MODEL'],
    port: process.env['MEMFORGE_PORT'],
    decayIntervalHours: process.env['DECAY_INTERVAL_HOURS'],
    reflectIntervalHours: process.env['REFLECT_INTERVAL_HOURS'],
    defaultTokenBudget: process.env['DEFAULT_TOKEN_BUDGET'],
    dedupThreshold: process.env['DEDUP_THRESHOLD'],
    conflictThreshold: process.env['CONFLICT_THRESHOLD'],
    hybridSearchEnabled: process.env['HYBRID_SEARCH_ENABLED'],
    rerankerEnabled: process.env['RERANKER_ENABLED'],
    llmExtractionEnabled: process.env['LLM_EXTRACTION_ENABLED'],
    azureOpenaiEndpoint: process.env['AZURE_OPENAI_ENDPOINT'],
    azureOpenaiKey: process.env['AZURE_OPENAI_KEY'],
    azureOpenaiModel: process.env['AZURE_OPENAI_MODEL'],
    azureEmbeddingEndpoint: process.env['AZURE_EMBEDDING_ENDPOINT'],
    queryDecompositionEnabled: process.env['QUERY_DECOMPOSITION_ENABLED'],
    agenticRecallEnabled: process.env['AGENTIC_RECALL_ENABLED'],
    azureChatEndpoint: process.env['AZURE_CHAT_ENDPOINT'],
  });

  const defaults = PROVIDER_DEFAULTS[raw.embeddingProvider] ?? PROVIDER_DEFAULTS['local']!;

  return {
    ...raw,
    dedupThreshold: raw.dedupThreshold ?? defaults.dedup,
    conflictThreshold: raw.conflictThreshold ?? defaults.conflict,
  };
}

/**
 * Returns the embedding dimension based on the configured provider.
 * local (all-MiniLM-L6-v2) = 384
 * openai (text-embedding-3-small) = 1536
 */
export function getEmbeddingDimension(provider: Config['embeddingProvider']): number {
  switch (provider) {
    case 'local':
      return 384;
    case 'openai':
      return 1536;
    case 'azure':
      return 3072;
  }
}
