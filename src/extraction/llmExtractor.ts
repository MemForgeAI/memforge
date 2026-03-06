/**
 * LLM-based memory extraction using Azure OpenAI.
 *
 * Extracts structured tags, entities, and query hints from memory content.
 * The query_hints are questions this memory could answer — when stored
 * alongside the memory, they dramatically improve recall by bridging
 * the vocabulary gap between queries and memories.
 *
 * Cost: ~$0.001 per extraction (gpt-5.2-chat / gpt-4o-mini)
 */

export interface LLMExtractionResult {
  categories: string[];
  entities: string[];
  query_hints: string[];
}

export interface LLMExtractorConfig {
  endpoint: string;  // Azure OpenAI endpoint URL
  apiKey: string;
  model: string;     // deployment name (e.g., 'gpt-5.2-chat')
}

const EXTRACTION_PROMPT = `Extract structured tags from this memory. Return ONLY valid JSON, no markdown.

Memory: "{MEMORY}"

Return:
{"categories": ["short category labels"], "entities": ["proper nouns and key terms"], "query_hints": ["2-3 questions this memory could answer"]}`;

/**
 * Extract structured metadata from a memory using an LLM.
 */
export async function llmExtract(
  content: string,
  config: LLMExtractorConfig,
): Promise<LLMExtractionResult> {
  const prompt = EXTRACTION_PROMPT.replace('{MEMORY}', content.replace(/"/g, '\\"'));

  const body = JSON.stringify({
    model: config.model,
    input: prompt,
    max_output_tokens: 250,
  });

  const response = await fetch(config.endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'api-key': config.apiKey,
    },
    body,
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`LLM extraction failed (${response.status}): ${error}`);
  }

  const data = await response.json() as {
    output?: Array<{
      type: string;
      content?: Array<{
        type: string;
        text?: string;
      }>;
    }>;
  };

  // Parse the response — Azure Responses API format
  for (const item of data.output ?? []) {
    if (item.type === 'message') {
      for (const content of item.content ?? []) {
        if (content.type === 'output_text' && content.text) {
          try {
            return JSON.parse(content.text) as LLMExtractionResult;
          } catch {
            // Try to extract JSON from text that might have extra content
            const match = content.text.match(/\{[\s\S]*\}/);
            if (match) {
              return JSON.parse(match[0]) as LLMExtractionResult;
            }
          }
        }
      }
    }
  }

  // Fallback if LLM didn't return valid JSON
  return { categories: [], entities: [], query_hints: [] };
}
