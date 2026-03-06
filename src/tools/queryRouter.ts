import type { Config } from '../config.js';

// ============================================================
// Query types
// ============================================================

export type QueryType = 'DIRECT' | 'AGGREGATION' | 'MULTI_HOP';

export interface QueryClassification {
  type: QueryType;
  subQueries?: string[];
}

// ============================================================
// Regex-based classification (fast, no LLM cost)
// ============================================================

const AGGREGATION_PATTERNS = [
  /\bwhat\b.+\bhas\b.+\b(done|read|painted|bought|made|visited|seen|attended|played|written|tried|started|taken|learned|joined|collected|adopted|rescued)\b/i,
  /\bwhat\b.+\b(activities|events|books|hobbies|pets|instruments|items|things|sports|classes|courses|projects|goals|interests)\b/i,
  /\bwhat\b.+\bdo(es)?\b.+\bdo\b/i,
  /\bhow many\b.+\b(times|events|activities|pets|books|hobbies|children|kids)\b/i,
  /\bwhat\s+(?:are|were)\s+\w+'?s?\s+(?:pet|children|kid|hobby|activit|instrument|book|sport|interest)/i,
  /\bwhere\s+has\b/i,
  /\blist\s+(?:all|the)\b/i,
  /\bwhat\s+(?:all|different)\b/i,
  /\bname\s+(?:all|the)\b/i,
  /\bcan\s+you\s+(?:list|name|tell\s+me\s+all)\b/i,
  /\bwho\s+supports?\b/i,
  /\bwhat\s+(?:type|kind|sort)s?\s+(?:of|are)\b/i,
  /\bwhat\s+(?:symbols?|instruments?|careers?)\s+(?:are|does|do|has|have)\b/i,
  /\bwhat\s+does\s+\w+\s+do\s+(?:to|for|with)\b/i,
];

const MULTI_HOP_PATTERNS = [
  /\bafter\s+(?:the|a|her|his|their)\b/i,
  /\bbefore\s+(?:the|a|her|his|their)\b/i,
  /\bwhen\s+did\s+\w+\s+\w+\s+after\b/i,
  /\bwho\s+supports?\b.+\bwhen\b/i,
  /\bwhat\s+does\s+\w+\s+do\s+(?:with|to|for|during)\b/i,
  /\bhow\s+did\s+\w+\s+(?:feel|react|respond)\s+(?:when|after|about)\b/i,
  /\bwhat\s+happened\s+(?:after|before|when|while)\b/i,
  /\bwhat\s+(?:was|is)\s+the\s+(?:result|outcome|consequence)\b/i,
];

function classifyByRegex(query: string): QueryType | null {
  for (const pattern of AGGREGATION_PATTERNS) {
    if (pattern.test(query)) return 'AGGREGATION';
  }
  // Multi-hop regex disabled: LOCOMO multi-hop questions are inference-type
  // ("Would Caroline likely...") not chain-type ("What happened after X").
  // These are better served by DIRECT with full context.
  return null;
}

// ============================================================
// LLM helper (Azure chat/completions)
// ============================================================

async function callLLM(
  prompt: string,
  config: Config,
  timeoutMs: number = 3000,
): Promise<string | null> {
  if (!config.azureOpenaiEndpoint || !config.azureOpenaiKey) return null;

  // Use Azure Responses API (same format as llmExtractor.ts)
  try {
    const res = await fetch(config.azureOpenaiEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'api-key': config.azureOpenaiKey,
      },
      body: JSON.stringify({
        model: config.azureOpenaiModel,
        input: prompt,
        max_output_tokens: 200,
      }),
      signal: AbortSignal.timeout(timeoutMs),
    });

    if (!res.ok) {
      const body = await res.text().catch(() => '');
      console.warn(`[memforge] LLM call failed (${res.status}): ${body.slice(0, 200)}`);
      return null;
    }

    // Parse Azure Responses API format
    const data = await res.json() as {
      output?: Array<{
        type: string;
        content?: Array<{ type: string; text?: string }>;
      }>;
    };

    for (const item of data.output ?? []) {
      if (item.type === 'message') {
        for (const content of item.content ?? []) {
          if (content.type === 'output_text' && content.text) {
            return content.text.trim();
          }
        }
      }
    }

    return null;
  } catch (err) {
    console.warn(`[memforge] LLM call error: ${(err as Error).message}`);
    return null;
  }
}

// ============================================================
// Chat Completions LLM helper (more reliable under concurrent load)
// ============================================================

/**
 * Call Azure OpenAI using the Chat Completions API.
 * Used by agentic recall gap analysis (15s timeout).
 */
export async function callChatLLM(
  systemPrompt: string,
  userPrompt: string,
  config: Config,
  timeoutMs: number = 15000,
): Promise<string | null> {
  const endpoint = config.azureChatEndpoint;
  const apiKey = config.azureOpenaiKey;
  if (!endpoint || !apiKey) return null;

  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'api-key': apiKey,
      },
      body: JSON.stringify({
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt },
        ],
        max_completion_tokens: 300,
      }),
      signal: AbortSignal.timeout(timeoutMs),
    });

    if (!res.ok) {
      const body = await res.text().catch(() => '');
      console.warn(`[memforge] Chat LLM failed (${res.status}): ${body.slice(0, 200)}`);
      return null;
    }

    const data = await res.json() as {
      choices?: Array<{ message?: { content?: string } }>;
    };
    return data.choices?.[0]?.message?.content?.trim() ?? null;
  } catch (err) {
    console.warn(`[memforge] Chat LLM error: ${(err as Error).message}`);
    return null;
  }
}

// ============================================================
// LLM classification fallback
// ============================================================

async function classifyByLLM(
  query: string,
  config: Config,
): Promise<QueryType | null> {
  const prompt = `Classify this memory retrieval query into exactly one category.

DIRECT: Simple single-answer question. Example: "What is Emily's job?" "Where does John live?"
AGGREGATION: Asks for ALL instances or a complete list. Example: "What books has Emily read?" "What are John's hobbies?" "How many pets does Sarah have?"
MULTI_HOP: Requires connecting multiple facts or reasoning across events. Example: "When did Emily start painting after moving to Portland?" "How did John react when Sarah got the promotion?"

Query: "${query}"
Return ONLY one word: DIRECT, AGGREGATION, or MULTI_HOP`;

  const response = await callLLM(prompt, config);
  if (!response) return null;

  const upper = response.toUpperCase().trim();
  if (upper.includes('AGGREGATION')) return 'AGGREGATION';
  if (upper.includes('MULTI_HOP')) return 'MULTI_HOP';
  if (upper.includes('DIRECT')) return 'DIRECT';

  // Try JSON parse as fallback
  try {
    const parsed = JSON.parse(response) as { type?: string };
    const type = parsed.type?.toUpperCase();
    if (type === 'AGGREGATION' || type === 'MULTI_HOP' || type === 'DIRECT') {
      return type;
    }
  } catch {
    // Not JSON, already handled above
  }

  return null;
}

// ============================================================
// Query decomposition for MULTI_HOP
// ============================================================

export async function decomposeQuery(
  query: string,
  config: Config,
): Promise<string[]> {
  const prompt = `Break this question into 2-3 simpler sub-questions that each retrieve one specific fact needed to answer the original. Each sub-question should be self-contained.

Question: "${query}"
Return ONLY a JSON object: {"sub_queries": ["question 1", "question 2"]}`;

  const response = await callLLM(prompt, config, 5000);
  if (!response) return [];

  try {
    // Extract JSON from response (may have markdown fences)
    const jsonMatch = response.match(/\{[\s\S]*\}/);
    if (!jsonMatch) return [];

    const parsed = JSON.parse(jsonMatch[0]) as { sub_queries?: string[] };
    if (Array.isArray(parsed.sub_queries) && parsed.sub_queries.length >= 2) {
      return parsed.sub_queries.slice(0, 3);
    }
  } catch {
    // Parse failure — fallback to no decomposition
  }

  return [];
}

// ============================================================
// Main classification entry point
// ============================================================

export async function classifyQuery(
  query: string,
  config: Config,
): Promise<QueryClassification> {
  if (!config.queryDecompositionEnabled) {
    return { type: 'DIRECT' };
  }

  // 1. Fast regex classification
  const regexType = classifyByRegex(query);
  if (regexType) {
    console.log(`[memforge] Query classified as ${regexType} (regex)`);
    return { type: regexType };
  }

  // 2. Default to DIRECT (no LLM classification — regex is fast and predictable)
  return { type: 'DIRECT' };
}
