/**
 * Memory Card — MemForge's standardized memory representation.
 *
 * Like Pantone assigns a precise code to every color, MemForge transforms
 * raw observations into standardized, self-describing memory cards that
 * are DESIGNED to be found by any query.
 *
 * A raw observation "Alex uses TypeScript with strict mode enabled for
 * Node.js projects" becomes:
 *
 * {
 *   atomic_facts: [
 *     "Alex uses TypeScript",
 *     "Alex enables TypeScript strict mode",
 *     "Alex uses Node.js"
 *   ],
 *   categories: ["programming_language", "configuration", "runtime"],
 *   entities: [
 *     { name: "Alex", type: "person" },
 *     { name: "TypeScript", type: "programming_language" },
 *     { name: "Node.js", type: "runtime" },
 *     { name: "strict mode", type: "configuration" }
 *   ],
 *   query_signatures: [
 *     "What programming languages does Alex use?",
 *     "Does Alex use TypeScript?",
 *     "What is Alex's TypeScript configuration?",
 *     "What does Alex use for backend development?",
 *     "Does Alex use strict mode?"
 *   ],
 *   temporal: {
 *     is_current: true,
 *     supersedes: null,
 *     time_expression: null
 *   },
 *   importance: 0.7,
 *   memory_type: "semantic"
 * }
 *
 * The query_signatures are the "Pantone codes" — precise questions that
 * this memory answers. At recall time, the query matches against these
 * signatures with near-perfect cosine similarity.
 */

export interface MemoryCard {
  /** The original raw observation */
  original: string;

  /** Atomic facts extracted — one fact per statement, no compounds */
  atomic_facts: string[];

  /** Standardized categories from a controlled vocabulary */
  categories: string[];

  /** Named entities with typed relationships */
  entities: Array<{
    name: string;
    type: string;
  }>;

  /** Questions this memory answers — the "Pantone codes" for retrieval */
  query_signatures: string[];

  /** Temporal awareness */
  temporal: {
    is_current: boolean;
    supersedes: string | null; // fact this replaces (e.g., "Alex used VS Code")
    time_expression: string | null; // "recently", "last week", "since 2024"
  };

  /** Estimated importance 0-1 */
  importance: number;

  /** Memory classification */
  memory_type: "semantic" | "episodic" | "procedural";
}

export interface MemoryCardConfig {
  endpoint: string;
  apiKey: string;
  model: string;
}

const MEMORY_CARD_PROMPT = `You are a memory extraction system. Transform the raw observation into a structured memory card.

IMPORTANT: Return ONLY valid JSON, no markdown, no explanation.

Raw observation: "{OBSERVATION}"

Extract:
{
  "atomic_facts": ["break compound statements into individual atomic facts - ALWAYS include dates/times in each fact if available"],
  "categories": ["use standardized categories: person, relationship, family, friend, hobby, art, music, sport, travel, event, career, education, health, identity, belief, preference, possession, pet, milestone, emotion, plan, community, tool, programming_language, framework, project, workflow"],
  "entities": [{"name": "exact name", "type": "person|place|organization|event|hobby|pet|item|tool|concept|programming_language|framework"}],
  "query_signatures": ["10-12 natural questions this memory could answer — IMPORTANT: generate at least 2 questions per atomic fact. Include aggregation queries ('What [category] does [person] have?'), detail queries, temporal queries, and synonym-based variants."],
  "temporal": {"is_current": true, "supersedes": null, "time_expression": null},
  "importance": 0.5,
  "memory_type": "semantic or episodic or procedural"
}

Rules:
- atomic_facts: Extract ONLY concrete, verifiable facts. Good: "Caroline attended the LGBTQ support group", "Melanie painted a sunrise in 2022", "Caroline is interested in counseling". Bad: "Caroline expressed confidence", "Melanie said she was proud", "Caroline believes everyone has unique paths". If a statement is a sentiment, compliment, or motivational expression, SKIP IT.
- atomic_facts: CRITICAL — if a date/time appears (e.g., "[7 May 2023]"), INCLUDE the date in every fact. Example: "[7 May 2023] Caroline: I went to the LGBTQ support group" → ["Caroline attended the LGBTQ support group on 7 May 2023"]. Convert relative dates to absolute.
- atomic_facts: Split compound statements. "I went camping and painted a sunrise" → two separate facts with dates.
- atomic_facts: Use third person and the speaker's name. NEVER use "the recipient" or "the person" — always use the actual name. "I went hiking" from Caroline → "Caroline went hiking on [date]"
- atomic_facts: Maximum 8 facts per observation. Extract ALL concrete facts, especially names, titles, numbers, and specific details.
- atomic_facts: ALWAYS preserve proper nouns EXACTLY as stated — book titles ("Nothing is Impossible", "Becoming Nicole"), song names, place names, organization names. Never paraphrase proper nouns.
- atomic_facts: ALWAYS include specific quantities and numbers — "two children", "three cats", "five years", "$500".
- query_signatures: Generate DIVERSE questions. Include "When did [person] [action]?", "What did [person] do?", "What [category] does [person] [verb]?", "Has [person] ever [action]?"
- If the observation mentions a CHANGE (switched, moved, stopped, started, now), set temporal.supersedes to what was replaced
- importance: 0.9 for life events/milestones, 0.7 for activities/hobbies/preferences, 0.5 for casual facts, 0.1 for greetings (which should be skipped from atomic_facts anyway)`;

/**
 * Transform a raw observation into a standardized Memory Card.
 */
export async function createMemoryCard(
  observation: string,
  config: MemoryCardConfig,
): Promise<MemoryCard> {
  const prompt = MEMORY_CARD_PROMPT.replace(
    "{OBSERVATION}",
    observation.replace(/"/g, '\\"'),
  );

  const body = JSON.stringify({
    model: config.model,
    input: prompt,
    max_output_tokens: 1200,
  });

  const response = await fetch(config.endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "api-key": config.apiKey,
    },
    body,
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(
      `Memory card extraction failed (${response.status}): ${error}`,
    );
  }

  const data = (await response.json()) as {
    output?: Array<{
      type: string;
      content?: Array<{
        type: string;
        text?: string;
      }>;
    }>;
  };

  for (const item of data.output ?? []) {
    if (item.type === "message") {
      for (const content of item.content ?? []) {
        if (content.type === "output_text" && content.text) {
          try {
            const card = JSON.parse(content.text) as Omit<
              MemoryCard,
              "original"
            >;
            return { ...card, original: observation };
          } catch {
            // Try extracting JSON object from the text
            const match = content.text.match(/\{[\s\S]*\}/);
            if (match) {
              try {
                const card = JSON.parse(match[0]) as Omit<
                  MemoryCard,
                  "original"
                >;
                return { ...card, original: observation };
              } catch {
                // Try to repair truncated JSON
                const repaired = repairJson(match[0]);
                if (repaired) {
                  const card = JSON.parse(repaired) as Omit<
                    MemoryCard,
                    "original"
                  >;
                  return { ...card, original: observation };
                }
              }
            }
          }
        }
      }
    }
  }

  // Fallback: return a basic card
  return {
    original: observation,
    atomic_facts: [observation],
    categories: [],
    entities: [],
    query_signatures: [],
    temporal: { is_current: true, supersedes: null, time_expression: null },
    importance: 0.5,
    memory_type: "semantic",
  };
}

/**
 * Attempt to repair truncated JSON from LLM output.
 * Handles common truncation cases: unclosed strings, arrays, objects.
 */
function repairJson(text: string): string | null {
  let json = text.trim();

  // Close any unclosed strings
  const quoteCount = (json.match(/(?<!\\)"/g) || []).length;
  if (quoteCount % 2 !== 0) {
    json += '"';
  }

  // Count unclosed brackets/braces
  let braces = 0;
  let brackets = 0;
  let inString = false;

  for (let i = 0; i < json.length; i++) {
    const ch = json[i];
    if (ch === '"' && (i === 0 || json[i - 1] !== "\\")) {
      inString = !inString;
      continue;
    }
    if (inString) continue;
    if (ch === "{") braces++;
    else if (ch === "}") braces--;
    else if (ch === "[") brackets++;
    else if (ch === "]") brackets--;
  }

  // Remove trailing comma before closing
  json = json.replace(/,\s*$/, "");

  // Close unclosed brackets and braces
  for (let i = 0; i < brackets; i++) json += "]";
  for (let i = 0; i < braces; i++) json += "}";

  try {
    JSON.parse(json);
    return json;
  } catch {
    return null;
  }
}
