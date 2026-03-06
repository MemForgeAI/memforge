/**
 * Entity extraction from memory content.
 *
 * Rule-based extraction for common entity types:
 *   - person: Names (capitalized words)
 *   - location: Cities, countries, places
 *   - organization: Company names
 *   - preference: Liked/disliked things
 *   - topic: Key nouns/concepts
 *
 * In later phases, this can be augmented with LLM-based extraction
 * (Claude Haiku / GPT-4o-mini) for higher accuracy.
 */

export interface ExtractedEntity {
  name: string;
  type: string;
  confidence: number;
}

export interface ExtractionResult {
  entities: ExtractedEntity[];
  relationships: Array<{
    from: { name: string; type: string };
    to: { name: string; type: string };
    relation: string;
  }>;
}

// ============================================================
// Pattern-based extraction
// ============================================================

// Common location keywords
const LOCATION_PATTERNS = [
  /\b(?:in|to|from|at|near|around|visiting)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b/g,
  /\b(New York|San Francisco|Los Angeles|Hong Kong|United Kingdom|United States|North America|South America|Central Park)\b/gi,
];

// Named entities (capitalized multi-word phrases not at sentence start)
const PERSON_PATTERNS = [
  /\b(?:name is|called|named|known as|goes by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b/g,
  /\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b/g,
  /\b(?:does|did|has|have|is|was|do|can|will|would|could|should)\s+([A-Z][a-z]+)\b/g,
  /\b([A-Z][a-z]+)(?:'s)\b/g,
];

// Organization patterns
const ORG_PATTERNS = [
  /\b(?:works?\s+(?:at|for)|employed\s+(?:at|by)|company\s+(?:is|called))\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b/g,
  /\b([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*)\s+(?:Inc|Corp|LLC|Ltd|Company|Co|Group|Foundation)\b/g,
];

// Preference extraction (what the user likes/dislikes)
const PREFERENCE_PATTERNS = [
  /\b(?:prefers?|likes?|loves?|enjoys?|favou?rites?)\s+(.+?)(?:\.|,|$)/gi,
  /\b(?:hates?|dislikes?|avoids?|allergic to)\s+(.+?)(?:\.|,|$)/gi,
];

// Topic extraction — key nouns
const TOPIC_PATTERNS = [
  /\b(?:about|regarding|concerning|related to)\s+(.+?)(?:\.|,|$)/gi,
];

// ============================================================
// Vocabulary-based extraction (tech terms, tools, frameworks)
// ============================================================

/**
 * Known tech vocabulary. When these terms appear in memory content,
 * extract them as entities regardless of surrounding syntax.
 * This catches "TypeScript", "Cursor", "PostgreSQL" that pattern-based
 * extraction misses because they don't follow "prefers X" patterns.
 */
const TECH_VOCABULARY: Record<string, { type: string; confidence: number }> = {
  // Programming languages
  'TypeScript': { type: 'technology', confidence: 0.8 },
  'JavaScript': { type: 'technology', confidence: 0.8 },
  'Go': { type: 'technology', confidence: 0.7 },
  'Rust': { type: 'technology', confidence: 0.8 },
  'Python': { type: 'technology', confidence: 0.8 },
  'Java': { type: 'technology', confidence: 0.7 },
  'Ruby': { type: 'technology', confidence: 0.8 },
  'Swift': { type: 'technology', confidence: 0.7 },
  'Kotlin': { type: 'technology', confidence: 0.8 },
  'PHP': { type: 'technology', confidence: 0.8 },
  'Deno': { type: 'technology', confidence: 0.8 },
  'Node.js': { type: 'technology', confidence: 0.8 },

  // Editors & IDEs
  'VS Code': { type: 'tool', confidence: 0.9 },
  'VSCode': { type: 'tool', confidence: 0.9 },
  'Cursor': { type: 'tool', confidence: 0.9 },
  'Vim': { type: 'tool', confidence: 0.8 },
  'Neovim': { type: 'tool', confidence: 0.8 },
  'IntelliJ': { type: 'tool', confidence: 0.8 },

  // Databases
  'PostgreSQL': { type: 'technology', confidence: 0.9 },
  'Redis': { type: 'technology', confidence: 0.9 },
  'MongoDB': { type: 'technology', confidence: 0.9 },
  'MySQL': { type: 'technology', confidence: 0.9 },
  'SQLite': { type: 'technology', confidence: 0.9 },

  // Frameworks & libraries
  'React': { type: 'technology', confidence: 0.8 },
  'Next.js': { type: 'technology', confidence: 0.8 },
  'Express': { type: 'technology', confidence: 0.7 },
  'Prisma': { type: 'technology', confidence: 0.9 },
  'GraphQL': { type: 'technology', confidence: 0.8 },
  'REST': { type: 'technology', confidence: 0.7 },
  'gRPC': { type: 'technology', confidence: 0.8 },

  // Testing
  'Jest': { type: 'tool', confidence: 0.8 },
  'Vitest': { type: 'tool', confidence: 0.8 },
  'pytest': { type: 'tool', confidence: 0.8 },
  'Cypress': { type: 'tool', confidence: 0.8 },
  'Playwright': { type: 'tool', confidence: 0.8 },

  // DevOps & infrastructure
  'Docker': { type: 'technology', confidence: 0.9 },
  'Kubernetes': { type: 'technology', confidence: 0.9 },
  'Helm': { type: 'tool', confidence: 0.8 },
  'Terraform': { type: 'tool', confidence: 0.8 },
  'GitHub Actions': { type: 'tool', confidence: 0.8 },
  'GitHub': { type: 'tool', confidence: 0.7 },
  'Git': { type: 'tool', confidence: 0.7 },

  // Package managers
  'npm': { type: 'tool', confidence: 0.7 },
  'pnpm': { type: 'tool', confidence: 0.8 },
  'yarn': { type: 'tool', confidence: 0.7 },

  // Code quality
  'ESLint': { type: 'tool', confidence: 0.8 },
  'Prettier': { type: 'tool', confidence: 0.8 },
  'EditorConfig': { type: 'tool', confidence: 0.8 },

  // Logging & monitoring
  'Pino': { type: 'tool', confidence: 0.8 },
  'OpenTelemetry': { type: 'technology', confidence: 0.8 },
  'Prometheus': { type: 'tool', confidence: 0.8 },

  // Feature management
  'LaunchDarkly': { type: 'tool', confidence: 0.9 },

  // Cloud
  'AWS': { type: 'technology', confidence: 0.8 },
  'ECS': { type: 'technology', confidence: 0.7 },
  'Fargate': { type: 'technology', confidence: 0.8 },

  // Communication
  'Slack': { type: 'tool', confidence: 0.8 },

  // Misc tools
  'Turborepo': { type: 'tool', confidence: 0.8 },
  'lazygit': { type: 'tool', confidence: 0.8 },
  'husky': { type: 'tool', confidence: 0.7 },
  'Stripe': { type: 'tool', confidence: 0.8 },
  'Kafka': { type: 'tool', confidence: 0.8 },
  'Neo4j': { type: 'technology', confidence: 0.8 },
  'Tailwind': { type: 'technology', confidence: 0.8 },
};

/**
 * Extract tech entities by vocabulary lookup.
 * Case-sensitive matching for proper nouns (TypeScript, not typescript).
 */
function extractByVocabulary(content: string): ExtractedEntity[] {
  const entities: ExtractedEntity[] = [];
  const seen = new Set<string>();

  for (const [term, meta] of Object.entries(TECH_VOCABULARY)) {
    // Case-sensitive check for the term in content
    if (content.includes(term)) {
      const key = term.toLowerCase();
      if (!seen.has(key)) {
        seen.add(key);
        entities.push({ name: term, type: meta.type, confidence: meta.confidence });
      }
    }
  }

  return entities;
}

// Words to skip in entity extraction
const STOPWORDS = new Set([
  'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be',
  'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
  'would', 'could', 'should', 'may', 'might', 'shall', 'can', 'need',
  'must', 'that', 'this', 'these', 'those', 'it', 'its', 'my', 'your',
  'his', 'her', 'our', 'their', 'i', 'you', 'he', 'she', 'we', 'they',
  'me', 'him', 'us', 'them', 'what', 'which', 'who', 'whom', 'when',
  'where', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
  'some', 'any', 'no', 'not', 'only', 'very', 'just', 'also', 'too',
  'user', 'agent', 'memory', 'always', 'never', 'usually', 'typically',
  'then', 'first', 'next', 'after', 'before', 'when', 'while', 'during',
  'since', 'until', 'because', 'if', 'although', 'though', 'even',
  'however', 'but', 'yet', 'so', 'therefore', 'thus', 'hence',
]);

function extractByPatterns(
  content: string,
  patterns: RegExp[],
  entityType: string,
  confidence: number,
): ExtractedEntity[] {
  const entities: ExtractedEntity[] = [];
  const seen = new Set<string>();

  for (const pattern of patterns) {
    // Reset lastIndex for global patterns
    pattern.lastIndex = 0;
    let match: RegExpExecArray | null;

    while ((match = pattern.exec(content)) !== null) {
      const name = (match[1] ?? match[0]).trim();

      // Clean up the extracted name
      const cleaned = name
        .replace(/[.!?,;:'"]+$/, '') // trailing punctuation
        .trim();

      // Skip empty, too short, or stopword-only results
      if (cleaned.length < 2) continue;
      const words = cleaned.toLowerCase().split(/\s+/);
      if (words.every((w) => STOPWORDS.has(w))) continue;

      const key = `${cleaned.toLowerCase()}:${entityType}`;
      if (!seen.has(key)) {
        seen.add(key);
        entities.push({ name: cleaned, type: entityType, confidence });
      }
    }
  }

  return entities;
}

// ============================================================
// Main extraction function
// ============================================================

/**
 * Extract entities and relationships from memory content.
 *
 * Uses rule-based patterns for fast extraction (~1ms).
 * Returns entities with types and confidence scores.
 */
export function extractEntities(content: string): ExtractionResult {
  const entities: ExtractedEntity[] = [];
  const relationships: ExtractionResult['relationships'] = [];

  // Extract by type (pattern-based)
  entities.push(...extractByPatterns(content, PERSON_PATTERNS, 'person', 0.7));
  entities.push(...extractByPatterns(content, LOCATION_PATTERNS, 'location', 0.6));
  entities.push(...extractByPatterns(content, ORG_PATTERNS, 'organization', 0.6));
  entities.push(...extractByPatterns(content, PREFERENCE_PATTERNS, 'preference', 0.5));
  entities.push(...extractByPatterns(content, TOPIC_PATTERNS, 'topic', 0.4));

  // Extract by vocabulary lookup (tech terms, tools, frameworks)
  entities.push(...extractByVocabulary(content));

  // Deduplicate across types (prefer more specific types)
  const typeRank: Record<string, number> = {
    person: 5,
    technology: 4,
    tool: 4,
    organization: 3,
    location: 2,
    preference: 1,
    topic: 0,
  };

  const deduped = new Map<string, ExtractedEntity>();
  for (const entity of entities) {
    const key = entity.name.toLowerCase();
    const existing = deduped.get(key);
    if (!existing || (typeRank[entity.type] ?? 0) > (typeRank[existing.type] ?? 0)) {
      deduped.set(key, entity);
    }
  }

  // Build relationships between co-occurring entities
  const entityList = [...deduped.values()];
  for (let i = 0; i < entityList.length; i++) {
    for (let j = i + 1; j < entityList.length; j++) {
      const a = entityList[i]!;
      const b = entityList[j]!;

      // Infer relationship type from entity types
      let relation = 'RELATED_TO';
      if (a.type === 'person' && b.type === 'location') relation = 'ASSOCIATED_WITH';
      if (a.type === 'person' && b.type === 'organization') relation = 'WORKS_AT';
      if (a.type === 'person' && b.type === 'preference') relation = 'HAS_PREFERENCE';

      relationships.push({
        from: { name: a.name, type: a.type },
        to: { name: b.name, type: b.type },
        relation,
      });
    }
  }

  return { entities: entityList, relationships };
}

// ============================================================
// Query-time entity extraction
// ============================================================

/** Words that should NOT be treated as person names in queries */
const QUERY_STOP_WORDS = new Set([
  'What', 'When', 'Where', 'How', 'Who', 'Which', 'Why',
  'Does', 'Did', 'Has', 'Have', 'Is', 'Was', 'Do', 'Can',
  'Will', 'Would', 'Could', 'Should', 'Are', 'Were', 'May',
  'The', 'This', 'That', 'These', 'Those', 'For', 'From',
  'With', 'About', 'Into', 'After', 'Before', 'During',
  'Between', 'Through', 'Over', 'Under', 'Since', 'Until',
  'Many', 'Much', 'Most', 'Some', 'Any', 'All', 'Each',
  'Every', 'Both', 'Few', 'Several', 'No', 'Not',
]);

/**
 * Extract entities from a query string.
 * Tuned for query-time: finds capitalized words that are likely person names.
 * More aggressive than extractEntities() — catches "Melanie" in "What does Melanie do?"
 */
export function extractQueryEntities(query: string): ExtractedEntity[] {
  const entities: ExtractedEntity[] = [];
  const seen = new Set<string>();

  // Find any capitalized word (2+ letters) that isn't a query stop word
  const capitalizedPattern = /\b([A-Z][a-z]{1,})\b/g;
  let match: RegExpExecArray | null;

  while ((match = capitalizedPattern.exec(query)) !== null) {
    const name = match[1]!;
    if (QUERY_STOP_WORDS.has(name)) continue;

    const key = name.toLowerCase();
    if (!seen.has(key)) {
      seen.add(key);
      entities.push({ name, type: 'person', confidence: 0.6 });
    }
  }

  return entities;
}
