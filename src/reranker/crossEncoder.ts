/**
 * Cross-encoder re-ranker using Xenova/ms-marco-MiniLM-L-6-v2.
 *
 * Scores (query, document) pairs for semantic relevance.
 * Uses AutoModelForSequenceClassification + AutoTokenizer because
 * the pipeline('text-classification') API does not pass text_pair
 * to the tokenizer, which cross-encoders require.
 *
 * ~22MB quantized ONNX model, runs on CPU.
 */

const MODEL_ID = "Xenova/ms-marco-MiniLM-L-6-v2";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let tokenizer: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let model: any = null;
let initialized = false;

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Pre-load the cross-encoder model. Call once at server startup.
 */
export async function initCrossEncoder(): Promise<void> {
  if (initialized) return;

  console.log(`[memforge] Loading cross-encoder model (${MODEL_ID})...`);
  const start = Date.now();

  const { AutoTokenizer, AutoModelForSequenceClassification } =
    await import("@xenova/transformers");

  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
  model = await AutoModelForSequenceClassification.from_pretrained(MODEL_ID);

  initialized = true;
  console.log(`[memforge] Cross-encoder loaded in ${Date.now() - start}ms`);
}

export interface CrossEncoderResult {
  index: number;
  logit: number;
  score: number;
}

/**
 * Score an array of (query, document) pairs.
 */
export async function scorePairs(
  query: string,
  documents: string[],
): Promise<CrossEncoderResult[]> {
  if (!initialized || !tokenizer || !model) {
    throw new Error(
      "Cross-encoder not initialized. Call initCrossEncoder() first.",
    );
  }

  if (documents.length === 0) return [];

  const queries = documents.map(() => query);

  const inputs = tokenizer(queries, {
    text_pair: documents,
    padding: true,
    truncation: true,
    max_length: 512,
  });

  const output = await model(inputs);

  const logits: Float32Array = output.logits.data;
  const numLabels: number = output.logits.dims[1] ?? 1;

  const results: CrossEncoderResult[] = [];
  for (let i = 0; i < documents.length; i++) {
    let logit: number;
    if (numLabels === 1) {
      logit = logits[i]!;
    } else {
      logit = logits[i * numLabels + 1]!;
    }
    results.push({
      index: i,
      logit,
      score: sigmoid(logit),
    });
  }

  return results;
}

export function isReady(): boolean {
  return initialized;
}
