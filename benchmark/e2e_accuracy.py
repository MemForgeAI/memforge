"""
End-to-end response accuracy test.

Measures what actually matters: given MemForge's recall context,
can an LLM correctly answer the query?

This is directly comparable to Mem0's published "response accuracy"
metric. Instead of measuring retrieval precision (is the right memory
in the top 5?), it measures answer correctness (did the agent get
the right answer using the full context?).

Usage:
    python benchmark/e2e_accuracy.py

Requires:
    - MemForge running at localhost:3100
    - AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY set
    - Benchmark data already seeded (run run_benchmark.py first)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.adapters.memforge_adapter import MemforgeAdapter

SEED_DIR = Path(__file__).parent / "seed"

# Azure OpenAI config
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_KEY = os.environ.get("AZURE_OPENAI_KEY", "")
AZURE_MODEL = os.environ.get("AZURE_OPENAI_MODEL", "gpt-5.2-chat")


def llm_answer(query: str, context: str) -> str:
    """Ask the LLM to answer a query using the provided context."""
    prompt = f"""You are an AI coding assistant with persistent memory about a developer named Alex.
Use the context below to answer the question as thoroughly as possible.

IMPORTANT INSTRUCTIONS:
- List EVERY relevant detail from the context, even if only mentioned once
- Include specific names of tools, technologies, frameworks, and people
- If the question asks about preferences or choices, mention BOTH current AND previous choices
- If the question is broad, cover ALL relevant topics from the context
- Be thorough — missing a detail is worse than being verbose

MEMORY CONTEXT:
{context}

QUESTION: {query}

DETAILED ANSWER:"""

    data = json.dumps({
        "model": AZURE_MODEL,
        "input": prompt,
        "max_output_tokens": 300,
    }).encode()

    req = urllib.request.Request(AZURE_ENDPOINT, data=data, headers={
        "Content-Type": "application/json",
        "api-key": AZURE_KEY,
    })

    resp = urllib.request.urlopen(req, timeout=30)
    result = json.loads(resp.read())

    for item in result.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    return content["text"]

    return ""


def check_accuracy(
    answer: str,
    expected_present: list[str],
    expected_absent: list[str],
) -> dict:
    """Check if the LLM's answer contains expected facts."""
    answer_lower = answer.lower()

    present_hits = []
    present_misses = []
    for term in expected_present:
        if term.lower() in answer_lower:
            present_hits.append(term)
        else:
            present_misses.append(term)

    absent_violations = []
    for term in expected_absent:
        if term.lower() in answer_lower:
            absent_violations.append(term)

    total_expected = len(expected_present)
    found = len(present_hits)
    accuracy = found / total_expected if total_expected > 0 else 1.0

    return {
        "accuracy": accuracy,
        "found": present_hits,
        "missed": present_misses,
        "violations": absent_violations,
        "correct": len(present_misses) == 0 and len(absent_violations) == 0,
    }


async def main() -> None:
    gt = json.load(open(SEED_DIR / "ground_truth.json"))

    # Filter to string_match queries with expected answers (skip LLM-judge-only queries)
    testable = [
        q for q in gt
        if q.get("expected_memories")
        and any(not e.get("should_be_superseded") for e in q["expected_memories"])
    ]

    print(f"End-to-End Response Accuracy Test")
    print(f"=================================")
    print(f"Queries: {len(testable)}")
    print(f"LLM: {AZURE_MODEL}")
    print()

    adapter = MemforgeAdapter()
    await adapter.setup()

    # First, seed the data
    conversations = json.load(open(SEED_DIR / "conversations.json"))
    print("Seeding memories...")
    count = 0
    for conv in conversations:
        for mem in conv.get("memories_to_extract", []):
            await adapter.store(
                mem,
                user_id=conv.get("user_id", "alex"),
                agent_id=conv.get("agent_id", "coding-assistant"),
                importance=conv.get("importance"),
                shared=conv.get("shared", False),
                task_id=conv.get("task_id"),
            )
            count += 1
    print(f"Seeded {count} memories.\n")

    results = []
    correct_count = 0
    total_accuracy = 0.0

    for q in testable:
        qid = q["query_id"]
        query = q["query"]

        # Get recall context from MemForge
        recall = await adapter.recall(
            query,
            user_id=q.get("user_id", "alex"),
            agent_id=q.get("agent_id", "coding-assistant"),
            token_budget=q.get("token_budget", 4000),
            include_shared=q.get("include_shared", False),
            task_id=q.get("task_id"),
        )

        context = recall.formatted_context or ""

        if not context:
            print(f"Q{qid:02d}: NO CONTEXT - {query[:50]}")
            results.append({"query_id": qid, "accuracy": 0, "correct": False})
            continue

        # Ask the LLM to answer using the context
        expected_present = [
            e["content_match"]
            for e in q["expected_memories"]
            if not e.get("should_be_superseded")
        ]
        expected_absent = q.get("expected_absent", [])

        try:
            answer = llm_answer(query, context)
            check = check_accuracy(answer, expected_present, expected_absent)

            status = "CORRECT" if check["correct"] else "PARTIAL" if check["accuracy"] > 0 else "WRONG"
            print(f"Q{qid:02d} [{status}] acc={check['accuracy']:.0%}: {query[:50]}")

            if check["missed"]:
                print(f"     missed: {check['missed']}")
            if check["violations"]:
                print(f"     violations: {check['violations']}")

            results.append({
                "query_id": qid,
                "accuracy": check["accuracy"],
                "correct": check["correct"],
                "found": check["found"],
                "missed": check["missed"],
                "answer_preview": answer[:100],
            })

            if check["correct"]:
                correct_count += 1
            total_accuracy += check["accuracy"]

        except Exception as exc:
            print(f"Q{qid:02d} [ERROR]: {exc}")
            results.append({"query_id": qid, "accuracy": 0, "correct": False})

    await adapter.teardown()

    # Summary
    n = len(results)
    avg_accuracy = total_accuracy / n if n > 0 else 0
    pct_correct = correct_count / n if n > 0 else 0

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Total queries tested:     {n}")
    print(f"Fully correct:            {correct_count}/{n} ({pct_correct:.0%})")
    print(f"Average accuracy:         {avg_accuracy:.1%}")
    print(f"(accuracy = % of expected facts found in answer)")
    print()

    # Save results
    results_path = Path(__file__).parent / "results" / "e2e_accuracy.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "summary": {
                "total_queries": n,
                "fully_correct": correct_count,
                "pct_correct": pct_correct,
                "avg_accuracy": avg_accuracy,
                "model": AZURE_MODEL,
            },
            "results": results,
        }, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    asyncio.run(main())
