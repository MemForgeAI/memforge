"""Main benchmark orchestrator.

Usage:
    python benchmark/run_benchmark.py                    # All systems
    python benchmark/run_benchmark.py --systems memforge   # Single system
    python benchmark/run_benchmark.py --systems memforge mem0  # Two systems
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.adapters.base import MemoryAdapter
from benchmark.adapters.memforge_adapter import MemforgeAdapter
from benchmark.adapters.mem0_adapter import Mem0Adapter
from benchmark.adapters.graphiti_adapter import GraphitiAdapter
from benchmark.metrics.scorer import (
    score_string_match,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_mrr,
    compute_noise_rejection,
    compute_token_efficiency,
    llm_judge_relevance,
)
from benchmark.metrics.cost_tracker import CostTracker
from benchmark.metrics.latency_tracker import LatencyTracker
from benchmark.models import ExpectedMemory, QueryResult

SEED_DIR = Path(__file__).parent / "seed"
RESULTS_DIR = Path(__file__).parent / "results"


def load_conversations() -> list[dict]:
    with open(SEED_DIR / "conversations.json") as f:
        return json.load(f)


def load_ground_truth() -> list[dict]:
    with open(SEED_DIR / "ground_truth.json") as f:
        return json.load(f)


def create_adapter(name: str) -> MemoryAdapter:
    if name == "memforge":
        url = os.environ.get("MEMFORGE_URL", "http://localhost:3100")
        return MemforgeAdapter(base_url=url)
    elif name == "mem0":
        azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        azure_llm_model = os.environ.get("AZURE_OPENAI_LLM_MODEL", "gpt-5.2-chat")
        azure_embed_model = os.environ.get(
            "AZURE_OPENAI_EMBED_MODEL", "text-embedding-3-large"
        )
        if azure_api_key:
            config = {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "embedding_model_dims": 3072,
                        "collection_name": "mem0_benchmark",
                    },
                },
                "llm": {
                    "provider": "azure_openai",
                    "config": {
                        "model": azure_llm_model,
                        "azure_kwargs": {
                            "api_key": azure_api_key,
                            "azure_deployment": azure_llm_model,
                            "azure_endpoint": azure_endpoint,
                            "api_version": "2025-04-01-preview",
                        },
                    },
                },
                "embedder": {
                    "provider": "azure_openai",
                    "config": {
                        "model": azure_embed_model,
                        "embedding_dims": 3072,
                        "azure_kwargs": {
                            "api_key": azure_api_key,
                            "azure_deployment": azure_embed_model,
                            "azure_endpoint": azure_endpoint,
                            "api_version": "2023-05-15",
                        },
                    },
                },
            }
            return Mem0Adapter(config=config)
        return Mem0Adapter()
    elif name == "graphiti":
        return GraphitiAdapter(
            neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
            neo4j_password=os.environ.get("NEO4J_PASSWORD", "demodemo"),
        )
    else:
        raise ValueError(f"Unknown system: {name}")


async def seed_adapter(
    adapter: MemoryAdapter,
    conversations: list[dict],
    cost_tracker: CostTracker,
    latency_tracker: LatencyTracker,
) -> None:
    print(f"  Seeding {adapter.name} with {len(conversations)} conversations...")
    total_memories = 0
    for conv in conversations:
        for memory_text in conv.get("memories_to_extract", []):
            result = await adapter.store(
                memory_text,
                user_id=conv.get("user_id", "alex"),
                agent_id=conv.get("agent_id", "coding-assistant"),
                importance=conv.get("importance"),
                shared=conv.get("shared", False),
                task_id=conv.get("task_id"),
            )
            cost_tracker.record_store(adapter.name, result.api_calls, result.latency_ms)
            latency_tracker.record_store(adapter.name, result.latency_ms)
            total_memories += 1
    print(f"  {adapter.name} seeded with {total_memories} memories.")


async def run_queries(
    adapter: MemoryAdapter,
    queries: list[dict],
    cost_tracker: CostTracker,
    latency_tracker: LatencyTracker,
) -> list[QueryResult]:
    results = []
    for q in queries:
        query_text = q["query"]
        token_budget = q.get("token_budget", 4000)

        recall = await adapter.recall(
            query_text,
            user_id=q.get("user_id", "alex"),
            agent_id=q.get("agent_id", "coding-assistant"),
            token_budget=token_budget,
            include_shared=q.get("include_shared", False),
            task_id=q.get("task_id"),
        )

        cost_tracker.record_recall(adapter.name, recall.api_calls, recall.latency_ms)
        latency_tracker.record_recall(adapter.name, recall.latency_ms)

        expected = [ExpectedMemory(**e) for e in q.get("expected_memories", [])]
        absent = q.get("expected_absent", [])
        scoring = q.get("scoring_method", "string_match")

        p5 = p10 = r10 = mrr = noise = None

        if scoring == "string_match" and expected:
            hits = score_string_match(recall.memories, expected)
            required_expected = [e for e in expected if e.required]
            total_relevant = len(required_expected)

            relevant_flags = []
            for mem in recall.memories:
                is_relevant = any(
                    e.content_match.lower() in mem.lower()
                    for e in expected
                    if not e.should_be_superseded
                )
                relevant_flags.append(is_relevant)

            p5 = compute_precision_at_k(relevant_flags, 5)
            p10 = compute_precision_at_k(relevant_flags, 10)
            r10 = compute_recall_at_k(hits, total_relevant, len(hits))
            mrr = compute_mrr(relevant_flags)
            noise = compute_noise_rejection(recall.memories, absent)

        elif scoring == "llm_judge":
            try:
                scores = await llm_judge_relevance(query_text, recall.memories)
                relevant_flags = [s >= 0.5 for s in scores]
                p5 = compute_precision_at_k(relevant_flags, 5)
                p10 = compute_precision_at_k(relevant_flags, 10)
                total_relevant = (
                    len([e for e in expected if e.required])
                    if expected
                    else len(relevant_flags)
                )
                r10 = compute_recall_at_k(
                    relevant_flags, total_relevant, len(relevant_flags)
                )
                mrr = compute_mrr(relevant_flags)
                noise = compute_noise_rejection(recall.memories, absent)
            except Exception as exc:
                print(f"  LLM judge failed for query {q['query_id']}: {exc}")

        temporal_correct = None
        if "temporal_supersession" in q.get("tests", []):
            current = [e for e in expected if e.required and e.rank == 1]
            if current and recall.memories:
                current_found = any(
                    current[0].content_match.lower() in m.lower()
                    for m in recall.memories[:3]
                )
                temporal_correct = current_found

        token_eff = None
        if recall.formatted_context and recall.total_tokens > 0:
            relevant_token_count = 0
            for mem in recall.memories:
                for e in expected:
                    if (
                        e.content_match.lower() in mem.lower()
                        and not e.should_be_superseded
                    ):
                        relevant_token_count += int(len(mem.split()) * 1.3)
                        break
            token_eff = compute_token_efficiency(
                relevant_token_count, recall.total_tokens
            )

        results.append(
            QueryResult(
                query_id=q["query_id"],
                system=adapter.name,
                recall_result=recall,
                precision_at_5=p5,
                precision_at_10=p10,
                recall_at_10=r10,
                mrr=mrr,
                temporal_correct=temporal_correct,
                noise_rejected=noise,
                token_efficiency=token_eff,
            )
        )

        status = f"P@5={p5:.2f}" if p5 is not None else "llm_judge"
        print(f"  [{adapter.name}] Q{q['query_id']:02d} ({q['category']}): {status}")

    return results


def generate_report(
    all_results: dict[str, list[QueryResult]],
    cost_tracker: CostTracker,
    latency_tracker: LatencyTracker,
    setup_times: dict[str, float],
) -> str:
    systems = list(all_results.keys())
    now = datetime.now().strftime("%Y-%m-%d")

    lines = [
        "# MemForge Benchmark Results -- Coding Assistant (50 Conversations)",
        "",
        f"Run date: {now}",
        "",
    ]

    lines.append("## Retrieval Quality")
    lines.append("")
    header = "| Metric | " + " | ".join(systems) + " |"
    sep = "|--------|" + "|".join(["--------"] * len(systems)) + "|"
    lines.extend([header, sep])

    for metric_name, metric_key in [
        ("Precision@5", "precision_at_5"),
        ("Precision@10", "precision_at_10"),
        ("Recall@10", "recall_at_10"),
        ("MRR", "mrr"),
    ]:
        row = f"| {metric_name} |"
        for sys_name in systems:
            vals = [
                getattr(r, metric_key)
                for r in all_results[sys_name]
                if getattr(r, metric_key) is not None
            ]
            avg = sum(vals) / len(vals) if vals else 0.0
            row += f" {avg:.3f} |"
        lines.append(row)

    row = "| Temporal Correctness |"
    for sys_name in systems:
        vals = [
            r.temporal_correct
            for r in all_results[sys_name]
            if r.temporal_correct is not None
        ]
        correct = sum(1 for v in vals if v)
        row += f" {correct}/{len(vals)} |"
    lines.append(row)

    row = "| Noise Rejection |"
    for sys_name in systems:
        vals = [
            r.noise_rejected
            for r in all_results[sys_name]
            if r.noise_rejected is not None
        ]
        avg = sum(vals) / len(vals) if vals else 0.0
        row += f" {avg:.3f} |"
    lines.append(row)

    lines.append("")
    lines.append("## Operational Efficiency")
    lines.append("")
    lines.extend([header, sep])

    row = "| Token Efficiency |"
    for sys_name in systems:
        vals = [
            r.token_efficiency
            for r in all_results[sys_name]
            if r.token_efficiency is not None
        ]
        if vals:
            avg = sum(vals) / len(vals)
            row += f" {avg:.3f} |"
        else:
            row += " N/A |"
    lines.append(row)

    row = "| Cost per 1000 ops |"
    for sys_name in systems:
        summary = cost_tracker.get_summary(sys_name)
        row += f" ${summary.cost_per_1000_ops:.4f} |"
    lines.append(row)

    row = "| Setup Time |"
    for sys_name in systems:
        row += f" {setup_times.get(sys_name, 0):.1f}s |"
    lines.append(row)

    for label, prop in [
        ("Recall Latency p50", "recall_p50"),
        ("Recall Latency p95", "recall_p95"),
    ]:
        row = f"| {label} |"
        for sys_name in systems:
            lat = latency_tracker.get_summary(sys_name)
            val = getattr(lat, prop)
            row += f" {val:.0f}ms |"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def write_csv(all_results: dict[str, list[QueryResult]], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "query_id",
                "system",
                "precision_at_5",
                "precision_at_10",
                "recall_at_10",
                "mrr",
                "temporal_correct",
                "noise_rejected",
                "token_efficiency",
                "latency_ms",
                "api_calls",
                "total_tokens",
            ]
        )
        for sys_name, results in all_results.items():
            for r in results:
                writer.writerow(
                    [
                        r.query_id,
                        r.system,
                        f"{r.precision_at_5:.3f}"
                        if r.precision_at_5 is not None
                        else "",
                        f"{r.precision_at_10:.3f}"
                        if r.precision_at_10 is not None
                        else "",
                        f"{r.recall_at_10:.3f}"
                        if r.recall_at_10 is not None
                        else "",
                        f"{r.mrr:.3f}" if r.mrr is not None else "",
                        r.temporal_correct
                        if r.temporal_correct is not None
                        else "",
                        f"{r.noise_rejected:.3f}"
                        if r.noise_rejected is not None
                        else "",
                        f"{r.token_efficiency:.3f}"
                        if r.token_efficiency is not None
                        else "",
                        f"{r.recall_result.latency_ms:.1f}",
                        r.recall_result.api_calls,
                        r.recall_result.total_tokens,
                    ]
                )


async def main() -> None:
    parser = argparse.ArgumentParser(description="MemForge Benchmark Suite")
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["memforge", "mem0", "graphiti"],
        choices=["memforge", "mem0", "graphiti"],
        help="Systems to benchmark",
    )
    args = parser.parse_args()

    conversations = load_conversations()
    queries = load_ground_truth()
    print(f"Loaded {len(conversations)} conversations, {len(queries)} queries")

    cost_tracker = CostTracker()
    latency_tracker = LatencyTracker()
    setup_times: dict[str, float] = {}
    all_results: dict[str, list[QueryResult]] = {}

    for sys_name in args.systems:
        print(f"\n{'=' * 60}")
        print(f"  Benchmarking: {sys_name.upper()}")
        print(f"{'=' * 60}")

        try:
            adapter = create_adapter(sys_name)
            setup_time = await adapter.setup()
            setup_times[sys_name] = setup_time
            print(f"  Setup time: {setup_time:.2f}s")

            await seed_adapter(adapter, conversations, cost_tracker, latency_tracker)
            results = await run_queries(
                adapter, queries, cost_tracker, latency_tracker
            )
            all_results[sys_name] = results

            await adapter.teardown()
        except Exception as exc:
            print(f"  ERROR: {sys_name} failed: {exc}")
            traceback.print_exc()
            continue

    if all_results:
        RESULTS_DIR.mkdir(exist_ok=True)

        report = generate_report(
            all_results, cost_tracker, latency_tracker, setup_times
        )
        report_path = RESULTS_DIR / "comparison_table.md"
        report_path.write_text(report)
        print(f"\nReport written to: {report_path}")
        print(report)

        csv_path = RESULTS_DIR / "per_query_results.csv"
        write_csv(all_results, csv_path)
        print(f"CSV written to: {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
