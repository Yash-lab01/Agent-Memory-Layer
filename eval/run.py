"""Benchmark runner executing baseline agents and computing evaluation metrics."""

import argparse
import asyncio
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from eval.baselines import (
    BaseBaseline,
    FullContextBaseline,
    MemLayerBaseline,
    NaiveRAGBaseline,
)
from eval.cache import DiskCache
from eval.datasets.loader import load_eval_dataset
from eval.datasets.models import EvalQuestion
from eval.judge import LLMJudge


def compute_metrics(
    eval_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute aggregate benchmark metrics across questions and categories."""
    if not eval_items:
        return {
            "total_questions": 0,
            "overall_accuracy": 0.0,
            "category_accuracy": {},
            "latency": {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0},
            "tokens": {"mean_prompt": 0.0, "mean_completion": 0.0, "total": 0},
            "estimated_cost_usd": 0.0,
        }

    total = len(eval_items)
    correct_count = sum(1 for item in eval_items if item["is_correct"])
    overall_acc = (correct_count / total) * 100.0

    # Per-category accuracy
    cat_items: dict[str, list[dict[str, Any]]] = {}
    for item in eval_items:
        cat = item["category"]
        if cat not in cat_items:
            cat_items[cat] = []
        cat_items[cat].append(item)

    category_accuracy: dict[str, dict[str, Any]] = {}
    for cat, items in cat_items.items():
        cat_total = len(items)
        cat_corr = sum(1 for it in items if it["is_correct"])
        category_accuracy[cat] = {
            "total": cat_total,
            "correct": cat_corr,
            "accuracy": (cat_corr / cat_total) * 100.0 if cat_total > 0 else 0.0,
        }

    # Latency distribution
    latencies = [item["latency_ms"] for item in eval_items]
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    mean_lat = float(np.mean(latencies))

    # Token counts
    prompt_tokens = [item["prompt_tokens"] for item in eval_items]
    completion_tokens = [item["completion_tokens"] for item in eval_items]
    total_tokens = sum(item["total_tokens"] for item in eval_items)
    mean_prompt = float(np.mean(prompt_tokens))
    mean_comp = float(np.mean(completion_tokens))

    # Estimated cost (default standard rates: $0.15/1M input, $0.60/1M output tokens)
    input_cost = (sum(prompt_tokens) / 1_000_000) * 0.15
    output_cost = (sum(completion_tokens) / 1_000_000) * 0.60
    est_cost = round(input_cost + output_cost, 6)

    return {
        "total_questions": total,
        "correct_count": correct_count,
        "overall_accuracy": round(overall_acc, 2),
        "category_accuracy": category_accuracy,
        "latency": {
            "mean_ms": round(mean_lat, 2),
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
        },
        "tokens": {
            "mean_prompt": round(mean_prompt, 2),
            "mean_completion": round(mean_comp, 2),
            "total": total_tokens,
        },
        "estimated_cost_usd": est_cost,
    }


def format_markdown_table(benchmark: str, baseline: str, metrics: dict[str, Any]) -> str:
    """Format evaluation metrics as a clean Markdown table."""
    lines = [
        f"### Benchmark Evaluation: `{benchmark}` | Baseline: `{baseline}`",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **Overall Accuracy** | **{metrics['overall_accuracy']}%** ({metrics.get('correct_count', 0)}/{metrics['total_questions']}) |",
        f"| **Latency (p50 / p95)** | {metrics['latency']['p50_ms']} ms / {metrics['latency']['p95_ms']} ms |",
        f"| **Mean Prompt Tokens** | {metrics['tokens']['mean_prompt']} tokens |",
        f"| **Mean Completion Tokens** | {metrics['tokens']['mean_completion']} tokens |",
        f"| **Total Tokens** | {metrics['tokens']['total']} tokens |",
        f"| **Estimated Cost** | ${metrics['estimated_cost_usd']} USD |",
        "",
        "#### Per-Category Accuracy Breakdown",
        "",
        "| Category | Questions | Correct | Accuracy |",
        "| :--- | :--- | :--- | :--- |",
    ]

    for cat, data in metrics["category_accuracy"].items():
        lines.append(
            f"| `{cat}` | {data['total']} | {data['correct']} | **{round(data['accuracy'], 1)}%** |"
        )

    return "\n".join(lines)


async def run_single_baseline(
    baseline_cls: type[BaseBaseline],
    baseline_name: str,
    questions: list[EvalQuestion],
    benchmark: str,
    model: str,
    judge_model: str,
    cache: DiskCache | None,
    output_dir: Path,
    extra_baseline_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute evaluation for a single baseline across questions."""
    judge = LLMJudge(model=judge_model, cache=cache)
    item_results: list[dict[str, Any]] = []

    print(f"\n[eval] Running baseline '{baseline_name}' on {len(questions)} items...")

    for idx, question in enumerate(questions, 1):
        baseline = baseline_cls(
            name=baseline_name,
            model=model,
            cache=cache,
            **(extra_baseline_kwargs or {}),
        )

        try:
            # 1. Setup background conversation
            await baseline.setup(question)

            # 2. Answer question
            output = await baseline.answer(question)

            # 3. Judge answer against ground truth
            judgment = await judge.evaluate(
                item=question,
                prediction=output.prediction,
                baseline_name=baseline_name,
            )

            record = {
                "question_id": question.id,
                "category": question.category,
                "user_id": question.user_id,
                "question": question.question,
                "ground_truth": question.ground_truth,
                "prediction": output.prediction,
                "is_correct": judgment.is_correct,
                "score": judgment.score,
                "rationale": judgment.rationale,
                "latency_ms": output.latency_ms,
                "prompt_tokens": output.prompt_tokens,
                "completion_tokens": output.completion_tokens,
                "total_tokens": output.total_tokens,
                "context_used": output.context_used,
            }
            item_results.append(record)

            mark = "PASS" if judgment.is_correct else "FAIL"
            print(f"  [{idx}/{len(questions)}] {question.id} ({question.category}): {mark}")

        finally:
            await baseline.teardown()

    metrics = compute_metrics(item_results)

    # Save run artifact
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = f"{benchmark}_{baseline_name}_{timestamp}"
    run_record = {
        "run_id": run_id,
        "benchmark": benchmark,
        "baseline": baseline_name,
        "model": model,
        "judge_model": judge_model,
        "timestamp": datetime.now(UTC).isoformat(),
        "metrics": metrics,
        "results": item_results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{run_id}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(run_record, f, indent=2)

    print(f"\n[eval] Saved run results to {out_file}")
    table = format_markdown_table(benchmark, baseline_name, metrics)
    print("\n" + table + "\n")

    return run_record


async def main_async(args: argparse.Namespace) -> None:
    """Async CLI entry point."""
    random.seed(args.seed)
    np.random.seed(args.seed)

    cache = None if args.no_cache else DiskCache(cache_dir=args.cache_dir)
    output_dir = Path(args.output_dir)

    questions = load_eval_dataset(
        benchmark=args.benchmark,
        split=args.split,
        category=None if args.category == "all" else args.category,
        limit=args.limit,
    )

    if not questions:
        print(f"[eval] No questions found for benchmark='{args.benchmark}' split='{args.split}'")
        return

    baseline_map: dict[str, type[BaseBaseline]] = {
        "full_context": FullContextBaseline,
        "naive_rag": NaiveRAGBaseline,
        "memlayer": MemLayerBaseline,
    }

    targets = (
        ["full_context", "naive_rag", "memlayer"] if args.baseline == "all" else [args.baseline]
    )

    for b_name in targets:
        b_cls = baseline_map[b_name]
        await run_single_baseline(
            baseline_cls=b_cls,
            baseline_name=b_name,
            questions=questions,
            benchmark=args.benchmark,
            model=args.model,
            judge_model=args.judge_model,
            cache=cache,
            output_dir=output_dir,
        )


def main() -> None:
    """Synchronous entry point parsing arguments."""
    parser = argparse.ArgumentParser(description="Agent Memory Layer Benchmark Runner")
    parser.add_argument(
        "--baseline",
        choices=["full_context", "naive_rag", "memlayer", "all"],
        default="memlayer",
        help="Baseline agent architecture to evaluate.",
    )
    parser.add_argument(
        "--benchmark",
        choices=["longmemeval", "locomo"],
        default="longmemeval",
        help="Evaluation benchmark dataset.",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="dev",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--category",
        choices=["all", "knowledge_update", "temporal_reasoning", "multi_session", "abstention"],
        default="all",
        help="Category filter.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum questions to run.",
    )
    parser.add_argument(
        "--model",
        default="gemini/gemini-1.5-flash",
        help="Model used for baseline agent completions.",
    )
    parser.add_argument(
        "--judge-model",
        default="gemini/gemini-1.5-pro",
        help="Model used for standardized LLM judge.",
    )
    parser.add_argument(
        "--cache-dir",
        default="eval/cache",
        help="Directory to store deterministic LLM completions cache.",
    )
    parser.add_argument(
        "--output-dir",
        default="eval/results",
        help="Directory to save run results JSON.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass deterministic disk caching.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
