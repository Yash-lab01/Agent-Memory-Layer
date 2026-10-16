"""Ablation study harness evaluating individual component contributions to MemLayer."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.baselines.memlayer_agent import MemLayerBaseline
from eval.cache import DiskCache
from eval.datasets.models import EvalQuestion
from eval.run import run_single_baseline
from memlayer.retrieve.scoring import ScoringWeights


def get_ablation_configs() -> dict[str, dict[str, Any]]:
    """Return dictionary of the 4 planned ablation parameter sets."""
    return {
        "memlayer_full": {
            "name": "memlayer",
            "description": "Full Agent Memory Layer (extraction, supersession, hybrid BM25+vector, recency decay, router)",
            "kwargs": {
                "enable_supersession": True,
                "enable_decay": True,
                "vector_only": False,
                "enable_router": True,
            },
        },
        "without_decay": {
            "name": "memlayer_without_decay",
            "description": "Ablation: Memory frequency decay disabled",
            "kwargs": {
                "enable_supersession": True,
                "enable_decay": False,
                "scoring_weights": ScoringWeights(a=0.7, b=0.0, c=0.3),
                "vector_only": False,
                "enable_router": True,
            },
        },
        "without_supersession": {
            "name": "memlayer_without_supersession",
            "description": "Ablation: Conflict detection & supersession disabled (outdated facts remain active)",
            "kwargs": {
                "enable_supersession": False,
                "enable_decay": True,
                "vector_only": False,
                "enable_router": True,
            },
        },
        "vector_only": {
            "name": "memlayer_vector_only",
            "description": "Ablation: Dense vector retrieval only (BM25 sparse scoring disabled)",
            "kwargs": {
                "enable_supersession": True,
                "enable_decay": True,
                "vector_only": True,
                "enable_router": True,
            },
        },
        "without_recency": {
            "name": "memlayer_without_recency",
            "description": "Ablation: Recency decay factor omitted in scoring formula (b=0.0)",
            "kwargs": {
                "enable_supersession": True,
                "enable_decay": True,
                "scoring_weights": ScoringWeights(a=0.7, b=0.0, c=0.3),
                "vector_only": False,
                "enable_router": True,
            },
        },
    }


def format_ablation_markdown_table(
    benchmark: str,
    ablation_metrics: dict[str, dict[str, Any]],
) -> str:
    """Format comparative ablation study table in Markdown."""
    lines = [
        f"### Ablation Study Results: `{benchmark}`",
        "",
        "| Configuration | Description | Overall Acc | Knowledge Update | Temporal | Multi-Session | Abstention | Delta vs MemLayer |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    baseline_acc = ablation_metrics.get("memlayer_full", {}).get("overall_accuracy", 0.0)

    for config_key, data in ablation_metrics.items():
        name = data.get("name", config_key)
        desc = data.get("description", "")
        acc = data.get("overall_accuracy", 0.0)
        cats = data.get("category_accuracy", {})

        ku_acc = cats.get("knowledge_update", {}).get("accuracy", 0.0)
        tr_acc = cats.get("temporal_reasoning", {}).get("accuracy", 0.0)
        ms_acc = cats.get("multi_session", {}).get("accuracy", 0.0)
        ab_acc = cats.get("abstention", {}).get("accuracy", 0.0)

        delta = acc - baseline_acc
        delta_str = "Ref (0.0%)" if config_key == "memlayer_full" else f"{delta:+.1f}%"

        lines.append(
            f"| **`{name}`** | {desc} | **{acc:.1f}%** | {ku_acc:.1f}% | {tr_acc:.1f}% | {ms_acc:.1f}% | {ab_acc:.1f}% | **{delta_str}** |"
        )

    return "\n".join(lines)


async def run_ablation_suite(
    questions: list[EvalQuestion],
    benchmark: str = "longmemeval",
    model: str = "gemini/gemini-1.5-flash",
    judge_model: str = "gemini/gemini-1.5-pro",
    cache: DiskCache | None = None,
    output_dir: Path | None = None,
    extra_baseline_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the full MemLayer baseline alongside all 4 systematic ablations."""
    out_dir = output_dir or Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = get_ablation_configs()
    ablation_metrics: dict[str, dict[str, Any]] = {}
    full_records: dict[str, Any] = {}

    print("\n=======================================================")
    print(f"Starting Ablation Study Suite on Benchmark: {benchmark}")
    print(f"Configurations to run: {len(configs)}")
    print("=======================================================\n")

    for config_key, cfg in configs.items():
        name = cfg["name"]
        print(f"\n--- Running Ablation: {name} ({cfg['description']}) ---")
        merged_kwargs = {**cfg["kwargs"], **(extra_baseline_kwargs or {})}

        record = await run_single_baseline(
            baseline_cls=MemLayerBaseline,
            baseline_name=name,
            questions=questions,
            benchmark=benchmark,
            model=model,
            judge_model=judge_model,
            cache=cache,
            output_dir=out_dir,
            extra_baseline_kwargs=merged_kwargs,
        )

        full_records[config_key] = record
        metrics = record["metrics"]
        ablation_metrics[config_key] = {
            "name": name,
            "description": cfg["description"],
            "overall_accuracy": metrics["overall_accuracy"],
            "category_accuracy": metrics["category_accuracy"],
            "latency": metrics["latency"],
            "tokens": metrics["tokens"],
            "estimated_cost_usd": metrics["estimated_cost_usd"],
        }

    table = format_ablation_markdown_table(benchmark, ablation_metrics)
    print("\n" + table + "\n")

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    suite_id = f"{benchmark}_ablation_suite_{timestamp}"
    suite_file = out_dir / f"{suite_id}.json"
    suite_record = {
        "suite_id": suite_id,
        "benchmark": benchmark,
        "timestamp": datetime.now(UTC).isoformat(),
        "ablation_metrics": ablation_metrics,
        "full_runs": full_records,
    }

    with open(suite_file, "w", encoding="utf-8") as f:
        json.dump(suite_record, f, indent=2)

    print(f"[eval] Saved complete ablation suite to {suite_file}")
    return suite_record
