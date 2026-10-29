#!/usr/bin/env python3
"""Cost estimation and budgeting script for benchmark evaluations.

Calculates token expenditures and estimated API costs across:
1. Ingestion: Fact extraction and embedding over conversation haystacks
2. Query: Prompt tokens and answer generation for each baseline & ablation
3. Judge: Evaluation prompt tokens and structured judgment completions
4. Break-even analysis: Minimum questions per haystack for memlayer cost advantage
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CostBreakdown:
    benchmark: str
    split: str
    questions_count: int
    haystacks_count: int
    ingestion_tokens_input: int
    ingestion_tokens_output: int
    ingestion_cost_usd: float
    query_tokens_full_context: int
    query_tokens_naive_rag: int
    query_tokens_bm25_rerank: int
    query_tokens_memlayer: int
    answer_cost_full_context_usd: float
    answer_cost_naive_rag_usd: float
    answer_cost_bm25_rerank_usd: float
    answer_cost_memlayer_usd: float
    judge_cost_usd: float
    ablations_cost_usd: float
    subtotal_cost_usd: float
    safety_buffer_usd: float
    total_budget_usd: float
    break_even_queries: float


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and parse YAML configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Invalid configuration format: root must be a mapping.")
    return data


def estimate_benchmark_cost(
    cfg: dict[str, Any],
    benchmark_key: str,
    split_key: str,
    buffer_ratio: float = 0.25,
) -> CostBreakdown:
    """Compute itemized cost estimation for a benchmark split."""
    b_cfg = cfg["benchmarks"][benchmark_key]
    split_cfg = b_cfg["splits"][split_key]
    haystack_info = b_cfg.get("haystack_characteristics", {})

    q_count = int(split_cfg.get("sample_size", 100))
    q_per_h = max(1, int(haystack_info.get("questions_per_haystack", 1)))
    mean_tokens_per_h = int(haystack_info.get("mean_tokens_per_haystack", 15000))
    h_count = max(1, q_count // q_per_h)

    # Models & Pricing (USD per 1M tokens)
    ext_pricing = cfg["models"]["extraction_model"]["pricing_per_1m_tokens"]
    ans_pricing = cfg["models"]["answer_model"]["pricing_per_1m_tokens"]
    judge_pricing = cfg["models"]["judge_model"]["pricing_per_1m_tokens"]

    # 1. Ingestion Phase (MemLayer only; extract once per haystack)
    ingestion_input_tokens = h_count * mean_tokens_per_h
    # Estimated output: ~8% atomic facts representation
    ingestion_output_tokens = int(ingestion_input_tokens * 0.08)
    ingestion_cost = (ingestion_input_tokens / 1_000_000) * ext_pricing["input_usd"] + (
        ingestion_output_tokens / 1_000_000
    ) * ext_pricing["output_usd"]

    # 2. Query Phase Tokens
    # Context sizes per query
    tokens_fc_prompt = mean_tokens_per_h + 100
    tokens_rag_prompt = (5 * 512) + 100  # 5 chunks of 512
    tokens_bm25_prompt = (5 * 512) + 100
    tokens_mem_prompt = int(cfg["baselines"]["memlayer_full"].get("token_budget", 1500)) + 100

    total_fc_prompt = q_count * tokens_fc_prompt
    total_rag_prompt = q_count * tokens_rag_prompt
    total_bm25_prompt = q_count * tokens_bm25_prompt
    total_mem_prompt = q_count * tokens_mem_prompt

    # Answer generation: ~120 completion tokens per answer
    answer_comp_tokens_per_q = 120
    total_answer_comp = q_count * answer_comp_tokens_per_q

    ans_fc_cost = (total_fc_prompt / 1_000_000) * ans_pricing["input_usd"] + (
        total_answer_comp / 1_000_000
    ) * ans_pricing["output_usd"]
    ans_rag_cost = (total_rag_prompt / 1_000_000) * ans_pricing["input_usd"] + (
        total_answer_comp / 1_000_000
    ) * ans_pricing["output_usd"]
    ans_bm25_cost = (total_bm25_prompt / 1_000_000) * ans_pricing["input_usd"] + (
        total_answer_comp / 1_000_000
    ) * ans_pricing["output_usd"]
    ans_mem_cost = (total_mem_prompt / 1_000_000) * ans_pricing["input_usd"] + (
        total_answer_comp / 1_000_000
    ) * ans_pricing["output_usd"]

    # 3. Judge Phase (4 baselines evaluated)
    systems_count = 4
    judge_prompt_tokens = 600 + answer_comp_tokens_per_q
    judge_comp_tokens = 80
    total_judge_in = q_count * systems_count * judge_prompt_tokens
    total_judge_out = q_count * systems_count * judge_comp_tokens
    judge_cost = (total_judge_in / 1_000_000) * judge_pricing["input_usd"] + (
        total_judge_out / 1_000_000
    ) * judge_pricing["output_usd"]

    # 4. Ablations Phase (6 retrieval-level ablations reusing ingested stores)
    # Reuses existing stores; query cost is similar to memlayer query + judging
    ablations_count = len(cfg.get("ablations", {}))
    ablations_query_cost = (
        q_count * ablations_count * tokens_mem_prompt / 1_000_000
    ) * ans_pricing["input_usd"] + (
        q_count * ablations_count * answer_comp_tokens_per_q / 1_000_000
    ) * ans_pricing["output_usd"]
    ablations_judge_cost = (
        q_count * ablations_count * judge_prompt_tokens / 1_000_000
    ) * judge_pricing["input_usd"] + (
        q_count * ablations_count * judge_comp_tokens / 1_000_000
    ) * judge_pricing["output_usd"]
    ablations_total_cost = ablations_query_cost + ablations_judge_cost

    # 5. Break-even calculation (Questions per haystack where MemLayer beats Full-Context)
    # Per-query difference: FC query cost - MemLayer query cost
    fc_query_unit_cost = (tokens_fc_prompt / 1_000_000) * ans_pricing["input_usd"] + (
        answer_comp_tokens_per_q / 1_000_000
    ) * ans_pricing["output_usd"]
    mem_query_unit_cost = (tokens_mem_prompt / 1_000_000) * ans_pricing["input_usd"] + (
        answer_comp_tokens_per_q / 1_000_000
    ) * ans_pricing["output_usd"]
    savings_per_q = max(1e-7, fc_query_unit_cost - mem_query_unit_cost)
    ingestion_per_h = ingestion_cost / max(1, h_count)
    break_even_q = ingestion_per_h / savings_per_q

    # Subtotal and buffer
    subtotal = (
        ingestion_cost
        + ans_fc_cost
        + ans_rag_cost
        + ans_bm25_cost
        + ans_mem_cost
        + judge_cost
        + ablations_total_cost
    )
    buffer_amt = subtotal * buffer_ratio
    total_budget = subtotal + buffer_amt

    return CostBreakdown(
        benchmark=b_cfg.get("name", benchmark_key),
        split=split_key,
        questions_count=q_count,
        haystacks_count=h_count,
        ingestion_tokens_input=ingestion_input_tokens,
        ingestion_tokens_output=ingestion_output_tokens,
        ingestion_cost_usd=round(ingestion_cost, 4),
        query_tokens_full_context=total_fc_prompt,
        query_tokens_naive_rag=total_rag_prompt,
        query_tokens_bm25_rerank=total_bm25_prompt,
        query_tokens_memlayer=total_mem_prompt,
        answer_cost_full_context_usd=round(ans_fc_cost, 4),
        answer_cost_naive_rag_usd=round(ans_rag_cost, 4),
        answer_cost_bm25_rerank_usd=round(ans_bm25_cost, 4),
        answer_cost_memlayer_usd=round(ans_mem_cost, 4),
        judge_cost_usd=round(judge_cost, 4),
        ablations_cost_usd=round(ablations_total_cost, 4),
        subtotal_cost_usd=round(subtotal, 4),
        safety_buffer_usd=round(buffer_amt, 4),
        total_budget_usd=round(total_budget, 4),
        break_even_queries=round(break_even_q, 1),
    )


def print_cost_report(breakdowns: list[CostBreakdown], buffer_pct: int) -> None:
    """Print structured cost summary table to console."""
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8", errors="replace")
    print("=" * 86)
    print("      AGENT MEMORY LAYER (`memlayer`) - BENCHMARK EVALUATION COST ESTIMATION")
    print("=" * 86)

    grand_subtotal = sum(b.subtotal_cost_usd for b in breakdowns)
    grand_buffer = sum(b.safety_buffer_usd for b in breakdowns)
    grand_total = sum(b.total_budget_usd for b in breakdowns)

    for b in breakdowns:
        print(f"\nBenchmark Split: {b.benchmark.upper()} [{b.split.upper()}]")
        print(f"  * Questions: {b.questions_count} | Haystacks: {b.haystacks_count}")
        print(
            f"  * Ingestion Tokens : {b.ingestion_tokens_input:,} in / {b.ingestion_tokens_output:,} out"
        )
        print(f"  * Ingestion Cost   : ${b.ingestion_cost_usd:.4f} USD")
        print("  * Query Generation Costs (Answer Model):")
        print(
            f"      - Full-Context  : ${b.answer_cost_full_context_usd:.4f} ({b.query_tokens_full_context:,} tokens)"
        )
        print(
            f"      - Naive RAG     : ${b.answer_cost_naive_rag_usd:.4f} ({b.query_tokens_naive_rag:,} tokens)"
        )
        print(
            f"      - BM25 + Rerank : ${b.answer_cost_bm25_rerank_usd:.4f} ({b.query_tokens_bm25_rerank:,} tokens)"
        )
        print(
            f"      - MemLayer Full : ${b.answer_cost_memlayer_usd:.4f} ({b.query_tokens_memlayer:,} tokens)"
        )
        print(f"  * LLM Judge Cost   : ${b.judge_cost_usd:.4f} USD")
        print(f"  * Ablations Suite  : ${b.ablations_cost_usd:.4f} USD")
        print(f"  * Subtotal Cost    : ${b.subtotal_cost_usd:.4f} USD")
        print(f"  * Break-even Point : {b.break_even_queries:.1f} queries / haystack")

    print("\n" + "-" * 86)
    print("EXPENDITURE SUMMARY ACROSS ALL SELECTED RUNS:")
    print(f"  Base Subtotal Estimated Cost : ${grand_subtotal:8.2f} USD")
    print(f"  Safety Buffer (+{buffer_pct}%)          : ${grand_buffer:8.2f} USD")
    print(f"  RECOMMENDED TOTAL BUDGET     : ${grand_total:8.2f} USD")
    print("-" * 86)
    print("Cost-Control Mitigations:")
    print("  1. Disk Cache enabled: Re-runs and judge reviews incur 0 additional API cost.")
    print("  2. Store Reuse: Extracted memory stores reused across all 8 ablation configurations.")
    print("  3. Pilot run: Execute 10 items first to verify per-item telemetry matches estimate.")
    print("=" * 86 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate API tokens and costs for memlayer benchmark runs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("eval/configs/main.yaml"),
        help="Path to main YAML configuration file.",
    )
    parser.add_argument(
        "--benchmark",
        choices=["all", "locomo", "longmemeval"],
        default="all",
        help="Target benchmark subset.",
    )
    parser.add_argument(
        "--split",
        choices=["all", "dev", "test"],
        default="all",
        help="Target evaluation split.",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=0.25,
        help="Safety buffer ratio (default 0.25 for +25%%).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured JSON instead of human-readable report.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    benchmarks_to_run = ["locomo", "longmemeval"] if args.benchmark == "all" else [args.benchmark]
    splits_to_run = ["dev", "test"] if args.split == "all" else [args.split]

    breakdowns: list[CostBreakdown] = []
    for b in benchmarks_to_run:
        for s in splits_to_run:
            breakdowns.append(estimate_benchmark_cost(cfg, b, s, buffer_ratio=args.buffer))

    if args.json:
        result = [asdict(b) for b in breakdowns]
        print(json.dumps(result, indent=2))
    else:
        print_cost_report(breakdowns, int(args.buffer * 100))


if __name__ == "__main__":
    main()
