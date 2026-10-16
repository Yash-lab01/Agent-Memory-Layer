"""Dataset loaders for benchmark evaluation splits."""

import json
from pathlib import Path

from eval.datasets.models import EvalQuestion

DEFAULT_FIXTURES_DIR = Path(__file__).parent / "fixtures"

BENCHMARK_FIXTURE_MAP = {
    "longmemeval": "longmemeval_sample.json",
    "locomo": "locomo_sample.json",
}


def load_eval_dataset(
    benchmark: str = "longmemeval",
    split: str = "dev",
    category: str | None = None,
    limit: int | None = None,
    fixtures_dir: Path | None = None,
    custom_file: Path | None = None,
) -> list[EvalQuestion]:
    """Load and validate benchmark evaluation questions.

    Args:
        benchmark: Name of the benchmark ("longmemeval" or "locomo").
        split: Dataset split ("dev" or "test").
        category: Optional category filter ("knowledge_update", "temporal_reasoning", etc.).
        limit: Optional maximum number of items to return.
        fixtures_dir: Directory containing fixture JSON files.
        custom_file: Direct path to a JSON file overriding default fixtures.

    Returns:
        List of validated EvalQuestion objects.
    """
    if custom_file:
        data_path = custom_file
    else:
        norm_name = benchmark.strip().lower()
        if norm_name not in BENCHMARK_FIXTURE_MAP:
            valid_keys = ", ".join(BENCHMARK_FIXTURE_MAP.keys())
            raise ValueError(f"Unknown benchmark '{benchmark}'. Supported: {valid_keys}")

        root = fixtures_dir or DEFAULT_FIXTURES_DIR
        data_path = root / BENCHMARK_FIXTURE_MAP[norm_name]

    if not data_path.exists():
        raise FileNotFoundError(f"Benchmark fixture not found at {data_path}")

    with open(data_path, encoding="utf-8") as f:
        raw_items = json.load(f)

    if not isinstance(raw_items, list):
        raise ValueError(f"Expected a JSON list in {data_path}, got {type(raw_items).__name__}")

    questions: list[EvalQuestion] = []
    for raw in raw_items:
        q = EvalQuestion.model_validate(raw)
        if split and q.split != split:
            continue
        if category and q.category != category:
            continue
        questions.append(q)
        if limit is not None and len(questions) >= limit:
            break

    return questions
