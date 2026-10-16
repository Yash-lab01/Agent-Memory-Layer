"""Comprehensive test suite for benchmark evaluation harness, baselines, judge, and cache."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from eval.baselines import FullContextBaseline, MemLayerBaseline, NaiveRAGBaseline
from eval.cache import DiskCache
from eval.datasets.loader import load_eval_dataset
from eval.datasets.models import EvalQuestion, EvalTurn
from eval.judge import LLMJudge
from eval.run import compute_metrics, format_markdown_table, run_single_baseline
from memlayer.extract.extractor import FactExtractor
from memlayer.llm.embedder import MockEmbedder
from memlayer.models import ExtractedFact, ExtractionResult


def _make_mock_extractor(facts: list[str]) -> FactExtractor:
    """Create a deterministic mock FactExtractor for offline tests."""
    extractor = FactExtractor.__new__(FactExtractor)
    extracted = [
        ExtractedFact(
            content=f,
            entity="User",
            confidence=0.95,
            importance=0.8,
            status="active",
        )
        for f in facts
    ]

    async def mock_extract(_messages: Any, _session_id: Any = None) -> ExtractionResult:
        return ExtractionResult(facts=extracted, raw_response="mocked")

    extractor.extract = mock_extract  # type: ignore[assignment]
    return extractor


def _sample_question() -> EvalQuestion:
    return EvalQuestion(
        id="test-q-1",
        benchmark="longmemeval",
        category="knowledge_update",
        user_id="user_test_1",
        split="dev",
        conversation=[
            EvalTurn(
                role="user",
                content="I lived in Paris before.",
                timestamp=datetime(2026, 1, 1, 10, 0, tzinfo=UTC),
                session_id="s1",
            ),
            EvalTurn(
                role="assistant",
                content="Paris is wonderful.",
                timestamp=datetime(2026, 1, 1, 10, 0, 5, tzinfo=UTC),
                session_id="s1",
            ),
            EvalTurn(
                role="user",
                content="Now I moved to Berlin permanently.",
                timestamp=datetime(2026, 2, 1, 12, 0, tzinfo=UTC),
                session_id="s2",
            ),
        ],
        question="Where do I currently live?",
        ground_truth="Berlin (moved from Paris).",
        metadata={"current_value": "Berlin", "superseded_value": "Paris"},
    )


# =============================================================================
# 1. Dataset Loader & Schema Tests
# =============================================================================


def test_load_dataset_longmemeval() -> None:
    questions = load_eval_dataset(benchmark="longmemeval", split="dev")
    assert len(questions) == 4
    categories = {q.category for q in questions}
    assert "knowledge_update" in categories
    assert "temporal_reasoning" in categories
    assert "multi_session" in categories
    assert "abstention" in categories


def test_load_dataset_locomo() -> None:
    questions = load_eval_dataset(benchmark="locomo", split="dev")
    assert len(questions) == 4
    for q in questions:
        assert q.benchmark == "locomo"
        assert len(q.conversation) > 0


def test_load_dataset_category_filter() -> None:
    questions = load_eval_dataset(
        benchmark="longmemeval",
        split="dev",
        category="knowledge_update",
    )
    assert len(questions) == 1
    assert questions[0].category == "knowledge_update"


def test_load_dataset_limit() -> None:
    questions = load_eval_dataset(benchmark="longmemeval", split="dev", limit=2)
    assert len(questions) == 2


def test_load_dataset_invalid_benchmark() -> None:
    with pytest.raises(ValueError, match="Unknown benchmark"):
        load_eval_dataset(benchmark="nonexistent_bench")


# =============================================================================
# 2. Disk Cache Tests
# =============================================================================


def test_disk_cache_operations(tmp_path: Path) -> None:
    cache = DiskCache(cache_dir=tmp_path / "cache")
    model = "test-model"
    prompt = "Hello AI"

    # Initially empty
    assert cache.get(model, prompt) is None
    assert not cache.has(model, prompt)

    # Set value
    cache.set(model, prompt, "Hello User!", temperature=0.0)
    assert cache.has(model, prompt, temperature=0.0)
    assert cache.get(model, prompt, temperature=0.0) == "Hello User!"

    # Different kwargs yields cache miss
    assert cache.get(model, prompt, temperature=0.7) is None

    # Clear cache
    deleted = cache.clear()
    assert deleted == 1
    assert cache.get(model, prompt, temperature=0.0) is None


# =============================================================================
# 3. Baseline Agent Tests
# =============================================================================


@pytest.mark.asyncio
async def test_full_context_baseline(tmp_path: Path) -> None:
    cache = DiskCache(cache_dir=tmp_path / "cache")
    mock_llm = lambda *_a, **_kw: ("I live in Berlin.", 50, 10)  # noqa: E731
    baseline = FullContextBaseline(
        name="full_context",
        cache=cache,
        llm_client=mock_llm,
    )

    q = _sample_question()
    await baseline.setup(q)
    output = await baseline.answer(q)

    assert output.prediction == "I live in Berlin."
    assert output.prompt_tokens == 50
    assert output.completion_tokens == 10
    assert "Berlin" in output.context_used
    assert "Paris" in output.context_used
    await baseline.teardown()


@pytest.mark.asyncio
async def test_naive_rag_baseline(tmp_path: Path) -> None:
    cache = DiskCache(cache_dir=tmp_path / "cache")
    mock_llm = lambda *_a, **_kw: ("You live in Berlin.", 30, 8)  # noqa: E731
    embedder = MockEmbedder(dimension=64)
    baseline = NaiveRAGBaseline(
        name="naive_rag",
        top_k=2,
        embedder=embedder,
        cache=cache,
        llm_client=mock_llm,
    )

    q = _sample_question()
    await baseline.setup(q)
    output = await baseline.answer(q)

    assert output.prediction == "You live in Berlin."
    assert output.total_tokens == 38
    assert output.latency_ms >= 0.0
    await baseline.teardown()


@pytest.mark.asyncio
async def test_memlayer_baseline(tmp_path: Path) -> None:
    cache = DiskCache(cache_dir=tmp_path / "cache")
    mock_llm = lambda *_a, **_kw: ("Your current city is Berlin.", 25, 7)  # noqa: E731
    embedder = MockEmbedder(dimension=64)
    extractor = _make_mock_extractor(["User moved to Berlin permanently."])

    baseline = MemLayerBaseline(
        name="memlayer",
        embedder=embedder,
        extractor=extractor,
        cache=cache,
        llm_client=mock_llm,
    )

    q = _sample_question()
    await baseline.setup(q)
    output = await baseline.answer(q)

    assert output.prediction == "Your current city is Berlin."
    assert "recalled_memories_count" in output.metadata
    await baseline.teardown()


# =============================================================================
# 4. Standardized Judge Tests
# =============================================================================


@pytest.mark.asyncio
async def test_llm_judge_json_parsing() -> None:
    valid_json = json.dumps(
        {
            "is_correct": True,
            "score": 1.0,
            "rationale": "Correctly updated to Berlin.",
        }
    )
    mock_judge_llm = lambda *_a, **_kw: valid_json  # noqa: E731
    judge = LLMJudge(llm_client=mock_judge_llm)

    q = _sample_question()
    res = await judge.evaluate(q, "The user lives in Berlin.", baseline_name="memlayer")
    assert res.is_correct is True
    assert res.score == 1.0
    assert "Berlin" in res.rationale


@pytest.mark.asyncio
async def test_llm_judge_abstention_heuristic() -> None:
    judge = LLMJudge(llm_client=None)  # forces fallback heuristic
    abs_q = EvalQuestion(
        id="abs-1",
        benchmark="longmemeval",
        category="abstention",
        user_id="u1",
        question="What is my cat's name?",
        ground_truth="User never mentioned having a cat.",
    )

    # Correct abstention
    res_pass = await judge.evaluate(abs_q, "That was not mentioned in our conversations.")
    assert res_pass.is_correct is True
    assert res_pass.score == 1.0

    # Hallucination failure
    res_fail = await judge.evaluate(abs_q, "Your cat's name is Whiskers.")
    assert res_fail.is_correct is False
    assert res_fail.score == 0.0


# =============================================================================
# 5. Metrics & Benchmark Runner Smoke Test
# =============================================================================


def test_compute_metrics() -> None:
    items = [
        {
            "category": "knowledge_update",
            "is_correct": True,
            "latency_ms": 100.0,
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        },
        {
            "category": "abstention",
            "is_correct": False,
            "latency_ms": 200.0,
            "prompt_tokens": 80,
            "completion_tokens": 10,
            "total_tokens": 90,
        },
    ]

    metrics = compute_metrics(items)
    assert metrics["total_questions"] == 2
    assert metrics["correct_count"] == 1
    assert metrics["overall_accuracy"] == 50.0
    assert metrics["category_accuracy"]["knowledge_update"]["accuracy"] == 100.0
    assert metrics["category_accuracy"]["abstention"]["accuracy"] == 0.0
    assert metrics["latency"]["mean_ms"] == 150.0
    assert metrics["tokens"]["total"] == 210

    table = format_markdown_table("longmemeval", "memlayer", metrics)
    assert "Overall Accuracy" in table
    assert "knowledge_update" in table


@pytest.mark.asyncio
async def test_runner_smoke_execution(tmp_path: Path) -> None:
    cache = DiskCache(cache_dir=tmp_path / "cache")
    mock_llm = lambda *_a, **_kw: ("Austin, Texas", 40, 5)  # noqa: E731
    questions = load_eval_dataset(
        benchmark="longmemeval",
        split="dev",
        category="knowledge_update",
        limit=1,
    )

    record = await run_single_baseline(
        baseline_cls=FullContextBaseline,
        baseline_name="full_context",
        questions=questions,
        benchmark="longmemeval",
        model="mock-model",
        judge_model="mock-judge",
        cache=cache,
        output_dir=tmp_path / "results",
        extra_baseline_kwargs={"llm_client": mock_llm},
    )

    assert record["baseline"] == "full_context"
    assert len(record["results"]) == 1
    assert record["metrics"]["total_questions"] == 1

    # Check output json file written
    saved_files = list((tmp_path / "results").glob("*.json"))
    assert len(saved_files) == 1


# =============================================================================
# 6. Ablation Study Tests
# =============================================================================


def test_ablation_configs() -> None:
    from eval.ablation import get_ablation_configs

    configs = get_ablation_configs()
    assert "memlayer_full" in configs
    assert "without_decay" in configs
    assert "without_supersession" in configs
    assert "vector_only" in configs
    assert "without_recency" in configs

    assert configs["without_supersession"]["kwargs"]["enable_supersession"] is False
    assert configs["without_decay"]["kwargs"]["enable_decay"] is False
    assert configs["vector_only"]["kwargs"]["vector_only"] is True


@pytest.mark.asyncio
async def test_ablation_suite_execution(tmp_path: Path) -> None:
    from eval.ablation import run_ablation_suite

    cache = DiskCache(cache_dir=tmp_path / "cache")
    mock_llm = lambda *_a, **_kw: ("Berlin (moved from Paris).", 30, 8)  # noqa: E731
    questions = load_eval_dataset(
        benchmark="longmemeval",
        split="dev",
        category="knowledge_update",
        limit=1,
    )

    suite = await run_ablation_suite(
        questions=questions,
        benchmark="longmemeval",
        model="mock-model",
        judge_model="mock-judge",
        cache=cache,
        output_dir=tmp_path / "results",
        extra_baseline_kwargs={
            "llm_client": mock_llm,
            "embedder": MockEmbedder(dimension=64),
            "extractor": _make_mock_extractor(["User moved to Berlin permanently."]),
        },
    )

    assert "suite_id" in suite
    assert len(suite["ablation_metrics"]) == 5
    assert "memlayer_full" in suite["ablation_metrics"]
    assert "without_supersession" in suite["ablation_metrics"]

    suite_files = list((tmp_path / "results").glob("*suite*.json"))
    assert len(suite_files) == 1
