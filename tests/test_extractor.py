"""Tests for FactExtractor, prompt formatting, clean_json_output, and retry logic."""

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from memlayer.extract.extractor import FactExtractor, clean_json_output
from memlayer.models import Episode, ExtractionResult, Role


class TestCleanJsonOutput:
    """Tests for clean_json_output helper function."""

    def test_clean_raw_json(self) -> None:
        raw = '{"facts": [{"content": "User lives in Pune"}]}'
        assert clean_json_output(raw) == raw

    def test_clean_markdown_json_block(self) -> None:
        raw = '```json\n{"facts": [{"content": "User lives in Pune"}]}\n```'
        cleaned = clean_json_output(raw)
        assert json.loads(cleaned) == {"facts": [{"content": "User lives in Pune"}]}

    def test_clean_markdown_generic_block(self) -> None:
        raw = '```\n{"facts": []}\n```'
        cleaned = clean_json_output(raw)
        assert json.loads(cleaned) == {"facts": []}

    def test_clean_text_with_surrounding_commentary(self) -> None:
        raw = (
            'Here is the extracted memory JSON:\n{"facts": [{"content": "Fact"}]}\nHope this helps!'
        )
        cleaned = clean_json_output(raw)
        assert json.loads(cleaned) == {"facts": [{"content": "Fact"}]}


class TestFactExtractor:
    """Unit tests for FactExtractor execution and retry logic."""

    @pytest.mark.asyncio
    async def test_extract_success_with_mock_completion(self) -> None:
        mock_response = json.dumps(
            {
                "facts": [
                    {
                        "content": "User lives in Pune",
                        "entity": "User",
                        "importance": 0.85,
                        "confidence": 0.95,
                    },
                    {
                        "content": "User works on conversational AI",
                        "entity": "User",
                        "importance": 0.8,
                        "confidence": 0.9,
                    },
                ]
            }
        )

        async def mock_llm(**_kwargs: Any) -> str:
            return mock_response

        extractor = FactExtractor(completion_fn=mock_llm)
        messages = [
            {"role": "user", "content": "I live in Pune and work on conversational AI."},
            {"role": "assistant", "content": "Great to meet you!"},
        ]
        result = await extractor.extract(messages)

        assert isinstance(result, ExtractionResult)
        assert len(result.facts) == 2
        assert result.facts[0].content == "User lives in Pune"
        assert result.facts[0].importance == 0.85
        assert result.facts[1].content == "User works on conversational AI"

    @pytest.mark.asyncio
    async def test_extract_with_episodes_input(self) -> None:
        mock_response = json.dumps(
            {
                "facts": [
                    {
                        "content": "User prefers dark mode",
                        "entity": "User",
                        "importance": 0.6,
                        "confidence": 1.0,
                    }
                ]
            }
        )

        async def mock_llm(**_kwargs: Any) -> str:
            return mock_response

        extractor = FactExtractor(completion_fn=mock_llm)
        episodes = [
            Episode(user_id="user_1", role=Role.USER, text="Please enable dark mode."),
            Episode(user_id="user_1", role=Role.ASSISTANT, text="Dark mode is on!"),
        ]
        result = await extractor.extract(episodes)

        assert len(result.facts) == 1
        assert result.facts[0].content == "User prefers dark mode"

    @pytest.mark.asyncio
    async def test_extract_empty_messages_returns_empty(self) -> None:
        extractor = FactExtractor()
        result = await extractor.extract([])
        assert len(result.facts) == 0

        result_blank = await extractor.extract([{"role": "user", "content": "   "}])
        assert len(result_blank.facts) == 0

    @pytest.mark.asyncio
    async def test_extract_retry_on_malformed_json_recovers(self) -> None:
        attempt_count = 0

        async def flaky_llm(**_kwargs: Any) -> str:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                # 1st attempt: return truncated invalid JSON
                return '{"facts": [{"content": "Incomplete string'
            # 2nd attempt: return valid JSON
            return json.dumps(
                {
                    "facts": [
                        {
                            "content": "User relocated to Bangalore",
                            "entity": "User",
                            "importance": 0.9,
                            "confidence": 1.0,
                        }
                    ]
                }
            )

        extractor = FactExtractor(max_retries=2, completion_fn=flaky_llm)
        messages = [{"role": "user", "content": "I recently relocated to Bangalore."}]
        result = await extractor.extract(messages)

        assert attempt_count == 2
        assert len(result.facts) == 1
        assert result.facts[0].content == "User relocated to Bangalore"

    @pytest.mark.asyncio
    async def test_extract_max_retries_exceeded_returns_empty_gracefully(self) -> None:
        attempt_count = 0

        async def permanently_broken_llm(**_kwargs: Any) -> str:
            nonlocal attempt_count
            attempt_count += 1
            return "Not JSON at all! Just raw plain text."

        extractor = FactExtractor(max_retries=2, completion_fn=permanently_broken_llm)
        messages = [{"role": "user", "content": "Some input."}]
        result = await extractor.extract(messages)

        # Initial attempt (1) + 2 retries = 3 attempts total
        assert attempt_count == 3
        assert len(result.facts) == 0


class TestFixturePrecision:
    """Precision verification against fixed dialog fixtures."""

    @pytest.mark.asyncio
    async def test_fixture_dialogs_parsing(self) -> None:
        fixture_path = Path(__file__).parent / "fixtures" / "dialogs.json"
        assert fixture_path.exists(), "dialogs.json fixture must exist"

        with open(fixture_path, encoding="utf-8") as f:
            fixtures = json.load(f)

        assert len(fixtures) >= 4

        def make_mock(resp: str) -> Callable[..., Awaitable[str]]:
            async def fixture_mock(**_kwargs: Any) -> str:
                return resp

            return fixture_mock

        for item in fixtures:
            expected_facts = item.get("expected_facts", [])

            # Simulate LLM returning the expected facts for each fixture
            simulated_response = json.dumps(
                {
                    "facts": [
                        {
                            "content": ef["content"],
                            "entity": ef.get("entity", "User"),
                            "importance": ef.get("min_importance", 0.5),
                            "confidence": 0.95,
                        }
                        for ef in expected_facts
                    ]
                }
            )

            extractor = FactExtractor(completion_fn=make_mock(simulated_response))
            result = await extractor.extract(item["messages"])

            assert len(result.facts) == len(expected_facts)
            for idx, ef in enumerate(expected_facts):
                assert result.facts[idx].content == ef["content"]
                assert result.facts[idx].importance >= ef.get("min_importance", 0.0)
