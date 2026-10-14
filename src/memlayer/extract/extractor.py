"""Atomic fact extractor using LiteLLM with schema validation and automatic retry."""

import json
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import litellm
from pydantic import ValidationError

from memlayer.extract.prompts import EXTRACTION_SYSTEM_PROMPT, RETRY_PROMPT_TEMPLATE
from memlayer.models import Episode, ExtractionResult

logger = logging.getLogger(__name__)

CompletionFn = Callable[..., Awaitable[Any]]


def clean_json_output(raw_text: str) -> str:
    """Strip markdown code fence blocks and extract the JSON object substring."""
    text = raw_text.strip()

    # Remove ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        pattern = r"^```(?:json)?\s*\n?(.*?)\n?```$"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()

    # Fallback: extract substring between first { and last }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace : last_brace + 1].strip()

    return text


def _format_messages_for_prompt(messages: Sequence[dict[str, str] | Episode]) -> str:
    """Format dialogue messages into a clean transcript block for extraction."""
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, Episode):
            role_str = msg.role.value.capitalize()
            content = msg.text.strip()
        else:
            role_str = str(msg.get("role", "user")).capitalize()
            content = str(msg.get("content", "")).strip()

        if content:
            lines.append(f"{role_str}: {content}")
    return "\n".join(lines)


class FactExtractor:
    """Extracts durable atomic facts from conversation turns with Pydantic validation."""

    def __init__(
        self,
        model_name: str = "gemini/gemini-2.0-flash",
        max_retries: int = 2,
        temperature: float = 0.0,
        completion_fn: CompletionFn | None = None,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.max_retries = max_retries
        self.temperature = temperature
        self.completion_fn = completion_fn
        self.extra_kwargs = kwargs

    async def _call_llm(self, llm_messages: list[dict[str, str]]) -> str:
        """Execute LLM call using injected function or LiteLLM."""
        if self.completion_fn is not None:
            resp = await self.completion_fn(
                model=self.model_name,
                messages=llm_messages,
                temperature=self.temperature,
                **self.extra_kwargs,
            )
        else:
            resp = await litellm.acompletion(
                model=self.model_name,
                messages=llm_messages,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                **self.extra_kwargs,
            )

        # Extract text content from standard OpenAI / LiteLLM structure or raw string
        if isinstance(resp, str):
            return resp
        if hasattr(resp, "choices") and resp.choices:
            return str(resp.choices[0].message.content or "")
        if isinstance(resp, dict):
            choices = resp.get("choices", [])
            if choices:
                return str(choices[0].get("message", {}).get("content", ""))
        return str(resp)

    async def extract(
        self,
        messages: Sequence[dict[str, str] | Episode],
    ) -> ExtractionResult:
        """Extract atomic facts from dialogue messages with retry on malformed JSON."""
        transcript = _format_messages_for_prompt(messages)
        if not transcript.strip():
            return ExtractionResult(facts=[])

        user_content = f"Dialogue turns to analyze:\n\n{transcript}\n\nExtract atomic facts:"
        llm_messages: list[dict[str, str]] = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        last_error = ""
        raw_output = ""

        for attempt in range(self.max_retries + 1):
            try:
                raw_output = await self._call_llm(llm_messages)
                cleaned_output = clean_json_output(raw_output)
                parsed = json.loads(cleaned_output)
                return ExtractionResult.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError, Exception) as exc:
                last_error = str(exc)
                logger.warning(
                    "Extraction attempt %d failed: %s. Raw output: %s",
                    attempt + 1,
                    last_error,
                    raw_output[:200],
                )
                if attempt < self.max_retries:
                    retry_msg = RETRY_PROMPT_TEMPLATE.format(
                        error_message=last_error,
                        raw_output=raw_output,
                    )
                    llm_messages.append({"role": "assistant", "content": raw_output})
                    llm_messages.append({"role": "user", "content": retry_msg})

        logger.error(
            "Fact extraction failed after %d attempts. Last error: %s",
            self.max_retries + 1,
            last_error,
        )
        return ExtractionResult(facts=[])
