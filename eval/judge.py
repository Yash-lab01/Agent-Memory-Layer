"""Standardized LLM Judge for memory benchmark evaluation."""

import json
import re
from collections.abc import Callable
from typing import Any

import litellm
from pydantic import BaseModel, Field

from eval.cache import DiskCache
from eval.datasets.models import EvalQuestion


class JudgeResult(BaseModel):
    """Result of an evaluation judgment for a single test item."""

    question_id: str
    baseline: str
    category: str
    score: float = Field(ge=0.0, le=1.0)
    is_correct: bool
    rationale: str


JUDGE_SYSTEM_PROMPT = """You are an impartial, highly rigorous evaluation judge for long-term memory benchmarks in conversational AI.
You evaluate whether an agent's answer to a user question accurately matches the Ground Truth information.

Category-Specific Evaluation Guidelines:
1. 'knowledge_update': The user changed or updated a previous fact. The prediction MUST state the updated fact and NOT the old, superseded fact. If the prediction gives the outdated fact, it is INCORRECT (score 0.0).
2. 'abstention': The user never mentioned this information. The prediction MUST explicitly state that the information was not mentioned, unknown, or not in the conversation history. If the agent invents or hallucinates an answer, it is INCORRECT (score 0.0).
3. 'temporal_reasoning': The user mentioned dates, durations, or temporal sequences. The prediction must correctly compute or identify the target temporal fact.
4. 'multi_session': The user shared facts across separate sessions. The prediction must correctly integrate facts across both sessions.

You MUST respond ONLY with a valid JSON object in the following format:
{
  "is_correct": true,
  "score": 1.0,
  "rationale": "Clear, concise reason explaining why the prediction matches or fails the ground truth."
}
"""


class LLMJudge:
    """Standardized LLM judge comparing baseline predictions against ground truth."""

    def __init__(
        self,
        model: str = "gemini/gemini-1.5-pro",
        cache: DiskCache | None = None,
        llm_client: Callable[..., Any] | None = None,
    ) -> None:
        self.model = model
        self.cache = cache
        self.llm_client = llm_client

    def _format_judge_prompt(self, item: EvalQuestion, prediction: str) -> str:
        return (
            f"Question: {item.question}\n"
            f"Category: {item.category}\n"
            f"Ground Truth: {item.ground_truth}\n"
            f"Model Prediction: {prediction}\n\n"
            f"Provide your JSON judgment:"
        )

    async def evaluate(
        self,
        item: EvalQuestion,
        prediction: str,
        baseline_name: str = "agent",
    ) -> JudgeResult:
        """Evaluate a prediction against ground truth for an EvalQuestion."""
        prompt = self._format_judge_prompt(item, prediction)

        # Check cache if available
        raw_response: str | None = None
        if self.cache:
            raw_response = self.cache.get(self.model, prompt)

        if raw_response is None:
            if self.llm_client is not None:
                res = self.llm_client(
                    prompt=prompt,
                    system_prompt=JUDGE_SYSTEM_PROMPT,
                    model=self.model,
                )
                if hasattr(res, "__await__"):
                    res = await res
                raw_response = str(res)
            else:
                try:
                    response = await litellm.acompletion(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.0,
                        max_tokens=256,
                    )
                    raw_response = response.choices[0].message.content or ""
                except Exception as err:
                    # Fallback heuristic if external LLM judge is unreachable
                    return self._fallback_evaluate(item, prediction, baseline_name, str(err))

            if self.cache and raw_response:
                self.cache.set(self.model, prompt, raw_response)

        # Parse JSON judgment
        return self._parse_judge_response(raw_response or "", item, baseline_name)

    def _parse_judge_response(
        self,
        raw_text: str,
        item: EvalQuestion,
        baseline_name: str,
    ) -> JudgeResult:
        """Parse structured judgment from JSON response."""
        try:
            cleaned = raw_text.strip()
            # Extract JSON block if wrapped in markdown fences
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(0)

            data = json.loads(cleaned)
            is_correct = bool(data.get("is_correct", False))
            score = float(data.get("score", 1.0 if is_correct else 0.0))
            rationale = str(data.get("rationale", "No rationale provided"))
            return JudgeResult(
                question_id=item.id,
                baseline=baseline_name,
                category=item.category,
                score=score,
                is_correct=is_correct,
                rationale=rationale,
            )
        except Exception:
            return self._fallback_evaluate(
                item,
                raw_text,
                baseline_name,
                "Failed to parse structured JSON from judge output",
            )

    def _fallback_evaluate(
        self,
        item: EvalQuestion,
        prediction: str,
        baseline_name: str,
        reason: str,
    ) -> JudgeResult:
        """Rule-based heuristic fallback for evaluation in offline environments."""
        pred_lower = prediction.lower()
        category = item.category

        if category == "abstention":
            abstention_keywords = [
                "not mentioned",
                "never mentioned",
                "unknown",
                "don't know",
                "do not know",
                "no mention",
                "not specified",
                "not provided",
            ]
            is_correct = any(kw in pred_lower for kw in abstention_keywords)
            score = 1.0 if is_correct else 0.0
            rationale = (
                f"Heuristic abstention check: {'passed' if is_correct else 'failed'}. ({reason})"
            )
            return JudgeResult(
                question_id=item.id,
                baseline=baseline_name,
                category=category,
                score=score,
                is_correct=is_correct,
                rationale=rationale,
            )

        # For knowledge update, check that current value is present and superseded is absent
        if category == "knowledge_update":
            curr = item.metadata.get("current_value", "").lower()
            old = item.metadata.get("superseded_value", "").lower()
            if curr and curr in pred_lower:
                if old and old in pred_lower and "originally" not in pred_lower:
                    is_correct = False
                    score = 0.0
                    rationale = f"Heuristic check: Found old superseded value '{old}' alongside new value. ({reason})"
                else:
                    is_correct = True
                    score = 1.0
                    rationale = f"Heuristic check: Found current updated value '{curr}'. ({reason})"
                return JudgeResult(
                    question_id=item.id,
                    baseline=baseline_name,
                    category=category,
                    score=score,
                    is_correct=is_correct,
                    rationale=rationale,
                )

        # Generic token overlap check with ground truth
        gt_words = {w for w in re.findall(r"\w+", item.ground_truth.lower()) if len(w) > 3}
        pred_words = {w for w in re.findall(r"\w+", pred_lower) if len(w) > 3}
        overlap = len(gt_words.intersection(pred_words))
        is_correct = overlap >= max(1, len(gt_words) // 3)
        score = 1.0 if is_correct else 0.0
        rationale = f"Heuristic word overlap ({overlap}/{len(gt_words)} words matching). ({reason})"

        return JudgeResult(
            question_id=item.id,
            baseline=baseline_name,
            category=category,
            score=score,
            is_correct=is_correct,
            rationale=rationale,
        )
