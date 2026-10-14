"""Context formatter and token-budgeted prompt injection utilities."""

from __future__ import annotations

from collections.abc import Sequence

from memlayer.models import ScoredMemory


def estimate_tokens(text: str) -> int:
    """Estimate token count for prompt injection budgeting."""
    if not text:
        return 0
    words = text.split()
    return max(1, int(len(words) * 1.3))


def format_memory_line(sm: ScoredMemory) -> str:
    """Format a single scored memory into a timestamped context string."""
    date_str = sm.memory.created_at.strftime("%Y-%m-%d")
    entity = sm.memory.entity.strip() if sm.memory.entity else ""

    if entity and entity.lower() not in {"user", "none"}:
        return f"[{date_str}] ({entity}) {sm.memory.content}"
    return f"[{date_str}] {sm.memory.content}"


def format_context(
    memories: Sequence[ScoredMemory],
    token_budget: int | None = None,
) -> tuple[str, list[ScoredMemory], int]:
    """Pack scored memories into a timestamped context block within a token budget.

    Args:
        memories: Sequence of ScoredMemory instances (assumed sorted by score descending).
        token_budget: Optional maximum token allowance.

    Returns:
        Tuple of (formatted_context_string, included_memories, estimated_token_count).
    """
    if not memories:
        return "", [], 0

    lines: list[str] = []
    included: list[ScoredMemory] = []
    current_tokens = 0

    for sm in memories:
        line = format_memory_line(sm)
        line_tokens = estimate_tokens(line)

        # If budget is specified, verify if adding this line exceeds the limit
        if token_budget is not None and (current_tokens + line_tokens) > token_budget:
            # If nothing has been added yet and budget is very small, we skip to strictly honor budget
            if not included:
                break
            break

        lines.append(line)
        included.append(sm)
        current_tokens += line_tokens

    formatted_context = "\n".join(lines)
    return formatted_context, included, current_tokens
