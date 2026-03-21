"""Utilities for Anthropic API responses: cost estimation, text extraction, JSON parsing.

Used for direct Anthropic API calls (generator, optimizer).
The Claude Code SDK provides `total_cost_usd` directly on ResultMessage.
"""

from __future__ import annotations

from typing import Any

# Anthropic model pricing (per million tokens) as of 2025-05
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4": {"input": 3.0, "output": 15.0},
    "claude-haiku-3-5": {"input": 0.80, "output": 4.0},
    "claude-haiku-3-5-20241022": {"input": 0.80, "output": 4.0},
    "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
    "claude-opus-4": {"input": 15.0, "output": 75.0},
}

# Fallback pricing if model not found (use sonnet pricing as default)
_DEFAULT_PRICING = {"input": 3.0, "output": 15.0}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost from token counts.

    Args:
        model: The model identifier (e.g., "claude-sonnet-4-20250514").
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.

    Returns:
        Estimated cost in USD.
    """
    pricing = MODEL_PRICING.get(model, _DEFAULT_PRICING)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def extract_response_text(content_blocks: list[Any]) -> str:
    """Extract concatenated text from Anthropic API response content blocks.

    Args:
        content_blocks: The response.content list from an Anthropic API call.

    Returns:
        Concatenated text from all text blocks.
    """
    return "".join(block.text for block in content_blocks if block.type == "text")


def strip_code_fences(text: str) -> str:
    """Strip markdown code fences from LLM output.

    Handles responses wrapped in ```json ... ``` or similar fenced blocks.

    Args:
        text: Raw text that may be wrapped in code fences.

    Returns:
        The inner content with fences removed, or the original text if no fences found.
    """
    text = text.strip()
    if not text.startswith("```"):
        return text

    lines = text.split("\n")
    # First line is the opening fence (e.g., "```json")
    start_idx = 1
    end_idx = len(lines)
    for i in range(start_idx, len(lines)):
        if lines[i].strip() == "```":
            end_idx = i
            break
    return "\n".join(lines[start_idx:end_idx])
