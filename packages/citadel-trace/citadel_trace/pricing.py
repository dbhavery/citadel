"""Model pricing database for LLM cost tracking.

Prices are per 1M tokens as (input_cost, output_cost) in USD.
"""

from __future__ import annotations

from typing import Optional

# Cost per 1M tokens: (input_usd, output_usd)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-6": (15.0, 75.0),
    "claude-haiku-4-5": (0.80, 4.0),
    # OpenAI
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1": (15.0, 60.0),
    "o1-mini": (3.0, 12.0),
    # Google
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.0-flash": (0.10, 0.40),
    # Local models are free
    "ollama/*": (0.0, 0.0),
}


def _match_model(model: str) -> Optional[tuple[float, float]]:
    """Look up pricing for a model, supporting wildcard patterns."""
    # Direct match
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]

    # Wildcard match: check if model starts with any wildcard prefix
    for pattern, pricing in MODEL_PRICING.items():
        if pattern.endswith("/*"):
            prefix = pattern[:-1]  # "ollama/*" -> "ollama/"
            if model.startswith(prefix):
                return pricing

    return None


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Calculate cost in USD for a given model and token counts.

    Args:
        model: The model identifier (e.g., "claude-sonnet-4-6", "ollama/qwen3:8b").
        input_tokens: Number of input (prompt) tokens.
        output_tokens: Number of output (completion) tokens.

    Returns:
        Cost in USD. Returns 0.0 for unknown models.
    """
    pricing = _match_model(model)
    if pricing is None:
        return 0.0

    input_cost_per_token = pricing[0] / 1_000_000.0
    output_cost_per_token = pricing[1] / 1_000_000.0

    return (input_tokens * input_cost_per_token) + (output_tokens * output_cost_per_token)
