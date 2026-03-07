"""Tests for model pricing calculations."""

from citadel_trace.pricing import calculate_cost


class TestPricing:
    """Test cost calculation for various models."""

    def test_known_model_cost(self) -> None:
        """Known models produce correct cost calculations."""
        # claude-sonnet-4-6: $3.00/1M input, $15.00/1M output
        cost = calculate_cost("claude-sonnet-4-6", input_tokens=1000, output_tokens=500)
        expected = (1000 * 3.0 / 1_000_000) + (500 * 15.0 / 1_000_000)
        assert abs(cost - expected) < 1e-9
        assert abs(cost - 0.01050) < 1e-9

        # gpt-4o: $2.50/1M input, $10.00/1M output
        cost_gpt = calculate_cost("gpt-4o", input_tokens=2000, output_tokens=1000)
        expected_gpt = (2000 * 2.50 / 1_000_000) + (1000 * 10.0 / 1_000_000)
        assert abs(cost_gpt - expected_gpt) < 1e-9

    def test_unknown_model_returns_zero(self) -> None:
        """Unknown models return 0.0 cost."""
        cost = calculate_cost("totally-unknown-model-v99", input_tokens=10000, output_tokens=5000)
        assert cost == 0.0

    def test_ollama_models_are_free(self) -> None:
        """All ollama/* models are free (cost = 0.0)."""
        cost1 = calculate_cost("ollama/qwen3:8b", input_tokens=50000, output_tokens=20000)
        assert cost1 == 0.0

        cost2 = calculate_cost("ollama/llama3:70b", input_tokens=100000, output_tokens=50000)
        assert cost2 == 0.0

        cost3 = calculate_cost("ollama/mistral:latest", input_tokens=1, output_tokens=1)
        assert cost3 == 0.0
