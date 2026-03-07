"""Tests for citadel_gateway.router."""

from citadel_gateway.router import Router, RoutingRule


class TestRouterDefaultRules:
    """Verify the built-in routing rules."""

    def test_claude_routes_to_anthropic(self) -> None:
        router = Router()
        result = router.resolve("claude-sonnet-4-20250514")
        assert result.provider == "anthropic"
        assert result.model == "claude-sonnet-4-20250514"

    def test_gemini_routes_to_google(self) -> None:
        router = Router()
        result = router.resolve("gemini-2.5-flash")
        assert result.provider == "google"
        assert result.model == "gemini-2.5-flash"

    def test_gpt_routes_to_openai(self) -> None:
        router = Router()
        result = router.resolve("gpt-4o")
        assert result.provider == "openai"
        assert result.model == "gpt-4o"

    def test_unknown_model_routes_to_ollama(self) -> None:
        router = Router()
        result = router.resolve("qwen3:8b")
        assert result.provider == "ollama"
        assert result.model == "qwen3:8b"

    def test_unknown_arbitrary_name_routes_to_ollama(self) -> None:
        router = Router()
        result = router.resolve("my-custom-model")
        assert result.provider == "ollama"


class TestRouterCustomRules:
    """Verify user-supplied rules take priority."""

    def test_custom_rule_overrides_default(self) -> None:
        custom = RoutingRule(
            pattern=r"claude-.*",
            provider="my-proxy",
            model="proxy-{model}",
            priority=100,
        )
        router = Router()
        router.add_rule(custom)
        result = router.resolve("claude-sonnet-4-20250514")
        assert result.provider == "my-proxy"
        assert result.model == "proxy-claude-sonnet-4-20250514"

    def test_priority_ordering(self) -> None:
        low = RoutingRule(pattern=r"test-.*", provider="low", model="{model}", priority=1)
        high = RoutingRule(pattern=r"test-.*", provider="high", model="{model}", priority=10)
        router = Router(rules=[low, high])
        result = router.resolve("test-model")
        assert result.provider == "high"


class TestCostAwareRouting:
    """Verify cost-aware routing picks the cheapest match."""

    def test_cheapest_picks_lowest_cost(self) -> None:
        rules = [
            RoutingRule(pattern=r".*", provider="expensive", model="{model}", priority=10, cost_per_1k_tokens=10.0),
            RoutingRule(pattern=r".*", provider="cheap", model="{model}", priority=5, cost_per_1k_tokens=0.1),
            RoutingRule(pattern=r".*", provider="mid", model="{model}", priority=7, cost_per_1k_tokens=2.0),
        ]
        router = Router(rules=rules)
        result = router.cheapest("any-model")
        assert result.provider == "cheap"
        assert result.cost_per_1k_tokens == 0.1

    def test_cost_annotation_present(self) -> None:
        router = Router()
        result = router.resolve("claude-sonnet")
        assert result.cost_per_1k_tokens is not None
        assert result.cost_per_1k_tokens > 0
