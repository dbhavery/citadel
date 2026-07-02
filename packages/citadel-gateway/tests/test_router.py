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


class TestResolveAll:
    """Verify the failover chain produced by resolve_all."""

    def test_resolve_all_returns_chain_in_priority_order(self) -> None:
        router = Router(
            rules=[
                RoutingRule(pattern=r"m-.*", provider="a", model="{model}", priority=10),
                RoutingRule(pattern=r"m-.*", provider="b", model="{model}", priority=5),
                RoutingRule(pattern=r"m-.*", provider="c", model="{model}", priority=1),
            ]
        )
        chain = router.resolve_all("m-1")
        assert [r.provider for r in chain] == ["a", "b", "c"]

    def test_resolve_all_dedupes_provider_keeping_highest_priority(self) -> None:
        router = Router(
            rules=[
                RoutingRule(pattern=r"m-.*", provider="a", model="hi-{model}", priority=10),
                RoutingRule(pattern=r"m-.*", provider="a", model="lo-{model}", priority=1),
                RoutingRule(pattern=r"m-.*", provider="b", model="{model}", priority=5),
            ]
        )
        chain = router.resolve_all("m-1")
        assert [r.provider for r in chain] == ["a", "b"]
        # Highest-priority rule for provider "a" wins.
        assert chain[0].model == "hi-m-1"

    def test_resolve_all_default_rules_include_ollama_fallback(self) -> None:
        # A claude model falls back to the ollama catch-all after anthropic.
        chain = Router().resolve_all("claude-sonnet-4-20250514")
        providers = [r.provider for r in chain]
        assert providers[0] == "anthropic"
        assert "ollama" in providers

    def test_resolve_all_raises_when_no_rule_matches(self) -> None:
        router = Router(
            rules=[RoutingRule(pattern=r"only", provider="a", model="{model}", priority=1)]
        )
        import pytest

        with pytest.raises(ValueError):
            router.resolve_all("nope")


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
