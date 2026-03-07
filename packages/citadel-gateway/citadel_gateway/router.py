"""Request routing — matches model strings to providers using regex rules."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class RoutingRule:
    """A single routing rule that maps a model name pattern to a provider."""

    pattern: str  # regex applied to the requested model name
    provider: str  # target provider key (e.g. "anthropic", "ollama")
    model: str  # actual model name to pass to the provider
    priority: int = 0  # higher = checked first
    cost_per_1k_tokens: Optional[float] = None  # for cost-aware routing

    def matches(self, model_name: str) -> bool:
        """Return True if *model_name* matches this rule's pattern."""
        return bool(re.fullmatch(self.pattern, model_name))


# Default routing rules — lowest priority so user rules win.
DEFAULT_RULES: list[RoutingRule] = [
    RoutingRule(
        pattern=r"claude-.*",
        provider="anthropic",
        model="{model}",
        priority=0,
        cost_per_1k_tokens=3.0,
    ),
    RoutingRule(
        pattern=r"gemini-.*",
        provider="google",
        model="{model}",
        priority=0,
        cost_per_1k_tokens=0.5,
    ),
    RoutingRule(
        pattern=r"gpt-.*",
        provider="openai",
        model="{model}",
        priority=0,
        cost_per_1k_tokens=2.0,
    ),
    RoutingRule(
        pattern=r".*",
        provider="ollama",
        model="{model}",
        priority=-1,
        cost_per_1k_tokens=0.0,
    ),
]


@dataclass
class RouteResult:
    """The resolved target for a request."""

    provider: str
    model: str
    cost_per_1k_tokens: Optional[float] = None


class Router:
    """Matches an incoming model string to a provider and concrete model name."""

    def __init__(self, rules: list[RoutingRule] | None = None) -> None:
        self._rules: list[RoutingRule] = list(rules) if rules else list(DEFAULT_RULES)
        self._sort_rules()

    def _sort_rules(self) -> None:
        """Keep rules sorted by descending priority."""
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def add_rule(self, rule: RoutingRule) -> None:
        """Add a routing rule and re-sort."""
        self._rules.append(rule)
        self._sort_rules()

    @property
    def rules(self) -> list[RoutingRule]:
        """Return current rules (read-only copy)."""
        return list(self._rules)

    def resolve(self, model_name: str) -> RouteResult:
        """Resolve *model_name* to a provider and concrete model.

        The ``{model}`` placeholder in a rule's ``model`` field is replaced
        with the original *model_name* so that pass-through routing works.

        Raises ``ValueError`` if no rule matches (shouldn't happen with default
        catch-all, but guards against misconfigured rule sets).
        """
        for rule in self._rules:
            if rule.matches(model_name):
                concrete_model = rule.model.replace("{model}", model_name)
                return RouteResult(
                    provider=rule.provider,
                    model=concrete_model,
                    cost_per_1k_tokens=rule.cost_per_1k_tokens,
                )
        raise ValueError(f"No routing rule matched model: {model_name!r}")

    def cheapest(self, model_name: str) -> RouteResult:
        """Like ``resolve`` but picks the cheapest matching rule."""
        matches: list[tuple[RoutingRule, RouteResult]] = []
        for rule in self._rules:
            if rule.matches(model_name):
                concrete_model = rule.model.replace("{model}", model_name)
                result = RouteResult(
                    provider=rule.provider,
                    model=concrete_model,
                    cost_per_1k_tokens=rule.cost_per_1k_tokens,
                )
                matches.append((rule, result))

        if not matches:
            raise ValueError(f"No routing rule matched model: {model_name!r}")

        # Sort by cost (None treated as infinity)
        matches.sort(key=lambda pair: pair[0].cost_per_1k_tokens or float("inf"))
        return matches[0][1]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Router":
        """Load routing rules from a YAML file.

        Expected format::

            rules:
              - pattern: "claude-.*"
                provider: anthropic
                model: "{model}"
                priority: 10
                cost_per_1k_tokens: 3.0
        """
        import yaml  # lazy import

        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Routing rules file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}

        raw_rules = data.get("rules", [])
        rules: list[RoutingRule] = []
        for entry in raw_rules:
            rules.append(
                RoutingRule(
                    pattern=entry["pattern"],
                    provider=entry["provider"],
                    model=entry.get("model", "{model}"),
                    priority=entry.get("priority", 0),
                    cost_per_1k_tokens=entry.get("cost_per_1k_tokens"),
                )
            )

        # Append default catch-all so there's always a fallback
        rules.extend(DEFAULT_RULES)
        return cls(rules=rules)
