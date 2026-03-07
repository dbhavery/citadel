"""Gateway configuration — loads from environment variables and/or YAML file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class GatewayConfig:
    """Central configuration for the Citadel Gateway."""

    host: str = "0.0.0.0"
    port: int = 8080

    # Provider configs: provider_name -> {"api_key": ..., "base_url": ...}
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Cache settings
    cache_enabled: bool = True
    cache_ttl: int = 3600
    cache_db_path: str = "./gateway_cache.db"

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_rpm: int = 60

    # Routing rules YAML path (optional)
    routing_rules_path: Optional[str] = None

    @classmethod
    def from_env(cls) -> "GatewayConfig":
        """Build a GatewayConfig from environment variables.

        Recognized env vars:
            GATEWAY_HOST, GATEWAY_PORT,
            GATEWAY_CACHE_ENABLED, GATEWAY_CACHE_TTL, GATEWAY_CACHE_DB_PATH,
            GATEWAY_RATE_LIMIT_ENABLED, GATEWAY_RATE_LIMIT_RPM,
            GATEWAY_ROUTING_RULES_PATH,
            ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY,
            OLLAMA_BASE_URL
        """
        providers: dict[str, dict[str, Any]] = {}

        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            providers["anthropic"] = {
                "api_key": anthropic_key,
                "base_url": os.environ.get(
                    "ANTHROPIC_BASE_URL", "https://api.anthropic.com"
                ),
            }

        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            providers["openai"] = {
                "api_key": openai_key,
                "base_url": os.environ.get(
                    "OPENAI_BASE_URL", "https://api.openai.com/v1"
                ),
            }

        google_key = os.environ.get("GOOGLE_API_KEY")
        if google_key:
            providers["google"] = {
                "api_key": google_key,
            }

        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        providers["ollama"] = {"base_url": ollama_url}

        def _bool(val: str | None, default: bool) -> bool:
            if val is None:
                return default
            return val.lower() in ("1", "true", "yes")

        return cls(
            host=os.environ.get("GATEWAY_HOST", "0.0.0.0"),
            port=int(os.environ.get("GATEWAY_PORT", "8080")),
            providers=providers,
            cache_enabled=_bool(os.environ.get("GATEWAY_CACHE_ENABLED"), True),
            cache_ttl=int(os.environ.get("GATEWAY_CACHE_TTL", "3600")),
            cache_db_path=os.environ.get("GATEWAY_CACHE_DB_PATH", "./gateway_cache.db"),
            rate_limit_enabled=_bool(
                os.environ.get("GATEWAY_RATE_LIMIT_ENABLED"), True
            ),
            rate_limit_rpm=int(os.environ.get("GATEWAY_RATE_LIMIT_RPM", "60")),
            routing_rules_path=os.environ.get("GATEWAY_ROUTING_RULES_PATH"),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GatewayConfig":
        """Load config from a YAML file, with env-var overrides applied on top."""
        import yaml  # lazy import — only needed when YAML config is used

        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}

        config = cls.from_env()

        # YAML values fill in anything not set by env
        for key in (
            "host",
            "port",
            "cache_enabled",
            "cache_ttl",
            "cache_db_path",
            "rate_limit_enabled",
            "rate_limit_rpm",
            "routing_rules_path",
        ):
            if key in data and os.environ.get(f"GATEWAY_{key.upper()}") is None:
                setattr(config, key, data[key])

        # Merge provider blocks
        yaml_providers: dict[str, dict[str, Any]] = data.get("providers", {})
        for name, settings in yaml_providers.items():
            if name not in config.providers:
                config.providers[name] = settings
            else:
                config.providers[name] = {**settings, **config.providers[name]}

        return config
