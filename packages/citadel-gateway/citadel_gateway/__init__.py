"""Citadel Gateway: LLM reverse proxy with routing, caching, rate limiting, and failover."""

# Defined before the import below on purpose: server.py imports __version__ from
# this package, so binding it first is what keeps that from being a circular
# import against a partially initialised module.
__version__ = "0.1.0"

from citadel_gateway.server import create_app  # noqa: E402

__all__ = ["create_app", "__version__"]
