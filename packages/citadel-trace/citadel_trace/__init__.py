"""citadel-trace: LLM observability -- auto-instrument SDKs, trace calls, track costs, monitor latency."""

__version__ = "0.1.0"

from .alerts import AlertManager, AlertRule
from .collector import TraceCollector
from .instrument import Instrumentor
from .metrics import MetricsCalculator
from .pricing import MODEL_PRICING, calculate_cost
from .span import Span, Trace

__all__ = [
    "AlertManager",
    "AlertRule",
    "Instrumentor",
    "MetricsCalculator",
    "MODEL_PRICING",
    "Span",
    "Trace",
    "TraceCollector",
    "calculate_cost",
]
