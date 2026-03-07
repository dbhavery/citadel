"""Built-in tools available to all agents.

Provides common utility tools: calculator, current_time, read_file,
write_file, and http_get.
"""

from __future__ import annotations

import ast
import ipaddress
import logging
import operator
import os
import socket
from datetime import datetime, timezone
from typing import Any, Callable, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

from citadel_agents.tool import ToolRegistry, _generate_schema


_SAFE_BINOPS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_SAFE_UNARYOPS: dict[type, Callable[..., Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> Union[int, float]:
    """Recursively evaluate an AST node using only safe arithmetic operations.

    Raises:
        ValueError: If the node contains unsupported operations.
    """
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
        return node.value
    if isinstance(node, ast.BinOp):
        op_fn = _SAFE_BINOPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return op_fn(left, right)
    if isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_UNARYOPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_fn(_eval_node(node.operand))
    raise ValueError(f"Unsupported operation in expression: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Supports basic arithmetic: +, -, *, /, **, %, //.
    Does NOT support function calls or imports.

    Args:
        expression: A mathematical expression string (e.g., "2 + 3 * 4").

    Returns:
        The result as a string.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree)
        return str(result)
    except (SyntaxError, TypeError, ZeroDivisionError, ValueError) as e:
        return f"Error: {e}"


def current_time() -> str:
    """Return the current date and time in ISO 8601 format (UTC).

    Returns:
        Current datetime string.
    """
    return datetime.now(timezone.utc).isoformat()


def _get_allowed_directories() -> list[str]:
    """Return the list of directories agents are allowed to access.

    Configured via CITADEL_AGENT_ALLOWED_DIRS (comma-separated paths).
    Defaults to current working directory if not set.
    """
    env_val = os.environ.get("CITADEL_AGENT_ALLOWED_DIRS", "")
    if env_val.strip():
        return [os.path.normpath(d.strip()) for d in env_val.split(",") if d.strip()]
    return [os.path.normpath(os.getcwd())]


def _is_path_allowed(path: str) -> bool:
    """Check if a path falls within one of the allowed directories."""
    normalized = os.path.normpath(os.path.abspath(path))
    for allowed in _get_allowed_directories():
        # Use os.path.commonpath to safely check containment
        try:
            common = os.path.commonpath([normalized, allowed])
            if common == allowed:
                return True
        except ValueError:
            # Different drives on Windows
            continue
    return False


def read_file(path: str) -> str:
    """Read and return the contents of a file.

    Args:
        path: Absolute path to the file to read.

    Returns:
        The file contents as a string, or an error message.
    """
    normalized = os.path.normpath(path)
    if ".." in normalized.split(os.sep):
        return "Error: Path traversal (..) is not allowed"
    if not _is_path_allowed(normalized):
        return "Error: Path is outside allowed directories"
    try:
        with open(normalized, "r", encoding="utf-8") as f:
            content = f.read()
        if len(content) > 100_000:
            return content[:100_000] + "\n... [truncated at 100,000 characters]"
        return content
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(path: str, content: str) -> str:
    """Write content to a file, creating directories as needed.

    Args:
        path: Absolute path to write to.
        content: Content to write.

    Returns:
        Success message or error string.
    """
    normalized = os.path.normpath(path)
    if ".." in normalized.split(os.sep):
        return "Error: Path traversal (..) is not allowed"
    if not _is_path_allowed(normalized):
        return "Error: Path is outside allowed directories"
    try:
        os.makedirs(os.path.dirname(normalized), exist_ok=True)
        with open(normalized, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def _is_url_safe(url: str) -> tuple[bool, str]:
    """Validate a URL is safe to fetch (no SSRF to internal networks).

    Returns:
        (is_safe, error_message) tuple.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False, f"Unsupported scheme: {parsed.scheme!r}. Only http/https allowed."
    hostname = parsed.hostname
    if not hostname:
        return False, "No hostname in URL"
    try:
        resolved = socket.getaddrinfo(hostname, None)
        for family, _, _, _, sockaddr in resolved:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False, f"URL resolves to private/reserved IP: {ip}"
    except (socket.gaierror, ValueError) as e:
        return False, f"Cannot resolve hostname: {e}"
    return True, ""


async def http_get(url: str) -> str:
    """Fetch a URL and return its content.

    Only allows http/https URLs that resolve to public IP addresses.

    Args:
        url: The URL to fetch.

    Returns:
        The response body as text, or an error message.
    """
    safe, err = _is_url_safe(url)
    if not safe:
        return f"Error: {err}"
    try:
        import httpx
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            text = resp.text
            if len(text) > 100_000:
                return text[:100_000] + "\n... [truncated at 100,000 characters]"
            return text
    except Exception as e:
        return f"Error fetching URL: {e}"


def register_builtin_tools(registry: ToolRegistry) -> None:
    """Register all built-in tools into a ToolRegistry.

    Args:
        registry: The registry to add tools to.
    """
    registry.register(
        name="calculator",
        description="Evaluate a mathematical expression safely. Supports +, -, *, /, **, %, //.",
        parameters=_generate_schema(calculator),
        handler=calculator,
    )
    registry.register(
        name="current_time",
        description="Return the current date and time in ISO 8601 format (UTC).",
        parameters=_generate_schema(current_time),
        handler=current_time,
    )
    registry.register(
        name="read_file",
        description="Read and return the contents of a file given its absolute path.",
        parameters=_generate_schema(read_file),
        handler=read_file,
    )
    registry.register(
        name="write_file",
        description="Write content to a file at the given absolute path.",
        parameters=_generate_schema(write_file),
        handler=write_file,
    )
    registry.register(
        name="http_get",
        description="Fetch a URL via HTTP GET and return the response body.",
        parameters=_generate_schema(http_get),
        handler=http_get,
    )
