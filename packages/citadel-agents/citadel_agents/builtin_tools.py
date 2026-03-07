"""Built-in tools available to all agents.

Provides common utility tools: calculator, current_time, read_file,
write_file, and http_get.
"""

from __future__ import annotations

import ast
import os
from datetime import datetime, timezone
from typing import Any

from citadel_agents.tool import ToolRegistry, ToolSpec, _generate_schema


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
        # Parse the expression into an AST
        tree = ast.parse(expression, mode="eval")
        # Validate that it only contains safe operations
        for node in ast.walk(tree):
            if isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                                 ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
                                 ast.Mod, ast.FloorDiv, ast.USub, ast.UAdd)):
                continue
            else:
                return f"Error: Unsupported operation in expression: {type(node).__name__}"
        result = eval(compile(tree, "<calc>", "eval"))  # noqa: S307 — validated AST
        return str(result)
    except (SyntaxError, TypeError, ZeroDivisionError) as e:
        return f"Error: {e}"


def current_time() -> str:
    """Return the current date and time in ISO 8601 format (UTC).

    Returns:
        Current datetime string.
    """
    return datetime.now(timezone.utc).isoformat()


def read_file(path: str) -> str:
    """Read and return the contents of a file.

    Args:
        path: Absolute path to the file to read.

    Returns:
        The file contents as a string, or an error message.
    """
    # Basic path validation
    normalized = os.path.normpath(path)
    if ".." in normalized.split(os.sep):
        return "Error: Path traversal (..) is not allowed"
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
    try:
        os.makedirs(os.path.dirname(normalized), exist_ok=True)
        with open(normalized, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error writing file: {e}"


async def http_get(url: str) -> str:
    """Fetch a URL and return its content.

    Args:
        url: The URL to fetch.

    Returns:
        The response body as text, or an error message.
    """
    try:
        import httpx
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
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
