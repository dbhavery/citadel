"""
Citadel CLI -- unified command-line interface for the Citadel AI Operations Platform.

Each command lazy-imports its package so the CLI works even if not all
packages are installed.
"""
from __future__ import annotations

import sys
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from citadel_cli import __version__

console = Console()


def _check_import(package: str, pip_name: str) -> object:
    """Lazy-import a package and give a helpful error if missing."""
    try:
        return __import__(package)
    except ImportError:
        console.print(
            f"[red]Error:[/red] Package [bold]{pip_name}[/bold] is not installed.\n"
            f"  Install it with: [cyan]pip install {pip_name}[/cyan]"
        )
        sys.exit(1)


# ============================================================
# CLI Group
# ============================================================

@click.group()
@click.version_option(__version__, prog_name="citadel")
def cli() -> None:
    """Citadel -- AI Operations Platform.

    Unified gateway, vector engine, agents, ingestion, tracing, and
    dashboard for LLM-powered applications.
    """


# ============================================================
# serve
# ============================================================

@cli.command()
@click.option("--port", default=8080, type=int, help="Gateway listen port.")
@click.option("--host", default="0.0.0.0", help="Bind address.")
@click.option("--trace-port", default=8081, type=int, help="Trace server port.")
@click.option("--dashboard-port", default=3000, type=int, help="Dashboard port.")
@click.option("--no-dashboard", is_flag=True, help="Skip starting the dashboard.")
def serve(
    port: int,
    host: str,
    trace_port: int,
    dashboard_port: int,
    no_dashboard: bool,
) -> None:
    """Start all Citadel services."""
    console.print(f"[bold cyan]Citadel[/bold cyan] v{__version__}")
    console.print(f"  Gateway     : {host}:{port}")
    console.print(f"  Trace Server: {host}:{trace_port}")
    if not no_dashboard:
        console.print(f"  Dashboard   : http://localhost:{dashboard_port}")
    console.print()

    gateway = _check_import("citadel_gateway", "citadel-gateway")

    try:
        trace_mod = _check_import("citadel_trace", "citadel-trace")
    except SystemExit:
        console.print("[yellow]Warning:[/yellow] citadel-trace not installed, tracing disabled.")
        trace_mod = None

    # Start gateway (blocks)
    try:
        if hasattr(gateway, "serve"):
            gateway.serve(host=host, port=port)
        elif hasattr(gateway, "create_app"):
            import uvicorn  # type: ignore[import-untyped]
            app = gateway.create_app()
            uvicorn.run(app, host=host, port=port)
        else:
            console.print("[red]Error:[/red] citadel_gateway has no serve() or create_app().")
            sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down.[/yellow]")


# ============================================================
# ingest
# ============================================================

@cli.command()
@click.argument("path")
@click.option(
    "--chunk-strategy",
    default="fixed",
    type=click.Choice(["fixed", "sentence", "paragraph", "semantic"]),
    help="Chunking strategy.",
)
@click.option("--chunk-size", default=500, type=int, help="Target chunk size in tokens.")
@click.option("--overlap", default=50, type=int, help="Chunk overlap in tokens.")
@click.option("--collection", default="default", help="Vector collection name.")
def ingest(
    path: str,
    chunk_strategy: str,
    chunk_size: int,
    overlap: int,
    collection: str,
) -> None:
    """Ingest documents into the vector store."""
    ingest_mod = _check_import("citadel_ingest", "citadel-ingest")
    vector_mod = _check_import("citadel_vector", "citadel-vector")

    import pathlib
    p = pathlib.Path(path)
    if not p.exists():
        console.print(f"[red]Error:[/red] Path does not exist: {path}")
        sys.exit(1)

    console.print(f"[bold]Ingesting[/bold] {path}")
    console.print(f"  Strategy   : {chunk_strategy}")
    console.print(f"  Chunk size : {chunk_size}")
    console.print(f"  Overlap    : {overlap}")
    console.print(f"  Collection : {collection}")
    console.print()

    try:
        if hasattr(ingest_mod, "ingest"):
            result = ingest_mod.ingest(
                path=path,
                chunk_strategy=chunk_strategy,
                chunk_size=chunk_size,
                overlap=overlap,
                collection=collection,
            )
        elif hasattr(ingest_mod, "Ingestor"):
            ing = ingest_mod.Ingestor(
                chunk_strategy=chunk_strategy,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            result = ing.ingest(path, collection=collection)
        else:
            console.print("[red]Error:[/red] citadel_ingest has no ingest() or Ingestor class.")
            sys.exit(1)

        console.print(f"[green]Done.[/green] Ingested {getattr(result, 'chunk_count', '?')} chunks.")
    except Exception as exc:
        console.print(f"[red]Error during ingestion:[/red] {exc}")
        sys.exit(1)


# ============================================================
# search
# ============================================================

@cli.command()
@click.argument("query")
@click.option("--k", default=5, type=int, help="Number of results to return.")
@click.option("--collection", default="default", help="Vector collection name.")
def search(query: str, k: int, collection: str) -> None:
    """Search ingested documents."""
    vector_mod = _check_import("citadel_vector", "citadel-vector")

    console.print(f'[bold]Searching[/bold] "{query}" (k={k}, collection={collection})\n')

    try:
        if hasattr(vector_mod, "search"):
            results = vector_mod.search(query, k=k, collection=collection)
        elif hasattr(vector_mod, "VectorStore"):
            store = vector_mod.VectorStore(collection=collection)
            results = store.search(query, k=k)
        else:
            console.print("[red]Error:[/red] citadel_vector has no search() or VectorStore class.")
            sys.exit(1)

        if not results:
            console.print("[yellow]No results found.[/yellow]")
            return

        table = Table(title=f"Top {k} Results")
        table.add_column("#", style="dim", width=4)
        table.add_column("Score", width=8)
        table.add_column("Source", width=30)
        table.add_column("Text", max_width=60)

        for i, r in enumerate(results, 1):
            score = getattr(r, "score", r.get("score", "?")) if isinstance(r, dict) else getattr(r, "score", "?")
            source = getattr(r, "source", r.get("source", "--")) if isinstance(r, dict) else getattr(r, "source", "--")
            text = getattr(r, "text", r.get("text", "")) if isinstance(r, dict) else getattr(r, "text", "")
            table.add_row(str(i), f"{score:.4f}" if isinstance(score, float) else str(score), str(source), text[:120])

        console.print(table)
    except Exception as exc:
        console.print(f"[red]Search error:[/red] {exc}")
        sys.exit(1)


# ============================================================
# agent
# ============================================================

@cli.command()
@click.argument("agent_path")
@click.option("--input", "-i", "user_input", prompt="Input", help="Input to send to the agent.")
@click.option("--verbose", "-v", is_flag=True, help="Show agent reasoning steps.")
def agent(agent_path: str, user_input: str, verbose: bool) -> None:
    """Run an agent from a YAML definition."""
    agents_mod = _check_import("citadel_agents", "citadel-agents")

    import pathlib
    p = pathlib.Path(agent_path)
    if not p.exists():
        console.print(f"[red]Error:[/red] Agent file not found: {agent_path}")
        sys.exit(1)

    console.print(f"[bold]Running agent[/bold] {agent_path}\n")

    try:
        if hasattr(agents_mod, "run_agent"):
            result = agents_mod.run_agent(agent_path, user_input, verbose=verbose)
        elif hasattr(agents_mod, "Agent"):
            ag = agents_mod.Agent.from_file(agent_path)
            result = ag.run(user_input, verbose=verbose)
        else:
            console.print("[red]Error:[/red] citadel_agents has no run_agent() or Agent class.")
            sys.exit(1)

        output = getattr(result, "output", result) if not isinstance(result, str) else result
        console.print(f"\n[bold green]Result:[/bold green]\n{output}")
    except Exception as exc:
        console.print(f"[red]Agent error:[/red] {exc}")
        sys.exit(1)


# ============================================================
# traces
# ============================================================

@cli.command()
@click.option("--limit", "-n", default=20, type=int, help="Number of traces to show.")
@click.option("--status", type=click.Choice(["ok", "error"]), default=None, help="Filter by status.")
@click.option("--model", default=None, help="Filter by model name.")
def traces(limit: int, status: Optional[str], model: Optional[str]) -> None:
    """View recent traces."""
    trace_mod = _check_import("citadel_trace", "citadel-trace")

    try:
        if hasattr(trace_mod, "get_traces"):
            data = trace_mod.get_traces(limit=limit, status=status, model=model)
        elif hasattr(trace_mod, "TraceStore"):
            store = trace_mod.TraceStore()
            data = store.list(limit=limit, status=status, model=model)
        else:
            console.print("[red]Error:[/red] citadel_trace has no get_traces() or TraceStore class.")
            sys.exit(1)

        if not data:
            console.print("[yellow]No traces found.[/yellow]")
            return

        table = Table(title=f"Recent Traces (limit={limit})")
        table.add_column("ID", style="cyan", width=18)
        table.add_column("Time", width=20)
        table.add_column("Model", width=28)
        table.add_column("Tokens", justify="right", width=8)
        table.add_column("Cost", justify="right", width=10)
        table.add_column("Latency", justify="right", width=10)
        table.add_column("Status", width=8)

        for t in data:
            tid = getattr(t, "id", t.get("id", "")) if isinstance(t, dict) else getattr(t, "id", "")
            ts = getattr(t, "timestamp", t.get("timestamp", "")) if isinstance(t, dict) else getattr(t, "timestamp", "")
            mdl = getattr(t, "model", t.get("model", "")) if isinstance(t, dict) else getattr(t, "model", "")
            tok = getattr(t, "total_tokens", t.get("total_tokens", 0)) if isinstance(t, dict) else getattr(t, "total_tokens", 0)
            cst = getattr(t, "cost", t.get("cost", 0)) if isinstance(t, dict) else getattr(t, "cost", 0)
            lat = getattr(t, "latency_ms", t.get("latency_ms", 0)) if isinstance(t, dict) else getattr(t, "latency_ms", 0)
            st = getattr(t, "status", t.get("status", "ok")) if isinstance(t, dict) else getattr(t, "status", "ok")

            st_style = "green" if st == "ok" else "red"
            table.add_row(
                str(tid)[:18],
                str(ts)[:20],
                str(mdl),
                str(tok),
                f"${float(cst):.4f}",
                f"{int(lat)}ms",
                f"[{st_style}]{st}[/{st_style}]",
            )

        console.print(table)
    except Exception as exc:
        console.print(f"[red]Error reading traces:[/red] {exc}")
        sys.exit(1)


# ============================================================
# cost
# ============================================================

@cli.command()
@click.option("--days", "-d", default=7, type=int, help="Number of days to show.")
def cost(days: int) -> None:
    """Show cost summary."""
    trace_mod = _check_import("citadel_trace", "citadel-trace")

    try:
        if hasattr(trace_mod, "get_cost_summary"):
            data = trace_mod.get_cost_summary(days=days)
        elif hasattr(trace_mod, "TraceStore"):
            store = trace_mod.TraceStore()
            data = store.cost_summary(days=days)
        else:
            console.print("[red]Error:[/red] citadel_trace has no cost reporting.")
            sys.exit(1)

        if isinstance(data, dict):
            total = data.get("total", 0)
            by_model = data.get("by_model", {})
            by_day = data.get("by_day", [])
        else:
            total = sum(getattr(d, "cost", 0) for d in data) if data else 0
            by_model = {}
            by_day = data or []

        console.print(f"\n[bold]Cost Summary[/bold] (last {days} days)")
        console.print(f"  Total: [bold]${float(total):.2f}[/bold]\n")

        if by_model:
            table = Table(title="Cost by Model")
            table.add_column("Model", width=30)
            table.add_column("Cost", justify="right", width=12)
            table.add_column("Requests", justify="right", width=10)
            for mdl, info in by_model.items():
                c = info.get("cost", 0) if isinstance(info, dict) else info
                r = info.get("requests", "--") if isinstance(info, dict) else "--"
                table.add_row(mdl, f"${float(c):.2f}", str(r))
            console.print(table)

        if by_day:
            console.print("\n[bold]Daily Breakdown:[/bold]")
            for d in by_day:
                date = d.get("date", "?") if isinstance(d, dict) else getattr(d, "date", "?")
                c = d.get("cost", 0) if isinstance(d, dict) else getattr(d, "cost", 0)
                bar = "#" * int(float(c) * 5)
                console.print(f"  {date}  ${float(c):>7.2f}  {bar}")

    except Exception as exc:
        console.print(f"[red]Error reading cost data:[/red] {exc}")
        sys.exit(1)


# ============================================================
# status
# ============================================================

@cli.command()
@click.option("--url", default="http://localhost:8080", help="Gateway URL.")
def status(url: str) -> None:
    """Check Citadel service status."""
    import urllib.request
    import json

    console.print(f"[bold]Citadel Status[/bold] (checking {url})\n")

    services = [
        ("Gateway",      f"{url}/health"),
        ("Trace Server", f"{url.rsplit(':', 1)[0]}:8081/health"),
    ]

    for name, endpoint in services:
        try:
            req = urllib.request.Request(endpoint, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read())
                version = body.get("version", "?")
                console.print(f"  [green]OK[/green]    {name:20s}  {endpoint}  (v{version})")
        except urllib.error.URLError:
            console.print(f"  [red]DOWN[/red]  {name:20s}  {endpoint}")
        except Exception as exc:
            console.print(f"  [red]ERR[/red]   {name:20s}  {endpoint}  ({exc})")

    # Check if dashboard is reachable
    try:
        req = urllib.request.Request("http://localhost:3000", method="HEAD")
        with urllib.request.urlopen(req, timeout=3):
            console.print(f"  [green]OK[/green]    {'Dashboard':20s}  http://localhost:3000")
    except Exception:
        console.print(f"  [yellow]--[/yellow]    {'Dashboard':20s}  http://localhost:3000  (not running)")

    console.print()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    cli()
