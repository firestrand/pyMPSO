"""Typer-based command-line interface for the PSO framework."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from . import __version__
from .cli_output import (
    build_complete_message,
    build_dry_run_summary_table,
    build_summary_table,
    write_output_if_requested,
)
from .cli_service import execute_cli_request

app = typer.Typer(help="Run PSO experiments from JSON/YAML configuration files.")
console = Console()


def _print_error(message: str) -> None:
    """Render error output consistently."""
    console.print(f"[bold red]Error:[/bold red] {message}")


def _execute_run(
    config: Path,
    output: Path | None = None,
    output_format: str | None = None,
    verbose: bool = False,
    dry_run: bool = False,
    no_color: bool = False,
    include_variants: str | None = None,
    exclude_variants: str | None = None,
) -> None:
    """Execute a benchmark or single-run PSO job from a config file."""
    if no_color:
        console.no_color = True

    selected_variants = (
        {name.strip() for name in include_variants.split(",") if name.strip()} if include_variants else None
    )
    excluded_variants = (
        {name.strip() for name in exclude_variants.split(",") if name.strip()} if exclude_variants else None
    )

    execution = execute_cli_request(
        config,
        output=output,
        output_format=output_format,
        verbose=verbose,
        dry_run=dry_run,
        selected_variants=selected_variants,
        excluded_variants=excluded_variants,
    )

    if not execution.success:
        _print_error(execution.message or "CLI execution failed.")
        raise typer.Exit(code=1) from None

    if dry_run and execution.config_data is not None:
        console.print("[yellow]Dry run completed:[/yellow] configuration parsed successfully.")
        console.print(build_dry_run_summary_table(execution.config_data))
        console.print("[yellow]Loaded keys:[/yellow] " + ", ".join(sorted(execution.config_data.keys())))
        return

    if execution.results is None:
        _print_error("No results returned from execution.")
        raise typer.Exit(code=1) from None

    try:
        write_output_if_requested(console, execution.results, output, output_format)
    except ValueError as exc:
        _print_error(str(exc))
        raise typer.Exit(code=1) from None

    console.print(build_summary_table(execution.results))
    console.print(build_complete_message())


@app.callback(invoke_without_command=True)
def main(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to a JSON or YAML configuration file.",
            readable=True,
            resolve_path=True,
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ],
    version: bool = typer.Option(  # noqa: B008
        False,
        "--version",
        "-V",
        help="Print version and exit.",
        is_eager=True,
    ),
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Optional path for writing results.",
        ),
    ] = None,
    output_format: Annotated[
        str | None,
        typer.Option(
            "--output-format",
            "-f",
            help="Output format: json or yaml. Defaults from output file extension.",
        ),
    ] = None,
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging during execution.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Parse and validate config without running optimization.",
    ),
    include_variants: str | None = typer.Option(
        None,
        "--include-variants",
        help="Comma-separated variant names to include from pso_variants.",
    ),
    exclude_variants: str | None = typer.Option(
        None,
        "--exclude-variants",
        help="Comma-separated variant names to exclude from pso_variants.",
    ),
    no_color: bool = typer.Option(
        False,
        "--no-color",
        help="Disable rich color output.",
    ),
) -> None:
    """Execute PSO experiments from a JSON/YAML config file."""
    if version:
        console.print(__version__)
        raise typer.Exit()

    _execute_run(
        config,
        output=output,
        output_format=output_format,
        verbose=verbose,
        dry_run=dry_run,
        no_color=no_color,
        include_variants=include_variants,
        exclude_variants=exclude_variants,
    )


if __name__ == "__main__":
    app()
