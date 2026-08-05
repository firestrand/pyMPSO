"""CLI presentation helpers for PSO output and result formatting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .utils.serialization import make_json_serializable

VALID_OUTPUT_FORMATS = {"json", "yaml"}


def normalize_output_format(output: Path, output_format: str | None = None) -> str:
    """Resolve the output format from explicit flag or file extension."""
    if output_format is not None:
        normalized = output_format.strip().lower()
        if normalized == "yml":
            normalized = "yaml"
        if normalized not in VALID_OUTPUT_FORMATS:
            raise ValueError("Unsupported output format. Use 'json' or 'yaml'.")
        return normalized

    extension = output.suffix.lower()
    if extension == ".json":
        return "json"
    if extension in {".yaml", ".yml"}:
        return "yaml"
    raise ValueError(
        f"Could not infer output format from file extension '{extension}'. Use --output-format to force json or yaml."
    )


def _format_count(value: Any) -> str:
    """Render an iteration or evaluation count, dropping a meaningless .0 suffix."""
    if value is None:
        return "n/a"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def build_summary_table(results: dict[str, Any]) -> Table:
    """Create a Rich table for benchmark summary output."""
    table = Table(title="PSO Benchmark Summary")
    table.add_column("Task", style="cyan")
    table.add_column("Best fitness", justify="right", style="green")
    table.add_column("Iterations", justify="right")
    table.add_column("Evaluations", justify="right")
    table.add_column("Status", style="magenta")

    task_rows = list(results.items())
    if "_benchmark_summary" in results and len(task_rows) > 1:
        for key, value in results.items():
            if key == "_benchmark_summary":
                continue
            if not isinstance(value, dict):
                continue
            summary = value.get("summary_statistics", {})
            best = value.get("best_fitness", summary.get("min_fitness", "n/a"))
            iterations = value.get("iterations", summary.get("mean_iterations"))
            evaluations = value.get("evaluations", summary.get("mean_evaluations"))
            status = "failed" if "error" in value else "ok"
            table.add_row(key, str(best), _format_count(iterations), _format_count(evaluations), status)
    else:
        status = "ok" if "error" not in results else "failed"
        table.add_row(
            "single run",
            str(results.get("best_fitness", "n/a")),
            _format_count(results.get("iterations")),
            _format_count(results.get("evaluations")),
            status,
        )

    return table


def build_dry_run_summary_table(config_data: dict[str, Any]) -> Table:
    """Create a compact plan summary for --dry-run output."""
    table = Table(title="Dry Run Plan")
    table.add_column("Task", style="cyan")
    table.add_column("Dimensions", justify="right")
    table.add_column("Runs", justify="right")

    problems = config_data.get("problems_to_benchmark", {})
    if not isinstance(problems, dict) or not problems:
        table.add_row("No tasks found", "N/A", "N/A")
        return table

    common_run_config = config_data.get("common_run_config", {})
    default_runs = common_run_config.get("num_runs", 1)
    for task_name, task_settings in problems.items():
        if not isinstance(task_settings, dict):
            table.add_row(str(task_name), "invalid", "invalid")
            continue
        dimensions = task_settings.get("dimensions", "n/a")
        task_runs = (
            task_settings.get("run_config", {}).get("num_runs", default_runs)
            if isinstance(task_settings.get("run_config"), dict)
            else default_runs
        )
        table.add_row(str(task_name), str(dimensions), str(task_runs))

    return table


def write_output_if_requested(
    console: Console,
    results: dict[str, Any],
    output: Path | None,
    output_format: str | None = None,
) -> None:
    """Write results to file using JSON or YAML."""
    if output is None:
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    fmt = normalize_output_format(output, output_format)

    payload = make_json_serializable(results)
    if fmt == "yaml":
        output.write_text(yaml.dump(payload, sort_keys=False), encoding="utf-8")
    else:
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    console.print(f"[green]Saved results to[/green] [bold]{output}[/bold]")


def build_complete_message() -> Panel:
    """Build the success message panel shown after completion."""
    return Panel.fit(
        Text(
            "Run complete. Use --output to persist results.",
            style="green",
        )
    )
