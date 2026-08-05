"""Service-layer orchestration helpers for CLI execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from . import run_pso as run_pso_module
from .cli_output import normalize_output_format

CliStatus = Literal[
    "ok",
    "load_error",
    "validation_error",
    "format_error",
    "execution_error",
    "unexpected_result",
]


@dataclass(frozen=True)
class CliExecutionResult:
    """Result envelope returned from CLI service execution."""

    status: CliStatus
    config_data: dict[str, Any] | None = None
    results: dict[str, Any] | None = None
    message: str | None = None

    @property
    def success(self) -> bool:
        """Whether the request completed successfully."""
        return self.status == "ok"


def execute_cli_request(
    config: Path,
    output: Path | None = None,
    output_format: str | None = None,
    verbose: bool = False,
    dry_run: bool = False,
    selected_variants: set[str] | None = None,
    excluded_variants: set[str] | None = None,
) -> CliExecutionResult:
    """Load, validate, and optionally execute a PSO config for CLI users."""
    config_path = str(config)
    try:
        config_data = run_pso_module.load_config(config_path)
    except Exception as exc:
        return CliExecutionResult(status="load_error", message=f"Failed to load configuration: {exc}")

    if dry_run:
        try:
            run_pso_module._validate_configuration(
                config_data,
                config_path,
                selected_variants=selected_variants,
                excluded_variants=excluded_variants,
                validate_vectorized=True,
            )
        except Exception as exc:
            return CliExecutionResult(
                status="validation_error",
                config_data=config_data,
                message=f"Invalid configuration: {exc}",
            )
        return CliExecutionResult(status="ok", config_data=config_data)

    try:
        if output_format is not None:
            normalize_output_format(output or Path("results.json"), output_format)
    except ValueError as exc:
        return CliExecutionResult(
            status="format_error",
            config_data=config_data,
            message=str(exc),
        )

    results = run_pso_module.run_pso_from_config(
        config_path,
        None,
        verbose,
        config_data=config_data,
        selected_variants=selected_variants,
        excluded_variants=excluded_variants,
    )
    if not isinstance(results, dict):
        return CliExecutionResult(
            status="unexpected_result",
            config_data=config_data,
            message="Unexpected result format returned by run_pso_from_config",
        )

    if "error" in results:
        return CliExecutionResult(
            status="execution_error",
            config_data=config_data,
            results=results,
            message=str(results["error"]),
        )

    return CliExecutionResult(status="ok", config_data=config_data, results=results)
