from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def benchmark_task_key(problem_name: str, dimensions: int, variant_name: str | None = None) -> str:
    """Build deterministic task identifiers with optional variant suffixes."""
    variant_suffix = f"[{variant_name}]" if variant_name else ""
    return f"{problem_name}_{dimensions}D{variant_suffix}"


@dataclass(frozen=True)
class BenchmarkTask:
    """Execution-time task configuration after merge and validation."""

    config_file: str
    task_name: str
    problem_name: str
    variant_name: str | None
    dimensions: int
    problem: dict[str, Any]
    swarm: dict[str, Any]
    pso_params: dict[str, Any]
    stopping_criteria: dict[str, Any]
    run_config: dict[str, Any]
    task_run_index: int
    random_seed: int | None


@dataclass(frozen=True)
class BenchmarkExecutionPlan:
    """Computed plan for a benchmark session."""

    task_order: list[str]
    task_plan: dict[str, dict[str, Any]]
    tasks: list[BenchmarkTask]
    parallel_enabled: bool
    parallel_num_cpus: int | None
    base_output_dir: str | None
    preexisting_results: dict[str, Any]
    config_file_name: str
