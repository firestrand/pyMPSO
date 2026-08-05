from __future__ import annotations

import logging
from typing import Any

from .execution.orchestrator import (
    execute_plan as _execute_plan_impl,
)
from .execution.orchestrator import (
    with_execution_strategy as _with_execution_strategy,
)
from .execution.protocols import RunExecutor
from .planning.domain import BenchmarkExecutionPlan
from .planning.planner import TaskPlanner
from .planning.planner import build_benchmark_plan as _build_benchmark_plan
from .variants.domain import VariantDescriptor


def build_benchmark_plan(
    config: dict[str, Any],
    config_path: str,
    common_swarm: dict[str, Any],
    common_pso_params: dict[str, Any],
    common_run_config: dict[str, Any],
    benchmark_seed: int | None,
    build_task_plan: TaskPlanner,
    *,
    pso_variants: tuple[VariantDescriptor, ...] | None = None,
) -> BenchmarkExecutionPlan:
    """Build benchmark plan."""
    return _build_benchmark_plan(
        config=config,
        config_path=config_path,
        common_swarm=common_swarm,
        common_pso_params=common_pso_params,
        common_run_config=common_run_config,
        benchmark_seed=benchmark_seed,
        build_task_plan=build_task_plan,
        pso_variants=pso_variants,
    )


def with_execution_strategy(
    plan: BenchmarkExecutionPlan,
    parallel_enabled: bool,
    parallel_num_cpus: int | None,
) -> BenchmarkExecutionPlan:
    """Attach parallel execution strategy controls to an already-built plan."""
    return _with_execution_strategy(plan, parallel_enabled, parallel_num_cpus)


def execute_plan(
    plan: BenchmarkExecutionPlan,
    serial_executor: RunExecutor,
    parallel_executor: RunExecutor | None = None,
    *,
    logger: logging.Logger,
    verbose: bool = False,
) -> dict[str, Any]:
    """Execute a benchmark plan with protocol-based executors."""

    return _execute_plan_impl(
        plan=plan,
        serial_executor=serial_executor,
        parallel_executor=parallel_executor,
        logger=logger,
        verbose=verbose,
    )
