from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from ..variants.domain import DEFAULT_VARIANT_NAME, VariantDescriptor
from .domain import BenchmarkExecutionPlan, BenchmarkTask

TaskPlanner = Callable[
    [
        str,
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        int | None,
        str,
        VariantDescriptor,
    ],
    tuple[str, dict[str, Any], list[BenchmarkTask]],
]


def build_benchmark_plan(
    config: dict[str, Any],
    config_path: str,
    common_swarm: dict[str, Any],
    common_pso_params: dict[str, Any],
    common_run_config: dict[str, Any],
    benchmark_seed: int | None,
    build_task_plan: TaskPlanner,
    *,
    pso_variants: Iterable[VariantDescriptor] | None = None,
) -> BenchmarkExecutionPlan:
    """Build deterministic execution plan without running optimization."""
    problems_to_run = config.get("problems_to_benchmark")
    if not problems_to_run or not isinstance(problems_to_run, dict):
        raise ValueError("Config must contain a 'problems_to_benchmark' dictionary.")

    base_output_dir = common_run_config.get("output_dir")
    task_order: list[str] = []
    task_plan: dict[str, dict[str, Any]] = {}
    tasks: list[BenchmarkTask] = []
    preexisting_results: dict[str, Any] = {}
    config_file_name = config_path

    variants = tuple(pso_variants) if pso_variants is not None else ()

    if not variants:
        variants = (VariantDescriptor.from_raw(DEFAULT_VARIANT_NAME, {}),)

    for problem_name, problem_settings in problems_to_run.items():
        for variant in variants:
            try:
                task_name, task_entry, specs = build_task_plan(
                    problem_name,
                    problem_settings,
                    common_swarm,
                    common_pso_params,
                    common_run_config,
                    benchmark_seed,
                    config_file_name,
                    variant,
                )
            except (TypeError, ValueError) as exc:
                key = f"{problem_name}::{variant.name}"
                preexisting_results[key] = {"error": str(exc), "variant": variant.name}
                continue

            task_order.append(task_name)
            task_plan[task_name] = task_entry
            tasks.extend(specs)

    return BenchmarkExecutionPlan(
        task_order=task_order,
        task_plan=task_plan,
        tasks=tasks,
        parallel_enabled=False,
        parallel_num_cpus=None,
        base_output_dir=base_output_dir,
        preexisting_results=preexisting_results,
        config_file_name=config_file_name,
    )
