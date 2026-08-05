from __future__ import annotations

import logging
import time
from typing import Any

from ..planning.domain import BenchmarkExecutionPlan
from .protocols import ExecutionResult, RunExecutor, RunExecutorError


def with_execution_strategy(
    plan: BenchmarkExecutionPlan,
    parallel_enabled: bool,
    parallel_num_cpus: int | None,
) -> BenchmarkExecutionPlan:
    """Return a plan clone with resolved parallel execution controls."""
    return BenchmarkExecutionPlan(
        task_order=plan.task_order,
        task_plan=plan.task_plan,
        tasks=plan.tasks,
        parallel_enabled=parallel_enabled,
        parallel_num_cpus=parallel_num_cpus,
        base_output_dir=plan.base_output_dir,
        preexisting_results=plan.preexisting_results,
        config_file_name=plan.config_file_name,
    )


def execute_plan(
    plan: BenchmarkExecutionPlan,
    serial_executor: RunExecutor,
    parallel_executor: RunExecutor | None = None,
    *,
    logger: logging.Logger,
    verbose: bool = False,
) -> dict[str, Any]:
    """Execute a prepared plan and return benchmark results."""

    plan_start = time.time()

    if not plan.task_order and not plan.tasks:
        summary = {
            "_benchmark_summary": {
                "total_wall_time": 0.0,
                "config_file": plan.config_file_name,
            },
        }
        summary.update(plan.preexisting_results)
        return summary

    if parallel_executor is not None and plan.parallel_enabled and len(plan.tasks) > 1:
        try:
            if verbose:
                logger.info("Executing %s runs in parallel.", len(plan.tasks))
            all_run_results = parallel_executor.execute_many(plan.tasks)
        except RunExecutorError as exc:
            if verbose:
                logger.error("Execution failed: %s", exc)
            return _failure_results_for_parallel_error(plan, str(exc))
    else:
        if verbose and len(plan.tasks) > 1:
            logger.info("Executing %s runs serially.", len(plan.tasks))
        all_run_results = serial_executor.execute_many(plan.tasks)

    grouped_results: dict[str, list[ExecutionResult]] = {}
    for run_result in all_run_results:
        task_name = run_result.get("benchmark_task")
        if task_name is None:
            if verbose:
                logger.warning("Run result missing task name: %s", run_result)
            continue
        grouped_results.setdefault(task_name, []).append(run_result)

    benchmark_suite_results: dict[str, Any] = dict(plan.preexisting_results)
    for task_name in plan.task_order:
        task_entry = plan.task_plan[task_name]
        task_run_results = grouped_results.get(task_name, [])
        task_run_count = task_entry["num_runs_requested"]
        task_problem_type = task_entry["problem"].get("type")
        task_problem_dimension = task_entry["dimensions"]
        task_num_particles = task_entry["swarm"].get("num_particles", 30)

        if not task_run_results:
            benchmark_suite_results[task_name] = {
                "error": "No run results produced",
                "config_file": task_entry["config_file"],
                "benchmark_task": task_name,
                "problem_type": task_problem_type,
                "problem_dimension": task_problem_dimension,
                "num_particles": task_num_particles,
            }
            continue

        if task_run_count == 1:
            task_payload = task_run_results[0]
            task_payload.pop("run_index", None)
            benchmark_suite_results[task_name] = task_payload
            if verbose and "best_fitness" in task_payload:
                logger.info("  Finished Task: %s", task_name)
                logger.info("    Best Fitness: %s", task_payload.get("best_fitness", float("nan")))
            if verbose and "error" in task_payload:
                logger.error("    ERROR: %s", task_payload["error"])
            continue

        task_total_time = 0.0
        for result in task_run_results:
            task_total_time += float(result.get("total_wall_time", 0.0))
        summary_stats = _calculate_summary_stats(task_run_results, task_total_time)
        benchmark_suite_results[task_name] = {
            "summary_statistics": summary_stats,
            "individual_runs": task_run_results,
            "config_file": task_entry["config_file"],
            "benchmark_task": task_name,
            "problem_type": task_problem_type,
            "problem_dimension": task_problem_dimension,
            "num_particles": task_num_particles,
            "num_runs_requested": task_run_count,
        }
        if verbose:
            logger.info("  Finished Task: %s", task_name)
            logger.info(
                "    Mean Fitness: %s",
                summary_stats.get("mean_fitness", float("nan")),
            )
            logger.info(
                "    Num Runs: %s/%s",
                summary_stats.get("num_runs_completed", "N/A"),
                task_run_count,
            )

    for task_name in plan.task_order:
        if task_name not in benchmark_suite_results:
            benchmark_suite_results[task_name] = {
                "error": "Task missing from execution output",
                "config_file": plan.config_file_name,
                "benchmark_task": task_name,
            }

    total_benchmark_time = time.time() - plan_start if plan.tasks else 0.0
    benchmark_suite_results["_benchmark_summary"] = {
        "total_wall_time": total_benchmark_time,
        "config_file": plan.config_file_name,
    }

    if verbose:
        logger.info("%s", "-" * 30)
        logger.info("Benchmark Suite Completed in %.2f seconds.", total_benchmark_time)
        logger.info("%s", "-" * 30)

    return benchmark_suite_results


def _calculate_summary_stats(all_run_results: list[ExecutionResult], total_time_multi: float) -> dict[str, Any]:
    import numpy as np

    best_fitnesses = [
        r["best_fitness"]
        for r in all_run_results
        if "best_fitness" in r and r["best_fitness"] is not None and np.isfinite(r["best_fitness"])
    ]
    if not best_fitnesses:
        summary_stats: dict[str, Any] = {}
    else:
        summary_stats = {
            "mean_fitness": float(np.mean(best_fitnesses)),
            "median_fitness": float(np.median(best_fitnesses)),
            "std_dev_fitness": float(np.std(best_fitnesses)),
            "min_fitness": float(np.min(best_fitnesses)),
            "max_fitness": float(np.max(best_fitnesses)),
        }

    # Runs that errored carry no counters; averaging over the runs that reported
    # them keeps a failed run from dragging the mean toward zero.
    for source_key, stat_key in (("iterations", "mean_iterations"), ("evaluations", "mean_evaluations")):
        counts = [r[source_key] for r in all_run_results if r.get(source_key) is not None]
        if counts:
            summary_stats[stat_key] = float(np.mean(counts))

    summary_stats["num_runs_completed"] = len(all_run_results)
    summary_stats["total_multi_run_wall_time"] = total_time_multi
    return summary_stats


def _failure_results_for_parallel_error(
    plan: BenchmarkExecutionPlan,
    error_message: str,
) -> dict[str, Any]:
    results: dict[str, Any] = dict(plan.preexisting_results)
    for task_name, task_entry in plan.task_plan.items():
        results[task_name] = {
            "error": error_message,
            "config_file": task_entry["config_file"],
            "benchmark_task": task_name,
            "problem_type": task_entry["problem"]["type"],
            "problem_dimension": task_entry["dimensions"],
            "num_particles": task_entry["swarm"].get("num_particles", 30),
            "num_runs_requested": task_entry["num_runs_requested"],
        }

    results["_benchmark_summary"] = {
        "total_wall_time": 0.0,
        "config_file": plan.config_file_name,
    }
    return results
