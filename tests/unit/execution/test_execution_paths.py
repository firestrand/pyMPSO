import logging
from dataclasses import replace

import pytest

from src.execution.orchestrator import execute_plan, with_execution_strategy
from src.execution.parallel import RayExecutor
from src.execution.protocols import RunExecutorError
from src.execution.serial import SerialExecutor
from src.planning.planner import build_benchmark_plan
from src.run_pso import _build_task_plan
from src.variants.domain import VariantDescriptor


@pytest.fixture
def plan():
    return build_benchmark_plan(
        {
            "problems_to_benchmark": {
                "sphere": {
                    "dimensions": 2,
                    "bounds": {"min": -2, "max": 2},
                    "stopping_criteria": {"type": "max_iterations", "max_iterations": 2},
                }
            }
        },
        "test.yaml",
        {},
        {},
        {"num_runs": 2},
        17,
        _build_task_plan,
    )


def test_serial_plan_preserves_results_and_aggregates_runs(plan, caplog):
    plan = replace(plan, preexisting_results={"bad": {"error": "invalid bounds"}})
    executor = SerialExecutor(
        lambda task: {
            "benchmark_task": task.task_name,
            "run_index": task.task_run_index,
            "best_fitness": float(task.task_run_index + 1),
            "total_wall_time": 0.5,
        }
    )
    with caplog.at_level(logging.INFO):
        results = execute_plan(plan, executor, logger=logging.getLogger(__name__), verbose=True)
    summary = results["sphere_2D"]["summary_statistics"]
    assert summary["mean_fitness"] == 1.5
    assert summary["total_multi_run_wall_time"] == 1.0
    assert results["bad"]["error"] == "invalid bounds"
    assert "serially" in caplog.text
    assert "Benchmark Suite Completed" in caplog.text


def test_parallel_plan_failure_keeps_task_metadata(plan, caplog):
    class FailingExecutor:
        def execute(self, task):
            del task
            raise RunExecutorError("worker unavailable")

        def execute_many(self, tasks):
            del tasks
            raise RunExecutorError("worker unavailable")

    parallel_plan = with_execution_strategy(plan, True, 2)
    assert plan.parallel_enabled is False
    with caplog.at_level(logging.INFO):
        results = execute_plan(
            parallel_plan,
            SerialExecutor(lambda _task: {}),
            FailingExecutor(),
            logger=logging.getLogger(__name__),
            verbose=True,
        )
    assert results["sphere_2D"]["error"] == "worker unavailable"
    assert results["sphere_2D"]["problem_dimension"] == 2
    assert results["sphere_2D"]["num_runs_requested"] == 2
    assert results["_benchmark_summary"]["total_wall_time"] == 0
    assert "Execution failed" in caplog.text


@pytest.mark.parametrize("verbose", [False, True])
def test_missing_task_results_and_empty_plan(plan, verbose, caplog):
    with caplog.at_level(logging.INFO):
        result = execute_plan(
            plan, SerialExecutor(lambda _task: {}), logger=logging.getLogger(__name__), verbose=verbose
        )
    assert result["sphere_2D"]["error"] == "No run results produced"
    assert ("missing task name" in caplog.text) is verbose
    empty = replace(plan, task_order=[], task_plan={}, tasks=[], preexisting_results={"invalid": {"error": "bad"}})
    result = execute_plan(empty, SerialExecutor(lambda _task: {}), logger=logging.getLogger(__name__))
    assert result == {
        "_benchmark_summary": {"total_wall_time": 0, "config_file": "test.yaml"},
        "invalid": {"error": "bad"},
    }


@pytest.mark.parametrize("payload", [{"best_fitness": 2.0}, {"error": "objective failed"}])
def test_single_task_result_removes_run_index(plan, payload):
    entries = {name: {**entry, "num_runs_requested": 1} for name, entry in plan.task_plan.items()}
    single = replace(plan, tasks=plan.tasks[:1], task_plan=entries)
    executor = SerialExecutor(lambda task: {"benchmark_task": task.task_name, "run_index": 0, **payload})
    result = execute_plan(single, executor, logger=logging.getLogger(__name__), verbose=True)
    assert "run_index" not in result["sphere_2D"]
    for key, value in payload.items():
        assert result["sphere_2D"][key] == value


def test_planner_keeps_valid_tasks_when_another_problem_is_invalid():
    config = {
        "problems_to_benchmark": {
            "missing": {},
            "sphere": {
                "dimensions": 2,
                "bounds": {"min": -1, "max": 1},
                "stopping_criteria": {"type": "max_iterations", "max_iterations": 1},
            },
        }
    }
    variants = [VariantDescriptor.from_raw("standard"), VariantDescriptor.from_raw("other")]
    plan = build_benchmark_plan(config, "test.yaml", {}, {}, {}, None, _build_task_plan, pso_variants=variants)
    assert plan.task_order == ["sphere_2D[standard]", "sphere_2D[other]"]
    assert set(plan.preexisting_results) == {"missing::standard", "missing::other"}
    assert len(plan.tasks) == 2
    with pytest.raises(ValueError, match="problems_to_benchmark"):
        build_benchmark_plan({}, "test.yaml", {}, {}, {}, None, _build_task_plan)


def test_ray_single_task_executes_directly_without_starting_cluster(plan):
    executor = RayExecutor(lambda task: {"benchmark_task": task.task_name, "best_fitness": 3.0})
    assert executor.execute(plan.tasks[0]) == {"benchmark_task": "sphere_2D", "best_fitness": 3.0}
