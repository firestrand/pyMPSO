"""Tests for per-task aggregation of multi-run benchmark results."""

from __future__ import annotations

from src.execution.orchestrator import _calculate_summary_stats
from src.execution.protocols import ExecutionResult


def _run(best_fitness: float, iterations: int, evaluations: int) -> ExecutionResult:
    return {
        "best_fitness": best_fitness,
        "iterations": iterations,
        "evaluations": evaluations,
        "total_wall_time": 0.5,
    }


def test_summary_aggregates_iterations_and_evaluations():
    runs = [_run(1.0, 500, 20000), _run(2.0, 500, 20000), _run(3.0, 500, 20000)]

    stats = _calculate_summary_stats(runs, total_time_multi=1.5)

    assert stats["mean_iterations"] == 500.0
    assert stats["mean_evaluations"] == 20000.0
    assert stats["num_runs_completed"] == 3


def test_summary_averages_runs_that_stop_at_different_iterations():
    runs = [_run(1.0, 400, 16000), _run(2.0, 500, 20000)]

    stats = _calculate_summary_stats(runs, total_time_multi=1.0)

    assert stats["mean_iterations"] == 450.0
    assert stats["mean_evaluations"] == 18000.0


def test_summary_omits_iteration_stats_when_runs_report_none():
    runs: list[ExecutionResult] = [{"best_fitness": 1.0, "total_wall_time": 0.1}]

    stats = _calculate_summary_stats(runs, total_time_multi=0.1)

    assert "mean_iterations" not in stats
    assert "mean_evaluations" not in stats
    assert stats["num_runs_completed"] == 1


def test_summary_ignores_runs_missing_counts_when_averaging():
    """A failed run without counts must not drag the mean toward zero."""
    failed: ExecutionResult = {"error": "boom"}
    runs: list[ExecutionResult] = [_run(1.0, 500, 20000), failed]

    stats = _calculate_summary_stats(runs, total_time_multi=0.2)

    assert stats["mean_iterations"] == 500.0
    assert stats["mean_evaluations"] == 20000.0
    assert stats["num_runs_completed"] == 2
