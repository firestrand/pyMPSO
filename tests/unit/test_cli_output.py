"""Tests for CLI summary rendering."""

from __future__ import annotations

from typing import Any

from src.cli_output import build_summary_table


def _cell_text(table, row_index: int, column_index: int) -> str:
    return str(table.columns[column_index]._cells[row_index])


def _multi_task_results(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "_benchmark_summary": {"total_wall_time": 1.0, "config_file": "cfg.yaml"},
        "sphere_10D": {"summary_statistics": summary, "individual_runs": []},
        "rastrigin_10D": {"summary_statistics": summary, "individual_runs": []},
    }


def test_multi_run_summary_reports_iterations_not_run_count():
    """The Iterations column must never fall back to the number of runs."""
    summary = {
        "min_fitness": 1.5,
        "num_runs_completed": 3,
        "mean_iterations": 500.0,
        "mean_evaluations": 20040.0,
    }
    table = build_summary_table(_multi_task_results(summary))

    assert _cell_text(table, 0, 2) == "500"
    assert _cell_text(table, 0, 3) == "20040"


def test_multi_run_summary_without_iteration_stats_reports_not_available():
    """Missing iteration data renders as n/a rather than an unrelated count."""
    summary = {"min_fitness": 1.5, "num_runs_completed": 3}
    table = build_summary_table(_multi_task_results(summary))

    assert _cell_text(table, 0, 2) == "n/a"
    assert _cell_text(table, 0, 3) == "n/a"


def test_fractional_means_are_preserved():
    """Runs that stop at different iterations keep their fractional mean."""
    summary = {
        "min_fitness": 1.5,
        "num_runs_completed": 3,
        "mean_iterations": 412.5,
        "mean_evaluations": 16500.5,
    }
    table = build_summary_table(_multi_task_results(summary))

    assert _cell_text(table, 0, 2) == "412.5"
    assert _cell_text(table, 0, 3) == "16500.5"


def test_single_run_table_reports_its_own_counts():
    """A single-task result renders its exact iteration and evaluation counts."""
    table = build_summary_table({"best_fitness": 0.25, "iterations": 120, "evaluations": 4800})

    assert _cell_text(table, 0, 2) == "120"
    assert _cell_text(table, 0, 3) == "4800"
