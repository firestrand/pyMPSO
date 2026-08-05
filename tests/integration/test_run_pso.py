from pathlib import Path

import pytest

from src.run_pso import (
    run_pso_from_config,
)

# Get the directory of the current test file
TEST_DIR = Path(__file__).resolve().parent
TEST_CONFIG_PATH = TEST_DIR / "test_config.yaml"


def test_run_pso_from_config_smoke_test():
    """Basic smoke test to ensure run_pso_from_config executes with a valid config."""
    assert TEST_CONFIG_PATH.exists(), f"Test config file not found: {TEST_CONFIG_PATH}"

    try:
        results = run_pso_from_config(config_path=str(TEST_CONFIG_PATH), verbose=False)
        # --- Assertions for Benchmark Result Structure ---
        assert isinstance(results, dict)
        assert "_benchmark_summary" in results  # Check for summary key
        assert "total_wall_time" in results["_benchmark_summary"]
        assert results["_benchmark_summary"]["config_file"] == TEST_CONFIG_PATH.name

        # Check for the specific task defined in test_config.yaml
        task_key = "sphere_2D"
        assert task_key in results
        task_results = results[task_key]

        # Assertions on the specific task's results
        assert isinstance(task_results, dict)
        assert "best_fitness" in task_results
        assert "best_position" in task_results
        assert "iterations" in task_results
        assert "evaluations" in task_results
        assert "total_wall_time" in task_results  # Wall time for this specific task
        assert "config_file" in task_results  # Base config file name
        assert "benchmark_task" in task_results
        assert task_results["benchmark_task"] == task_key
        assert task_results["problem_dimension"] == 2  # From test_config.yaml
        assert task_results["num_particles"] == 5  # From test_config.yaml

    except Exception as e:
        pytest.fail(f"run_pso_from_config raised an exception unexpectedly: {e}")
