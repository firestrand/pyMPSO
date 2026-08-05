"""End-to-end checks for the vectorized SPSO 2011 component stack."""

from __future__ import annotations

import numpy as np
import pytest

from src.run_pso import run_pso_from_config

SPSO2011_PSO_PARAMS = {
    "topology": "random",
    "topology_params": {"k": 3, "rebuild_probability": 1.0},
    "velocity_strategy": "hypersphere",
    "influence_strategy": "single_best",
    "boundary_handler": "damped_reflection",
    "initialization_strategy": "bounds_aware",
}


def _config(max_iterations: int, num_runs: int, seed: int | None, dimensions: int = 10) -> dict:
    return {
        "problems_to_benchmark": {
            "sphere": {
                "dimensions": dimensions,
                "bounds": {"min": -100.0, "max": 100.0},
                "stopping_criteria": [{"type": "max_iterations", "max_iterations": max_iterations}],
            }
        },
        "common_swarm": {"num_particles": 20},
        "common_pso_params": dict(SPSO2011_PSO_PARAMS),
        "common_run_config": {"num_runs": num_runs, "random_seed": seed, "verbose": False},
    }


def _best_fitness(results: dict, task: str = "sphere_10D") -> float:
    task_result = results[task]
    if "best_fitness" in task_result:
        return float(task_result["best_fitness"])
    return float(task_result["summary_statistics"]["min_fitness"])


@pytest.mark.integration
def test_spso2011_stack_runs_end_to_end():
    results = run_pso_from_config(config_path="in-memory", config_data=_config(50, 1, seed=11))

    assert np.isfinite(_best_fitness(results))


@pytest.mark.integration
def test_spso2011_stack_converges_on_sphere():
    """200 iterations must beat the best of the random initial swarm by orders of magnitude."""
    baseline = _best_fitness(run_pso_from_config(config_path="in-memory", config_data=_config(1, 1, seed=5)))
    converged = _best_fitness(run_pso_from_config(config_path="in-memory", config_data=_config(200, 1, seed=5)))

    assert converged < baseline / 100.0


@pytest.mark.integration
def test_seeded_spso2011_runs_are_reproducible():
    """The random topology must draw from the seeded run RNG, not a fresh one."""
    first = _best_fitness(run_pso_from_config(config_path="in-memory", config_data=_config(60, 1, seed=2024)))
    second = _best_fitness(run_pso_from_config(config_path="in-memory", config_data=_config(60, 1, seed=2024)))

    assert first == pytest.approx(second, rel=1e-12)


@pytest.mark.integration
def test_different_seeds_explore_differently():
    first = _best_fitness(run_pso_from_config(config_path="in-memory", config_data=_config(60, 1, seed=1)))
    second = _best_fitness(run_pso_from_config(config_path="in-memory", config_data=_config(60, 1, seed=99)))

    assert first != pytest.approx(second, rel=1e-12)


@pytest.mark.integration
def test_ring_topology_stack_runs_end_to_end():
    config = _config(50, 1, seed=7)
    config["common_pso_params"]["topology"] = "ring"
    config["common_pso_params"]["topology_params"] = {"k": 2}

    results = run_pso_from_config(config_path="in-memory", config_data=config)

    assert np.isfinite(_best_fitness(results))
