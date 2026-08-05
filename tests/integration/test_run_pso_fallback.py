from src.run_pso import run_pso_from_config


def test_run_pso_rejects_non_vectorized_configuration():
    """Verify run_pso_from_config fails fast when vectorized execution cannot support the config."""

    # 'clpso' velocity has no vectorized implementation yet, so it must be rejected.
    config_data = {
        "problems_to_benchmark": {
            "sphere": {
                "dimensions": 2,
                "bounds": {"min": -5.0, "max": 5.0},
                "stopping_criteria": {"type": "max_iterations", "max_iterations": 2},
            }
        },
        "common_swarm": {"num_particles": 5},
        "common_pso_params": {
            "topology": "global",
            "velocity_strategy": "clpso",  # Not vectorized
            "influence_strategy": "single_best",
            "boundary_handler": "clamping",
            "initialization_strategy": "random_uniform",
        },
        "common_run_config": {"num_runs": 1, "random_seed": 42, "verbose": False},
    }

    results = run_pso_from_config(config_path="dummy_config.json", verbose=False, config_data=config_data)

    assert "error" not in results
    assert "sphere_2D" in results

    task_results = results["sphere_2D"]
    assert "error" in task_results
    assert "non-vectorized execution has been removed" in task_results["error"].lower()
    assert task_results["num_particles"] == 5
