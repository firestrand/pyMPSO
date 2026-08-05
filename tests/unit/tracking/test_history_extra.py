import tempfile
from pathlib import Path

import numpy as np

from src.tracking.history import ExperimentHistory, IterationMetrics


def test_empty_history_methods():
    history = ExperimentHistory()
    assert history._calculate_diversity(np.array([])) == 0.0
    assert history.get_convergence_curve() == ([], [])
    assert history.get_diversity_curve() == ([], [])
    assert history.get_summary_statistics() == {}


def test_convergence_90_iter_and_no_improvement():
    history = ExperimentHistory()
    history.initial_best_fitness = 100.0
    history.final_best_fitness = 95.0  # Total improvement is 5.0, 90% is 4.5

    # Not enough improvement yet
    metrics1 = IterationMetrics(
        iteration=0,
        best_fitness=98.0,
        mean_fitness=98.0,
        worst_fitness=98.0,
        fitness_std=0.0,
        global_best_position=np.array([1.0]),
        diversity=0.0,
        mean_velocity_norm=0.0,
        max_velocity_norm=0.0,
        min_velocity_norm=0.0,
        improvement=2.0,
        evaluations=1,
        elapsed_time=0.1,
    )
    history.iterations.append(metrics1)

    summary = history.get_summary_statistics()
    assert summary["convergence_90_iteration"] is None

    # Missing initial_best_fitness
    history.initial_best_fitness = None
    summary_no_init = history.get_summary_statistics()
    assert summary_no_init["convergence_90_iteration"] is None
    assert summary_no_init["total_improvement"] == 0


def test_to_dict_track_positions_large():
    history = ExperimentHistory(track_positions=True)

    # Add 15 positions to test the > 10 branch
    for i in range(15):
        history.particle_positions.append(np.array([[float(i)]]))
        history.iterations.append(
            IterationMetrics(
                iteration=i,
                best_fitness=1.0,
                mean_fitness=1.0,
                worst_fitness=1.0,
                fitness_std=0.0,
                global_best_position=np.array([1.0]),
                diversity=0.0,
                mean_velocity_norm=0.0,
                max_velocity_norm=0.0,
                min_velocity_norm=0.0,
                improvement=0.0,
                evaluations=1,
                elapsed_time=0.1,
            )
        )

    d = history.to_dict()
    assert "particle_positions_sample" in d
    assert "first_5" in d["particle_positions_sample"]
    assert "last_5" in d["particle_positions_sample"]


def test_save_and_load_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_history.json"

        history = ExperimentHistory(track_positions=False)
        history.initial_best_fitness = 10.0
        history.final_best_fitness = 1.0
        history.total_evaluations = 50
        history.total_iterations = 5
        history.convergence_iteration = 3

        metrics = IterationMetrics(
            iteration=1,
            best_fitness=5.0,
            mean_fitness=6.0,
            worst_fitness=7.0,
            fitness_std=0.5,
            global_best_position=np.array([2.0, 3.0]),
            diversity=1.2,
            mean_velocity_norm=0.1,
            max_velocity_norm=0.2,
            min_velocity_norm=0.0,
            improvement=5.0,
            evaluations=10,
            elapsed_time=0.5,
        )
        history.iterations.append(metrics)

        history.save_to_json(str(filepath))

        assert filepath.exists()

        loaded = ExperimentHistory.load_from_json(str(filepath))

        assert loaded.total_iterations == 5
        assert loaded.total_evaluations == 50
        assert loaded.initial_best_fitness == 10.0
        assert loaded.final_best_fitness == 1.0
        assert loaded.convergence_iteration == 3

        assert len(loaded.iterations) == 1
        m = loaded.iterations[0]
        assert m.iteration == 1
        assert m.best_fitness == 5.0
        assert m.evaluations == 10
        np.testing.assert_array_equal(m.global_best_position, np.array([2.0, 3.0]))
