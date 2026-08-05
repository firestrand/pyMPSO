import numpy as np

from src.particles.swarm_state import SwarmState
from src.tracking.history import ExperimentHistory


def test_experiment_history_as_subscriber():
    """Verify ExperimentHistory records metrics correctly via event subscription."""

    # Create dummy states for two iterations
    state1 = SwarmState(
        positions=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        velocities=np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]),
        pbest_positions=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        pbest_fitness=np.array([10.0, 8.0, 15.0]),
        current_fitness=np.array([10.0, 8.0, 15.0]),
    )

    state2 = SwarmState(
        positions=np.array([[1.5, 1.5], [2.5, 2.5], [3.5, 3.5]]),
        velocities=np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]),
        pbest_positions=np.array([[1.5, 1.5], [2.0, 2.0], [3.0, 3.0]]),
        pbest_fitness=np.array([5.0, 8.0, 15.0]),  # P0 improved
        current_fitness=np.array([5.0, 9.0, 20.0]),
    )

    history = ExperimentHistory(track_positions=True)

    # Trigger events
    history.on_run_start(state1)

    # Iteration 0
    history.on_iteration_end(0, state1, evaluations_used=3)

    # Iteration 1
    history.on_iteration_end(1, state2, evaluations_used=6)

    history.on_run_end(state2)

    # Verify metrics
    assert len(history.iterations) == 2

    # Iteration 0 checks
    iter0 = history.iterations[0]
    assert iter0.iteration == 0
    assert iter0.evaluations == 3
    assert iter0.best_fitness == 8.0  # From P1
    np.testing.assert_array_equal(iter0.global_best_position, np.array([2.0, 2.0]))

    # Iteration 1 checks
    iter1 = history.iterations[1]
    assert iter1.iteration == 1
    assert iter1.evaluations == 6
    assert iter1.best_fitness == 5.0  # From P0
    np.testing.assert_array_equal(iter1.global_best_position, np.array([1.5, 1.5]))

    # Verify improvement
    assert iter1.improvement == 3.0  # 8.0 - 5.0

    # Summary stats
    summary = history.get_summary_statistics()
    assert summary["total_iterations"] == 1  # Final iteration index
    assert summary["total_evaluations"] == 6
    assert summary["initial_best_fitness"] == 8.0
    assert summary["final_best_fitness"] == 5.0
    assert summary["total_improvement"] == 3.0

    # Check that serialization works
    as_dict = history.to_dict()
    assert len(as_dict["iterations"]) == 2
    assert as_dict["summary"]["total_evaluations"] == 6
    assert "particle_positions" in as_dict
