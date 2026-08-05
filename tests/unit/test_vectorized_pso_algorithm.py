from typing import Any

import numpy as np

from src.boundary.vectorized_position_clamping import VectorizedPositionClampingBoundaryHandler
from src.clamping.velocity import MaxNormVelocityClampingStrategy
from src.influence.vectorized_single_best import VectorizedSingleBestInfluence
from src.initialization.random_uniform import RandomUniformInitialization
from src.particles.swarm_state import SwarmState
from src.position.standard import StandardPositionUpdate
from src.problem.sphere import SphereFunction
from src.stopping.base import StoppingCriteria
from src.topology.config import GlobalTopologyConfig
from src.topology.vectorized_global import VectorizedGlobalTopology
from src.tracking.events import EventSubscriber
from src.vectorized_pso_algorithm import VectorizedPSOAlgorithm
from src.velocity.config import StandardVelocityConfig
from src.velocity.vectorized_standard import VectorizedStandardVelocityUpdate


class MockStoppingCriteria(StoppingCriteria):
    def __init__(self, max_iters: int):
        self.max_iters = max_iters

    def should_stop(self, swarm_state: dict[str, Any]) -> bool:
        return swarm_state.get("current_iteration", 0) >= self.max_iters


class MockEventSubscriber(EventSubscriber):
    def __init__(self):
        self.run_start_called = 0
        self.iter_start_called = 0
        self.iter_end_called = 0
        self.run_end_called = 0

    def on_run_start(self, swarm_state: SwarmState, **kwargs: Any) -> None:  # noqa: ARG002
        self.run_start_called += 1

    def on_iteration_start(self, iteration: int, swarm_state: SwarmState, **kwargs: Any) -> None:  # noqa: ARG002
        self.iter_start_called += 1

    def on_iteration_end(self, iteration: int, swarm_state: SwarmState, **kwargs: Any) -> None:  # noqa: ARG002
        self.iter_end_called += 1

    def on_run_end(self, swarm_state: SwarmState, **kwargs: Any) -> None:  # noqa: ARG002
        self.run_end_called += 1


def test_vectorized_pso_algorithm_runs_and_fires_events():
    """Verify VectorizedPSOAlgorithm runs end-to-end and fires all observer events."""
    dimension = 2
    bounds = np.array([[-5.0, 5.0]] * dimension)
    problem = SphereFunction(dimension=dimension, bounds=bounds)

    stopping_criteria = MockStoppingCriteria(max_iters=5)

    pso = VectorizedPSOAlgorithm(
        problem=problem,
        stopping_criteria=stopping_criteria,
        initialization_strategy=RandomUniformInitialization(bounds=bounds, seed=42),
        topology=VectorizedGlobalTopology(),
        topology_config=GlobalTopologyConfig(),
        influence_strategy=VectorizedSingleBestInfluence(),
        velocity_strategy=VectorizedStandardVelocityUpdate(),
        velocity_config=StandardVelocityConfig(c1=1.5, c2=1.5),
        position_strategy=StandardPositionUpdate(),
        boundary_handler=VectorizedPositionClampingBoundaryHandler(),
        num_particles=10,
        verbose=False,
    )

    subscriber = MockEventSubscriber()
    pso.event_publisher.add_subscriber(subscriber)

    results = pso.run()

    assert "best_fitness" in results
    assert "best_position" in results
    assert results["iterations"] == 5

    assert subscriber.run_start_called == 1
    assert subscriber.iter_start_called == 5
    assert subscriber.iter_end_called == 5
    assert subscriber.run_end_called == 1


def test_velocity_clamping_is_applied_to_velocity_matrix():
    """Configured vectorized velocity clamping should be applied after velocity updates."""
    dimension = 2
    bounds = np.array([[-5.0, 5.0]] * dimension)
    problem = SphereFunction(dimension=dimension, bounds=bounds)

    stopping_criteria = MockStoppingCriteria(max_iters=1)
    clamping = MaxNormVelocityClampingStrategy(max_velocity=1.0)

    pso = VectorizedPSOAlgorithm(
        problem=problem,
        stopping_criteria=stopping_criteria,
        initialization_strategy=RandomUniformInitialization(bounds=bounds, seed=42),
        topology=VectorizedGlobalTopology(),
        topology_config=GlobalTopologyConfig(),
        influence_strategy=VectorizedSingleBestInfluence(),
        velocity_strategy=VectorizedStandardVelocityUpdate(),
        velocity_config=StandardVelocityConfig(c1=0.0, c2=0.0, velocity_clamping=clamping),
        position_strategy=StandardPositionUpdate(),
        boundary_handler=VectorizedPositionClampingBoundaryHandler(),
        num_particles=2,
        verbose=False,
    )

    raw_velocities = np.array([[2.5, -4.2], [0.2, -0.5]], dtype=np.float64)
    clamped = pso._apply_velocity_clamping(raw_velocities)

    expected = np.array([[1.0, -1.0], [0.2, -0.5]])
    np.testing.assert_allclose(clamped, expected)
