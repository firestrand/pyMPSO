import numpy as np

from src.particles.swarm_state import SwarmState
from src.position.base import PositionUpdateStrategy


class MockPositionUpdate:
    def update_positions(self, swarm_state: SwarmState, **_kwargs) -> np.ndarray:
        return swarm_state.positions + swarm_state.velocities


def test_position_update_strategy_protocol():
    strategy: PositionUpdateStrategy = MockPositionUpdate()

    positions = np.array([[1.0, 2.0], [3.0, 4.0]])
    velocities = np.array([[0.5, 0.5], [1.0, 1.0]])

    state = SwarmState(
        positions=positions,
        velocities=velocities,
        pbest_positions=np.zeros_like(positions),
        pbest_fitness=np.zeros(2),
        current_fitness=np.zeros(2),
    )

    new_positions = strategy.update_positions(state)
    expected = np.array([[1.5, 2.5], [4.0, 5.0]])
    np.testing.assert_array_equal(new_positions, expected)

    assert isinstance(strategy, PositionUpdateStrategy)
