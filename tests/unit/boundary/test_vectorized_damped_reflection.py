"""Tests for the vectorized SPSO 2011 damped reflection boundary handler."""

from __future__ import annotations

import numpy as np
import pytest

from src.boundary.vectorized_damped_reflection import VectorizedDampedReflectionBoundaryHandler
from src.particles.swarm_state import SwarmState


def _swarm(positions, velocities) -> SwarmState:
    positions = np.asarray(positions, dtype=float)
    n = positions.shape[0]
    return SwarmState(
        positions=positions,
        velocities=np.asarray(velocities, dtype=float),
        pbest_positions=positions.copy(),
        pbest_fitness=np.zeros(n),
        current_fitness=np.zeros(n),
    )


BOUNDS = np.array([[-5.0, 5.0], [-5.0, 5.0]])


def test_positions_are_clamped_to_the_bounds():
    swarm = _swarm([[10.0, -8.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]])

    VectorizedDampedReflectionBoundaryHandler().apply(swarm, BOUNDS)

    assert swarm.positions == pytest.approx(np.array([[5.0, -5.0], [0.0, 1.0]]))


def test_violating_velocity_components_are_reflected_and_damped():
    swarm = _swarm([[10.0, -8.0]], [[2.0, -4.0]])

    VectorizedDampedReflectionBoundaryHandler().apply(swarm, BOUNDS)

    assert swarm.velocities == pytest.approx(np.array([[-1.0, 2.0]]))


def test_in_bounds_components_are_left_untouched():
    swarm = _swarm([[10.0, 1.0]], [[2.0, 3.0]])

    VectorizedDampedReflectionBoundaryHandler().apply(swarm, BOUNDS)

    assert swarm.positions == pytest.approx(np.array([[5.0, 1.0]]))
    assert swarm.velocities == pytest.approx(np.array([[-1.0, 3.0]]))


def test_damping_factor_is_configurable():
    swarm = _swarm([[10.0, 0.0]], [[2.0, 0.0]])

    VectorizedDampedReflectionBoundaryHandler(damping_factor=-1.0).apply(swarm, BOUNDS)

    assert swarm.velocities == pytest.approx(np.array([[-2.0, 0.0]]))


def test_missing_bounds_is_a_no_op():
    swarm = _swarm([[10.0, -8.0]], [[2.0, -4.0]])

    VectorizedDampedReflectionBoundaryHandler().apply(swarm, None)

    assert swarm.positions == pytest.approx(np.array([[10.0, -8.0]]))
    assert swarm.velocities == pytest.approx(np.array([[2.0, -4.0]]))


def test_particles_exactly_on_the_boundary_are_not_reflected():
    swarm = _swarm([[5.0, -5.0]], [[2.0, -4.0]])

    VectorizedDampedReflectionBoundaryHandler().apply(swarm, BOUNDS)

    assert swarm.positions == pytest.approx(np.array([[5.0, -5.0]]))
    assert swarm.velocities == pytest.approx(np.array([[2.0, -4.0]]))


def test_whole_swarm_is_handled_in_one_call():
    positions = np.array([[9.0, 0.0], [-9.0, 0.0], [0.0, 9.0], [1.0, 2.0]])
    swarm = _swarm(positions, np.ones((4, 2)))

    VectorizedDampedReflectionBoundaryHandler().apply(swarm, BOUNDS)

    assert np.all(swarm.positions >= BOUNDS[:, 0])
    assert np.all(swarm.positions <= BOUNDS[:, 1])
    assert swarm.velocities == pytest.approx(np.array([[-0.5, 1.0], [-0.5, 1.0], [1.0, -0.5], [1.0, 1.0]]))
