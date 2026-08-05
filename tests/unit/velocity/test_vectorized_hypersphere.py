"""Tests for the vectorized SPSO 2011 hypersphere velocity update."""

from __future__ import annotations

import numpy as np
import pytest

from src.constants import SPSO2011_C1, SPSO2011_C2, SPSO2011_W
from src.particles.swarm_state import SwarmState
from src.velocity.config import HypersphereVelocityConfig
from src.velocity.vectorized_hypersphere import VectorizedHypersphereVelocity


def _swarm(positions, velocities, pbest_positions) -> SwarmState:
    positions = np.asarray(positions, dtype=float)
    n = positions.shape[0]
    return SwarmState(
        positions=positions,
        velocities=np.asarray(velocities, dtype=float),
        pbest_positions=np.asarray(pbest_positions, dtype=float),
        pbest_fitness=np.zeros(n),
        current_fitness=np.zeros(n),
    )


def _config(**kwargs) -> HypersphereVelocityConfig:
    kwargs.setdefault("rng", np.random.RandomState(7))
    return HypersphereVelocityConfig(**kwargs)


def test_defaults_follow_the_spso2011_constants():
    config = HypersphereVelocityConfig()

    assert config.c1 == pytest.approx(SPSO2011_C1)
    assert config.c2 == pytest.approx(SPSO2011_C2)
    assert config.w == pytest.approx(SPSO2011_W)


def test_shape_is_preserved():
    swarm = _swarm(np.zeros((6, 4)), np.ones((6, 4)), np.ones((6, 4)))

    new_velocities = VectorizedHypersphereVelocity().update_velocities(swarm, np.ones((6, 4)), _config())

    assert new_velocities.shape == (6, 4)


def test_zero_radius_row_decays_by_inertia_only():
    """When the gravity centre coincides with the position, v(t+1) = w*v(t)."""
    swarm = _swarm(positions=[[1.0, 2.0]], velocities=[[3.0, -4.0]], pbest_positions=[[1.0, 2.0]])
    informants = np.array([[1.0, 2.0]])

    new_velocities = VectorizedHypersphereVelocity().update_velocities(swarm, informants, _config(w=0.5))

    assert new_velocities == pytest.approx(np.array([[1.5, -2.0]]))


def test_displacement_stays_within_twice_the_gravity_radius():
    """The sampled point lies inside the hypersphere of radius |G-x| centred at G."""
    rng = np.random.RandomState(3)
    positions = rng.uniform(-10, 10, size=(50, 5))
    pbest = rng.uniform(-10, 10, size=(50, 5))
    informants = rng.uniform(-10, 10, size=(50, 5))
    swarm = _swarm(positions, np.zeros((50, 5)), pbest)

    new_velocities = VectorizedHypersphereVelocity().update_velocities(swarm, informants, _config(w=0.0))

    c1, c2 = SPSO2011_C1, SPSO2011_C2
    gravity = positions + (c1 / 3.0) * (pbest - positions) + (c2 / 3.0) * (informants - positions)
    radius = np.linalg.norm(gravity - positions, axis=1)
    displacement = np.linalg.norm(new_velocities, axis=1)

    assert np.all(displacement <= 2 * radius + 1e-9)


def test_row_with_pbest_equal_to_informant_uses_the_halved_cognitive_form():
    """SPSO 2011 collapses to G = x + (c1/2)(p-x) when the informant is the pbest."""
    position = np.array([[0.0, 0.0]])
    pbest = np.array([[2.0, 0.0]])
    swarm = _swarm(position, np.zeros((1, 2)), pbest)

    new_velocities = VectorizedHypersphereVelocity().update_velocities(swarm, pbest.copy(), _config(w=0.0))

    expected_radius = np.linalg.norm(0.5 * SPSO2011_C1 * (pbest - position))
    assert np.linalg.norm(new_velocities) <= 2 * expected_radius + 1e-9
    assert np.linalg.norm(new_velocities) > 0.0


def test_same_seed_reproduces_the_same_velocities():
    swarm_a = _swarm(np.zeros((8, 3)), np.ones((8, 3)), np.full((8, 3), 2.0))
    swarm_b = _swarm(np.zeros((8, 3)), np.ones((8, 3)), np.full((8, 3), 2.0))
    informants = np.full((8, 3), -1.0)

    a = VectorizedHypersphereVelocity().update_velocities(swarm_a, informants, _config())
    b = VectorizedHypersphereVelocity().update_velocities(swarm_b, informants, _config())

    assert a == pytest.approx(b)


def test_inertia_scales_the_carried_velocity():
    """With pbest and informant at the position, only the inertia term survives."""
    swarm = _swarm(positions=[[0.0, 0.0]], velocities=[[2.0, 4.0]], pbest_positions=[[0.0, 0.0]])
    informants = np.zeros((1, 2))

    new_velocities = VectorizedHypersphereVelocity().update_velocities(swarm, informants, _config(w=0.25))

    assert new_velocities == pytest.approx(np.array([[0.5, 1.0]]))


def test_uniform_in_sphere_distribution_is_supported():
    """distrib=-1 selects the uniform-in-volume radius scaling."""
    swarm = _swarm(np.zeros((30, 4)), np.zeros((30, 4)), np.ones((30, 4)))
    informants = np.full((30, 4), 3.0)

    new_velocities = VectorizedHypersphereVelocity().update_velocities(swarm, informants, _config(distrib=-1, w=0.0))

    assert np.isfinite(new_velocities).all()
    assert np.any(new_velocities != 0.0)
