"""Tests for the vectorized SPSO 2011 random topology."""

from __future__ import annotations

import numpy as np
import pytest

from src.particles.swarm_state import SwarmState
from src.topology.config import RandomTopologyConfig
from src.topology.vectorized_random import VectorizedRandomTopology


def _swarm(n: int, d: int = 2) -> SwarmState:
    return SwarmState(
        positions=np.zeros((n, d)),
        velocities=np.zeros((n, d)),
        pbest_positions=np.zeros((n, d)),
        pbest_fitness=np.zeros(n),
        current_fitness=np.zeros(n),
    )


def _config(**kwargs) -> RandomTopologyConfig:
    kwargs.setdefault("rng", np.random.RandomState(12345))
    return RandomTopologyConfig(**kwargs)


def test_every_particle_always_informs_itself():
    matrix = VectorizedRandomTopology().get_topology_matrix(_swarm(20), _config(k=3))

    assert np.diag(matrix).all()


def test_matrix_is_boolean_and_square():
    matrix = VectorizedRandomTopology().get_topology_matrix(_swarm(12), _config(k=3))

    assert matrix.dtype == np.bool_
    assert matrix.shape == (12, 12)


def test_same_seed_reproduces_the_same_topology():
    a = VectorizedRandomTopology().get_topology_matrix(_swarm(15), _config(k=3))
    b = VectorizedRandomTopology().get_topology_matrix(_swarm(15), _config(k=3))

    assert np.array_equal(a, b)


def test_expected_degree_matches_the_spso2011_formula():
    """Expected informants per particle is 1 + (S-1)*(1-((S-1)/S)^K)."""
    n, k = 40, 3
    strategy = VectorizedRandomTopology()
    config = _config(k=k, rebuild_probability=1.0)

    degrees = [strategy.get_topology_matrix(_swarm(n), config).sum(axis=1).mean() for _ in range(200)]

    inform_prob = 1 - ((n - 1) / n) ** k
    expected = 1 + (n - 1) * inform_prob
    assert np.mean(degrees) == pytest.approx(expected, rel=0.02)


def test_rebuild_probability_zero_reuses_the_cached_topology():
    strategy = VectorizedRandomTopology()
    config = _config(k=3, rebuild_probability=0.0)
    swarm = _swarm(15)

    first = strategy.get_topology_matrix(swarm, config).copy()
    for _ in range(5):
        assert np.array_equal(strategy.get_topology_matrix(swarm, config), first)


def test_rebuild_probability_one_resamples_every_call():
    strategy = VectorizedRandomTopology()
    config = _config(k=3, rebuild_probability=1.0)
    swarm = _swarm(30)

    first = strategy.get_topology_matrix(swarm, config).copy()
    second = strategy.get_topology_matrix(swarm, config)

    assert not np.array_equal(first, second)


def test_cached_topology_is_discarded_when_the_swarm_size_changes():
    strategy = VectorizedRandomTopology()
    config = _config(k=3, rebuild_probability=0.0)

    strategy.get_topology_matrix(_swarm(10), config)
    resized = strategy.get_topology_matrix(_swarm(25), config)

    assert resized.shape == (25, 25)


def test_single_particle_is_its_own_neighbour():
    matrix = VectorizedRandomTopology().get_topology_matrix(_swarm(1), _config(k=3))

    assert np.array_equal(matrix, np.array([[True]]))
