"""Tests for the vectorized ring topology."""

from __future__ import annotations

import numpy as np
import pytest

from src.particles.swarm_state import SwarmState
from src.topology.config import RingTopologyConfig
from src.topology.vectorized_ring import VectorizedRingTopology


def _swarm(n: int, d: int = 2) -> SwarmState:
    return SwarmState(
        positions=np.zeros((n, d)),
        velocities=np.zeros((n, d)),
        pbest_positions=np.zeros((n, d)),
        pbest_fitness=np.zeros(n),
        current_fitness=np.zeros(n),
    )


def test_k1_connects_each_particle_to_itself_and_both_immediate_neighbours():
    matrix = VectorizedRingTopology().get_topology_matrix(_swarm(5), RingTopologyConfig(k=1))

    expected = np.array(
        [
            [True, True, False, False, True],
            [True, True, True, False, False],
            [False, True, True, True, False],
            [False, False, True, True, True],
            [True, False, False, True, True],
        ]
    )
    assert np.array_equal(matrix, expected)


def test_neighbourhood_size_is_2k_plus_1():
    matrix = VectorizedRingTopology().get_topology_matrix(_swarm(9), RingTopologyConfig(k=2))

    assert matrix.sum(axis=1).tolist() == [5] * 9


def test_topology_is_symmetric():
    matrix = VectorizedRingTopology().get_topology_matrix(_swarm(7), RingTopologyConfig(k=2))

    assert np.array_equal(matrix, matrix.T)


def test_matrix_is_boolean_and_square():
    matrix = VectorizedRingTopology().get_topology_matrix(_swarm(6), RingTopologyConfig(k=1))

    assert matrix.dtype == np.bool_
    assert matrix.shape == (6, 6)


def test_wide_neighbourhood_saturates_to_fully_connected():
    """k large enough to wrap past every particle degrades to the global topology."""
    matrix = VectorizedRingTopology().get_topology_matrix(_swarm(5), RingTopologyConfig(k=4))

    assert matrix.all()


def test_single_particle_is_its_own_neighbour():
    matrix = VectorizedRingTopology().get_topology_matrix(_swarm(1), RingTopologyConfig(k=1))

    assert np.array_equal(matrix, np.array([[True]]))


def test_every_particle_always_informs_itself():
    matrix = VectorizedRingTopology().get_topology_matrix(_swarm(8), RingTopologyConfig(k=3))

    assert np.diag(matrix).all()


def test_k_below_one_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        VectorizedRingTopology().get_topology_matrix(_swarm(5), RingTopologyConfig(k=0))
