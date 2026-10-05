"""Scalar influence fitness selection and random stream contracts."""

import numpy as np

from src.influence.fips import FIPSInfluenceStrategy
from src.influence.single_best import SingleBestInfluence
from src.particles.standard import StandardParticle


def test_single_best_keeps_self_on_ties_and_selects_strict_improvement():
    particle = StandardParticle(np.array([1.0]), np.zeros(1))
    neighbor = StandardParticle(np.array([2.0]), np.zeros(1))
    particle.pbest_fitness = neighbor.pbest_fitness = 3.0
    influence = SingleBestInfluence()
    np.testing.assert_array_equal(influence.get_informant_position(particle, []), [1])
    np.testing.assert_array_equal(influence.get_informant_position(particle, [neighbor]), [1])
    neighbor.pbest_fitness = 2.0
    np.testing.assert_array_equal(influence.get_informant_position(particle, [neighbor]), [2])


def test_self_informant_consumes_rng_until_another_index_is_drawn():
    particle = StandardParticle(np.ones(1), np.zeros(1))
    rng = np.random.RandomState(1)
    reference = np.random.RandomState(1)
    assert reference.random() < 0.5
    assert reference.random() >= 0.5
    influence = SingleBestInfluence()
    influence.get_informant_position(
        particle, [], {"consume_random_for_self_informant": True, "rng": rng, "swarm_size": 2, "particle_index": 0}
    )
    assert rng.random() == reference.random()
    before = np.random.RandomState(5)
    reference = np.random.RandomState(5)
    influence.get_informant_position(
        particle, [], {"consume_random_for_self_informant": True, "rng": before, "swarm_size": 1}
    )
    assert before.random() == reference.random()
    influence.get_informant_position(particle, [], {"consume_random_for_self_informant": True})


def test_fips_returns_independent_best_neighbor_or_self_position():
    particle = StandardParticle(np.array([1.0]), np.zeros(1))
    neighbor = StandardParticle(np.array([2.0]), np.zeros(1))
    neighbor.pbest_fitness = 2.0
    influence = FIPSInfluenceStrategy()
    result = influence.get_informant_position(particle, [])
    result[:] = 99
    np.testing.assert_array_equal(particle.pbest, [1])
    result = influence.get_informant_position(particle, [particle, neighbor])
    np.testing.assert_array_equal(result, [2])
    result[:] = 99
    np.testing.assert_array_equal(neighbor.pbest, [2])
    assert str(influence) == "FIPSInfluenceStrategy"
    assert repr(influence) == "FIPSInfluenceStrategy()"
