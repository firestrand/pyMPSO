"""Scalar topology neighborhood and cache contracts."""

import numpy as np
import pytest

from src.particles.base import ParticleBase
from src.particles.standard import StandardParticle
from src.topology.global_topology import GlobalTopology
from src.topology.random_topology import RandomTopology
from src.topology.ring_topology import RingTopology


def test_global_excludes_self_and_ring_wraps_in_both_directions():
    particles: list[ParticleBase] = [StandardParticle(np.array([float(i)]), np.zeros(1)) for i in range(5)]
    global_neighbors = GlobalTopology().get_neighbors(particles)
    assert global_neighbors[particles[0]] == particles[1:]
    ring_neighbors = RingTopology(k=2).get_neighbors(particles)
    assert ring_neighbors[particles[0]] == [particles[4], particles[1], particles[3], particles[2]]
    assert ring_neighbors[particles[4]] == [particles[3], particles[0], particles[2], particles[1]]


@pytest.mark.parametrize("topology", [GlobalTopology(), RingTopology(), RandomTopology(seed=1)])
def test_topologies_handle_empty_and_single_particle_swarms(topology):
    assert topology.get_neighbors([]) == {}
    particle = StandardParticle(np.zeros(1), np.zeros(1))
    expected = [particle] if isinstance(topology, RandomTopology) else []
    assert topology.get_neighbors([particle]) == {particle: expected}


def test_ring_rejects_nonpositive_connectivity():
    with pytest.raises(ValueError, match="at least 1"):
        RingTopology(k=0)


def test_random_topology_reuses_cache_and_forces_rebuild_with_seeded_rng():
    particles: list[ParticleBase] = [StandardParticle(np.array([float(i)]), np.zeros(1)) for i in range(6)]
    topology = RandomTopology(k=2, rebuild_probability=0.0, rng=np.random.RandomState(12))
    initial = topology.get_neighbors(particles)
    assert all(particle in initial[particle] for particle in particles)
    assert topology.get_neighbors(particles, rebuild=False) is initial
    assert topology.get_neighbors(particles) is initial
    rebuilt = topology.get_neighbors(particles, rebuild=True)
    assert rebuilt is not initial
    assert rebuilt != initial
    equivalent = RandomTopology(k=2, seed=12).get_neighbors(particles)
    assert equivalent == initial
    always_rebuild = RandomTopology(k=0, rebuild_probability=1.0)
    first = always_rebuild.get_neighbors(particles)
    assert first == {particle: [particle] for particle in particles}
    assert always_rebuild.get_neighbors(particles) is not first
