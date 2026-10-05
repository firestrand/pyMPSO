"""Behavioral contracts for scalar particles and their creation helpers."""

import numpy as np
import pytest

from src.initialization.bounds_aware import BoundsAwareInitialization
from src.initialization.random_uniform import RandomUniformInitialization
from src.particles.standard import StandardParticle
from src.particles.utils import create_particle, create_swarm, restart_particle


def test_particle_copies_initial_state_and_tracks_only_position_changes():
    position = np.array([1.0, 2.0])
    velocity = np.array([3.0, 4.0])
    particle = StandardParticle(position, velocity)
    position[:] = 9
    velocity[:] = 9
    np.testing.assert_array_equal(particle.position, [1, 2])
    np.testing.assert_array_equal(particle.velocity, [3, 4])
    particle.position = np.array([1.0, 2.0])
    assert particle.update_count == 0
    particle.position = np.array([2.0, 1.0])
    assert particle.update_count == 1
    particle.metadata["restart"] = True
    assert particle.metadata == {"restart": True}
    assert "age=1" in repr(particle)


def test_personal_best_tracks_improvement_and_stagnation_without_aliasing():
    particle = StandardParticle(np.array([2.0]), np.zeros(1))
    particle.fitness = 4
    particle.update_pbest()
    assert particle.pbest_fitness == 4.0
    particle.position = np.array([3.0])
    particle.fitness = 4.0
    particle.update_pbest()
    particle.fitness = 9.0
    particle.update_pbest()
    assert particle.iterations_since_improvement == 2
    np.testing.assert_array_equal(particle.pbest, [2])
    particle.position = np.array([1.0])
    particle.fitness = 1.0
    particle.update_pbest()
    particle.position[:] = 8
    np.testing.assert_array_equal(particle.pbest, [1])
    assert particle.iterations_since_improvement == 0
    assert particle.pbest_fitness == 1.0


@pytest.mark.parametrize("attribute", ["position", "velocity", "pbest"])
def test_particle_rejects_incompatible_vector_assignment(attribute):
    particle = StandardParticle(np.ones(2), np.zeros(2))
    with pytest.raises(ValueError, match="shape"):
        setattr(particle, attribute, np.ones(3))
    with pytest.raises(ValueError, match="numpy array"):
        setattr(particle, attribute, [1, 2])
    setattr(particle, attribute, np.array([3.0, 4.0]))
    np.testing.assert_array_equal(getattr(particle, attribute), [3, 4])


@pytest.mark.parametrize("attribute", ["fitness", "pbest_fitness"])
def test_particle_requires_numeric_fitness(attribute):
    particle = StandardParticle(np.ones(1), np.zeros(1))
    with pytest.raises(TypeError, match="float or int"):
        setattr(particle, attribute, "1")
    setattr(particle, attribute, 2)
    assert getattr(particle, attribute) == 2.0


@pytest.mark.parametrize("position,velocity", [(np.ones((1, 2)), np.zeros(2)), (np.ones(2), np.zeros(3))])
def test_particle_rejects_invalid_initial_shapes(position, velocity):
    with pytest.raises(ValueError, match="1D numpy array"):
        StandardParticle(position, velocity)


def test_creation_helpers_preserve_seeded_sequence_and_restart_bounds():
    bounds = np.array([[-2.0, 2.0], [0.0, 4.0]])
    strategy = BoundsAwareInitialization(bounds, seed=13)
    reference = BoundsAwareInitialization(bounds, seed=13)
    particles = create_swarm(strategy, 3)
    particles.append(create_particle(strategy))
    for particle in particles:
        expected_position, expected_velocity = reference.generate_next_particle()
        np.testing.assert_array_equal(particle.position, expected_position)
        np.testing.assert_array_equal(particle.velocity, expected_velocity)
    fixed_bounds = np.array([[1.0, 1.0], [2.0, 2.0]])
    restarted = restart_particle(strategy, fixed_bounds, np.zeros(2))
    np.testing.assert_array_equal(restarted.position, [1, 2])
    np.testing.assert_array_equal(restarted.velocity, [0, 0])
    ordinary_restart = restart_particle(RandomUniformInitialization(bounds, seed=1))
    np.testing.assert_array_equal(ordinary_restart.velocity, [0, 0])
