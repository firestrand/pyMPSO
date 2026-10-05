"""Initialization bounds, reproducibility, and restart behavior."""

import numpy as np
import pytest

from src.initialization.bounds_aware import BoundsAwareInitialization
from src.initialization.random_uniform import RandomUniformInitialization


@pytest.mark.parametrize("strategy_type", [RandomUniformInitialization, BoundsAwareInitialization])
def test_seeded_initialization_respects_anisotropic_bounds(strategy_type):
    bounds = np.array([[-3.0, 4.0], [10.0, 10.0]])
    strategy = strategy_type(bounds, seed=42)
    reference = strategy_type(bounds, rng=np.random.RandomState(42))
    positions, velocities = strategy.initialize_swarm(5)
    assert strategy.dimensions == 2
    for position, velocity in zip(positions, velocities, strict=True):
        expected_position, expected_velocity = reference.generate_next_particle()
        np.testing.assert_array_equal(position, expected_position)
        np.testing.assert_array_equal(velocity, expected_velocity)
        assert np.all(position >= bounds[:, 0]) and np.all(position <= bounds[:, 1])
        assert velocity[1] == 0
        if strategy_type is BoundsAwareInitialization:
            assert np.all(position + velocity >= bounds[:, 0])
            assert np.all(position + velocity <= bounds[:, 1])
        else:
            assert np.all(np.abs(velocity) <= bounds[:, 1] - bounds[:, 0])
    exposed_bounds = strategy.bounds
    exposed_bounds[:] = 99
    np.testing.assert_array_equal(strategy.bounds, bounds)


@pytest.mark.parametrize("strategy_type", [RandomUniformInitialization, BoundsAwareInitialization])
def test_overridden_restart_copies_rng_state_without_advancing_original(strategy_type):
    bounds = np.array([[-10.0, 10.0]])
    override = np.array([[2.0, 4.0]])
    strategy = strategy_type(bounds, seed=2)
    reference = strategy_type(bounds, seed=2)
    expected = strategy_type(override, seed=2).generate_next_particle()
    restarted = strategy.generate_restart_state(override, np.array([0.0]))
    for actual, wanted in zip(restarted, expected, strict=True):
        np.testing.assert_array_equal(actual, wanted)
    for actual, wanted in zip(strategy.generate_next_particle(), reference.generate_next_particle(), strict=True):
        np.testing.assert_array_equal(actual, wanted)
    np.testing.assert_array_equal(strategy.bounds, bounds)


@pytest.mark.parametrize("strategy_type", [RandomUniformInitialization, BoundsAwareInitialization])
@pytest.mark.parametrize("local_search", [False, True])
def test_restart_local_search_is_clipped_and_velocity_policy_is_preserved(strategy_type, local_search, monkeypatch):
    bounds = np.array([[0.0, 10.0], [-10.0, 0.0]])
    strategy = strategy_type(bounds, seed=10)
    monkeypatch.setattr(strategy, "_should_use_local_search", lambda: local_search)
    position, velocity = strategy.generate_restart_state(_current_best_position=np.array([0.0, 0.0]))
    if local_search:
        assert 0 <= position[0] <= 1
        assert -1 <= position[1] <= 0
    else:
        assert np.all(position >= bounds[:, 0]) and np.all(position <= bounds[:, 1])
    if strategy_type is RandomUniformInitialization:
        np.testing.assert_array_equal(velocity, [0, 0])
    else:
        assert np.all(position + velocity >= bounds[:, 0])
        assert np.all(position + velocity <= bounds[:, 1])


@pytest.mark.parametrize("strategy_type", [RandomUniformInitialization, BoundsAwareInitialization])
def test_default_restart_sampling_uses_seeded_local_search_decision(strategy_type):
    strategy = strategy_type(np.array([[0.0, 10.0]]), seed=5)
    position, _ = strategy.generate_restart_state(_current_best_position=np.array([0.0]))
    assert 0 <= position[0] <= 1


@pytest.mark.parametrize("bounds", [np.ones(2), np.ones((2, 3)), np.array([[2.0, 1.0]])])
def test_invalid_bounds_are_rejected(bounds):
    with pytest.raises(ValueError, match="Bounds|Lower bounds"):
        RandomUniformInitialization(bounds)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5])
def test_nonpositive_or_noninteger_batch_sizes_are_rejected(batch_size):
    strategy = RandomUniformInitialization(np.array([[0.0, 1.0]]))
    with pytest.raises(ValueError, match="positive integer"):
        strategy.initialize_swarm(batch_size)


@pytest.mark.parametrize("override", [np.ones((2, 2)), np.array([[2.0, 1.0]])])
def test_invalid_override_bounds_are_rejected(override):
    strategy = RandomUniformInitialization(np.array([[0.0, 1.0]]))
    with pytest.raises(ValueError, match="Override bounds|Lower bounds"):
        strategy.with_bounds_override(override)
    with pytest.raises(ValueError, match="Override bounds|Lower bounds"):
        strategy._generate_states_with_optional_bounds(1, override)


def test_temporary_bounds_are_restored_after_failed_batch():
    bounds = np.array([[0.0, 1.0]])
    strategy = RandomUniformInitialization(bounds, seed=3)
    override = np.array([[2.0, 2.0]])
    positions, _ = strategy._generate_states_with_optional_bounds(2, override)
    np.testing.assert_array_equal(positions, [[2], [2]])
    with pytest.raises(ValueError, match="positive integer"):
        strategy._generate_states_with_optional_bounds(0, override)
    np.testing.assert_array_equal(strategy.bounds, bounds)
    positions, _ = strategy._generate_states_with_optional_bounds(1)
    assert 0 <= positions[0][0] <= 1
