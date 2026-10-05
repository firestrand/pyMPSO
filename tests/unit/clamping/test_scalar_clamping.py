"""Clamping and reflection contracts for scalar particles."""

import numpy as np
import pytest

from src.boundary.damped_reflection import DampedReflectionBoundaryHandler
from src.boundary.position_clamping import PositionClampingBoundaryHandler
from src.clamping.base import PositionClampingStrategy
from src.clamping.position import (
    BasicPositionClampingStrategy,
    VelocityResetPositionClampingStrategy,
    apply_position_clamping,
)
from src.clamping.velocity import (
    MaxNormVelocityClampingStrategy,
    NoClampingStrategy,
    RelativeBoundsVelocityClampingStrategy,
)
from src.particles.standard import StandardParticle


@pytest.mark.parametrize("reset_velocity", [False, True])
def test_position_boundary_clamps_only_outside_coordinates(reset_velocity):
    particle = StandardParticle(np.array([-2.0, 1.0, 4.0]), np.array([-4.0, 5.0, 6.0]))
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0], [0.0, 3.0]])
    PositionClampingBoundaryHandler(reset_velocity).apply(particle, bounds)
    np.testing.assert_array_equal(particle.position, [-1, 1, 3])
    np.testing.assert_array_equal(particle.velocity, [0, 5, 0] if reset_velocity else [-4, 5, 6])


def test_basic_clamping_preserves_velocity_even_when_reset_requested():
    particle = StandardParticle(np.array([-2.0, 4.0]), np.array([-4.0, 6.0]))
    bounds = np.array([[-1.0, 1.0], [0.0, 3.0]])
    apply_position_clamping(particle, bounds, BasicPositionClampingStrategy(), reset_velocity=True, hyperparams={})
    np.testing.assert_array_equal(particle.position, [-1, 3])
    np.testing.assert_array_equal(particle.velocity, [-4, 6])
    position, mask = VelocityResetPositionClampingStrategy().clamp(np.array([-1.0, 4.0]), bounds, {})
    np.testing.assert_array_equal(position, [-1, 3])
    np.testing.assert_array_equal(mask, [False, True])


def test_damped_reflection_changes_only_violating_coordinates_and_copies_inputs():
    position = np.array([-2.0, 1.0, 4.0])
    velocity = np.array([-4.0, 5.0, 6.0])
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0], [0.0, 3.0]])
    handler = DampedReflectionBoundaryHandler(bounds, damping_factor=-0.25)
    result, reflected = handler.enforce_bounds(position, velocity)
    np.testing.assert_array_equal(result, [-1, 1, 3])
    np.testing.assert_array_equal(reflected, [1, 5, -1.5])
    np.testing.assert_array_equal(position, [-2, 1, 4])
    np.testing.assert_array_equal(velocity, [-4, 5, 6])
    _, absent_velocity = handler.enforce_bounds(position)
    assert absent_velocity is None
    unbounded = DampedReflectionBoundaryHandler()
    same_position, same_velocity = unbounded.enforce_bounds(position, velocity)
    assert same_position is position and same_velocity is velocity
    particle = StandardParticle(position, velocity)
    handler.apply(particle, bounds)
    np.testing.assert_array_equal(particle.position, result)
    np.testing.assert_array_equal(particle.velocity, reflected)


@pytest.mark.parametrize(
    "fixed,params,expected",
    [(2.0, {"max_velocity": 1.0}, [-2, 1, 2]), (None, {"max_velocity": 3.0}, [-3, 1, 3]), (None, {}, [-4, 1, 8])],
)
def test_absolute_velocity_clamping_fixed_value_precedes_hyperparameters(fixed, params, expected):
    values = np.array([-4.0, 1.0, 8.0])
    result = MaxNormVelocityClampingStrategy(fixed).clamp(values, np.zeros((3, 2)), params)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(values, [-4, 1, 8])


def test_relative_velocity_clamping_uses_each_dimension_range_and_override():
    values = np.array([-4.0, 8.0])
    bounds = np.array([[0.0, 10.0], [-20.0, 20.0]])
    strategy = RelativeBoundsVelocityClampingStrategy(0.1)
    np.testing.assert_array_equal(strategy.clamp(values, bounds, {}), [-1, 4])
    np.testing.assert_array_equal(strategy.clamp(values, bounds, {"max_velocity_ratio": 0.2}), [-2, 8])
    assert NoClampingStrategy().clamp(values, bounds, {}) is values


class ClipPlugin(PositionClampingStrategy):
    """A plugin that can return an explicit velocity reset mask."""

    def __init__(self, return_mask: bool):
        self.return_mask = return_mask

    def clamp(
        self, values: np.ndarray, bounds: np.ndarray, hyperparams: dict
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        clamped = np.clip(values, bounds[:, 0], bounds[:, 1])
        clamped *= hyperparams.get("scale", 1.0)
        if self.return_mask:
            return clamped, clamped != values
        return clamped


@pytest.mark.parametrize("return_mask", [False, True])
@pytest.mark.parametrize("reset_velocity", [False, True])
def test_custom_clamping_plugin_array_and_mask_results(return_mask, reset_velocity):
    particle = StandardParticle(np.array([-2.0, 0.5]), np.array([-4.0, 6.0]))
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    apply_position_clamping(particle, bounds, ClipPlugin(return_mask), reset_velocity, {"scale": 1.0})
    np.testing.assert_array_equal(particle.position, [-1, 0.5])
    np.testing.assert_array_equal(particle.velocity, [0, 6] if return_mask and reset_velocity else [-4, 6])
