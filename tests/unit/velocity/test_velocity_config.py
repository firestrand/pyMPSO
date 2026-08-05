import numpy as np

from src.clamping.velocity import MaxNormVelocityClampingStrategy
from src.velocity.config import (
    AdaptivePSOVelocityConfig,
    ConstrictionVelocityConfig,
    QuantumPSOVelocityConfig,
    StandardVelocityConfig,
)


def test_standard_velocity_config_defaults():
    config = StandardVelocityConfig()
    assert config.c1 == 2.0
    assert config.c2 == 2.0
    assert config.bounds is None
    assert isinstance(config.rng, np.random.RandomState)
    assert config.velocity_clamping is not None


def test_standard_velocity_config_custom():
    bounds = np.array([[-10, 10], [-5, 5]])
    rng = np.random.RandomState(42)
    config = StandardVelocityConfig(c1=1.5, c2=1.5, bounds=bounds, rng=rng)

    assert config.c1 == 1.5
    assert config.c2 == 1.5
    np.testing.assert_array_equal(config.bounds, bounds)
    assert config.rng is rng
    assert config.velocity_clamping is not None


def test_constriction_velocity_config_defaults():
    config = ConstrictionVelocityConfig()
    assert config.phi1 == 2.05
    assert config.phi2 == 2.05
    assert config.chi is None
    assert config.bounds is None
    assert isinstance(config.rng, np.random.RandomState)


def test_quantum_velocity_config_defaults():
    config = QuantumPSOVelocityConfig()
    assert config.beta == 1.0
    assert config.bounds is None
    assert isinstance(config.rng, np.random.RandomState)


def test_adaptive_velocity_config_defaults():
    config = AdaptivePSOVelocityConfig()
    assert config.w_max == 0.9
    assert config.w_min == 0.4
    assert config.c1_max == 2.5
    assert config.c1_min == 0.5
    assert config.c2_max == 2.5
    assert config.c2_min == 0.5
    assert config.bounds is None
    assert isinstance(config.rng, np.random.RandomState)
    assert config.velocity_clamping is not None


def test_velocity_config_accepts_velocity_clamping():
    clamping = MaxNormVelocityClampingStrategy(max_velocity=1.0)
    config = StandardVelocityConfig(c1=1.5, c2=1.5, velocity_clamping=clamping)

    assert config.velocity_clamping is clamping
