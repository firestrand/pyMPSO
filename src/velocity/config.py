from dataclasses import dataclass, field

import numpy as np

from ..clamping.base import VelocityClampingStrategy
from ..clamping.velocity import NoClampingStrategy
from ..constants import SPSO2011_C1, SPSO2011_C2, SPSO2011_W


@dataclass
class VelocityConfig:
    """Base configuration for velocity strategies."""

    bounds: np.ndarray | None = None
    rng: np.random.RandomState = field(default_factory=np.random.RandomState)
    velocity_clamping: VelocityClampingStrategy | None = field(default_factory=NoClampingStrategy)


@dataclass
class StandardVelocityConfig(VelocityConfig):
    """Configuration for Standard Velocity update."""

    c1: float = 2.0
    c2: float = 2.0


@dataclass
class ConstrictionVelocityConfig(VelocityConfig):
    """Configuration for Constriction Velocity update."""

    phi1: float = 2.05
    phi2: float = 2.05
    chi: float | None = None


@dataclass
class HypersphereVelocityConfig(VelocityConfig):
    """Configuration for the SPSO 2011 hypersphere velocity update."""

    c1: float = SPSO2011_C1
    c2: float = SPSO2011_C2
    w: float = SPSO2011_W
    # SPSO 2011 `param.distrib`: 0 samples the radius uniformly, -1 samples
    # uniformly by volume (r = u^(1/D)). The C reference defaults to 0.
    distrib: int = 0


@dataclass
class QuantumPSOVelocityConfig(VelocityConfig):
    """Configuration for Quantum-behaved PSO velocity update."""

    beta: float = 1.0
    beta_min: float | None = None
    beta_max: float | None = None


@dataclass
class AdaptivePSOVelocityConfig(VelocityConfig):
    """Configuration for Adaptive PSO (APSO)-style scheduling."""

    w_max: float = 0.9
    w_min: float = 0.4
    c1_max: float = 2.5
    c1_min: float = 0.5
    c2_max: float = 2.5
    c2_min: float = 0.5
