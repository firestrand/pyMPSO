from dataclasses import dataclass, field

import numpy as np


@dataclass
class TopologyConfig:
    """Base configuration for topology strategies."""

    # Stochastic topologies must draw from the run's generator, otherwise a
    # configured random_seed does not make the run reproducible.
    rng: np.random.RandomState = field(default_factory=np.random.RandomState)


@dataclass
class GlobalTopologyConfig(TopologyConfig):
    """Configuration for Global Topology."""

    pass


@dataclass
class RingTopologyConfig(TopologyConfig):
    """Configuration for Ring (lbest) Topology."""

    k: int = 1


@dataclass
class RandomTopologyConfig(TopologyConfig):
    """Configuration for Random Topology."""

    k: int = 3
    rebuild_probability: float = 1.0
