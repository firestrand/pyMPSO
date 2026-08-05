"""
PSO Framework - A modular Particle Swarm Optimization library for research.

This framework provides a comprehensive, modular implementation of various PSO algorithms
with pluggable components for velocity updates, topologies, boundary handling, and more.
"""

from .components.factory import (
    create_component,
    get_registered_component_names,
    register_component,
)
from .components.validation import validate_component_config
from .problem.ackley import AckleyFunction
from .problem.rastrigin import RastriginFunction
from .problem.rosenbrock import RosenbrockFunction
from .problem.sphere import SphereFunction
from .stopping.composite import AllStoppingCriteria, AnyStoppingCriteria
from .stopping.max_evaluations import MaxEvaluationsStopping
from .stopping.max_iterations import MaxIterationsStopping
from .stopping.target_fitness import TargetFitnessStopping
from .topology.global_topology import GlobalTopology
from .topology.random_topology import RandomTopology
from .topology.ring_topology import RingTopology
from .velocity.constriction import ConstrictionCoefficientVelocityUpdate
from .velocity.hypersphere import HypersphereVelocity
from .velocity.standard import StandardVelocityUpdate

__version__ = "0.1.0"
__author__ = "PSO Framework Contributors"
__license__ = "MIT"
# Import commonly used classes
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Component factory surface
    "create_component",
    "get_registered_component_names",
    "register_component",
    "validate_component_config",
    # Factory functions
    # Common problems
    "SphereFunction",
    "RastriginFunction",
    "RosenbrockFunction",
    "AckleyFunction",
    # Topologies
    "GlobalTopology",
    "RingTopology",
    "RandomTopology",
    # Velocity strategies
    "StandardVelocityUpdate",
    "ConstrictionCoefficientVelocityUpdate",
    "HypersphereVelocity",
    # Stopping criteria
    "MaxIterationsStopping",
    "MaxEvaluationsStopping",
    "TargetFitnessStopping",
    "AllStoppingCriteria",
    "AnyStoppingCriteria",
]
