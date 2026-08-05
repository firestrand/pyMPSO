"""
Registry module for the PSO algorithm components.

This module provides factory functions to create various components used by the PSO algorithm,
such as objective functions, topologies, velocity updates, and boundary handlers.
"""

from typing import Any

from .boundary.base import BoundaryHandler
from .boundary.damped_reflection import DampedReflectionBoundaryHandler
from .boundary.position_clamping import PositionClampingBoundaryHandler
from .clamping.base import VelocityClampingStrategy
from .clamping.velocity import (
    MaxNormVelocityClampingStrategy,
    NoClampingStrategy,
    RelativeBoundsVelocityClampingStrategy,
)
from .influence.base import NeighborhoodInfluenceStrategy
from .influence.fips import FIPSInfluenceStrategy
from .influence.orthogonal_learning import OrthogonalLearningInfluence
from .influence.single_best import SingleBestInfluence
from .initialization.base import InitializationStrategy
from .initialization.bounds_aware import BoundsAwareInitialization
from .initialization.random_uniform import RandomUniformInitialization
from .problem.ackley import AckleyFunction
from .problem.base import Problem
from .problem.griewank import GriewankFunction
from .problem.rastrigin import RastriginFunction
from .problem.rosenbrock import RosenbrockFunction
from .problem.schaffer_f6 import SchafferF6Function
from .problem.schwefel import SchwefelFunction
from .problem.sphere import SphereFunction
from .problem.step import StepFunction
from .problem.styblinski_tang import StyblinskiTangFunction
from .topology.base import NeighborhoodTopology
from .topology.global_topology import GlobalTopology
from .topology.random_topology import RandomTopology
from .topology.ring_topology import RingTopology
from .velocity.bare_bones import BareBonesPSOVelocityUpdate
from .velocity.base import VelocityUpdateStrategy
from .velocity.clpso import CLPSOVelocityUpdate
from .velocity.constriction import ConstrictionCoefficientVelocityUpdate
from .velocity.de_hybrid import DEHybridVelocityUpdate
from .velocity.fips import FIPSVelocityUpdate
from .velocity.hypersphere import HypersphereVelocity
from .velocity.inertial_weight import InertiaWeightVelocityUpdate
from .velocity.orthogonal_mutation import OrthogonalMutationVelocityUpdate
from .velocity.standard import StandardVelocityUpdate

# Use relative imports within the src package.
# from .topology.von_neumann import VonNeumannTopology
# Import other strategies if needed
# from .velocity.inertia_decay import InertiaDecayVelocityUpdate
# from .boundary.reflecting import ReflectingBoundaryHandler
# Import others if needed
# from .boundary.periodic import PeriodicBoundaryHandler
# Factory functions for objective functions


def get_objective_function(name: str, **kwargs) -> Problem:
    """
    Factory function to create an objective function.
    Args:
        name: Name of the objective function
        **kwargs: Additional parameters for the objective function (e.g., dimension, bias, bounds)
    Returns:
        An instance of the requested objective function (Problem subclass)
    Raises:
        ValueError: If the objective function is not found
    """
    objective_functions: dict[str, type[Problem]] = {
        "sphere": SphereFunction,
        "rastrigin": RastriginFunction,
        "rosenbrock": RosenbrockFunction,
        "ackley": AckleyFunction,
        "schaffer_f6": SchafferF6Function,
        "griewank": GriewankFunction,
        "schwefel": SchwefelFunction,
        "step": StepFunction,
        "styblinski_tang": StyblinskiTangFunction,
    }
    if name.lower() not in objective_functions:
        raise ValueError(f"Unknown objective function: {name}. Available options: {list(objective_functions.keys())}")
    return objective_functions[name.lower()](**kwargs)


def get_registered_component_names(component_type: str) -> list[str]:
    """Get supported implementation names for a component type."""
    component_map: dict[str, list[str]] = {
        "problem": [
            "sphere",
            "rastrigin",
            "rosenbrock",
            "ackley",
            "schaffer_f6",
            "griewank",
            "schwefel",
            "step",
            "styblinski_tang",
        ],
        "topology": ["global", "ring", "random"],
        "velocity_strategy": [
            "standard",
            "constriction",
            "hypersphere",
            "inertia_weight",
            "fips",
            "bare_bones",
            "clpso",
            "orthogonal_mutation",
            "de_hybrid",
            "qpso",
            "apso",
        ],
        "boundary_handler": ["clamping", "damped_reflection"],
        "initialization_strategy": ["random_uniform", "bounds_aware"],
        "influence_strategy": ["single_best", "fips", "orthogonal_learning"],
        "stopping_criteria": [
            "max_iterations",
            "max_evaluations",
            "target_fitness",
            "any",
            "all",
        ],
    }
    return component_map.get(component_type, [])


def get_stopping_criteria(name: str, **kwargs):
    """Create a stopping criterion instance."""
    from .stopping.max_evaluations import MaxEvaluationsStopping
    from .stopping.max_iterations import MaxIterationsStopping
    from .stopping.target_fitness import TargetFitnessStopping

    criterion_name = name.lower()
    if criterion_name == "max_iterations":
        return MaxIterationsStopping(**kwargs)
    if criterion_name == "max_evaluations":
        return MaxEvaluationsStopping(**kwargs)
    if criterion_name == "target_fitness":
        return TargetFitnessStopping(**kwargs)
    raise ValueError(
        f"Unknown stopping criterion: {name}. Available options: ['max_iterations', 'max_evaluations', 'target_fitness']"
    )


# Factory functions for topologies


def get_topology(name: str, **kwargs) -> NeighborhoodTopology:
    """
    Factory function to create a topology.
    Args:
        name: Name of the topology
        **kwargs: Additional parameters for the topology
    Returns:
        An instance of the requested topology
    Raises:
        ValueError: If the topology is not found
    """
    topologies: dict[str, type[NeighborhoodTopology]] = {
        "global": GlobalTopology,
        "ring": RingTopology,
        "random": RandomTopology,
        # Add others if needed and imported
        # 'von_neumann': VonNeumannTopology
    }
    if name.lower() not in topologies:
        raise ValueError(f"Unknown topology: {name}. Available options: {list(topologies.keys())}")
    return topologies[name.lower()](**kwargs)


# Factory functions for velocity clamping strategies


def get_velocity_clamping(name: str, **kwargs) -> VelocityClampingStrategy:
    """
    Factory function to create a velocity clamping strategy.
    Args:
        name: Name of the velocity clamping strategy
        **kwargs: Additional parameters for the velocity clamping strategy
    Returns:
        An instance of the requested velocity clamping strategy
    Raises:
        ValueError: If the velocity clamping strategy is not found
    """
    velocity_clampings: dict[str, type[VelocityClampingStrategy]] = {
        "none": NoClampingStrategy,
        "max_norm": MaxNormVelocityClampingStrategy,
        "relative_bounds": RelativeBoundsVelocityClampingStrategy,
    }
    if name.lower() not in velocity_clampings:
        raise ValueError(
            f"Unknown velocity clamping strategy: {name}. Available options: {list(velocity_clampings.keys())}"
        )
    return velocity_clampings[name.lower()](**kwargs)


# Utility function to create a velocity clamping strategy from a config object


def create_velocity_clamping_from_config(
    config: dict[str, Any] | None,
) -> VelocityClampingStrategy:
    """
    Create a velocity clamping strategy from a configuration dictionary.
    Args:
        config: Dictionary containing clamping configuration.
               Must include 'strategy' key and other parameters.
    Returns:
        A VelocityClampingStrategy instance.
    """
    if config is None:
        return NoClampingStrategy()
    if not isinstance(config, dict) or "strategy" not in config:
        raise ValueError("Velocity clamping config must be a dictionary with a 'strategy' key")
    strategy_name = config["strategy"]
    # Extract parameters, excluding the strategy name
    params = {k: v for k, v in config.items() if k != "strategy"}
    return get_velocity_clamping(strategy_name, **params)


# Factory functions for velocity update strategies


def get_velocity_update(name: str, **kwargs) -> VelocityUpdateStrategy:
    """
    Factory function to create a velocity update strategy.
    Args:
        name: Name of the velocity update strategy
        **kwargs: Additional parameters for the velocity update strategy.
            For standard velocity, you can specify:
            - clamping_strategy_name: Name of the velocity clamping strategy
            - clamping_strategy: Pass a pre - created clamping strategy instance
            - velocity_clamping: Dictionary with strategy configuration
            For constriction velocity, you can specify:
            - clamping_strategy_name: Name of the velocity clamping strategy
            - clamping_strategy: Pass a pre - created clamping strategy instance
            - velocity_clamping: Dictionary with strategy configuration
            - c1, c2: Cognitive and social coefficients (typically 2.05 each for SPSO 2007)
            - constriction: Optional pre - calculated constriction coefficient
            For hypersphere velocity, you can specify:
            - clamping_strategy_name: Name of the velocity clamping strategy
            - clamping_strategy: Pass a pre - created clamping strategy instance
            - velocity_clamping: Dictionary with strategy configuration
            - c1, c2: Cognitive and social coefficients (default: 0.5 + ln(2) each for SPSO 2011)
    Returns:
        An instance of the requested velocity update strategy
    Raises:
        ValueError: If the velocity update strategy is not found
    """
    velocity_updates: dict[str, type[VelocityUpdateStrategy]] = {
        "standard": StandardVelocityUpdate,
        "constriction": ConstrictionCoefficientVelocityUpdate,
        "hypersphere": HypersphereVelocity,
        "inertia_weight": InertiaWeightVelocityUpdate,
        "fips": FIPSVelocityUpdate,
        "bare_bones": BareBonesPSOVelocityUpdate,
        "clpso": CLPSOVelocityUpdate,
        "orthogonal_mutation": OrthogonalMutationVelocityUpdate,
        "de_hybrid": DEHybridVelocityUpdate,
        # Add others if needed
        # 'inertia_decay': InertiaDecayVelocityUpdate
    }
    if name.lower() not in velocity_updates:
        raise ValueError(
            f"Unknown velocity update strategy: {name}. Available options: {list(velocity_updates.keys())}"
        )
    # Handle clamping strategy for all velocity update types
    if name.lower() in [
        "standard",
        "constriction",
        "hypersphere",
        "inertia_weight",
        "fips",
        "bare_bones",
        "clpso",
        "orthogonal_mutation",
        "de_hybrid",
    ]:
        # Check if a velocity_clamping config object is provided
        if "velocity_clamping" in kwargs:
            clamping_config = kwargs.pop("velocity_clamping")
            clamping_strategy = create_velocity_clamping_from_config(clamping_config)
            strategy_class = velocity_updates[name.lower()]
            return strategy_class(clamping_strategy=clamping_strategy)  # type: ignore
        # Check if a clamping strategy instance is provided
        elif "clamping_strategy" in kwargs:
            clamping_strategy = kwargs.pop("clamping_strategy")
            strategy_class = velocity_updates[name.lower()]
            return strategy_class(clamping_strategy=clamping_strategy)  # type: ignore
        # Check if a clamping strategy name is provided
        elif "clamping_strategy_name" in kwargs:
            clamping_name = kwargs.pop("clamping_strategy_name")
            clamping_kwargs = {k.replace("clamping_", ""): v for k, v in kwargs.items() if k.startswith("clamping_")}
            # Remove the clamping prefixed kwargs to avoid passing them twice
            for k in list(kwargs.keys()):
                if k.startswith("clamping_"):
                    kwargs.pop(k)
            clamping_strategy = get_velocity_clamping(clamping_name, **clamping_kwargs)
            strategy_class = velocity_updates[name.lower()]
            return strategy_class(clamping_strategy=clamping_strategy)  # type: ignore
    # Default case - pass through all kwargs
    return velocity_updates[name.lower()](**kwargs)


# Factory functions for boundary handlers


def get_boundary_handler(name: str, **kwargs) -> BoundaryHandler:
    """
    Factory function to create a boundary handler.
    Args:
        name: Name of the boundary handler
        **kwargs: Additional parameters for the boundary handler
    Returns:
        An instance of the requested boundary handler
    Raises:
        ValueError: If the boundary handler is not found
    """
    boundary_handlers: dict[str, type[BoundaryHandler]] = {
        "clamping": PositionClampingBoundaryHandler,
        "damped_reflection": DampedReflectionBoundaryHandler,
        # 'reflecting': ReflectingBoundaryHandler,
        # Add others if needed
        # 'periodic': PeriodicBoundaryHandler
    }
    if name.lower() not in boundary_handlers:
        raise ValueError(f"Unknown boundary handler: {name}. Available options: {list(boundary_handlers.keys())}")
    return boundary_handlers[name.lower()](**kwargs)


# Factory functions for Initialization Strategy


def get_initialization_strategy(name: str, **kwargs) -> InitializationStrategy:
    """
    Factory function to create an initialization strategy.
    Args:
        name: Name of the initialization strategy
        **kwargs: Additional parameters for the initialization strategy
            - bounds: Search space bounds as numpy array of shape (dimensions, 2)
            - seed: Random seed for reproducibility
    Returns:
        An instance of the requested initialization strategy
    Raises:
        ValueError: If the initialization strategy is not found
    """
    initialization_strategies = {
        "random_uniform": RandomUniformInitialization,
        "bounds_aware": BoundsAwareInitialization,
    }
    if name.lower() not in initialization_strategies:
        raise ValueError(
            f"Unknown initialization strategy: {name}. Available options: {list(initialization_strategies.keys())}"
        )
    return initialization_strategies[name.lower()](**kwargs)


# Factory function for Particle Factory

# Particle factory pattern eliminated - use direct particle creation
# Factory functions for Influence Strategy


def get_influence_strategy(name: str, **kwargs) -> NeighborhoodInfluenceStrategy:
    """
    Factory function to create a neighborhood influence strategy.
    Args:
        name: Name of the strategy (e.g., 'single_best')
        **kwargs: Parameters for the strategy (if any)
    Returns:
        An instance of the NeighborhoodInfluenceStrategy.
    Raises:
        ValueError: If the strategy name is unknown.
    """
    strategies: dict[str, type[NeighborhoodInfluenceStrategy]] = {
        "single_best": SingleBestInfluence,
        "fips": FIPSInfluenceStrategy,
        "orthogonal_learning": OrthogonalLearningInfluence,
        # Add other strategies here when implemented
    }
    if name.lower() not in strategies:
        raise ValueError(f"Unknown influence strategy: {name}. Available: {list(strategies.keys())}")
    return strategies[name.lower()](**kwargs)


# Factory functions for Position Update Strategy

# Position strategy uses direct state update; no dedicated strategy object.
