from __future__ import annotations

import inspect
import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from typing import Any

from ..boundary.damped_reflection import DampedReflectionBoundaryHandler
from ..boundary.position_clamping import PositionClampingBoundaryHandler
from ..boundary.vectorized_damped_reflection import VectorizedDampedReflectionBoundaryHandler
from ..boundary.vectorized_position_clamping import VectorizedPositionClampingBoundaryHandler
from ..influence.fips import FIPSInfluenceStrategy
from ..influence.orthogonal_learning import OrthogonalLearningInfluence
from ..influence.single_best import SingleBestInfluence
from ..influence.vectorized_single_best import VectorizedSingleBestInfluence
from ..initialization.bounds_aware import BoundsAwareInitialization
from ..initialization.random_uniform import RandomUniformInitialization
from ..problem.ackley import AckleyFunction
from ..problem.griewank import GriewankFunction
from ..problem.rastrigin import RastriginFunction
from ..problem.rosenbrock import RosenbrockFunction
from ..problem.schaffer_f6 import SchafferF6Function
from ..problem.schwefel import SchwefelFunction
from ..problem.sphere import SphereFunction
from ..problem.step import StepFunction
from ..problem.styblinski_tang import StyblinskiTangFunction
from ..topology.config import GlobalTopologyConfig, RandomTopologyConfig, RingTopologyConfig
from ..topology.global_topology import GlobalTopology
from ..topology.random_topology import RandomTopology
from ..topology.ring_topology import RingTopology
from ..topology.vectorized_global import VectorizedGlobalTopology
from ..topology.vectorized_random import VectorizedRandomTopology
from ..topology.vectorized_ring import VectorizedRingTopology
from ..velocity.bare_bones import BareBonesPSOVelocityUpdate
from ..velocity.clpso import CLPSOVelocityUpdate
from ..velocity.config import (
    AdaptivePSOVelocityConfig,
    ConstrictionVelocityConfig,
    HypersphereVelocityConfig,
    QuantumPSOVelocityConfig,
    StandardVelocityConfig,
)
from ..velocity.constriction import ConstrictionCoefficientVelocityUpdate
from ..velocity.de_hybrid import DEHybridVelocityUpdate
from ..velocity.fips import FIPSVelocityUpdate
from ..velocity.hypersphere import HypersphereVelocity
from ..velocity.inertial_weight import InertiaWeightVelocityUpdate
from ..velocity.orthogonal_mutation import OrthogonalMutationVelocityUpdate
from ..velocity.standard import StandardVelocityUpdate
from ..velocity.vectorized_apso import VectorizedAdaptivePSOVelocityUpdate
from ..velocity.vectorized_constriction import VectorizedConstrictionVelocityUpdate
from ..velocity.vectorized_hypersphere import VectorizedHypersphereVelocity
from ..velocity.vectorized_qpso import VectorizedQPSOVelocityUpdate
from ..velocity.vectorized_standard import VectorizedStandardVelocityUpdate

logger = logging.getLogger(__name__)

Factory = Callable[..., Any]

_COMPONENT_REGISTRY: dict[str, dict[str, Factory]] = defaultdict(dict)
_VECTORIZED_REGISTRY: dict[str, dict[str, Factory]] = defaultdict(dict)
_CONFIG_REGISTRY: dict[str, dict[str, Factory]] = defaultdict(dict)


def register_component(component_type: str, name: str) -> Callable[[Factory], Factory]:
    """Register a component implementation under a namespaced component type."""

    normalized_type = component_type.strip().lower()
    normalized_name = name.strip().lower()

    def decorator(factory: Factory) -> Factory:
        type_map = _COMPONENT_REGISTRY[normalized_type]
        if normalized_name in type_map:
            raise ValueError(
                f"Duplicate registration for component '{normalized_type}:{normalized_name}'. "
                "A prior implementation already exists."
            )
        type_map[normalized_name] = factory
        return factory

    return decorator


def register_vectorized_component(component_type: str, name: str) -> Callable[[Factory], Factory]:
    normalized_type = component_type.strip().lower()
    normalized_name = name.strip().lower()

    def decorator(factory: Factory) -> Factory:
        type_map = _VECTORIZED_REGISTRY[normalized_type]
        if normalized_name in type_map:
            raise ValueError(f"Duplicate registration for vectorized component '{normalized_type}:{normalized_name}'.")
        type_map[normalized_name] = factory
        return factory

    return decorator


def register_typed_config(component_type: str, name: str) -> Callable[[Factory], Factory]:
    normalized_type = component_type.strip().lower()
    normalized_name = name.strip().lower()

    def decorator(factory: Factory) -> Factory:
        type_map = _CONFIG_REGISTRY[normalized_type]
        if normalized_name in type_map:
            raise ValueError(f"Duplicate registration for typed config '{normalized_type}:{normalized_name}'.")
        type_map[normalized_name] = factory
        return factory

    return decorator


def get_registered_component_names(component_type: str) -> list[str]:
    """Return sorted names for a registered component type."""
    base_type = component_type.strip().lower()
    names = set(_COMPONENT_REGISTRY.get(base_type, {}).keys())

    if base_type == "velocity_strategy":
        names.update(_VECTORIZED_REGISTRY.get(base_type, {}).keys())

    return sorted(names)


def supports_vectorization(config: dict[str, Any]) -> bool:
    """Check if all components in the configuration support vectorized execution."""
    return not get_vectorization_issues(config)


def get_vectorization_issues(config: dict[str, Any]) -> list[str]:
    """Return reasons a configuration is not currently supported by vectorized execution."""
    pso_params = config.get("pso_params", {})
    if not isinstance(pso_params, dict):
        raise TypeError("'pso_params' must be a dictionary.")

    # Defaults
    velocity = pso_params.get("velocity_strategy", "standard")
    topology = pso_params.get("topology", "global")
    influence = pso_params.get("influence_strategy", "single_best")
    boundary = pso_params.get("boundary_handler", "clamping")

    v_type = velocity.get("type") if isinstance(velocity, dict) else velocity
    if not isinstance(v_type, str):
        return ["velocity_strategy must be a string or dict with a string 'type' field."]

    t_type = topology.get("type") if isinstance(topology, dict) else topology
    if not isinstance(t_type, str):
        return ["topology must be a string or dict with a string 'type' field."]

    i_type = influence.get("type") if isinstance(influence, dict) else influence
    if not isinstance(i_type, str):
        return ["influence_strategy must be a string or dict with a string 'type' field."]

    b_type = boundary.get("type") if isinstance(boundary, dict) else boundary
    if not isinstance(b_type, str):
        return ["boundary_handler must be a string or dict with a string 'type' field."]

    vectorized_velocity = tuple(sorted(_VECTORIZED_REGISTRY.get("velocity_strategy", {}).keys()))
    supported_velocity = vectorized_velocity or ("standard", "constriction")

    topology_set = set(sorted(_VECTORIZED_REGISTRY.get("topology", {}).keys()))
    influence_set = set(sorted(_VECTORIZED_REGISTRY.get("influence_strategy", {}).keys()))
    boundary_set = set(sorted(_VECTORIZED_REGISTRY.get("boundary_handler", {}).keys()))
    issues: list[str] = []

    if v_type not in supported_velocity:
        issues.append(
            f"velocity_strategy '{v_type}' is not vectorized. "
            f"Supported velocity strategies: {', '.join(sorted(supported_velocity))}."
        )

    if t_type not in topology_set:
        issues.append(
            f"topology '{t_type}' is not vectorized. Supported topology: {', '.join(sorted(topology_set)) or 'global'}."
        )

    if i_type not in influence_set:
        issues.append(
            f"influence_strategy '{i_type}' is not vectorized. "
            f"Supported influence strategy: {', '.join(sorted(influence_set)) or 'single_best'}."
        )

    if b_type not in boundary_set:
        issues.append(
            f"boundary_handler '{b_type}' is not vectorized. "
            f"Supported boundary handlers: {', '.join(sorted(boundary_set)) or 'clamping, position_clamping'}."
        )

    return issues


def _register_default_components() -> None:
    """Register built-in implementations for first-class extension points."""

    register_component("problem", "sphere")(SphereFunction)
    register_component("problem", "rastrigin")(RastriginFunction)
    register_component("problem", "rosenbrock")(RosenbrockFunction)
    register_component("problem", "ackley")(AckleyFunction)
    register_component("problem", "schaffer_f6")(SchafferF6Function)
    register_component("problem", "griewank")(GriewankFunction)
    register_component("problem", "schwefel")(SchwefelFunction)
    register_component("problem", "step")(StepFunction)
    register_component("problem", "styblinski_tang")(StyblinskiTangFunction)

    register_component("topology", "global")(GlobalTopology)
    register_component("topology", "ring")(RingTopology)
    register_component("topology", "random")(RandomTopology)

    register_component("velocity_strategy", "standard")(StandardVelocityUpdate)
    register_component("velocity_strategy", "constriction")(ConstrictionCoefficientVelocityUpdate)
    register_component("velocity_strategy", "hypersphere")(HypersphereVelocity)
    register_component("velocity_strategy", "inertia_weight")(InertiaWeightVelocityUpdate)
    register_component("velocity_strategy", "fips")(FIPSVelocityUpdate)
    register_component("velocity_strategy", "bare_bones")(BareBonesPSOVelocityUpdate)
    register_component("velocity_strategy", "clpso")(CLPSOVelocityUpdate)
    register_component("velocity_strategy", "orthogonal_mutation")(OrthogonalMutationVelocityUpdate)
    register_component("velocity_strategy", "de_hybrid")(DEHybridVelocityUpdate)

    register_component("boundary_handler", "clamping")(PositionClampingBoundaryHandler)
    register_component("boundary_handler", "position_clamping")(PositionClampingBoundaryHandler)
    register_component("boundary_handler", "damped_reflection")(DampedReflectionBoundaryHandler)

    register_component("initialization_strategy", "random_uniform")(RandomUniformInitialization)
    register_component("initialization_strategy", "bounds_aware")(BoundsAwareInitialization)

    register_component("influence_strategy", "single_best")(SingleBestInfluence)
    register_component("influence_strategy", "fips")(FIPSInfluenceStrategy)
    register_component("influence_strategy", "orthogonal_learning")(OrthogonalLearningInfluence)

    # Vectorized components
    register_vectorized_component("velocity_strategy", "standard")(VectorizedStandardVelocityUpdate)
    register_typed_config("velocity_strategy", "standard")(StandardVelocityConfig)

    register_vectorized_component("velocity_strategy", "constriction")(VectorizedConstrictionVelocityUpdate)
    register_typed_config("velocity_strategy", "constriction")(ConstrictionVelocityConfig)
    register_vectorized_component("velocity_strategy", "qpso")(VectorizedQPSOVelocityUpdate)
    register_typed_config("velocity_strategy", "qpso")(QuantumPSOVelocityConfig)
    register_vectorized_component("velocity_strategy", "apso")(VectorizedAdaptivePSOVelocityUpdate)
    register_typed_config("velocity_strategy", "apso")(AdaptivePSOVelocityConfig)
    register_vectorized_component("velocity_strategy", "hypersphere")(VectorizedHypersphereVelocity)
    register_typed_config("velocity_strategy", "hypersphere")(HypersphereVelocityConfig)

    register_vectorized_component("topology", "global")(VectorizedGlobalTopology)
    register_typed_config("topology", "global")(GlobalTopologyConfig)
    register_vectorized_component("topology", "ring")(VectorizedRingTopology)
    register_typed_config("topology", "ring")(RingTopologyConfig)
    register_vectorized_component("topology", "random")(VectorizedRandomTopology)
    register_typed_config("topology", "random")(RandomTopologyConfig)

    register_vectorized_component("influence_strategy", "single_best")(VectorizedSingleBestInfluence)

    register_vectorized_component("boundary_handler", "clamping")(VectorizedPositionClampingBoundaryHandler)
    register_vectorized_component("boundary_handler", "position_clamping")(VectorizedPositionClampingBoundaryHandler)
    register_vectorized_component("boundary_handler", "damped_reflection")(VectorizedDampedReflectionBoundaryHandler)


def _filter_supported_kwargs(factory: Any, params: dict[str, Any]) -> dict[str, Any]:
    """Return keyword arguments supported by a factory constructor."""
    try:
        signature = inspect.signature(factory.__init__)
    except (TypeError, ValueError):
        # Be conservative for unusual callables.
        return {}

    params_kwargs = list(signature.parameters.values())
    accepts_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params_kwargs)
    if accepts_var_keyword:
        return {
            name: value
            for name, value in params.items()
            if name
            not in {
                "self",
            }
            and name
            in {
                param.name
                for param in params_kwargs
                if param.kind
                not in {
                    inspect.Parameter.VAR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                }
            }
        }

    allowed = {param.name for param in params_kwargs if param.name != "self"}
    return {name: value for name, value in params.items() if name in allowed}


def create_component(component_type: str, config: dict[str, Any], bounds_arr: Any | None = None, **kwargs: Any) -> Any:
    """Create a component instance from a validated config with ``type`` key."""
    if "type" not in config:
        raise ValueError(f"Missing 'type' in {component_type} configuration: {config}")

    component_name = config["type"]
    if not isinstance(component_name, str):
        raise TypeError(f"Invalid component type name for '{component_type}': expected string.")

    normalized_name = component_name.strip().lower()
    normalized_type = component_type.strip().lower()
    type_map = _COMPONENT_REGISTRY.get(normalized_type)
    if type_map is None or normalized_name not in type_map:
        raise ValueError(
            f"Unknown {component_type} '{component_name}'. Available options: {list(type_map.keys()) if type_map else []}"
        )

    params = {k: v for k, v in config.items() if k != "type"}
    params.update(kwargs)

    if normalized_type == "problem":
        if "dimensions" not in params:
            raise ValueError("Missing 'dimensions' for problem component")
        params = {
            "dimension": params.pop("dimensions"),
            "bias": params.pop("bias", 0.0),
            "bounds": bounds_arr,
            **params,
        }
        params["use_cec_shift"] = bool(params.get("use_cec_shift", False))
        if bounds_arr is not None:
            params["bounds"] = bounds_arr

    return type_map[normalized_name](**params)


def create_vectorized_component(
    component_type: str, config: dict[str, Any], _bounds_arr: Any | None = None, **_kwargs: Any
) -> Any:
    if "type" not in config:
        raise ValueError(f"Missing 'type' in {component_type} configuration: {config}")

    component_name = config["type"]
    normalized_name = component_name.strip().lower()
    normalized_type = component_type.strip().lower()

    type_map = _VECTORIZED_REGISTRY.get(normalized_type)
    if type_map is None or normalized_name not in type_map:
        raise ValueError(
            f"Unknown vectorized {component_type} '{component_name}'. Available options: {list(type_map.keys()) if type_map else []}"
        )

    params = {k: v for k, v in config.items() if k != "type"}
    params.update(_kwargs)
    filtered_params = _filter_supported_kwargs(type_map[normalized_name], params)

    return type_map[normalized_name](**filtered_params)


def create_typed_config(component_type: str, config: dict[str, Any], **kwargs: Any) -> Any:
    if "type" not in config:
        raise ValueError(f"Missing 'type' in {component_type} configuration: {config}")

    component_name = config["type"]
    normalized_name = component_name.strip().lower()
    normalized_type = component_type.strip().lower()

    type_map = _CONFIG_REGISTRY.get(normalized_type)
    if type_map is None or normalized_name not in type_map:
        # If there's no config class, we assume it doesn't need one (e.g. SingleBestInfluence)
        return None

    config_class = type_map[normalized_name]
    params = {k: v for k, v in config.items() if k != "type"}
    params.update(kwargs)

    if is_dataclass(config_class):
        # A variant that overrides only the strategy still inherits the common
        # `*_params` block, which may be tuned for a different strategy. Drop the
        # keys this config cannot accept, but say so: silently discarding a
        # hyperparameter would change the experiment without changing the report.
        accepted = {field.name for field in dataclass_fields(config_class) if field.init}
        dropped = sorted(set(params) - accepted)
        if dropped:
            logger.warning(
                "Ignoring %s parameter(s) %s not accepted by '%s' configuration.",
                normalized_type,
                ", ".join(dropped),
                normalized_name,
            )
            params = {name: value for name, value in params.items() if name in accepted}

    return config_class(**params)


_register_default_components()
