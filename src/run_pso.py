#!/usr/bin/env python3
"""
Main entry point and helpers for running PSO configurations.

This module loads JSON or YAML configuration files and runs the PSO algorithm
with the specified components and parameters.
"""

import hashlib
import importlib
import json
import logging
import time
from copy import deepcopy  # Needed for merging configs
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from .components.factory import create_component as factory_create_component
from .components.factory import (
    create_typed_config,
    create_vectorized_component,
    get_registered_component_names,
    get_vectorization_issues,
)
from .execution.parallel import RayExecutor
from .execution.protocols import ExecutionResult, RunExecutor
from .execution.serial import SerialExecutor
from .planning.domain import BenchmarkTask, benchmark_task_key
from .position.standard import StandardPositionUpdate
from .pso_orchestrator import build_benchmark_plan, execute_plan, with_execution_strategy

# Import stopping-criteria factory for execution policy assembly.
from .registry import create_velocity_clamping_from_config, get_stopping_criteria
from .stopping.base import StoppingCriteria
from .stopping.composite import AllStoppingCriteria, AnyStoppingCriteria
from .utils.serialization import make_json_serializable
from .variants.domain import (
    DEFAULT_VARIANT_NAME,
    VariantDescriptor,
    extract_pso_variants,
    filter_pso_variants,
)
from .vectorized_pso_algorithm import VectorizedPSOAlgorithm

logger = logging.getLogger(__name__)

_PSO_COMPONENT_KEYS: tuple[str, ...] = (
    "topology",
    "initialization_strategy",
    "influence_strategy",
    "velocity_strategy",
    "boundary_handler",
)

_PSO_COMPONENT_DEFAULT_TYPES: dict[str, str] = {
    "topology": "global",
    "initialization_strategy": "random_uniform",
    "influence_strategy": "single_best",
    "velocity_strategy": "standard",
    "boundary_handler": "clamping",
}
_ALLOWED_PSO_PARAM_KEYS = (
    set(_PSO_COMPONENT_KEYS) | {f"{key}_params" for key in _PSO_COMPONENT_KEYS} | {"velocity_clamping"}
)


def _get_valid_component_names(component_type: str) -> list[str]:
    return get_registered_component_names(component_type)


def _normalize_component_config(raw_config: str | dict[str, Any] | None, default_type: str) -> dict[str, Any]:
    """Normalize a component entry into a dict with an explicit ``type`` key."""
    if raw_config is None:
        return {"type": default_type}
    if isinstance(raw_config, str):
        return {"type": raw_config}
    if isinstance(raw_config, dict):
        if "type" not in raw_config:
            raise TypeError("Component configuration dictionary must include a 'type' key.")
        return dict(raw_config)
    raise TypeError(f"Unsupported component config type: {type(raw_config)!r}")


def _normalize_topology_params(
    topology_config: dict[str, Any],
) -> dict[str, Any]:
    """Normalize topology config for random-topology defaults."""
    normalized = dict(topology_config)
    topology_type = normalized.get("type", "random")

    if topology_type != "random":
        # Non-random topologies do not use topology-rebuild parameters.
        return normalized

    # Keep default topology rebuild behavior explicit.
    if "rebuild_probability" not in normalized:
        normalized["rebuild_probability"] = 1.0

    return normalized


def _extract_max_iterations(stopping_config: Any) -> int | None:
    """Extract the maximum iteration budget for APSO-style schedules."""
    if isinstance(stopping_config, dict):
        if stopping_config.get("type") == "max_iterations":
            max_iterations = stopping_config.get("max_iterations")
            if isinstance(max_iterations, int) and max_iterations > 0:
                return max_iterations
            return None

        candidates: list[int] = []
        for key in ("any", "all"):
            nested = stopping_config.get(key)
            if isinstance(nested, list):
                for nested_item in nested:
                    candidate = _extract_max_iterations(nested_item)
                    if candidate is not None:
                        candidates.append(candidate)
        return max(candidates, default=None)

    if isinstance(stopping_config, list):
        values: list[int] = []
        for item in stopping_config:
            candidate = _extract_max_iterations(item)
            if candidate is not None:
                values.append(candidate)
        return max(values, default=None)

    return None


def _normalize_parallel_config(
    run_config: dict[str, Any],
) -> dict[str, int | bool | None]:
    """Normalize benchmark-level parallel execution settings."""
    parallel_config = run_config.get("parallel", {})
    if parallel_config is None:
        return {"enabled": False, "num_cpus": None}
    if not isinstance(parallel_config, dict):
        raise TypeError("'parallel' must be a dictionary.")
    enabled = bool(parallel_config.get("enabled", False))
    num_cpus = parallel_config.get("num_cpus")
    if num_cpus is None:
        return {"enabled": enabled, "num_cpus": None}
    if not isinstance(num_cpus, int) or num_cpus <= 0:
        raise ValueError("'parallel.num_cpus' must be a positive integer.")
    return {"enabled": enabled, "num_cpus": num_cpus}


def _derive_run_seed(base_seed: int, task_name: str, run_index: int) -> int:
    """Derive a deterministic run seed from a base seed."""
    salt = f"{base_seed}:{task_name}:{run_index}".encode()
    digest = hashlib.blake2b(salt, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def _run_single_benchmark_execution(task: BenchmarkTask) -> ExecutionResult:
    """Execute one benchmark run from a fully materialized run payload."""
    task_name = task.task_name
    config_file = task.config_file
    task_problem_config = task.problem
    task_swarm_config = task.swarm
    task_pso_params = task.pso_params
    task_stop_criteria_config = task.stopping_criteria
    task_run_settings = task.run_config
    task_verbose = task_run_settings.get("verbose", False)
    run_seed = task.random_seed
    run_index = task.task_run_index

    run_rng = np.random.RandomState(run_seed if run_seed is not None else None)

    task_result: dict[str, Any]
    num_particles = task_swarm_config.get("num_particles", 30)
    try:
        bounds_arr = create_bounds(task_problem_config)
        problem = create_component("problem", task_problem_config, bounds_arr=bounds_arr)
        stopping_criteria = create_stopping_criteria(task_stop_criteria_config)
        max_iterations = _extract_max_iterations(task_stop_criteria_config)

        vectorization_issues = get_vectorization_issues({"pso_params": task_pso_params})
        if vectorization_issues:
            issues_text = "; ".join(vectorization_issues)
            raise ValueError(
                "Current configuration is not supported by the vectorized execution engine. "
                f"Non-vectorized execution has been removed. {issues_text}"
            )

        initialization_strategy = create_component(
            "initialization_strategy",
            _build_component_config(task_pso_params, "initialization_strategy", "random_uniform"),
            bounds=bounds_arr,
            seed=run_seed,
            rng=run_rng,
        )

        topology_config_dict = _build_component_config(task_pso_params, "topology", "global")
        if "rng" not in topology_config_dict:
            topology_config_dict["rng"] = run_rng
        topology = create_vectorized_component("topology", topology_config_dict)
        topology_config = create_typed_config("topology", topology_config_dict)

        influence_strategy = create_vectorized_component(
            "influence_strategy",
            _build_component_config(task_pso_params, "influence_strategy", "single_best"),
        )

        velocity_config_dict = _build_component_config(task_pso_params, "velocity_strategy", "standard")
        velocity_config_dict.setdefault("bounds", bounds_arr)
        if "rng" not in velocity_config_dict:
            velocity_config_dict["rng"] = run_rng
        velocity_strategy = create_vectorized_component("velocity_strategy", velocity_config_dict)
        velocity_config = create_typed_config("velocity_strategy", velocity_config_dict)

        boundary_strategy = _build_component_config(
            task_pso_params,
            "boundary_handler",
            "clamping",
        )
        boundary_handler = create_vectorized_component("boundary_handler", boundary_strategy)

        position_strategy = StandardPositionUpdate()

        pso = VectorizedPSOAlgorithm(
            problem=problem,
            stopping_criteria=stopping_criteria,
            initialization_strategy=initialization_strategy,
            topology=topology,
            topology_config=topology_config,
            influence_strategy=influence_strategy,
            velocity_strategy=velocity_strategy,
            velocity_config=velocity_config,
            position_strategy=position_strategy,
            boundary_handler=boundary_handler,
            num_particles=num_particles,
            max_iterations=max_iterations,
            verbose=task_verbose,
        )

        start_time = time.time()
        results = pso.run()
        total_time = time.time() - start_time
        results["total_wall_time"] = total_time
        results["run_index"] = run_index
        results["seed_used"] = run_seed
        results["config_file"] = config_file
        results["benchmark_task"] = task_name
        results["problem_type"] = task_problem_config["type"]
        results["problem_dimension"] = task_problem_config["dimensions"]
        results["num_particles"] = num_particles
        task_result = results
    except Exception as exc:
        task_result = {
            "benchmark_task": task_name,
            "run_index": run_index,
            "seed_used": run_seed,
            "config_file": config_file,
            "problem_type": task_problem_config.get("type"),
            "problem_dimension": task_problem_config.get("dimensions"),
            "num_particles": task_swarm_config.get("num_particles", 30),
            "error": str(exc),
        }

    return cast(ExecutionResult, task_result)


def _build_component_config(
    pso_params: dict[str, Any],
    component_key: str,
    default_type: str,
) -> dict[str, Any]:
    """Build a normalized component config by merging base config and params blocks."""
    base_config = _normalize_component_config(
        pso_params.get(component_key),
        default_type=default_type,
    )
    params_key = f"{component_key}_params"
    params = pso_params.get(params_key, {})
    if params is not None and not isinstance(params, dict):
        raise TypeError(f"Expected '{params_key}' to be a dictionary, got {type(params)!r}.")

    merged = dict(base_config)
    merged.update(params)

    if component_key == "topology":
        merged = _normalize_topology_params(merged)

    if component_key == "velocity_strategy":
        velocity_clamping = pso_params.get("velocity_clamping")
        if velocity_clamping is not None and "velocity_clamping" not in merged:
            if not isinstance(velocity_clamping, dict):
                raise TypeError("'velocity_clamping' must be a dictionary if provided.")
            merged["velocity_clamping"] = create_velocity_clamping_from_config(velocity_clamping)
        elif "velocity_clamping" in merged and isinstance(merged["velocity_clamping"], dict):
            merged["velocity_clamping"] = create_velocity_clamping_from_config(merged["velocity_clamping"])

    return merged


def _build_task_plan(
    problem_name: str,
    problem_settings: dict[str, Any],
    common_swarm: dict[str, Any],
    common_pso_params: dict[str, Any],
    common_run_config: dict[str, Any],
    benchmark_seed: int | None,
    config_file: str,
    variant: VariantDescriptor,
) -> tuple[str, dict[str, Any], list[BenchmarkTask]]:
    """Build task metadata and concrete run specs for one benchmark problem."""
    dim = problem_settings.get("dimensions")
    if dim is None:
        raise ValueError(f"Problem '{problem_name}' is missing mandatory 'dimensions' key.")

    variant_name = variant.name
    task_name = benchmark_task_key(
        problem_name,
        dim,
        None if variant_name == DEFAULT_VARIANT_NAME else variant_name,
    )
    task_config = {
        "config_file": config_file,
        "problem": {"type": problem_name, "dimensions": dim},
        "swarm": {
            **deepcopy(common_swarm),
            **variant.swarm,
        },
        "pso_params": {
            **deepcopy(common_pso_params),
            **variant.pso_params,
        },
        "stopping_criteria": None,  # Placeholder
        "run_config": {
            **deepcopy(common_run_config),
            **variant.run_config,
        },
    }

    reserved_task_keys = {
        "dimensions",
        "bounds",
        "problem",
        "swarm",
        "pso_params",
        "stopping_criteria",
        "run_config",
    }
    task_config["problem"].update({k: deepcopy(v) for k, v in problem_settings.items() if k not in reserved_task_keys})
    task_config["problem"].update(problem_settings.get("problem", {}))
    task_config["problem"]["bounds"] = problem_settings.get("bounds")
    task_config["swarm"].update(problem_settings.get("swarm", {}))
    task_config["pso_params"].update(problem_settings.get("pso_params", {}))
    task_config["stopping_criteria"] = problem_settings.get("stopping_criteria")
    task_config["run_config"].update(problem_settings.get("run_config", {}))

    task_problem_config = task_config["problem"]
    task_swarm_config = task_config["swarm"]
    task_pso_params = task_config["pso_params"]
    task_stop_criteria_config = task_config["stopping_criteria"]
    task_run_settings = task_config["run_config"]

    if not task_problem_config.get("bounds") or not task_stop_criteria_config:
        raise ValueError(f"Task '{task_name}' is missing required 'bounds' or 'stopping_criteria'.")

    # Fail fast on task-level component/configuration issues before launching runs.
    bounds_arr = create_bounds(task_problem_config)
    create_stopping_criteria(task_stop_criteria_config)
    create_component("problem", task_problem_config, bounds_arr=bounds_arr)

    num_runs = task_run_settings.get("num_runs", 1)
    if not isinstance(num_runs, int) or num_runs <= 0:
        raise ValueError(f"'num_runs' for task '{task_name}' must be a positive integer.")

    task_seed = task_run_settings.get("random_seed", benchmark_seed)
    task_order_entry = {
        "num_runs_requested": num_runs,
        "config_file": config_file,
        "swarm": task_swarm_config,
        "problem": task_problem_config,
        "dimensions": dim,
        "pso_variant": variant_name,
    }

    run_specs: list[BenchmarkTask] = []
    for run_index in range(num_runs):
        if num_runs > 1 and task_seed is not None:
            run_seed = _derive_run_seed(task_seed, task_name, run_index)
        else:
            run_seed = task_seed

        run_specs.append(
            BenchmarkTask(
                config_file=config_file,
                task_name=task_name,
                problem_name=problem_name,
                variant_name=None if variant_name == DEFAULT_VARIANT_NAME else variant_name,
                dimensions=dim,
                problem=deepcopy(task_problem_config),
                swarm=deepcopy(task_swarm_config),
                pso_params=deepcopy(task_pso_params),
                stopping_criteria=deepcopy(task_stop_criteria_config),
                run_config=deepcopy(task_run_settings),
                random_seed=run_seed,
                task_run_index=run_index,
            )
        )

    return task_name, task_order_entry, run_specs


def load_class(class_path: str) -> type:
    """
    Dynamically load a class from its path.

    Args:
        class_path: A string of the form 'module.submodule.ClassName'

    Returns:
        The loaded class
    """
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import {class_path}: {e}") from e


def create_component(
    component_type: str, config: dict[str, Any], bounds_arr: np.ndarray | None = None, **kwargs
) -> Any:
    """Create a component instance through the modular component factory."""
    return factory_create_component(component_type=component_type, config=config, bounds_arr=bounds_arr, **kwargs)


def create_bounds(config: dict[str, Any]) -> np.ndarray | None:
    """
    Create bounds array from config.

    Args:
        config: Configuration dictionary with bounds information

    Returns:
        NumPy array of shape (dimensions, 2) with bounds for each dimension
    """
    dimensions = config["dimensions"]

    if "bounds" in config:
        bounds_config = config["bounds"]

        if isinstance(bounds_config, dict):
            # Uniform bounds for all dimensions
            if "min" in bounds_config and "max" in bounds_config:
                min_val = bounds_config["min"]
                max_val = bounds_config["max"]
                return np.array([[min_val, max_val]] * dimensions)
            else:
                raise ValueError("Bounds dict must contain 'min' and 'max' keys")

        elif isinstance(bounds_config, list):
            # Per-dimension bounds
            if len(bounds_config) != dimensions:
                raise ValueError(f"Bounds list length ({len(bounds_config)}) must match dimensions ({dimensions})")

            bounds = np.zeros((dimensions, 2))
            for i, bound in enumerate(bounds_config):
                if isinstance(bound, list) and len(bound) == 2:
                    bounds[i, 0] = bound[0]  # min
                    bounds[i, 1] = bound[1]  # max
                elif isinstance(bound, dict) and "min" in bound and "max" in bound:
                    bounds[i, 0] = bound["min"]
                    bounds[i, 1] = bound["max"]
                else:
                    raise ValueError(f"Invalid bounds format at index {i}")

            return bounds
        else:
            raise ValueError("Bounds must be a dict or list")
    else:
        # Default bounds are handled by the problem implementations
        return None


def _validate_configured_component(
    component_type: str,
    component_value: Any,
    config_scope: str,
) -> None:
    """Validate a component selection in a typed config block."""
    if component_value is None:
        return

    try:
        component_config = _ensure_dict_config(
            component_value,
            default_type=_PSO_COMPONENT_DEFAULT_TYPES.get(component_type, "standard"),
        )
    except TypeError as exc:
        raise TypeError(f"Invalid format for '{component_type}' in {config_scope}: {exc}") from exc

    component_name = component_config.get("type")
    if not component_name:
        raise ValueError(f"Component configuration for '{component_type}' in {config_scope} is missing a 'type' key.")

    valid_component_names = _get_valid_component_names(component_type)
    if valid_component_names and component_name not in valid_component_names:
        raise ValueError(
            f"Invalid component type '{component_name}' specified for '{component_type}' in {config_scope}. "
            f"Valid options: {valid_component_names}"
        )


def _validate_component_block(
    pso_params: dict[str, Any],
    context: str,
) -> None:
    """Validate a pso_params block in a config section."""
    unknown_keys = set(pso_params) - _ALLOWED_PSO_PARAM_KEYS
    if unknown_keys:
        allowed_keys = ", ".join(sorted(_ALLOWED_PSO_PARAM_KEYS))
        raise TypeError(
            f"Unsupported keys in pso params for {context}. Allowed keys are: {allowed_keys}. "
            f"Unknown keys: {sorted(unknown_keys)}"
        )

    for comp_key in _PSO_COMPONENT_KEYS:
        if comp_key in pso_params:
            _validate_configured_component(
                component_type=comp_key,
                component_value=pso_params[comp_key],
                config_scope=context,
            )

        params_key = f"{comp_key}_params"
        if params_key in pso_params and not isinstance(pso_params[params_key], dict):
            raise TypeError(f"Invalid format for '{params_key}' in {context}: expected a dictionary.")

    if "velocity_clamping" in pso_params and not isinstance(pso_params["velocity_clamping"], dict):
        raise TypeError(f"'velocity_clamping' in {context} must be a dictionary.")


def create_stopping_criteria(config: Any) -> StoppingCriteria:
    """
    Create stopping criteria from config, handling single, composite (any/all dict),
    or list-based configurations.

    Args:
        config: Configuration for stopping criteria. Can be:
                - A dict for a single criterion (must contain 'type').
                - A dict for composite criteria (must contain 'any' or 'all' key).
                - A list of dicts, implicitly treated as an 'any' composite.

    Returns:
        A StoppingCriteria instance
    """
    if isinstance(config, dict):
        # Handle explicit composite criteria ('any' or 'all') first.
        if "any" in config or "all" in config:
            if "type" in config:
                raise ValueError("Composite stopping criteria should not mix 'type' with 'any/all'.")
            if "any" in config:
                criteria_configs = config["any"]
                composite_type = "any"
            else:
                criteria_configs = config["all"]
                composite_type = "all"
            criteria = [create_stopping_criteria(criteria_config) for criteria_config in criteria_configs]
            if composite_type == "any":
                return AnyStoppingCriteria(criteria=criteria)
            return AllStoppingCriteria(criteria=criteria)

        if "type" not in config:
            raise ValueError(
                "Invalid dictionary-based stopping criteria configuration. Must contain 'type', 'any', or 'all'."
            )

        criterion_name = config["type"]
        # Use dynamic loading for stopping criteria base types for now
        params = {k: v for k, v in config.items() if k != "type"}

        if criterion_name in {"max_iterations", "max_evaluations", "target_fitness"}:
            return get_stopping_criteria(criterion_name, **params)

        raise ValueError(f"Unknown stopping_criteria type: {config.get('type')}")

    elif isinstance(config, list):
        if not config:
            raise ValueError("Stopping criteria list cannot be empty.")

        # If list has one item, treat it as a single criterion config
        if len(config) == 1:
            return create_stopping_criteria(config[0])

        # If list has multiple items, treat as implicit 'any' composite
        else:
            criteria = [create_stopping_criteria(criteria_config) for criteria_config in config]
            return AnyStoppingCriteria(criteria=criteria)

    else:
        raise TypeError("Invalid type for stopping criteria configuration. Expected dict or list.")


def _ensure_dict_config(
    config_val: Any,
    default_type: str,
) -> dict[str, Any]:
    """Convert type shorthand and dictionary forms to normalized dict form."""
    if isinstance(config_val, str):
        return {"type": config_val}
    elif isinstance(config_val, dict):
        if "type" not in config_val:
            raise TypeError("Component configuration dictionary must include a 'type' key.")
        return config_val
    elif config_val is None:
        # Handle case where config is explicitly null or missing, use default
        return {"type": default_type}
    else:
        raise TypeError(f"Unexpected config format: {config_val}")


def load_config(config_path: str) -> dict[str, Any]:
    """Load a configuration file as YAML or JSON based on file extension/content."""
    path = Path(config_path)
    with path.open("r") as f:
        if path.suffix.lower() in {".json"}:
            return json.load(f)
        if path.suffix.lower() in {".yml", ".yaml"}:
            return yaml.safe_load(f)
        raise ValueError("Unsupported configuration extension. Use .yaml, .yml, or .json.")


def _validate_configuration(
    config: dict[str, Any],
    config_path: str,
    *,
    selected_variants: set[str] | None = None,
    excluded_variants: set[str] | None = None,
    validate_vectorized: bool = False,
) -> None:
    """Perform validation checks on the loaded configuration dictionary."""
    if "problems_to_benchmark" not in config or not isinstance(config["problems_to_benchmark"], dict):
        raise ValueError(f"Config '{config_path}' must contain a top-level 'problems_to_benchmark' dictionary.")

    problems_to_run = config["problems_to_benchmark"]
    if not problems_to_run:
        raise ValueError("'problems_to_benchmark' dictionary cannot be empty.")

    common_swarm = config.get("common_swarm", {})
    common_pso_params = config.get("common_pso_params", {})
    common_run_config = config.get("common_run_config", {})

    # --- Validate Common Sections ---
    if (
        "num_particles" in common_swarm
        and not isinstance(common_swarm["num_particles"], int)
        or common_swarm.get("num_particles", 1) <= 0
    ):
        raise ValueError("'common_swarm.num_particles' must be a positive integer.")
    if (
        "num_runs" in common_run_config
        and not isinstance(common_run_config["num_runs"], int)
        or common_run_config.get("num_runs", 1) <= 0
    ):
        raise ValueError("'common_run_config.num_runs' must be a positive integer.")
    if (
        "random_seed" in common_run_config
        and common_run_config["random_seed"] is not None
        and not isinstance(common_run_config["random_seed"], int)
    ):
        raise ValueError("'common_run_config.random_seed' must be an integer or null.")
    try:
        _normalize_parallel_config(common_run_config)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid 'common_run_config.parallel' configuration: {exc}") from exc

    if not isinstance(common_pso_params, dict):
        raise TypeError("'common_pso_params' must be a dictionary.")

    _validate_component_block(common_pso_params, "common_pso_params")

    variants_for_validation = (
        extract_pso_variants(config)
        if "pso_variants" in config
        else [VariantDescriptor.from_raw(DEFAULT_VARIANT_NAME, {})]
    )
    try:
        variants_for_validation = filter_pso_variants(
            variants_for_validation,
            selected_variants=selected_variants,
            excluded_variants=excluded_variants,
        )
    except ValueError as exc:
        raise ValueError(f"Invalid variant selection: {exc}") from exc

    for variant in variants_for_validation:
        _validate_component_block(
            variant.pso_params,
            f"pso_variants['{variant.name}']",
        )

    # --- Validate Each Problem Definition ---
    for problem_name, settings in problems_to_run.items():
        if not isinstance(settings, dict):
            raise TypeError(f"Settings for problem '{problem_name}' must be a dictionary.")

        # Check required keys
        if "dimensions" not in settings:
            raise ValueError(f"Problem '{problem_name}' is missing mandatory 'dimensions' key.")
        if "bounds" not in settings:
            raise ValueError(f"Problem '{problem_name}' is missing mandatory 'bounds' key.")
        if "stopping_criteria" not in settings:
            raise ValueError(f"Problem '{problem_name}' is missing mandatory 'stopping_criteria' key.")

        # Check dimensions format (expecting single positive integer)
        dims = settings["dimensions"]
        if not isinstance(dims, int) or dims <= 0:
            raise TypeError(f"Problem '{problem_name}' dimensions must be a single positive integer.")

        # Basic bounds check (more detailed check happens in create_bounds)
        if not isinstance(settings["bounds"], (dict, list)):
            raise TypeError(f"Problem '{problem_name}' bounds must be a dictionary or a list.")

        # Basic stopping criteria check (more detail in create_stopping_criteria)
        if not isinstance(settings["stopping_criteria"], (dict, list)):
            raise TypeError(f"Problem '{problem_name}' stopping_criteria must be a dictionary or a list.")

        problem_pso_params = settings.get("pso_params", {})
        if not isinstance(problem_pso_params, dict):
            raise TypeError(f"Problem '{problem_name}' pso_params must be a dictionary if provided.")
        _validate_component_block(
            problem_pso_params,
            f"problem '{problem_name}' pso_params",
        )

        if "run_config" in settings and not isinstance(settings["run_config"], dict):
            raise TypeError(f"Problem '{problem_name}' run_config must be a dictionary if provided.")

        if "swarm" in settings and not isinstance(settings["swarm"], dict):
            raise TypeError(f"Problem '{problem_name}' swarm must be a dictionary if provided.")

    variant_contexts = variants_for_validation
    if not validate_vectorized:
        logger.debug("Configuration validation passed for '%s'.", config_path)
        return

    for variant in variant_contexts:
        for problem_name, settings in problems_to_run.items():
            merged_pso_params = {
                **common_pso_params,
                **variant.pso_params,
                **settings.get("pso_params", {}),
            }
            vectorization_issues = get_vectorization_issues({"pso_params": merged_pso_params})
            if vectorization_issues:
                issues_text = "; ".join(vectorization_issues)
                raise ValueError(
                    f"Problem '{problem_name}' and variant '{variant.name}' is not supported by vectorized execution. "
                    f"{issues_text}"
                )

    logger.debug("Configuration validation passed for '%s'.", config_path)


def run_pso_from_config(
    config_path: str,
    output_path: str | None = None,
    verbose: bool = False,
    *,
    config_data: dict[str, Any] | None = None,
    selected_variants: set[str] | None = None,
    excluded_variants: set[str] | None = None,
) -> dict[str, Any]:
    """
    Load configuration from a YAML file and run PSO experiments/benchmarks.
    Always expects the 'problems_to_benchmark' structure.

    Args:
        config_path: Path to YAML configuration file
        output_path: Optional path to write results (JSON, YAML, or basic TXT)
        verbose: Whether to print progress and summary information

    Returns:
        Dictionary containing results for each benchmark task.
    """
    if config_data is None:
        # Load configuration
        try:
            config = load_config(config_path)
        except FileNotFoundError:
            logger.error("Configuration file not found: %s", config_path)
            return {"error": f"File not found: {config_path}"}
        except (ValueError, yaml.YAMLError, json.JSONDecodeError) as e:
            logger.error("Error parsing config file '%s': %s", config_path, e)
            return {"error": f"Config parsing error: {e}"}
        except OSError as e:
            logger.error("Failed to read config file '%s': %s", config_path, e)
            return {"error": f"Config read error: {e}"}
    else:
        if not isinstance(config_data, dict):
            logger.error("Invalid configuration object for '%s': expected a dictionary.", config_path)
            return {"error": f"Invalid configuration data for '{config_path}': expected dict."}
        config = config_data

    # --- Validate Configuration ---
    try:
        _validate_configuration(
            config,
            config_path,
            selected_variants=selected_variants,
            excluded_variants=excluded_variants,
            validate_vectorized=False,
        )
    except (ValueError, TypeError) as e:
        logger.error("Configuration Error in '%s': %s", config_path, e)
        return {"error": f"Configuration error: {e}"}

    # --- Run Benchmark Mode (Unified approach) ---
    if verbose:
        logger.info("Loading Configuration: %s", config_path)
    benchmark_results = run_benchmark_mode(
        config,
        config_path,
        verbose,
        output_path,
        selected_variants=selected_variants,
        excluded_variants=excluded_variants,
    )
    return benchmark_results


def _build_execution_strategy(
    parallel_enabled: bool,
    parallel_num_cpus: int | None,
) -> tuple[RunExecutor, RunExecutor | None]:
    serial_executor: RunExecutor = SerialExecutor(_run_single_benchmark_execution)
    parallel_executor: RunExecutor | None = None
    if parallel_enabled and parallel_num_cpus is not None:
        parallel_executor = RayExecutor(_run_single_benchmark_execution, parallel_num_cpus)
    elif parallel_enabled:
        parallel_executor = RayExecutor(_run_single_benchmark_execution)

    return serial_executor, parallel_executor


def run_benchmark_mode(
    config: dict[str, Any],
    config_path: str,
    verbose: bool,
    output_path: str | None,
    selected_variants: set[str] | None = None,
    excluded_variants: set[str] | None = None,
) -> dict[str, Any]:
    """Run one or more benchmark tasks with optional Ray-backed parallelism."""
    problems_to_run = config.get("problems_to_benchmark")  # Expect this key
    if not problems_to_run or not isinstance(problems_to_run, dict):
        logger.error("Config must contain a 'problems_to_benchmark' dictionary.")
        return {"error": "Config missing 'problems_to_benchmark' dict"}

    common_swarm = config.get("common_swarm", {})
    common_pso_params = config.get("common_pso_params", {})
    common_run_config = config.get("common_run_config", {})
    benchmark_suite_results: dict[str, Any] = {}
    pso_variants = extract_pso_variants(config)
    try:
        pso_variants = filter_pso_variants(
            pso_variants,
            selected_variants=selected_variants,
            excluded_variants=excluded_variants,
        )
    except ValueError as exc:
        return {"error": str(exc)}

    if not pso_variants:
        return {"error": "No matching variants found for the requested selector."}

    base_output_dir = common_run_config.get("output_dir")
    if base_output_dir:
        Path(base_output_dir).mkdir(parents=True, exist_ok=True)
        if verbose:
            logger.info("Results will be saved under: %s", base_output_dir)

    benchmark_seed = common_run_config.get("random_seed")
    try:
        parallel_config = _normalize_parallel_config(common_run_config)
    except (TypeError, ValueError) as exc:
        return {"error": str(exc)}

    parallel_enabled = bool(parallel_config["enabled"])
    parallel_num_cpus = parallel_config["num_cpus"]

    if verbose:
        logger.info("--- Starting Benchmark Tasks --- ")
        logger.info("Problems to run: %s", list(problems_to_run.keys()))
        if parallel_enabled:
            logger.info("Parallel execution enabled (num_cpus=%s)", parallel_num_cpus or "auto")
        logger.info("%s", "-" * 30)

    plan = build_benchmark_plan(
        config=config,
        config_path=Path(config_path).name,
        common_swarm=common_swarm,
        common_pso_params=common_pso_params,
        common_run_config=common_run_config,
        benchmark_seed=benchmark_seed,
        build_task_plan=_build_task_plan,
        pso_variants=tuple(pso_variants),
    )

    plan = with_execution_strategy(plan, parallel_enabled, parallel_num_cpus)
    serial_executor, parallel_executor = _build_execution_strategy(parallel_enabled, parallel_num_cpus)
    benchmark_suite_results = execute_plan(
        plan=plan,
        serial_executor=serial_executor,
        parallel_executor=parallel_executor,
        logger=logger,
        verbose=verbose,
    )
    benchmark_suite_results["_benchmark_summary"]["config_file"] = Path(config_path).name

    final_output_path = output_path
    if not final_output_path and base_output_dir:
        final_output_path = Path(base_output_dir) / "_benchmark_results.yaml"

    if final_output_path:
        _save_results(benchmark_suite_results, final_output_path, 0, verbose, is_benchmark_summary=True)

    return benchmark_suite_results


# --- Helper functions for printing and saving ---


def _save_results(
    results_data: dict[str, Any],
    output_path: Path | str,
    num_runs: int,
    verbose: bool,
    is_benchmark_summary: bool = False,
):
    """Helper function to save results to a file (JSON, YAML, or TXT)."""

    serializable_output = make_json_serializable(results_data)
    output_path_obj = Path(output_path)
    extension = output_path_obj.suffix.lower()
    try:
        with output_path_obj.open("w", encoding="utf-8") as f:
            if extension == ".yaml" or extension == ".yml":
                yaml.dump(serializable_output, f, default_flow_style=False, sort_keys=False)
            elif extension == ".json":
                import json

                json.dump(serializable_output, f, indent=2)
            else:  # Plain text output
                if is_benchmark_summary:
                    f.write("--- PSO Benchmark Suite Summary ---\n")
                    summary = results_data.get("_benchmark_summary", {})
                    f.write(f"Config File: {summary.get('config_file', 'N/A')}\n")
                    f.write(f"Total Time: {summary.get('total_wall_time', 'N/A'):.2f}s\n")
                    f.write("\nRefer to saved YAML/JSON for detailed results per task.\n")
                elif num_runs == 1:
                    f.write("--- Single Run Result ---\n")
                    f.write(f"Best fitness: {results_data.get('best_fitness', 'N/A'):.6e}\n")
                    f.write(f"Iterations: {results_data.get('iterations', 'N/A')}\n")
                    f.write(f"Evaluations: {results_data.get('evaluations', 'N/A')}\n")
                    f.write(f"Wall Time: {results_data.get('total_wall_time', 'N/A'):.2f}s\n")
                    f.write(f"Seed Used: {results_data.get('seed_used', 'N/A')}\n")
                elif num_runs > 1:
                    f.write("--- Multi-Run Summary ---")
                    num_completed = results_data.get("num_runs_completed", "N/A")
                    num_requested = results_data.get("num_runs_requested", "N/A")
                    f.write(f"Runs Completed: {num_completed}/{num_requested}\n")
                    if "mean_fitness" in results_data["summary_statistics"]:
                        summary_statistics = results_data["summary_statistics"]
                        f.write(f"Mean Best Fitness:   {summary_statistics['mean_fitness']:.6e}\n")
                        f.write(f"Median Best Fitness: {summary_statistics['median_fitness']:.6e}\n")
                        f.write(f"Std Dev Fitness:   {summary_statistics['std_dev_fitness']:.6e}\n")
                        f.write(f"Min Best Fitness:    {summary_statistics['min_fitness']:.6e}\n")
                        f.write(f"Max Best Fitness:    {summary_statistics['max_fitness']:.6e}\n")
                    f.write(f"Total Time: {results_data['summary_statistics']['total_multi_run_wall_time']:.2f}s\n")
        if verbose:
            logger.info("Results saved to %s", output_path_obj)
    except (TypeError, ValueError, yaml.YAMLError, OSError) as exc:
        logger.error("Failed to write results to %s: %s", output_path_obj, exc)


if __name__ == "__main__":
    # Legacy direct invocation keeps using Typer entrypoint for consistency.
    from . import cli

    cli.app()
