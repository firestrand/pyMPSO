from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.clamping.velocity import MaxNormVelocityClampingStrategy
from src.problem.sphere import SphereFunction
from src.run_pso import (
    _build_component_config,
    _build_task_plan,
    _ensure_dict_config,
    _validate_component_block,
    _validate_configuration,
    create_bounds,
    create_stopping_criteria,
    load_class,
)

# Import stopping criteria classes used directly by create_stopping_criteria
from src.stopping.base import StoppingCriteria
from src.stopping.composite import AnyStoppingCriteria
from src.topology.global_topology import GlobalTopology
from src.variants.domain import VariantDescriptor, extract_pso_variants, filter_pso_variants

# --- Tests for load_class ---


def test_load_class_valid():
    """Test loading existing classes using their full path."""
    loaded_sphere = load_class("src.problem.sphere.SphereFunction")
    assert loaded_sphere is SphereFunction

    loaded_global_topo = load_class("src.topology.global_topology.GlobalTopology")
    assert loaded_global_topo is GlobalTopology


def test_load_class_invalid_class_name():
    """Test loading a non-existent class within an existing module."""
    with pytest.raises(ImportError, match="Could not import src.problem.sphere.NonExistentClass"):
        load_class("src.problem.sphere.NonExistentClass")


def test_load_class_invalid_module_path():
    """Test loading from a non-existent module path."""
    with pytest.raises(ImportError, match="Could not import src.non_existent_module.SomeClass"):
        load_class("src.non_existent_module.SomeClass")


def test_load_class_invalid_format():
    """Test loading with an improperly formatted class path."""
    with pytest.raises(ValueError):  # .rsplit will fail
        load_class("InvalidFormatNoDots")
    with pytest.raises(ImportError):  # Module path is just 'src' without proper submodule
        load_class("src.NonExistentClass")


# --- Tests for _ensure_dict_config ---


def test_ensure_dict_config_string_input():
    """Test converting a simple string (shorthand) to a dict."""
    assert _ensure_dict_config("clamping", "boundary_default") == {"type": "clamping"}
    assert _ensure_dict_config("global", "topology_default") == {"type": "global"}


def test_ensure_dict_config_dict_input():
    """Test passing a dictionary through (should remain unchanged)."""
    input_dict = {"type": "my_type", "param1": 123}
    assert _ensure_dict_config(input_dict, "default") is input_dict  # Should be the same object


def test_ensure_dict_config_dict_input_missing_type():
    """Test passing a dictionary without a 'type' key."""
    input_dict = {"param1": 123}
    with pytest.raises(TypeError, match="Component configuration dictionary must include a 'type' key"):
        _ensure_dict_config(input_dict, "default")


def test_ensure_dict_config_none_input():
    """Test passing None (should return dict with default type)."""
    assert _ensure_dict_config(None, "default_for_none") == {"type": "default_for_none"}


def test_ensure_dict_config_invalid_input_type():
    """Test passing an invalid type (e.g., list, int)."""
    with pytest.raises(TypeError, match="Unexpected config format"):
        _ensure_dict_config(["list"], "default")
    with pytest.raises(TypeError, match="Unexpected config format"):
        _ensure_dict_config(123, "default")


def test_build_component_config_uses_velocity_strategy_params():
    """Velocity strategy params should be passed through unchanged."""
    pso_params = {
        "velocity_strategy": "standard",
        "velocity_strategy_params": {"c1": 1.49445, "c2": 1.49445},
        "velocity_clamping": {"strategy": "max_norm", "max_velocity": 3.0},
    }

    velocity_config = _build_component_config(pso_params, "velocity_strategy", "standard")

    assert velocity_config["type"] == "standard"
    assert velocity_config["c1"] == 1.49445
    assert velocity_config["c2"] == 1.49445
    assert isinstance(velocity_config["velocity_clamping"], MaxNormVelocityClampingStrategy)


def test_build_component_config_merges_constriction_velocity_params():
    """Constriction velocity params should stay namespaced under velocity_strategy_params."""
    pso_params = {
        "velocity_strategy": "constriction",
        "velocity_strategy_params": {"phi1": 2.05, "phi2": 2.05, "chi": 0.7},
    }

    velocity_config = _build_component_config(pso_params, "velocity_strategy", "standard")

    assert velocity_config["type"] == "constriction"
    assert velocity_config["phi1"] == 2.05
    assert velocity_config["phi2"] == 2.05
    assert velocity_config["chi"] == 0.7


# --- Tests for create_bounds ---


def test_create_bounds_uniform():
    """Test creating bounds with uniform min/max for all dimensions."""
    config = {"dimensions": 3, "bounds": {"min": -10.0, "max": 10.0}}
    expected_bounds = np.array([[-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0]])
    bounds = create_bounds(config)
    assert isinstance(bounds, np.ndarray)
    np.testing.assert_array_equal(bounds, expected_bounds)


def test_create_bounds_per_dimension_list():
    """Test creating bounds with a list of [min, max] lists."""
    config = {"dimensions": 2, "bounds": [[-5.0, 5.0], [-1.0, 1.0]]}
    expected_bounds = np.array([[-5.0, 5.0], [-1.0, 1.0]])
    bounds = create_bounds(config)
    assert isinstance(bounds, np.ndarray)
    np.testing.assert_array_equal(bounds, expected_bounds)


def test_create_bounds_per_dimension_dict():
    """Test creating bounds with a list of {'min': x, 'max': y} dicts."""
    config = {"dimensions": 2, "bounds": [{"min": -5.0, "max": 5.0}, {"min": -1.0, "max": 1.0}]}
    expected_bounds = np.array([[-5.0, 5.0], [-1.0, 1.0]])
    bounds = create_bounds(config)
    assert isinstance(bounds, np.ndarray)
    np.testing.assert_array_equal(bounds, expected_bounds)


def test_create_bounds_missing_bounds_key():
    """Test config missing the 'bounds' key entirely (should return None)."""
    config = {"dimensions": 5}
    bounds = create_bounds(config)
    assert bounds is None


def test_create_bounds_uniform_missing_min_max():
    """Test uniform bounds dict missing 'min' or 'max'."""
    config_no_min = {"dimensions": 2, "bounds": {"max": 10.0}}
    config_no_max = {"dimensions": 2, "bounds": {"min": -10.0}}
    with pytest.raises(ValueError, match="Bounds dict must contain 'min' and 'max' keys"):
        create_bounds(config_no_min)
    with pytest.raises(ValueError, match="Bounds dict must contain 'min' and 'max' keys"):
        create_bounds(config_no_max)


def test_create_bounds_per_dimension_wrong_length():
    """Test per-dimension bounds list having the wrong length."""
    config = {
        "dimensions": 3,
        "bounds": [[-1, 1], [-2, 2]],  # Only 2 items for 3 dimensions
    }
    # Fixed warning by escaping parentheses
    with pytest.raises(ValueError, match=r"Bounds list length \(2\) must match dimensions \(3\)"):
        create_bounds(config)


def test_create_bounds_per_dimension_invalid_format():
    """Test invalid format within per-dimension bounds list."""
    config_invalid_list = {
        "dimensions": 1,
        "bounds": [[-1, 1, 0]],  # List has 3 items
    }
    config_invalid_dict = {
        "dimensions": 1,
        "bounds": [{"minimum": -1, "maximum": 1}],  # Wrong keys
    }
    config_invalid_type = {"dimensions": 1, "bounds": ["not a list or dict"]}
    with pytest.raises(ValueError, match="Invalid bounds format at index 0"):
        create_bounds(config_invalid_list)
    with pytest.raises(ValueError, match="Invalid bounds format at index 0"):
        create_bounds(config_invalid_dict)
    with pytest.raises(ValueError, match="Invalid bounds format at index 0"):
        create_bounds(config_invalid_type)


def test_create_bounds_invalid_top_level_type():
    """Test when the 'bounds' value is not a dict or list."""
    config = {"dimensions": 2, "bounds": "not a dict or list"}
    with pytest.raises(ValueError, match="Bounds must be a dict or list"):
        create_bounds(config)


# --- Tests for create_stopping_criteria ---


# Mock StoppingCriteria subclasses for isolation
class MockStoppingCriterion(StoppingCriteria):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def should_stop(self, swarm_state: dict[str, Any]) -> bool:
        _ = swarm_state
        return False


def test_create_stopping_criteria_single_dict(monkeypatch):
    """Test creating a single criterion from a dict."""
    mock_instance = MockStoppingCriterion()
    config = {"type": "max_iterations", "max_iterations": 100}

    # Mock registry factory for this specific type
    def mock_get_stopping_criteria(name: str, **kwargs):
        assert name == "max_iterations"
        assert kwargs == {"max_iterations": 100}
        return mock_instance

    monkeypatch.setattr("src.run_pso.get_stopping_criteria", mock_get_stopping_criteria)

    result = create_stopping_criteria(config)
    assert result is mock_instance


def test_create_stopping_criteria_single_max_evaluations_dict(monkeypatch):
    """Test creating max-evaluation criterion from a single dict."""
    mock_instance = MockStoppingCriterion(max_evaluations=123)

    def mock_get_stopping_criteria(name: str, **kwargs):
        assert name == "max_evaluations"
        assert kwargs == {"max_evaluations": 123}
        return mock_instance

    monkeypatch.setattr("src.run_pso.get_stopping_criteria", mock_get_stopping_criteria)
    config = {"type": "max_evaluations", "max_evaluations": 123}
    result = create_stopping_criteria(config)
    assert result is mock_instance


def test_create_stopping_criteria_single_list(monkeypatch):
    """Test creating a single criterion from a single-item list."""
    mock_instance = MockStoppingCriterion()
    config_item = {"type": "max_iterations", "max_iterations": 100}
    config_list = [config_item]

    # Mock registry factory for this specific type
    def mock_get_stopping_criteria(name: str, **kwargs):
        assert name == "max_iterations"
        assert kwargs == {"max_iterations": 100}
        return mock_instance

    monkeypatch.setattr("src.run_pso.get_stopping_criteria", mock_get_stopping_criteria)

    result = create_stopping_criteria(config_list)
    assert result is mock_instance


def test_create_stopping_criteria_implicit_any_list(monkeypatch):
    """Test creating an implicit 'any' composite from a multi-item list."""
    mock_criterion1 = MockStoppingCriterion(type="max_iterations", max_iterations=100)
    mock_criterion2 = MockStoppingCriterion(type="max_iterations", max_iterations=200)
    mock_any_criteria = MagicMock(spec=AnyStoppingCriteria)

    config = [
        {"type": "max_iterations", "max_iterations": 100},
        {"type": "max_iterations", "max_iterations": 200},
    ]

    # Mock registry factory for the base type
    def mock_get_stopping_criteria(name: str, **kwargs):
        assert name == "max_iterations"
        if kwargs.get("max_iterations") == 100:
            return mock_criterion1
        if kwargs.get("max_iterations") == 200:
            return mock_criterion2
        pytest.fail("Unexpected kwargs in mock get_stopping_criteria")

    # Mock the AnyStoppingCriteria constructor
    def mock_any_criteria_init(criteria: list[StoppingCriteria]):
        assert len(criteria) == 2
        assert criteria[0] is mock_criterion1
        assert criteria[1] is mock_criterion2
        return mock_any_criteria

    monkeypatch.setattr("src.run_pso.get_stopping_criteria", mock_get_stopping_criteria)
    monkeypatch.setattr("src.run_pso.AnyStoppingCriteria", mock_any_criteria_init)

    result = create_stopping_criteria(config)
    assert result is mock_any_criteria


def test_create_stopping_criteria_multi_item_list_uses_max_when_target_unreachable():
    """Test implicit-any list semantics enforce max-iterations even if target is unreachable."""
    config = [
        {"type": "max_iterations", "max_iterations": 3},
        {"type": "target_fitness", "target": -1.0},
    ]

    result = create_stopping_criteria(config)
    assert isinstance(result, AnyStoppingCriteria)

    # At iterations 0-2, max-iterations is not yet met.
    assert not result.should_stop({"current_iteration": 0, "best_fitness": 0.0})
    assert not result.should_stop({"current_iteration": 2, "best_fitness": 0.0})
    # At iteration 3, max-iterations is reached, so stop should occur.
    assert result.should_stop({"current_iteration": 3, "best_fitness": 0.0})


def test_create_stopping_criteria_error_empty_list():
    """Test error handling for an empty list config."""
    with pytest.raises(ValueError, match="Stopping criteria list cannot be empty"):
        create_stopping_criteria([])


def test_create_stopping_criteria_error_invalid_dict():
    """Test error handling for a dict missing type/any/all."""
    config = {"invalid_key": "some_value"}
    with pytest.raises(ValueError, match="Invalid dictionary-based stopping criteria configuration"):
        create_stopping_criteria(config)


def test_create_stopping_criteria_error_invalid_type():
    """Test error handling for invalid top-level config type."""
    with pytest.raises(TypeError, match="Invalid type for stopping criteria configuration"):
        create_stopping_criteria(123)
    with pytest.raises(TypeError, match="Invalid type for stopping criteria configuration"):
        create_stopping_criteria("a string")


def test_build_task_plan_propagates_top_level_problem_fields():
    """Task-level benchmark keys should flow into the resolved problem config."""
    task_name, task_entry, run_specs = _build_task_plan(
        problem_name="sphere",
        problem_settings={
            "dimensions": 10,
            "bounds": {"min": -100, "max": 100},
            "bias": -450.0,
            "use_cec_shift": True,
            "stopping_criteria": [{"type": "max_iterations", "max_iterations": 1}],
        },
        common_swarm={"num_particles": 12},
        common_pso_params={},
        common_run_config={"num_runs": 1},
        benchmark_seed=123,
        config_file="tmp.yaml",
        variant=VariantDescriptor.from_raw("default"),
    )

    assert task_name == "sphere_10D"
    assert task_entry["problem"]["bias"] == -450.0
    assert task_entry["problem"]["use_cec_shift"] is True
    assert run_specs[0].problem["type"] == "sphere"
    assert run_specs[0].problem["bias"] == -450.0
    assert run_specs[0].problem["use_cec_shift"] is True


def test_build_task_plan_supports_variant_overrides_and_suffixes():
    """Variant-specific overrides should be merged before task-specific settings."""
    task_name, task_entry, run_specs = _build_task_plan(
        problem_name="sphere",
        problem_settings={
            "dimensions": 10,
            "bounds": {"min": -100, "max": 100},
            "stopping_criteria": [{"type": "max_iterations", "max_iterations": 1}],
            "pso_params": {
                "velocity_strategy": "constriction",
            },
        },
        common_swarm={"num_particles": 12},
        common_pso_params={"topology": "global", "velocity_strategy": "standard"},
        common_run_config={"num_runs": 1},
        benchmark_seed=123,
        config_file="tmp.yaml",
        variant=VariantDescriptor.from_raw(
            "hypersphere",
            {
                "pso_params": {
                    "topology": "ring",
                    "velocity_strategy": "hypersphere",
                },
                "swarm": {
                    "num_particles": 24,
                },
            },
        ),
    )

    assert task_name == "sphere_10D[hypersphere]"
    assert task_entry["pso_variant"] == "hypersphere"
    assert task_entry["swarm"]["num_particles"] == 24
    assert run_specs[0].pso_params["topology"] == "ring"
    assert run_specs[0].pso_params["velocity_strategy"] == "constriction"


def test_extract_pso_variants_with_structured_entries():
    """Variant extraction should parse structured variant descriptors."""
    variants = extract_pso_variants(
        {
            "pso_variants": {
                "hypersphere": {
                    "pso_params": {"velocity_strategy": "hypersphere"},
                },
                "constriction": {
                    "pso_params": {"velocity_strategy": "constriction"},
                    "swarm": {"num_particles": 30},
                },
            }
        }
    )

    assert [variant.name for variant in variants] == ["hypersphere", "constriction"]
    assert variants[0].pso_params["velocity_strategy"] == "hypersphere"
    assert variants[1].swarm["num_particles"] == 30


def test_extract_pso_variants_rejects_unknown_variant_keys():
    """Variant blocks must use only supported descriptor sections."""
    with pytest.raises(TypeError, match="contains unknown keys"):
        extract_pso_variants(
            {
                "pso_variants": {
                    "legacy": {
                        "velocity_strategy": "hypersphere",
                    },
                }
            }
        )


def test_validate_configuration_with_variants():
    """validate_configuration should accept valid variant declarations."""
    config = {
        "problems_to_benchmark": {
            "sphere": {
                "dimensions": 2,
                "bounds": {"min": -1, "max": 1},
                "stopping_criteria": [{"type": "max_iterations", "max_iterations": 1}],
            }
        },
        "common_run_config": {"num_runs": 1},
        "pso_variants": {
            "hypersphere": {
                "pso_params": {"velocity_strategy": "hypersphere"},
            },
        },
    }

    _validate_configuration(config, "tmp.yaml")


def test_validate_component_block_rejects_unknown_top_level_keys():
    """Strict validation should reject unsupported keys in pso_params blocks."""
    with pytest.raises(TypeError, match="Unsupported keys in pso params"):
        _validate_component_block(
            {"velocity_strategy": "standard", "typo": 1},
            "problem 'sphere' pso_params",
        )


def test_validate_configuration_rejects_non_vectorized_components():
    """Vectorized execution validation should reject unsupported component combinations."""
    config = {
        "problems_to_benchmark": {
            "sphere": {
                "dimensions": 2,
                "bounds": {"min": -1, "max": 1},
                "stopping_criteria": [{"type": "max_iterations", "max_iterations": 1}],
            }
        },
        "common_pso_params": {"topology": "global", "velocity_strategy": "clpso", "boundary_handler": "clamping"},
        "common_run_config": {"num_runs": 1},
    }

    with pytest.raises(ValueError, match="not supported by vectorized execution"):
        _validate_configuration(config, "tmp.yaml", validate_vectorized=True)


def test_validate_configuration_without_vectorization_allows_unsupported_components():
    """Execution config validation should pass when strict vectorized checks are disabled."""
    config = {
        "problems_to_benchmark": {
            "sphere": {
                "dimensions": 2,
                "bounds": {"min": -1, "max": 1},
                "stopping_criteria": [{"type": "max_iterations", "max_iterations": 1}],
            }
        },
        "common_pso_params": {"topology": "global", "velocity_strategy": "clpso", "boundary_handler": "clamping"},
        "common_run_config": {"num_runs": 1},
    }

    _validate_configuration(config, "tmp.yaml")


def test_validate_configuration_validates_selected_variants_only_for_vectorized_check():
    """Vectorized validation should honor variant selection filters."""
    config = {
        "problems_to_benchmark": {
            "sphere": {
                "dimensions": 2,
                "bounds": {"min": -1, "max": 1},
                "stopping_criteria": [{"type": "max_iterations", "max_iterations": 1}],
            }
        },
        "common_run_config": {"num_runs": 1},
        "pso_variants": {
            "clpso_velocity": {"pso_params": {"velocity_strategy": "clpso"}},
            "global_topology": {"pso_params": {"topology": "global"}},
        },
    }

    with pytest.raises(ValueError, match="not supported by vectorized execution"):
        _validate_configuration(config, "tmp.yaml", validate_vectorized=True)

    assert (
        _validate_configuration(
            config,
            "tmp.yaml",
            selected_variants={"global_topology"},
            validate_vectorized=True,
        )
        is None
    )


def test_validate_configuration_rejects_invalid_variant_type():
    """Invalid variant payload types should fail validation."""
    config = {
        "problems_to_benchmark": {
            "sphere": {
                "dimensions": 2,
                "bounds": {"min": -1, "max": 1},
                "stopping_criteria": [{"type": "max_iterations", "max_iterations": 1}],
            }
        },
        "common_run_config": {"num_runs": 1},
        "pso_variants": {
            "bad": ["not", "a", "dict"],
        },
    }

    with pytest.raises(TypeError):
        _validate_configuration(config, "tmp.yaml")


def test_filter_pso_variants_includes_and_excludes():
    """Variant filter should honor include and exclude sets."""
    variants = [
        VariantDescriptor.from_raw("default"),
        VariantDescriptor.from_raw("hypersphere", {"pso_params": {"velocity_strategy": "hypersphere"}}),
    ]

    filtered_hypersphere = filter_pso_variants(variants, selected_variants={"hypersphere"})
    assert filtered_hypersphere == [
        VariantDescriptor.from_raw("hypersphere", {"pso_params": {"velocity_strategy": "hypersphere"}})
    ]
    assert filtered_hypersphere[0].name == "hypersphere"
    assert filtered_hypersphere[0].pso_params["velocity_strategy"] == "hypersphere"

    filtered_excluded_default = filter_pso_variants(variants, excluded_variants={"default"})
    assert len(filtered_excluded_default) == 1
    assert filtered_excluded_default[0].name == "hypersphere"

    filtered_default_only = filter_pso_variants(variants, selected_variants={"default"})
    assert len(filtered_default_only) == 1
    assert filtered_default_only[0].name == "default"
    assert filtered_default_only[0].pso_params == {}


def test_filter_pso_variants_rejects_unknown_selection():
    """Selecting unknown variants should fail loudly."""
    variants = [VariantDescriptor.from_raw("default")]

    with pytest.raises(ValueError, match="Unknown pso variants requested"):
        filter_pso_variants(variants, selected_variants={"missing"})
