from collections import defaultdict

import numpy as np
import pytest

from src.components import factory


@pytest.fixture
def isolated_registry(monkeypatch):
    for name in ["_COMPONENT_REGISTRY", "_VECTORIZED_REGISTRY", "_CONFIG_REGISTRY"]:
        mapping = getattr(factory, name)
        monkeypatch.setattr(factory, name, defaultdict(dict, {key: dict(value) for key, value in mapping.items()}))


@pytest.mark.parametrize(
    "registration,creation",
    [
        (factory.register_component, factory.create_component),
        (factory.register_vectorized_component, factory.create_vectorized_component),
        (factory.register_typed_config, factory.create_typed_config),
    ],
)
def test_plugin_registration_normalizes_names_and_rejects_duplicates(isolated_registry, registration, creation):
    del isolated_registry

    class ScaledValue:
        def __init__(self, scale: float = 1.0):
            self.scale = scale

        def apply(self, value: float) -> float:
            return self.scale * value

    registration(" Plugin ", " Scaled ")(ScaledValue)
    instance = creation("PLUGIN", {"type": "SCALED", "scale": 2.0})
    assert instance.apply(3.0) == 6.0
    with pytest.raises(ValueError, match="Duplicate registration"):
        registration("plugin", "scaled")(ScaledValue)


@pytest.mark.parametrize(
    "creation", [factory.create_component, factory.create_vectorized_component, factory.create_typed_config]
)
def test_factory_requires_explicit_type(creation):
    with pytest.raises(ValueError, match="Missing 'type'"):
        creation("topology", {})


@pytest.mark.parametrize("creation", [factory.create_component, factory.create_vectorized_component])
def test_factory_rejects_unknown_component_names(creation):
    with pytest.raises(ValueError, match="Unknown"):
        creation("missing", {"type": "missing"})


def test_problem_factory_validates_dimension_and_applies_explicit_bounds():
    with pytest.raises(TypeError, match="expected string"):
        factory.create_component("problem", {"type": 3})
    with pytest.raises(ValueError, match="Missing 'dimensions'"):
        factory.create_component("problem", {"type": "sphere"})
    bounds = np.array([[-2.0, 3.0]])
    problem = factory.create_component("problem", {"type": "sphere", "dimensions": 1, "bias": 10}, bounds_arr=bounds)
    assert problem.evaluate(np.array([2.0])) == 14.0
    np.testing.assert_array_equal(problem.bounds, bounds)
    assert factory.create_typed_config("influence_strategy", {"type": "single_best"}) is None


@pytest.mark.parametrize("key", ["velocity_strategy", "topology", "influence_strategy", "boundary_handler"])
def test_vectorization_validation_reports_malformed_type(key):
    issues = factory.get_vectorization_issues({"pso_params": {key: {"type": None}}})
    assert len(issues) == 1 and issues[0].startswith(f"{key} must be a string")
    with pytest.raises(TypeError, match="must be a dictionary"):
        factory.get_vectorization_issues({"pso_params": []})
