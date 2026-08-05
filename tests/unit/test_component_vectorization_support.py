import pytest

import src.components.factory as component_factory
from src.components.factory import get_vectorization_issues, supports_vectorization


def test_supports_vectorization_accepts_default_components() -> None:
    assert supports_vectorization({}) is True


def test_supports_vectorization_uses_default_components_explicitly() -> None:
    config = {
        "pso_params": {
            "velocity_strategy": "standard",
            "topology": "global",
            "influence_strategy": "single_best",
            "boundary_handler": "clamping",
        }
    }

    assert supports_vectorization(config) is True
    assert get_vectorization_issues(config) == []


def test_get_vectorization_issues_for_unsupported_velocity() -> None:
    config = {"pso_params": {"velocity_strategy": "clpso"}}
    issues = get_vectorization_issues(config)

    assert any("velocity_strategy 'clpso' is not vectorized" in issue for issue in issues)


def test_get_vectorization_issues_for_adaptive_velocity_variants() -> None:
    for velocity_strategy in ("qpso", "apso"):
        config = {"pso_params": {"velocity_strategy": velocity_strategy}}
        issues = get_vectorization_issues(config)
        assert issues == []


def test_get_vectorization_issues_accepts_the_spso2011_stack() -> None:
    """Random topology, hypersphere velocity and damped reflection are vectorized."""
    config = {
        "pso_params": {
            "velocity_strategy": "hypersphere",
            "topology": "random",
            "influence_strategy": "single_best",
            "boundary_handler": "damped_reflection",
        }
    }

    assert get_vectorization_issues(config) == []
    assert supports_vectorization(config) is True


def test_get_vectorization_issues_accepts_ring_topology() -> None:
    config = {"pso_params": {"topology": "ring"}}

    assert get_vectorization_issues(config) == []


def test_get_vectorization_issues_collects_multiple_failures() -> None:
    config = {
        "pso_params": {
            "velocity_strategy": "bare_bones",
            "topology": "global",
            "influence_strategy": "fips",
            "boundary_handler": "clamping",
        }
    }
    issues = get_vectorization_issues(config)

    assert any("velocity_strategy 'bare_bones'" in issue for issue in issues)
    assert any("influence_strategy 'fips'" in issue for issue in issues)
    assert supports_vectorization(config) is False


def test_create_vectorized_component_supports_constructor_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Vectorized component factory should pass keyword arguments into constructors."""
    original_registry = {key: dict(value) for key, value in component_factory._VECTORIZED_REGISTRY.items()}
    monkeypatch.setattr(component_factory, "_VECTORIZED_REGISTRY", original_registry)

    class DummyVectorizedTopology:
        def __init__(self, label: str):
            self.label = label

    component_factory.register_vectorized_component("topology", "recording")(DummyVectorizedTopology)
    result = component_factory.create_vectorized_component("topology", {"type": "recording", "label": "abc"})

    assert isinstance(result, DummyVectorizedTopology)
    assert result.label == "abc"
