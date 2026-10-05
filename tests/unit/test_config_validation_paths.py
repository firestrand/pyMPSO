from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from src import run_pso
from src.components.validation import ComponentConfigError, validate_component_config
from src.stopping.composite import AllStoppingCriteria, AnyStoppingCriteria


@pytest.fixture
def config() -> dict[str, Any]:
    return {
        "problems_to_benchmark": {
            "sphere": {
                "dimensions": 2,
                "bounds": {"min": -2, "max": 2},
                "stopping_criteria": {"type": "max_iterations", "max_iterations": 2},
            }
        },
        "common_swarm": {"num_particles": 4},
        "common_run_config": {"random_seed": 17},
    }


@pytest.mark.parametrize(
    ("section", "value", "message"),
    [
        ("problems_to_benchmark", {}, "cannot be empty"),
        ("problems_to_benchmark", [], "top-level"),
        ("common_swarm", {"num_particles": 0}, "positive integer"),
        ("common_swarm", {"num_particles": 1.5}, "positive integer"),
        ("common_run_config", {"num_runs": 0}, "positive integer"),
        ("common_run_config", {"num_runs": "2"}, "positive integer"),
        ("common_run_config", {"random_seed": "17"}, "integer or null"),
        ("common_run_config", {"parallel": []}, "parallel"),
        ("common_run_config", {"parallel": {"num_cpus": 0}}, "parallel"),
        ("common_pso_params", [], "must be a dictionary"),
        ("common_pso_params", {"topology": 2}, "Invalid format"),
        ("common_pso_params", {"topology": {"type": ""}}, "missing a 'type'"),
        ("common_pso_params", {"velocity_clamping": 3}, "must be a dictionary"),
        ("pso_variants", {"custom": {"pso_params": {"topology": "unknown"}}}, "Invalid component"),
    ],
)
def test_invalid_common_settings(config, section, value, message):
    config[section] = value
    with pytest.raises((TypeError, ValueError), match=message):
        run_pso._validate_configuration(config, "test.yaml")


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("dimensions", None, "positive integer"),
        ("dimensions", -1, "positive integer"),
        ("bounds", None, "dictionary or a list"),
        ("stopping_criteria", 1, "dictionary or a list"),
        ("pso_params", [], "must be a dictionary"),
        ("run_config", [], "must be a dictionary"),
        ("swarm", [], "must be a dictionary"),
    ],
)
def test_invalid_problem_settings(config, key, value, message):
    config["problems_to_benchmark"]["sphere"][key] = value
    with pytest.raises((TypeError, ValueError), match=message):
        run_pso._validate_configuration(config, "test.yaml")


@pytest.mark.parametrize("key", ["dimensions", "bounds", "stopping_criteria"])
def test_missing_required_problem_settings(config, key):
    del config["problems_to_benchmark"]["sphere"][key]
    with pytest.raises(ValueError, match=f"missing mandatory '{key}'"):
        run_pso._validate_configuration(config, "test.yaml")


def test_invalid_problem_and_variant_selection(config):
    config["problems_to_benchmark"]["sphere"] = []
    with pytest.raises(TypeError, match="Settings for problem"):
        run_pso._validate_configuration(config, "test.yaml")
    with pytest.raises(ValueError, match="Invalid variant selection"):
        run_pso._validate_configuration(config, "test.yaml", selected_variants={"missing"})


@pytest.mark.parametrize("value", [None, "global", {"type": "GLOBAL"}])
def test_registered_component_validation(value):
    validate_component_config("topology", value, "test")


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        ({}, ComponentConfigError, "Empty dict"),
        ({"k": 2}, ComponentConfigError, "missing a 'type'"),
        ([], TypeError, "Invalid component config type"),
        ("missing", ComponentConfigError, "Valid options"),
    ],
)
def test_invalid_component_validation(value, error, message):
    with pytest.raises(error, match=message):
        validate_component_config("topology", value, "test")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ([], None),
        ({"type": "max_iterations", "max_iterations": 0}, None),
        ({"type": "max_iterations", "max_iterations": "4"}, None),
        ({"all": [{"type": "max_iterations", "max_iterations": 4}, {"type": "target_fitness"}]}, 4),
        ([{"any": [{"type": "max_iterations", "max_iterations": 8}]}, {}], 8),
    ],
)
def test_iteration_budget_extraction(value, expected):
    assert run_pso._extract_max_iterations(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"parallel": None}, {"enabled": False, "num_cpus": None}),
        ({}, {"enabled": False, "num_cpus": None}),
        ({"parallel": {"enabled": True, "num_cpus": 2}}, {"enabled": True, "num_cpus": 2}),
    ],
)
def test_parallel_settings_normalization(value, expected):
    assert run_pso._normalize_parallel_config(value) == expected


def test_component_normalization_and_clamping():
    original = {"type": "ring", "k": 2}
    normalized = run_pso._normalize_component_config(original, "global")
    normalized["k"] = 3
    assert original["k"] == 2
    assert run_pso._normalize_component_config(None, "global") == {"type": "global"}
    assert run_pso._normalize_topology_params({"type": "random"})["rebuild_probability"] == 1
    with pytest.raises(TypeError, match="Expected 'topology_params'"):
        run_pso._build_component_config({"topology_params": 3}, "topology", "global")
    with pytest.raises(TypeError, match="velocity_clamping"):
        run_pso._build_component_config({"velocity_clamping": 3}, "velocity_strategy", "standard")
    built = run_pso._build_component_config(
        {"velocity_strategy": {"type": "standard", "velocity_clamping": {"strategy": "max_norm", "max_velocity": 2}}},
        "velocity_strategy",
        "standard",
    )
    assert built["velocity_clamping"].max_velocity == 2


def test_stopping_composites_and_errors():
    leaf = {"type": "max_iterations", "max_iterations": 2}
    assert isinstance(run_pso.create_stopping_criteria({"all": [leaf]}), AllStoppingCriteria)
    assert isinstance(run_pso.create_stopping_criteria([leaf, leaf]), AnyStoppingCriteria)
    for value in [{}, {"any": [], "type": "max_iterations"}, {"type": "missing"}, []]:
        with pytest.raises(ValueError):
            run_pso.create_stopping_criteria(value)
    with pytest.raises(TypeError):
        run_pso.create_stopping_criteria(None)


@pytest.mark.parametrize(("suffix", "contents"), [(".json", "{"), (".yaml", "["), (".txt", "text")])
def test_config_parse_errors(tmp_path: Path, suffix, contents):
    path = tmp_path / f"config{suffix}"
    path.write_text(contents)
    assert run_pso.run_pso_from_config(str(path))["error"].startswith("Config parsing error:")


def test_config_file_access_errors(tmp_path: Path):
    assert run_pso.run_pso_from_config(str(tmp_path / "missing.yaml"))["error"].startswith("File not found:")
    assert run_pso.run_pso_from_config(str(tmp_path))["error"].startswith("Config read error:")
    assert run_pso.run_pso_from_config("memory", config_data={})["error"].startswith("Configuration error:")


def test_seeded_small_benchmark_and_output(config, tmp_path: Path):
    config["common_run_config"].update({"num_runs": 2, "output_dir": str(tmp_path)})
    original = deepcopy(config)
    result = run_pso.run_pso_from_config("test.yaml", config_data=config, verbose=True)
    task = result["sphere_2D"]
    assert task["summary_statistics"]["num_runs_completed"] == 2
    assert len({r["seed_used"] for r in task["individual_runs"]}) == 2
    assert (tmp_path / "_benchmark_results.yaml").exists()
    assert config == original
    repeat = run_pso.run_pso_from_config("test.yaml", config_data=config)
    assert [r["best_fitness"] for r in task["individual_runs"]] == [
        r["best_fitness"] for r in repeat["sphere_2D"]["individual_runs"]
    ]


@pytest.mark.parametrize("value", [{}, []])
def test_invalid_normalized_component(value):
    with pytest.raises(TypeError):
        run_pso._normalize_component_config(value, "global")
