from pathlib import Path
from typing import cast

import pytest

from src import run_pso
from src.components.factory import get_vectorization_issues
from src.planning.domain import benchmark_task_key
from src.variants.domain import extract_pso_variants

VARIANT_CONFIG_DIR = Path("examples/variants")
VARIANT_PRESET_DIR = VARIANT_CONFIG_DIR / "presets"
VARIANT_CONFIGS = sorted(
    list(VARIANT_CONFIG_DIR.glob("*.yaml")) + [path for path in VARIANT_PRESET_DIR.glob("*.yaml") if path.is_file()]
)


def _has_vectorization_issues(config: dict) -> bool:
    """Determine whether any variant/problem combination is unsupported by vectorization."""
    common_pso_params = config.get("common_pso_params", {})
    problems = config["problems_to_benchmark"]
    for variant in extract_pso_variants(config):
        for problem_name, settings in problems.items():
            merged_pso_params = {
                **common_pso_params,
                **variant.pso_params,
                **settings.get("pso_params", {}),
            }
            if get_vectorization_issues({"pso_params": merged_pso_params}):
                return True
            _ = problem_name
    return False


@pytest.mark.parametrize("config_path", VARIANT_CONFIGS)
def test_variant_configs_validate(config_path: Path) -> None:
    config = run_pso.load_config(str(config_path))
    if _has_vectorization_issues(config):
        with pytest.raises(ValueError, match="not supported by vectorized execution"):
            run_pso._validate_configuration(config, str(config_path), validate_vectorized=True)
    else:
        run_pso._validate_configuration(config, str(config_path), validate_vectorized=True)


def _extract_variant_matrix(config: dict) -> list[tuple[str, dict[str, object], dict[str, object]]]:
    """Flatten variants and their expected normalized PSO/run parameters."""
    variants = extract_pso_variants(config)
    common_pso_params = config.get("common_pso_params", {})
    common_run_config = config.get("common_run_config", {})
    return [
        (
            variant.name,
            {
                **common_pso_params,
                **variant.pso_params,
            },
            {
                **common_run_config,
                **variant.run_config,
            },
        )
        for variant in variants
    ]


def _task_names_for_variants(config: dict, variant_name: str) -> list[str]:
    """Build expected task keys for each problem under one variant."""
    task_names: list[str] = []
    for problem_name, settings in config["problems_to_benchmark"].items():
        dimensions = settings["dimensions"]
        task_names.append(
            benchmark_task_key(problem_name, dimensions, None if variant_name == "default" else variant_name)
        )
    return task_names


def _run_single_variant(config_path: Path, config_data: dict, variant: str) -> dict:
    """Execute one variant by name through the public config entrypoint."""
    config_for_variant = dict(config_data)
    if "pso_variants" in config_data and isinstance(config_data["pso_variants"], dict):
        variants = config_data["pso_variants"]
        if variant in variants:
            config_for_variant["pso_variants"] = {variant: variants[variant]}
    return run_pso.run_pso_from_config(
        config_path=str(config_path),
        config_data=config_for_variant,
        verbose=False,
        selected_variants={variant},
    )


@pytest.mark.parametrize("config_path", VARIANT_CONFIGS)
def test_variant_configs_preserve_execution_contract(config_path: Path) -> None:
    """
    Each variant config should either execute successfully (vectorized)
    or fail with a clear vectorization-removed error.
    """
    config = run_pso.load_config(str(config_path))
    run_pso._validate_configuration(config, str(config_path))

    for variant_name, merged_pso_params, merged_run_config in _extract_variant_matrix(config):
        task_names = _task_names_for_variants(config, variant_name)
        vectorization_issues = get_vectorization_issues({"pso_params": merged_pso_params})
        num_runs_requested = cast(int, merged_run_config.get("num_runs", 1))
        execution_results = _run_single_variant(config_path, config, variant_name)

        assert "_benchmark_summary" in execution_results
        for task_name in task_names:
            assert task_name in execution_results
            task_results = execution_results[task_name]
            assert isinstance(task_results, dict)

            if vectorization_issues:
                if num_runs_requested == 1:
                    assert "error" in task_results
                    assert task_results["error"].startswith(
                        "Current configuration is not supported by the vectorized execution engine."
                    )
                else:
                    assert "summary_statistics" in task_results
                    individual_runs = task_results["individual_runs"]
                    assert isinstance(individual_runs, list)
                    assert len(individual_runs) == task_results["num_runs_requested"]
                    assert len(individual_runs) >= 1
                    assert task_results["summary_statistics"].get("num_runs_completed") == len(individual_runs)
                    for run in individual_runs:
                        assert "error" in run
                        assert (
                            "Current configuration is not supported by the vectorized execution engine." in run["error"]
                        )
            else:
                if num_runs_requested == 1:
                    assert "error" not in task_results
                    assert "best_fitness" in task_results
                    assert "iterations" in task_results
                else:
                    assert "summary_statistics" in task_results
                    individual_runs = task_results["individual_runs"]
                    assert isinstance(individual_runs, list)
                    assert len(individual_runs) >= 1
                    assert len(individual_runs) == task_results["num_runs_requested"]
                    assert task_results["summary_statistics"].get("num_runs_completed") == len(individual_runs)
                    for run in individual_runs:
                        assert "error" not in run
                        assert "best_fitness" in run
                        assert "iterations" in run
