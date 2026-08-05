"""Unit tests for CLI orchestration service."""

from pathlib import Path

from src import run_pso as run_pso_module
from src.cli_service import CliExecutionResult, execute_cli_request


def test_execute_cli_request_dry_run_success(monkeypatch):
    """Validate that a dry-run request loads config and validates successfully."""
    config_data = {
        "problems_to_benchmark": {"sphere": {"dimensions": 2}},
    }
    monkeypatch.setattr(run_pso_module, "load_config", lambda _: config_data)
    monkeypatch.setattr(run_pso_module, "_validate_configuration", lambda *_args, **_kwargs: None)

    result = execute_cli_request(config=Path("config.yaml"), dry_run=True)

    assert isinstance(result, CliExecutionResult)
    assert result.success
    assert result.status == "ok"
    assert result.config_data == config_data
    assert result.results is None


def test_execute_cli_request_invalid_configuration_for_dry_run(monkeypatch):
    """Validate failed validation returns a validation error for dry runs."""
    config_data = {
        "problems_to_benchmark": {"sphere": {"dimensions": 2}},
    }

    def _raise_validation_error(*_args, **_kwargs) -> None:
        raise ValueError("bad config")

    monkeypatch.setattr(run_pso_module, "load_config", lambda _: config_data)
    monkeypatch.setattr(run_pso_module, "_validate_configuration", _raise_validation_error)

    result = execute_cli_request(config=Path("config.yaml"), dry_run=True)

    assert not result.success
    assert result.status == "validation_error"
    assert result.message == "Invalid configuration: bad config"
    assert result.config_data == config_data


def test_execute_cli_request_checks_output_format(monkeypatch):
    """Validate an unsupported output format fails fast before execution."""
    config_data = {
        "problems_to_benchmark": {"sphere": {"dimensions": 2}},
    }
    monkeypatch.setattr(run_pso_module, "load_config", lambda *_args, **_kwargs: config_data)

    result = execute_cli_request(
        config=Path("config.yaml"),
        output=Path("results.bin"),
        output_format="toml",
        verbose=False,
        dry_run=False,
    )

    assert not result.success
    assert result.status == "format_error"


def test_execute_cli_request_executes_and_returns_results(monkeypatch):
    """Validate successful execution returns normalized results."""
    config_data = {
        "problems_to_benchmark": {"sphere": {"dimensions": 2}},
    }
    results_payload = {"sphere_2D": {"best_fitness": 0.1}}

    def fake_run_config(*_args, **_kwargs):
        return results_payload

    monkeypatch.setattr(run_pso_module, "load_config", lambda *_args, **_kwargs: config_data)
    monkeypatch.setattr(run_pso_module, "run_pso_from_config", fake_run_config)

    result = execute_cli_request(config=Path("config.yaml"), verbose=False, dry_run=False)

    assert result.success
    assert result.status == "ok"
    assert result.config_data == config_data
    assert result.results == results_payload


def test_execute_cli_request_with_variant_filters(monkeypatch):
    """Validate variant filters are passed through to run_pso_from_config."""
    config_data = {
        "problems_to_benchmark": {"sphere": {"dimensions": 2}},
    }
    captured = {}

    def fake_run_config(*_args, **_kwargs):
        captured["selected_variants"] = _kwargs.get("selected_variants")
        captured["excluded_variants"] = _kwargs.get("excluded_variants")
        return {"sphere_2D": {"best_fitness": 0.1}}

    monkeypatch.setattr(run_pso_module, "load_config", lambda *_args, **_kwargs: config_data)
    monkeypatch.setattr(run_pso_module, "run_pso_from_config", fake_run_config)

    result = execute_cli_request(
        config=Path("config.yaml"),
        verbose=False,
        selected_variants={"hypersphere"},
        excluded_variants={"default"},
    )

    assert result.success
    assert captured["selected_variants"] == {"hypersphere"}
    assert captured["excluded_variants"] == {"default"}


def test_execute_cli_request_handles_execution_error(monkeypatch):
    """Validate execution-time errors are surfaced as execution_error."""
    config_data = {
        "problems_to_benchmark": {"sphere": {"dimensions": 2}},
    }

    monkeypatch.setattr(run_pso_module, "load_config", lambda *_args, **_kwargs: config_data)
    monkeypatch.setattr(
        run_pso_module,
        "run_pso_from_config",
        lambda *_args, **_kwargs: {"error": "failed to run"},
    )

    result = execute_cli_request(config=Path("config.yaml"), dry_run=False)

    assert not result.success
    assert result.status == "execution_error"
    assert result.message == "failed to run"
    assert result.config_data == config_data
