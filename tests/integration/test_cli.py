"""Integration tests for the Typer-based CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from src.cli import app

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[2]
ACKLEY_CONFIG = REPO_ROOT / "examples" / "ackley_config.yaml"
ACKLEY_CONFIG_JSON = REPO_ROOT / "examples" / "ackley_config.json"
SPHERE_CONFIG_JSON = REPO_ROOT / "examples" / "sphere_config.json"


def test_cli_dry_run_is_lightweight_and_successful():
    """Validate that dry-run parses config and does not execute optimization."""
    result = runner.invoke(
        app,
        ["--config", str(ACKLEY_CONFIG), "--dry-run", "--no-color"],
    )
    assert result.exit_code == 0
    assert "Dry run completed" in result.stdout
    assert "Configuration validation passed" in result.stdout or "configuration parsed successfully" in result.stdout


def test_cli_runs_with_json_config(tmp_path: Path):
    """Validate the CLI can execute from a JSON configuration file."""
    output_path = tmp_path / "results.json"
    result = runner.invoke(
        app,
        [
            "--config",
            str(ACKLEY_CONFIG_JSON),
            "--output",
            str(output_path),
            "--output-format",
            "json",
            "--no-color",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    payload = json.loads(output_path.read_text())
    assert "ackley_10D" in payload
    assert "best_fitness" in payload["ackley_10D"]


def test_cli_runs_with_sphere_json_config(tmp_path: Path):
    """Validate JSON configuration path works for a simple 10D sphere benchmark."""
    output_path = tmp_path / "sphere_results.json"
    result = runner.invoke(
        app,
        [
            "--config",
            str(SPHERE_CONFIG_JSON),
            "--output",
            str(output_path),
            "--output-format",
            "json",
            "--no-color",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    payload = json.loads(output_path.read_text())
    assert "sphere_10D" in payload
    sphere_result = payload["sphere_10D"]
    if "summary_statistics" in sphere_result:
        assert "mean_fitness" in sphere_result["summary_statistics"]
    else:
        assert "best_fitness" in sphere_result


def test_cli_dry_run_does_not_require_output_format_extension():
    """Dry-run should not require output-format inference, even with extensionless output path."""
    result = runner.invoke(
        app,
        [
            "--config",
            str(ACKLEY_CONFIG),
            "--dry-run",
            "--no-color",
            "--output",
            "tmp-output",
        ],
    )
    assert result.exit_code == 0


def test_cli_requires_output_format_for_extensionless_output_path():
    """Non-dry-run must infer or validate output format when path has no extension."""
    result = runner.invoke(
        app,
        ["--config", str(ACKLEY_CONFIG), "--output", "tmp-output", "--no-color"],
    )
    assert result.exit_code == 1
    assert "Could not infer output format" in result.stdout


def test_cli_run_writes_json_results_and_is_serializable(tmp_path: Path):
    """Run a quick config and verify JSON output is produced with serializable types."""
    output_path = tmp_path / "results.json"
    result = runner.invoke(
        app,
        [
            "--config",
            str(ACKLEY_CONFIG),
            "--output",
            str(output_path),
            "--output-format",
            "json",
            "--no-color",
        ],
    )
    assert result.exit_code == 0
    assert output_path.exists()

    payload = json.loads(output_path.read_text())
    assert isinstance(payload, dict)
    assert "ackley_10D" in payload
    assert isinstance(payload["ackley_10D"]["best_position"], list)


def test_cli_run_writes_yaml_results_by_extension(tmp_path: Path):
    """Run with yaml extension and infer format from extension when no output-format is set."""
    output_path = tmp_path / "results.yaml"
    result = runner.invoke(
        app,
        [
            "--config",
            str(ACKLEY_CONFIG),
            "--output",
            str(output_path),
            "--no-color",
        ],
    )
    assert result.exit_code == 0
    assert output_path.exists()

    payload = yaml.safe_load(output_path.read_text())
    assert isinstance(payload, dict)
    assert "ackley_10D" in payload
    assert "best_fitness" in payload["ackley_10D"]


@pytest.mark.parametrize("bad_format", ["toml", "xml"])
def test_cli_rejects_invalid_output_format(tmp_path: Path, bad_format: str):
    """Validate output format is restricted to json/yaml."""
    result = runner.invoke(
        app,
        [
            "--config",
            str(ACKLEY_CONFIG),
            "--output",
            str(tmp_path / "results.json"),
            "--output-format",
            bad_format,
            "--no-color",
        ],
    )
    assert result.exit_code == 1
    assert "Unsupported output format" in result.stdout
