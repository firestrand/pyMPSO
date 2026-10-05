import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.run_pso import _save_results
from src.utils.serialization import make_json_serializable


def test_numpy_results_are_json_serializable():
    payload = {
        "int": np.int64(4),
        "float": np.float64(1.5),
        "complex": np.complex128(2 + 3j),
        "array": np.array([[1, 2]]),
        "bool": np.bool_(True),
        "void": np.void(b"x"),
        "tuple": (None, "text"),
        "set": {3},
    }
    result = json.loads(json.dumps(make_json_serializable(payload)))
    assert result == {
        "int": 4,
        "float": 1.5,
        "complex": {"real": 2.0, "imag": 3.0},
        "array": [[1, 2]],
        "bool": True,
        "void": None,
        "tuple": [None, "text"],
        "set": [3],
    }
    assert isinstance(result["bool"], bool)


def test_scalar_conversion_fallbacks():
    class Scalar:
        def item(self):
            return 4

    class ListLike:
        def tolist(self):
            return [4]

    class BrokenList(Scalar):
        def tolist(self):
            raise ValueError("not a list")

    assert make_json_serializable(Scalar()) == 4
    assert make_json_serializable(ListLike()) == [4]
    assert make_json_serializable(BrokenList()) == 4
    unchanged = object()
    assert make_json_serializable(unchanged) is unchanged
    for text in [b"bytes", bytearray(b"bytes")]:
        assert make_json_serializable(text) is text


@pytest.mark.parametrize("suffix", ["json", "yaml", "yml"])
def test_result_files_round_trip(tmp_path: Path, suffix):
    path = tmp_path / f"result.{suffix}"
    payload = {"position": np.array([1.0, 2.0]), "fitness": np.float64(5)}
    _save_results(payload, path, 1, True)
    loaded = json.loads(path.read_text()) if suffix == "json" else yaml.safe_load(path.read_text())
    assert loaded == {"position": [1.0, 2.0], "fitness": 5.0}


@pytest.mark.parametrize("kind", ["suite", "single", "multiple", "failed_multiple"])
def test_text_result_summaries(tmp_path: Path, kind):
    path = tmp_path / "result.txt"
    stats = {
        "mean_fitness": 2,
        "median_fitness": 2,
        "std_dev_fitness": 1,
        "min_fitness": 1,
        "max_fitness": 3,
        "total_multi_run_wall_time": 0.5,
    }
    if kind == "suite":
        payload = {"_benchmark_summary": {"config_file": "test.yaml", "total_wall_time": 0.5}}
    elif kind == "single":
        payload = {"best_fitness": 2, "iterations": 3, "evaluations": 4, "total_wall_time": 0.5, "seed_used": 17}
    else:
        payload = {"num_runs_completed": 2, "num_runs_requested": 2, "summary_statistics": stats}
        if kind == "failed_multiple":
            payload["summary_statistics"] = {"total_multi_run_wall_time": 0.5}
    _save_results(payload, path, 1 if kind == "single" else 2, True, is_benchmark_summary=kind == "suite")
    contents = path.read_text()
    assert "0.50s" in contents
    if kind == "suite":
        assert "Config File: test.yaml" in contents
    elif kind == "single":
        assert "Seed Used: 17" in contents
    else:
        assert "Runs Completed: 2/2" in contents
        assert ("Mean Best Fitness" in contents) is (kind == "multiple")


def test_failed_output_write_is_logged(tmp_path: Path, caplog):
    _save_results({}, tmp_path / "missing" / "result.json", 1, False)
    assert "Failed to write results" in caplog.text
