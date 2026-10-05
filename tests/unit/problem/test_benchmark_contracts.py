"""Analytic values and objective validation independent of optimization runs."""

import importlib
import math
from typing import Any

import numpy as np
import pytest

from src.problem.ackley import AckleyFunction
from src.problem.cec_data import CEC_SHIFT_DATA, get_cec_shift_vector
from src.problem.griewank import GriewankFunction
from src.problem.rastrigin import RastriginFunction
from src.problem.rosenbrock import RosenbrockFunction
from src.problem.schaffer_f6 import SchafferF6Function
from src.problem.schwefel import SchwefelFunction
from src.problem.sphere import SphereFunction
from src.problem.step import StepFunction
from src.problem.styblinski_tang import StyblinskiTangFunction

SHIFTED = [
    SphereFunction,
    AckleyFunction,
    GriewankFunction,
    RastriginFunction,
    RosenbrockFunction,
    SchwefelFunction,
    StepFunction,
]
VALIDATED = [*SHIFTED, StyblinskiTangFunction]


@pytest.mark.parametrize("problem_class", VALIDATED)
@pytest.mark.parametrize(
    "position, message",
    [
        ([0.0, 0.0], "numpy array"),
        (np.zeros(3), "shape"),
        (np.zeros((2, 1)), "shape"),
        (np.array([math.nan, 0.0]), "NaN or infinite"),
        (np.array([0.0, math.inf]), "NaN or infinite"),
        (np.array([-math.inf, 0.0]), "NaN or infinite"),
    ],
)
def test_evaluation_rejects_invalid_positions(problem_class: Any, position: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        problem_class(2).evaluate(position)


@pytest.mark.parametrize("problem_class", VALIDATED)
@pytest.mark.parametrize("dimension", [0, -1])
def test_dimensions_must_be_positive(problem_class: Any, dimension: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        problem_class(dimension)


@pytest.mark.parametrize("problem_class", VALIDATED)
def test_custom_bounds_are_copied(problem_class: Any) -> None:
    bounds = np.array([[-2.0, 3.0], [-4.0, 5.0]])
    problem = problem_class(2, bounds=bounds)
    bounds[:] = 100.0
    actual = problem.bounds
    np.testing.assert_array_equal(actual, [[-2.0, 3.0], [-4.0, 5.0]])
    assert actual is not None
    actual[:] = 99.0
    np.testing.assert_array_equal(problem.bounds, [[-2.0, 3.0], [-4.0, 5.0]])


@pytest.mark.parametrize(
    "bounds, message",
    [
        ([[-1.0, 1.0]] * 2, "numpy array"),
        (np.zeros((3, 2)), "shape"),
        (np.array([[2.0, 1.0], [-1.0, 1.0]]), "Lower bounds"),
    ],
)
def test_base_bounds_validation(bounds: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SphereFunction(2, bounds=bounds)


def test_sphere_unbounded_and_bias_properties() -> None:
    problem = SphereFunction(2, bias=7.0)
    assert problem.bounds is None
    assert problem.shift_vector is None
    assert problem.dimension == 2
    assert problem.bias == 7.0
    assert problem.evaluate(np.array([3.0, -4.0])) == 32.0


@pytest.mark.parametrize(
    "problem_class, position, expected",
    [
        (AckleyFunction, [1.0, -1.0], 20.0 * (1.0 - math.exp(-0.2))),
        (GriewankFunction, [math.pi, 0.0], 2.0 + math.pi**2 / 4000.0),
        (RastriginFunction, [0.5, -0.5], 40.5),
        (RosenbrockFunction, [2.0, 3.0], 101.0),
        (SchwefelFunction, [0.0, 0.0], 837.9658),
        (StepFunction, [-0.51, 1.5], 5.0),
        (StyblinskiTangFunction, [1.0, -1.0], -15.0),
    ],
)
def test_benchmark_analytic_values_include_bias(problem_class: Any, position: list[float], expected: float) -> None:
    problem = problem_class(2, bias=3.25)
    point = np.array(position)
    before = point.copy()
    assert problem.evaluate(point) == pytest.approx(expected + 3.25)
    np.testing.assert_array_equal(point, before)


def test_ackley_custom_coefficients() -> None:
    problem = AckleyFunction(2, a=4.0, b=0.5, c=math.pi)
    assert (problem.a, problem.b, problem.c) == (4.0, 0.5, math.pi)
    expected = 4.0 * (1.0 - math.exp(-0.5)) + math.e - math.exp(-1.0)
    assert problem.evaluate(np.ones(2)) == pytest.approx(expected)


@pytest.mark.parametrize("problem_class", SHIFTED)
def test_loaded_shift_moves_optimum_and_is_copied(problem_class: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    shift = np.array([3.0, -4.0])
    module = importlib.import_module(problem_class.__module__)
    monkeypatch.setattr(module, "get_cec_shift_vector", lambda _name, _dimension: shift)
    problem = problem_class(2, bias=2.5, use_cec_shift=True)
    shift[:] = 0.0
    np.testing.assert_array_equal(problem.shift_vector, [3.0, -4.0])
    exposed = problem.shift_vector
    assert exposed is not None
    exposed[:] = 100.0
    assert problem.evaluate(np.array([3.0, -4.0])) == pytest.approx(2.5, abs=1e-12)
    assert problem.use_cec_shift


@pytest.mark.parametrize("problem_class", SHIFTED)
@pytest.mark.parametrize("failure", ["absent", "wrong_length", "value_error", "key_error"])
def test_unavailable_shift_falls_back_to_unshifted_objective(
    problem_class: Any,
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def load_shift(_name: str, _dimension: int) -> np.ndarray | None:
        if failure == "value_error":
            raise ValueError("unsupported dimension")
        if failure == "key_error":
            raise KeyError("missing data")
        return None if failure == "absent" else np.array([1.0])

    module = importlib.import_module(problem_class.__module__)
    monkeypatch.setattr(module, "get_cec_shift_vector", load_shift)
    problem = problem_class(2, use_cec_shift=True)
    reference = problem_class(2)
    point = np.array([1.0, -2.0])
    assert not problem.use_cec_shift
    assert problem.shift_vector is None
    assert problem.evaluate(point) == pytest.approx(reference.evaluate(point))
    assert "unshifted version" in caplog.text


@pytest.mark.parametrize("problem_class", [cls for cls in SHIFTED if cls is not StepFunction])
def test_real_cec_data_places_optimum_at_shift(problem_class: Any) -> None:
    problem = problem_class(3, bias=-100.0, use_cec_shift=True)
    shift = problem.shift_vector
    assert shift is not None
    assert problem.evaluate(shift) == pytest.approx(-100.0, abs=1e-12)


def test_cec_data_lookup_normalizes_names_and_rejects_missing_dimensions(caplog: pytest.LogCaptureFixture) -> None:
    np.testing.assert_array_equal(get_cec_shift_vector("SPHERE", 2), CEC_SHIFT_DATA["sphere"][:2])
    assert get_cec_shift_vector("unknown", 2) is None
    assert get_cec_shift_vector("sphere", len(CEC_SHIFT_DATA["sphere"]) + 1) is None
    assert "No shift applied" in caplog.text


def test_step_without_cec_data_falls_back() -> None:
    problem = StepFunction(2, use_cec_shift=True)
    assert not problem.use_cec_shift
    assert problem.evaluate(np.array([-0.5, 0.49])) == 0.0
    assert problem.evaluate(np.array([0.5, -0.50001])) == 2.0


def test_rosenbrock_one_dimension_contract() -> None:
    assert RosenbrockFunction(1, bias=4.0).evaluate(np.array([3.0])) == 8.0


@pytest.mark.parametrize(
    "position, dim, expected",
    [
        ([0.0, 0.0], None, 0.0),
        ([1.0, 1.0], None, 2.0),
        ([0.5, -0.5], None, 12.5),
        ([0.5, -0.5, 1.0], 3, 13.5),
    ],
)
def test_rastrigin_dimension_override(position: list[float], dim: int | None, expected: float) -> None:
    problem = RastriginFunction(2, A=3.0, bias=1.0)
    assert problem.A == 3.0
    problem.set_test_mode("local_minima")
    assert problem.evaluate_with_dim(np.array(position), dim) == pytest.approx(expected + 1.0)
    problem.set_test_mode("normal")
    with pytest.raises(ValueError, match="does not match dimension"):
        problem.evaluate_with_dim(np.zeros(4))


def test_schaffer_values_and_custom_bounds() -> None:
    problem = SchafferF6Function(bias=2.0)
    assert problem.evaluate(np.zeros(2)) == 2.0
    expected = 2.5 - 0.5 / (1.0 + 0.001 * math.pi**2) ** 2
    assert problem.evaluate(np.array([math.pi, 0.0])) == pytest.approx(expected)
    np.testing.assert_array_equal(problem.bounds, [[-100.0, 100.0]] * 2)
    bounds = np.array([[-1.0, 1.0], [-2.0, 2.0]])
    custom = SchafferF6Function(bounds=bounds)
    bounds[:] = 0.0
    np.testing.assert_array_equal(custom.bounds, [[-1.0, 1.0], [-2.0, 2.0]])
    with pytest.raises(ValueError, match="only for 2 dimensions"):
        SchafferF6Function(3)
    with pytest.raises(ValueError, match="Bounds shape"):
        SchafferF6Function(bounds=np.zeros((3, 2)))
    with pytest.raises(ValueError, match="Input position"):
        problem.evaluate(np.zeros(3))
