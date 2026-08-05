import math

import numpy as np

from src.problem.ackley import AckleyFunction
from src.problem.griewank import GriewankFunction
from src.problem.rastrigin import RastriginFunction
from src.problem.rosenbrock import RosenbrockFunction
from src.problem.schwefel import SchwefelFunction
from src.problem.sphere import SphereFunction


def test_function_isolation_sphere_matches_formula():
    sphere = SphereFunction(dimension=3, bounds=np.array([[-100.0, 100.0]] * 3), bias=-12.5)
    point = np.array([1.0, -2.0, 3.0])
    expected = float(np.sum(point**2) - 12.5)
    assert sphere.evaluate(point) == expected


def test_function_isolation_sphere_cec_shifted_at_shift_vector():
    problem = SphereFunction(
        dimension=3,
        bounds=np.array([[-100.0, 100.0]] * 3),
        bias=-450.0,
        use_cec_shift=True,
    )
    assert problem.shift_vector is not None
    assert np.isclose(problem.evaluate(problem.shift_vector), -450.0, atol=1e-12)


def test_function_isolation_rastrigin_formula():
    rastrigin = RastriginFunction(dimension=3)
    point = np.array([1.0, 2.0, 3.0])
    expected = 10 * 3 + np.sum(point**2 - 10 * np.cos(2 * np.pi * point))
    assert np.isclose(rastrigin.evaluate(point), expected, atol=1e-12)


def test_function_isolation_rosenbrock_formula():
    rosenbrock = RosenbrockFunction(dimension=3)
    point = np.array([1.0, 2.0, 3.0])
    expected = (
        100.0 * (point[1] - point[0] ** 2) ** 2
        + (1.0 - point[0]) ** 2
        + 100.0 * (point[2] - point[1] ** 2) ** 2
        + (1.0 - point[1]) ** 2
    )
    assert np.isclose(rosenbrock.evaluate(point), expected, atol=1e-12)


def test_function_isolation_ackley_formula():
    ackley = AckleyFunction(dimension=3)
    point = np.array([1.0, 2.0, 3.0])
    n = 3.0
    a = 20.0
    b = 0.2
    c = 2 * np.pi
    expected = -a * np.exp(-b * math.sqrt(np.sum(point**2) / n)) - np.exp(np.sum(np.cos(c * point)) / n) + a + np.e
    assert np.isclose(ackley.evaluate(point), expected, atol=1e-12)


def test_function_isolation_griewank_formula():
    griewank = GriewankFunction(dimension=3)
    point = np.array([1.0, 2.0, -3.0])
    expected = np.sum(point**2) / 4000.0 - np.prod(np.cos(point / np.sqrt(np.arange(1, point.size + 1)))) + 1.0
    assert np.isclose(griewank.evaluate(point), expected, atol=1e-12)


def test_function_isolation_schwefel_formula():
    schwefel = SchwefelFunction(dimension=2)
    point = np.array([420.9687, 420.9687])
    abs_pos = np.abs(point)
    sqrt_abs = np.sqrt(np.maximum(abs_pos, 1e-10))
    expected = 2 * 418.9829 - np.sum(point * np.sin(sqrt_abs))
    assert np.isclose(schwefel.evaluate(point), expected, atol=1e-8)


def test_function_isolation_schwefel_cec_shift_at_shift_vector_is_zero():
    problem = SchwefelFunction(
        dimension=3,
        bounds=np.array([[-100.0, 100.0]] * 3),
        use_cec_shift=True,
    )
    assert problem.shift_vector is not None
    assert np.isclose(problem.evaluate(problem.shift_vector), 0.0, atol=1e-12)
