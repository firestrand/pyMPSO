"""Threshold and composite stopping behavior at boundary values."""

import numpy as np
import pytest

from src.stopping.base import StoppingCriteria
from src.stopping.composite import AllStoppingCriteria, AnyStoppingCriteria
from src.stopping.max_evaluations import MaxEvaluationsStopping
from src.stopping.max_iterations import MaxIterationsStopping
from src.stopping.target_fitness import TargetFitnessStopping


@pytest.mark.parametrize(
    "criterion_type,key,property_name",
    [
        (MaxIterationsStopping, "current_iteration", "max_iterations"),
        (MaxEvaluationsStopping, "evaluations_used", "max_evaluations"),
    ],
)
def test_budget_criteria_stop_at_threshold_and_reject_nonpositive_limits(criterion_type, key, property_name):
    criterion = criterion_type(3)
    assert getattr(criterion, property_name) == 3
    assert not criterion.should_stop({})
    assert not criterion.should_stop({key: 2})
    assert criterion.should_stop({key: 3})
    assert criterion.should_stop({key: 4})
    for limit in [0, -1]:
        with pytest.raises(ValueError, match="greater than 0"):
            criterion_type(limit)


@pytest.mark.parametrize(
    "fitness,expected",
    [
        (None, False),
        ("0", False),
        (np.nan, False),
        (np.inf, False),
        (-np.inf, True),
        (2.0, False),
        (1.0, True),
        (0.0, True),
    ],
)
def test_strict_target_fitness_handles_nonfinite_and_invalid_values(fitness, expected):
    criterion = TargetFitnessStopping(1.0)
    assert criterion.target == 1.0
    assert criterion.tolerance == 0.0
    assert bool(criterion.should_stop({"best_fitness": fitness})) is expected


@pytest.mark.parametrize("fitness,expected", [(0.5, False), (0.75, True), (1.25, True), (1.5, False), (-np.inf, False)])
def test_target_tolerance_is_symmetric_and_requires_finite_fitness(fitness, expected):
    criterion = TargetFitnessStopping(1.0, tolerance=0.25)
    assert criterion.tolerance == 0.25
    assert bool(criterion.should_stop({"best_fitness": fitness})) is expected


@pytest.mark.parametrize("invalid", ["invalid", None])
def test_target_rejects_invalid_parameters_and_reports_unsupported_maximization(caplog, invalid):
    with pytest.raises(TypeError, match="Target fitness"):
        TargetFitnessStopping(invalid)
    with pytest.raises(TypeError, match="Tolerance"):
        TargetFitnessStopping(1.0, tolerance=invalid)
    with pytest.raises(ValueError, match="non-negative"):
        TargetFitnessStopping(1.0, tolerance=-1.0)
    criterion = TargetFitnessStopping(1.0, is_minimization=False)
    assert "only minimization" in caplog.text
    assert not criterion.should_stop({"best_fitness": 2.0})


@pytest.mark.parametrize("composite", [AnyStoppingCriteria, AllStoppingCriteria])
def test_composites_reject_empty_and_invalid_children(composite):
    with pytest.raises(ValueError, match="cannot be empty"):
        composite([])
    with pytest.raises(ValueError, match="instances of StoppingCriteria"):
        composite([object()])


@pytest.mark.parametrize(
    "iterations,fitness,any_result,all_result",
    [(2, 2.0, False, False), (3, 2.0, True, False), (2, 1.0, True, False), (3, 1.0, True, True)],
)
def test_composite_conditions_combine_real_budget_and_fitness_criteria(iterations, fitness, any_result, all_result):
    children: list[StoppingCriteria] = [MaxIterationsStopping(3), TargetFitnessStopping(1.0)]
    state = {"current_iteration": iterations, "best_fitness": fitness}
    any_criterion = AnyStoppingCriteria(children)
    all_criterion = AllStoppingCriteria(children)
    assert any_criterion.children == children
    assert bool(any_criterion.should_stop(state)) is any_result
    assert bool(all_criterion.should_stop(state)) is all_result
