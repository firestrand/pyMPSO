"""Maximum evaluations stopping criteria implementation."""

from typing import Any

from .base import StoppingCriteria


class MaxEvaluationsStopping(StoppingCriteria):
    """
    Stop when the number of fitness evaluations reaches a budget.

    This mirrors SPSO implementations that define stopping by
    ``evalMax`` rather than maximum iteration count.
    """

    def __init__(self, max_evaluations: int):
        if max_evaluations <= 0:
            raise ValueError("max_evaluations must be greater than 0")
        self._max_evaluations = max_evaluations

    @property
    def max_evaluations(self) -> int:
        """Get the maximum number of fitness evaluations."""
        return self._max_evaluations

    def should_stop(self, swarm_state: dict[str, Any]) -> bool:
        evaluations = int(swarm_state.get("evaluations_used", 0))
        return bool(evaluations >= self._max_evaluations)
