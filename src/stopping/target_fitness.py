import logging
from typing import Any

import numpy as np

from .base import StoppingCriteria

"""Stopping criterion based on reaching a target fitness value."""

# Set up module logger
logger = logging.getLogger(__name__)


class TargetFitnessStopping(StoppingCriteria):
    """
    Stopping criterion that terminates PSO when a target fitness value is reached or surpassed.
    Assumes minimization by default (stops when best_fitness <= target).

    Supports both strict target mode and tolerance mode used by existing benchmarks.
    objective checks:
    - strict mode: best_fitness <= target
    - tolerance mode: abs(best_fitness - target) <= tolerance
    """

    def __init__(self, target: float, is_minimization: bool = True, tolerance: float = 0.0):
        """
        Initialize the stopping criteria with a target fitness value.
        Args:
            target: The target fitness value to reach.
            is_minimization: Flag indicating if the problem is minimization (True)
                             or maximization (False). Currently, only minimization
                             is fully supported in comparison logic.
            tolerance: Optional non-negative tolerance around the target. When
                greater than zero, stopping uses:
                    abs(best_fitness - target) <= tolerance
        Raises:
            TypeError: If target is not a float or cannot be converted.
        """
        try:
            self._target = float(target)
        except (ValueError, TypeError) as e:
            raise TypeError(f"Target fitness must be a number (float), got {target}. Error: {e}") from e
        try:
            self._tolerance = float(tolerance)
        except (ValueError, TypeError) as e:
            raise TypeError(f"Tolerance must be a number (float), got {tolerance}. Error: {e}") from e
        if self._tolerance < 0:
            raise ValueError("Tolerance must be non-negative.")
        # Note: Maximization logic may be implemented in future versions
        if not is_minimization:
            logger.warning(
                "is_minimization=False is set, but only minimization logic (<= target) is currently implemented in TargetFitnessStopping."
            )
        self._is_minimization = is_minimization
        self._use_tolerance = self._tolerance > 0.0

    @property
    def target(self) -> float:
        """Get the target fitness value."""
        return self._target

    @property
    def tolerance(self) -> float:
        """Get the stopping tolerance around the target."""
        return self._tolerance

    def should_stop(self, swarm_state: dict[str, Any]) -> bool:
        """
        Check if the best fitness has reached the target value.
        Args:
            swarm_state: Dictionary containing current state of the optimization.
                        Must contain 'best_fitness' key with a numeric value.
        Returns:
            True if best_fitness <= target (for minimization), False otherwise.
            If best_fitness is not in the state dict or is not finite, returns False.
        """
        best_fitness = swarm_state.get("best_fitness")
        if best_fitness is None:
            return False
        # Check for NaN explicitly, as it breaks comparisons
        if not isinstance(best_fitness, (int, float)) or np.isnan(best_fitness):
            return False
        # Handle minimization
        if self._is_minimization:
            if self._use_tolerance:
                return np.isfinite(best_fitness) and abs(best_fitness - self._target) <= self._tolerance
            # Allow -inf, check if finite or <= target
            if np.isneginf(best_fitness):
                return True  # -inf is always <= target
            # If not -inf, must be finite and <= target to stop
            return np.isfinite(best_fitness) and best_fitness <= self._target
        else:
            # Placeholder / Warning: Implement maximization logic
            # Should handle +inf similarly
            # if np.isposinf(best_fitness): return True
            # return np.isfinite(best_fitness) and best_fitness >= self._target
            return False
