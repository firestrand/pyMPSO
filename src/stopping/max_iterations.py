from typing import Any

from .base import StoppingCriteria

"""Maximum iterations stopping criteria implementation."""


class MaxIterationsStopping(StoppingCriteria):
    """
    Stopping criteria that terminates PSO after a maximum number of iterations.
    This is one of the simplest stopping criteria, useful as a fallback to ensure
    the algorithm terminates even if other criteria (like fitness convergence)
    are not met.
    """

    def __init__(self, max_iterations: int):
        """
        Initialize the stopping criteria with a maximum iteration count.
        Args:
            max_iterations: The maximum number of iterations to allow.
                          Must be greater than 0.
        Raises:
            ValueError: If max_iterations is not positive.
        """
        if max_iterations <= 0:
            raise ValueError("max_iterations must be greater than 0")
        self._max_iterations = max_iterations

    @property
    def max_iterations(self) -> int:
        """Get the maximum number of iterations."""
        return self._max_iterations

    def should_stop(self, swarm_state: dict[str, Any]) -> bool:
        """
        Check if the maximum number of iterations has been reached.
        Args:
            swarm_state: Dictionary containing current state of the optimization.
                        Must contain 'current_iteration' key with an integer value.
        Returns:
            True if current_iteration >= max_iterations, False otherwise.
            If current_iteration is not in the state dict, returns False.
        """
        current_iteration = int(swarm_state.get("current_iteration", 0))
        return bool(current_iteration >= self._max_iterations)
