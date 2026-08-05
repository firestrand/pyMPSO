from typing import Any

from .base import StoppingCriteria

"""Composite Stopping Criteria implementations."""


class BaseCompositeStoppingCriteria(StoppingCriteria):
    """Base class for composite stopping criteria."""

    def __init__(self, criteria: list[StoppingCriteria]):
        """
        Initialize the composite criteria with a list of child criteria.
        Args:
            criteria: A list of StoppingCriteria instances.
        Raises:
            ValueError: If the criteria list is empty or contains non - StoppingCriteria objects.
        """
        if not criteria:
            raise ValueError("Criteria list cannot be empty for composite stopping criteria.")
        if not all(isinstance(c, StoppingCriteria) for c in criteria):
            raise ValueError("All items in the criteria list must be instances of StoppingCriteria.")
        self._criteria = criteria

    @property
    def children(self) -> list[StoppingCriteria]:
        """Get the list of child criteria."""
        return self._criteria


class AnyStoppingCriteria(BaseCompositeStoppingCriteria):
    """
    Composite stopping criteria that stops if ANY of its child criteria are met.
    Implements OR logic for combining multiple stopping conditions.
    """

    def should_stop(self, swarm_state: dict[str, Any]) -> bool:
        """
        Check if any of the child criteria indicate that the algorithm should stop.
        Args:
            swarm_state: Dictionary containing the current state of the optimization.
        Returns:
            True if at least one child criterion returns True, False otherwise.
        """
        return any(criterion.should_stop(swarm_state) for criterion in self._criteria)


class AllStoppingCriteria(BaseCompositeStoppingCriteria):
    """
    Composite stopping criteria that stops only if ALL of its child criteria are met.
    Implements AND logic for combining multiple stopping conditions.
    """

    def should_stop(self, swarm_state: dict[str, Any]) -> bool:
        """
        Check if all of the child criteria indicate that the algorithm should stop.
        Args:
            swarm_state: Dictionary containing the current state of the optimization.
        Returns:
            True if all child criteria return True, False otherwise.
        """
        return all(criterion.should_stop(swarm_state) for criterion in self._criteria)
