from abc import ABC, abstractmethod
from typing import Any

"""Abstract Base Class for PSO stopping criteria."""


class StoppingCriteria(ABC):
    """
    Abstract Base Class for PSO stopping criteria.
    Stopping criteria determine when the PSO algorithm should terminate its search.
    This could be based on various factors such as:
    - Maximum number of iterations reached
    - Maximum number of fitness evaluations reached
    - Target fitness value achieved
    - No improvement in best fitness for N iterations
    - Convergence criteria met (e.g., swarm diversity below threshold)
    - Time limit reached
    - etc.
    The swarm_state dictionary passed to should_stop() contains all necessary
    information about the current state of the optimization, including:
    - current_iteration: Current iteration number
    - best_fitness: Best fitness found so far
    - best_position: Best position found so far
    - evaluations_used: Number of fitness evaluations used
    - iterations_without_improvement: Iterations since last improvement
    - swarm_diversity: Measure of particle diversity in the swarm
    - elapsed_time: Time elapsed since optimization started
    - etc.
    """

    @abstractmethod
    def should_stop(self, swarm_state: dict[str, Any]) -> bool:
        """
        Determine if the PSO algorithm should stop based on the current swarm state.
        Args:
            swarm_state: Dictionary containing current state of the optimization.
                        The exact contents depend on what information the PSO
                        algorithm tracks and what the stopping criteria needs.
        Returns:
            True if the algorithm should stop, False otherwise.
        """
        pass
