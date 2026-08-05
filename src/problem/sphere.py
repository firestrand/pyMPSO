"""Sphere function implementation as an optimization problem."""

import logging

import numpy as np

from .base import Problem
from .cec_data import get_cec_shift_vector

logger = logging.getLogger(__name__)


class SphereFunction(Problem):
    """
    Sphere function implementation.
    Classic benchmark function: f(x) = sum(x_i^2).
    Minimum is at x_i = 0 for all i, with f(x) = 0.
    If use_cec_shift is True, the function minimum is shifted according to
    the CEC benchmark standards using predefined shift vectors.
    """

    def __init__(
        self,
        dimension: int,
        bias: float = 0.0,
        bounds: np.ndarray | None = None,
        use_cec_shift: bool = False,
    ):
        """
        Initializes the SphereFunction.
        Args:
            dimension: The dimensionality of the problem.
            bias: Optional constant value added to the function evaluation.
            bounds: Optional numpy array of shape (dimension, 2) specifying [min, max]
                   bounds for each dimension.
            use_cec_shift: Whether to apply the CEC shift vector.
        """
        self.use_cec_shift = use_cec_shift
        shift_vector = None
        if use_cec_shift:
            try:
                # Assuming 'sphere' is the key for its shift vector in cec_data
                shift_vector = get_cec_shift_vector("sphere", dimension)
                if shift_vector is None:
                    logger.warning(
                        "No CEC shift vector found for Sphere function with %s dimensions. Using unshifted version.",
                        dimension,
                    )
                    self.use_cec_shift = False
                elif len(shift_vector) != dimension:
                    raise ValueError(
                        f"Shift vector length {len(shift_vector)} does not match problem dimensions {dimension}."
                    )
            except (ValueError, KeyError) as e:
                logger.warning(
                    "Could not load or validate CEC shift vector for Sphere: %s. Using unshifted version.",
                    e,
                )
                self.use_cec_shift = False
                shift_vector = None

        # Initialize the Problem base class with all necessary parameters.
        super().__init__(dimension=dimension, bias=bias, bounds=bounds, shift_vector=shift_vector)

    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluates the Sphere function for a given position.
        Args:
            position: A numpy array representing the position (particle's coordinates).
        Returns:
            The calculated fitness value.
        Raises:
            ValueError: If the input position dimension doesn't match the problem dimension.
        """
        # Use common validation from base class
        self._validate_position(position)
        if self.use_cec_shift and self._shift_vector is not None:
            # Apply shift before evaluation.
            shifted_position = position - self._shift_vector
            return float(np.sum(shifted_position**2) + self._bias)
        # Evaluate unshifted function
        return float(np.sum(position**2) + self._bias)
