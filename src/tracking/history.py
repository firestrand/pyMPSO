import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..particles.swarm_state import SwarmState
from .events import EventSubscriber

"""History tracking for PSO experiments."""


@dataclass
class IterationMetrics:
    """Metrics for a single PSO iteration."""

    iteration: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    fitness_std: float
    global_best_position: np.ndarray
    diversity: float
    mean_velocity_norm: float
    max_velocity_norm: float
    min_velocity_norm: float
    improvement: float
    evaluations: int
    elapsed_time: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "iteration": self.iteration,
            "best_fitness": float(self.best_fitness),
            "mean_fitness": float(self.mean_fitness),
            "worst_fitness": float(self.worst_fitness),
            "fitness_std": float(self.fitness_std),
            "global_best_position": self.global_best_position.tolist(),
            "diversity": float(self.diversity),
            "mean_velocity_norm": float(self.mean_velocity_norm),
            "max_velocity_norm": float(self.max_velocity_norm),
            "min_velocity_norm": float(self.min_velocity_norm),
            "improvement": float(self.improvement),
            "evaluations": self.evaluations,
            "elapsed_time": float(self.elapsed_time),
        }


class ExperimentHistory(EventSubscriber):
    """
    Tracks detailed metrics throughout a PSO experiment run.
    This class collects comprehensive data at each iteration for analysis,
    visualization, and comparison of different PSO variants.
    """

    def __init__(self, track_positions: bool = False):
        """
        Initialize the experiment history tracker.
        Args:
            track_positions: If True, store all particle positions at each iteration
                           (can be memory intensive for large swarms)
        """
        self.track_positions = track_positions
        self.start_time: float | None = None
        self.iterations: list[IterationMetrics] = []
        # Optional detailed tracking
        if track_positions:
            self.particle_positions: list[np.ndarray] = []
            self.particle_velocities: list[np.ndarray] = []
            self.particle_fitnesses: list[np.ndarray] = []
        # Summary statistics
        self.total_evaluations = 0
        self.total_iterations = 0
        self.initial_best_fitness: float | None = None
        self.final_best_fitness: float | None = None
        self.convergence_iteration: int | None = None  # Iteration where convergence criteria met

    def on_run_start(self, swarm_state: SwarmState, **kwargs: Any) -> None:  # noqa: ARG002
        """Mark the start of tracking."""
        self.start_time = time.time()

    def on_iteration_end(self, iteration: int, swarm_state: SwarmState, **kwargs: Any) -> None:
        """Called at the end of each iteration to record metrics."""
        evaluations_used = kwargs.get("evaluations_used", 0)
        global_best_position, global_best_fitness = swarm_state.get_global_best()

        self.record_iteration(
            iteration,
            swarm_state,
            global_best_fitness,
            global_best_position,
            evaluations_used,
        )

    def record_iteration(
        self,
        iteration: int,
        swarm_state: SwarmState,
        global_best_fitness: float,
        global_best_position: np.ndarray,
        evaluations_used: int,
    ):
        """
        Record metrics for a single iteration.
        """
        # Calculate fitness statistics
        fitnesses = swarm_state.current_fitness
        mean_fitness = float(np.mean(fitnesses))
        worst_fitness: float = float(np.max(fitnesses))
        fitness_std = float(np.std(fitnesses))

        # Calculate velocity statistics
        velocities = swarm_state.velocities
        velocity_norms = np.linalg.norm(velocities, axis=1)
        mean_velocity_norm = float(np.mean(velocity_norms))
        max_velocity_norm: float = float(np.max(velocity_norms))
        min_velocity_norm: float = float(np.min(velocity_norms))

        # Calculate swarm diversity (average distance from centroid)
        positions = swarm_state.positions
        diversity = self._calculate_diversity(positions)

        # Calculate improvement from previous iteration
        improvement = 0.0
        if len(self.iterations) > 0:
            improvement = self.iterations[-1].best_fitness - global_best_fitness
        elif self.initial_best_fitness is not None:
            improvement = self.initial_best_fitness - global_best_fitness

        # Track initial fitness
        if self.initial_best_fitness is None:
            self.initial_best_fitness = global_best_fitness

        # Calculate elapsed time
        elapsed_time = time.time() - self.start_time if self.start_time is not None else 0.0

        # Create iteration metrics
        metrics = IterationMetrics(
            iteration=iteration,
            best_fitness=global_best_fitness,
            mean_fitness=mean_fitness,
            worst_fitness=worst_fitness,
            fitness_std=fitness_std,
            global_best_position=global_best_position.copy(),
            diversity=diversity,
            mean_velocity_norm=mean_velocity_norm,
            max_velocity_norm=max_velocity_norm,
            min_velocity_norm=min_velocity_norm,
            improvement=improvement,
            evaluations=evaluations_used,
            elapsed_time=elapsed_time,
        )
        self.iterations.append(metrics)

        # Optional detailed tracking
        if self.track_positions:
            self.particle_positions.append(positions.copy())
            self.particle_velocities.append(velocities.copy())
            self.particle_fitnesses.append(fitnesses.copy())

        # Update summary statistics
        self.total_evaluations = evaluations_used
        self.total_iterations = iteration
        self.final_best_fitness = global_best_fitness

    def _calculate_diversity(self, positions: np.ndarray) -> float:
        """
        Calculate swarm diversity as average distance from centroid.
        Args:
            positions: Array of particle positions (n_particles x n_dimensions)
        Returns:
            Diversity measure
        """
        if len(positions) == 0:
            return 0.0
        centroid = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)
        return float(np.mean(distances))

    def mark_convergence(self, iteration: int):
        """Mark the iteration where convergence was achieved."""
        self.convergence_iteration = iteration

    def get_convergence_curve(self) -> tuple:
        """
        Get the convergence curve (best fitness over iterations).
        Returns:
            Tuple of (iterations, best_fitness_values)
        """
        if not self.iterations:
            return [], []
        iterations = [m.iteration for m in self.iterations]
        best_fitness = [m.best_fitness for m in self.iterations]
        return iterations, best_fitness

    def get_diversity_curve(self) -> tuple:
        """
        Get the diversity curve over iterations.
        Returns:
            Tuple of (iterations, diversity_values)
        """
        if not self.iterations:
            return [], []
        iterations = [m.iteration for m in self.iterations]
        diversity = [m.diversity for m in self.iterations]
        return iterations, diversity

    def get_summary_statistics(self) -> dict[str, Any]:
        """
        Get summary statistics for the entire run.
        Returns:
            Dictionary of summary statistics
        """
        if not self.iterations:
            return {}
        # Find iteration with maximum improvement
        improvements = [m.improvement for m in self.iterations]
        max_improvement_idx = int(np.argmax(improvements)) if improvements else 0
        # Calculate convergence rate (iterations to reach 90% of final improvement)
        if self.initial_best_fitness and self.final_best_fitness:
            total_improvement = self.initial_best_fitness - self.final_best_fitness
            target_improvement = total_improvement * 0.9
            convergence_90_iter = None
            for m in self.iterations:
                current_improvement = self.initial_best_fitness - m.best_fitness
                if current_improvement >= target_improvement:
                    convergence_90_iter = m.iteration
                    break
        else:
            convergence_90_iter = None
        return {
            "total_iterations": self.total_iterations,
            "total_evaluations": self.total_evaluations,
            "initial_best_fitness": self.initial_best_fitness,
            "final_best_fitness": self.final_best_fitness,
            "total_improvement": self.initial_best_fitness - self.final_best_fitness
            if self.initial_best_fitness and self.final_best_fitness
            else 0,
            "convergence_iteration": self.convergence_iteration,
            "convergence_90_iteration": convergence_90_iter,
            "max_improvement_iteration": self.iterations[max_improvement_idx].iteration if self.iterations else None,
            "max_improvement_value": max(improvements) if improvements else 0,
            "final_diversity": self.iterations[-1].diversity if self.iterations else None,
            "mean_diversity": np.mean([m.diversity for m in self.iterations]) if self.iterations else None,
            "total_time": self.iterations[-1].elapsed_time if self.iterations else 0,
        }

    def to_dict(self) -> dict[str, Any]:
        """
        Convert history to dictionary for serialization.
        Returns:
            Dictionary representation of history
        """
        result = {
            "summary": self.get_summary_statistics(),
            "iterations": [m.to_dict() for m in self.iterations],
        }
        if self.track_positions:
            # Only include first and last few iterations to save space
            if len(self.particle_positions) > 10:
                result["particle_positions_sample"] = {
                    "first_5": [p.tolist() for p in self.particle_positions[:5]],
                    "last_5": [p.tolist() for p in self.particle_positions[-5:]],
                }
            else:
                result["particle_positions"] = [p.tolist() for p in self.particle_positions]
        return result

    def save_to_json(self, filepath: str):
        """
        Save history to JSON file.
        Args:
            filepath: Path to save the JSON file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str) -> "ExperimentHistory":
        """
        Load history from JSON file.
        Args:
            filepath: Path to the JSON file
        Returns:
            ExperimentHistory object
        """
        with open(filepath) as f:
            data = json.load(f)
        history = cls()
        # Reconstruct iterations
        for iter_data in data.get("iterations", []):
            metrics = IterationMetrics(
                iteration=iter_data["iteration"],
                best_fitness=iter_data["best_fitness"],
                mean_fitness=iter_data["mean_fitness"],
                worst_fitness=iter_data["worst_fitness"],
                fitness_std=iter_data["fitness_std"],
                global_best_position=np.array(iter_data["global_best_position"]),
                diversity=iter_data["diversity"],
                mean_velocity_norm=iter_data["mean_velocity_norm"],
                max_velocity_norm=iter_data["max_velocity_norm"],
                min_velocity_norm=iter_data["min_velocity_norm"],
                improvement=iter_data["improvement"],
                evaluations=iter_data["evaluations"],
                elapsed_time=iter_data["elapsed_time"],
            )
            history.iterations.append(metrics)
        # Restore summary statistics
        summary = data.get("summary", {})
        history.total_iterations = summary.get("total_iterations", 0)
        history.total_evaluations = summary.get("total_evaluations", 0)
        history.initial_best_fitness = summary.get("initial_best_fitness")
        history.final_best_fitness = summary.get("final_best_fitness")
        history.convergence_iteration = summary.get("convergence_iteration")
        return history
