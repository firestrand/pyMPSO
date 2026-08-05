import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .boundary.vectorized_base import VectorizedBoundaryHandler
from .influence.vectorized_base import VectorizedInfluenceStrategy
from .initialization.base import InitializationStrategy
from .particles.swarm_state import SwarmState
from .position.base import PositionUpdateStrategy
from .problem.base import Problem
from .stopping.base import StoppingCriteria
from .topology.config import TopologyConfig
from .topology.vectorized_base import VectorizedTopologyStrategy
from .tracking.events import PSOEventPublisher
from .velocity.config import VelocityConfig
from .velocity.vectorized_base import VectorizedVelocityUpdateStrategy


class VectorizedPSOAlgorithm:
    """
    High-performance, fully vectorized PSO orchestrator.
    Executes the main optimization loop using matrix operations (NumPy)
    instead of per-particle Python loops. Dispatches events to an Observer
    pattern instead of hardcoding history tracking.
    """

    def __init__(
        self,
        problem: Problem,
        stopping_criteria: StoppingCriteria,
        initialization_strategy: InitializationStrategy,
        topology: VectorizedTopologyStrategy,
        topology_config: TopologyConfig,
        influence_strategy: VectorizedInfluenceStrategy,
        velocity_strategy: VectorizedVelocityUpdateStrategy,
        velocity_config: VelocityConfig,
        position_strategy: PositionUpdateStrategy,
        boundary_handler: VectorizedBoundaryHandler,
        num_particles: int,
        max_iterations: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.problem = problem
        self.stopping_criteria = stopping_criteria
        self.initialization_strategy = initialization_strategy
        self.topology = topology
        self.topology_config = topology_config
        self.influence_strategy = influence_strategy
        self.velocity_strategy = velocity_strategy
        self.velocity_config = velocity_config
        self.position_strategy = position_strategy
        self.boundary_handler = boundary_handler
        self.max_iterations = max_iterations

        if num_particles <= 0:
            raise ValueError("Number of particles must be positive.")

        self.num_particles = num_particles
        self.verbose = verbose

        self.bounds = initialization_strategy.bounds
        self.dimensions = initialization_strategy.dimensions

        # Telemetry via Observer pattern
        self.event_publisher = PSOEventPublisher()

        # State tracking
        self.global_best_position: NDArray[np.float64] | None = None
        self.global_best_fitness: float = np.inf
        self.iteration: int = 0
        self.evaluations_used: int = 0
        self.start_time: float = 0.0
        self.elapsed_time: float = 0.0
        self.iterations_without_improvement: int = 0
        self.swarm_state_dict: dict[str, Any] = {}

    def _initialize_swarm_state(self) -> SwarmState:
        # Generate initial positions using the initialization strategy
        pos_list = []
        vel_list = []
        for _ in range(self.num_particles):
            pos, vel = self.initialization_strategy.generate_next_particle()
            pos_list.append(pos)
            vel_list.append(vel)

        positions = np.array(pos_list)
        velocities = np.array(vel_list)

        # Evaluate initial fitness (batch mode for performance)
        # Note: If the problem doesn't support batch evaluation, we must map it.
        # But we aim for vectorized evaluation if supported. For now, evaluate each.
        current_fitness = np.array([self.problem.evaluate(pos) for pos in positions])
        self.evaluations_used += self.num_particles

        pbest_positions = positions.copy()
        pbest_fitness = current_fitness.copy()

        return SwarmState(
            positions=positions,
            velocities=velocities,
            pbest_positions=pbest_positions,
            pbest_fitness=pbest_fitness,
            current_fitness=current_fitness,
        )

    def _update_global_best(self, swarm_state: SwarmState) -> None:
        best_pos, best_fit = swarm_state.get_global_best()
        if best_fit < self.global_best_fitness:
            self.global_best_fitness = best_fit
            self.global_best_position = best_pos
            self.iterations_without_improvement = 0
        else:
            self.iterations_without_improvement += 1

    def _build_orchestrator_state(self) -> dict[str, Any]:
        self.elapsed_time = time.time() - self.start_time
        return {
            "current_iteration": self.iteration,
            "best_fitness": self.global_best_fitness,
            "best_position": self.global_best_position,
            "evaluations_used": self.evaluations_used,
            "iterations_without_improvement": self.iterations_without_improvement,
            "elapsed_time": self.elapsed_time,
        }

    def run(self) -> dict[str, Any]:
        """Execute the vectorized PSO optimization loop."""
        self.start_time = time.time()
        self.iteration = 0
        self.evaluations_used = 0
        self.iterations_without_improvement = 0
        self.global_best_fitness = np.inf
        self.global_best_position = None

        swarm_state = self._initialize_swarm_state()
        self._update_global_best(swarm_state)

        self.event_publisher.notify_run_start(swarm_state)

        while True:
            orch_state = self._build_orchestrator_state()
            if self.stopping_criteria.should_stop(orch_state):
                break

            self.event_publisher.notify_iteration_start(self.iteration, swarm_state)

            # 1. Topology updates (who sees whom)
            topology_matrix = self.topology.get_topology_matrix(swarm_state, self.topology_config)

            # 2. Influence updates (who do I follow)
            informant_matrix = self.influence_strategy.get_informant_matrix(swarm_state, topology_matrix)

            # 3. Velocity updates
            new_velocities = self.velocity_strategy.update_velocities(
                swarm_state,
                informant_matrix,
                self.velocity_config,
                current_iteration=self.iteration,
                max_iterations=self.max_iterations,
            )
            new_velocities = self._apply_velocity_clamping(new_velocities)
            swarm_state.velocities = new_velocities

            # 4. Position updates
            new_positions = self.position_strategy.update_positions(swarm_state)
            swarm_state.positions = new_positions

            # 5. Boundary Handling (applied in-place to SwarmState)
            self.boundary_handler.apply(swarm_state, self.bounds)

            # 6. Evaluation (Compatibility with un-vectorized objective functions)
            # Ideally: swarm_state.current_fitness = self.problem.evaluate(swarm_state.positions)
            current_fitness = np.array([self.problem.evaluate(pos) for pos in swarm_state.positions])
            swarm_state.current_fitness = current_fitness
            self.evaluations_used += self.num_particles

            # 7. Update pbests
            improved_mask = swarm_state.current_fitness < swarm_state.pbest_fitness
            swarm_state.pbest_fitness[improved_mask] = swarm_state.current_fitness[improved_mask]
            swarm_state.pbest_positions[improved_mask] = swarm_state.positions[improved_mask]

            # 8. Update gbest
            self._update_global_best(swarm_state)

            self.event_publisher.notify_iteration_end(
                self.iteration, swarm_state, evaluations_used=self.evaluations_used
            )
            self.iteration += 1

        self.event_publisher.notify_run_end(swarm_state)

        return {
            "best_position": self.global_best_position,
            "best_fitness": self.global_best_fitness,
            "iterations": self.iteration,
            "evaluations": self.evaluations_used,
            "elapsed_time": self.elapsed_time,
        }

    def _apply_velocity_clamping(self, velocities: np.ndarray) -> np.ndarray:
        """Apply configured velocity clamping strategy if configured."""
        clamping_strategy = getattr(self.velocity_config, "velocity_clamping", None)
        if clamping_strategy is None or self.bounds is None:
            return velocities

        if len(velocities) == 0:
            return velocities

        return np.asarray(
            [clamping_strategy.clamp(velocity, self.bounds, {}) for velocity in velocities],
            dtype=np.float64,
        )
