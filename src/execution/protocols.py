from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, TypedDict

from ..planning.domain import BenchmarkTask


class ExecutionResult(TypedDict, total=False):
    """Canonical execution payload returned by an engine."""

    benchmark_task: str
    config_file: str
    problem_type: str | None
    problem_dimension: int | None
    num_particles: int
    run_index: int
    seed_used: int | None
    best_fitness: float
    iterations: int
    evaluations: int
    total_wall_time: float
    error: str


class RunExecutorError(RuntimeError):
    """Execution-time error from a strategy or backend."""


@dataclass(frozen=True)
class RunExecutorErrorContext:
    """Context captured when summarizing failed parallel execution."""

    task_name: str
    reason: str


class RunExecutor(Protocol):
    """Protocol for execution strategies."""

    def execute(self, task: BenchmarkTask) -> ExecutionResult:
        """Run one benchmark task and return a single task result."""
        ...

    def execute_many(self, tasks: Sequence[BenchmarkTask]) -> list[ExecutionResult]:
        """Run a batch of benchmark tasks.

        Backends that only support single execution may implement this via
        repeated single-task execution.
        """
        ...


def default_execute_many(executor: RunExecutor, tasks: Sequence[BenchmarkTask]) -> list[ExecutionResult]:
    """Utility used by concrete strategies to provide a trivial batching policy."""

    return [executor.execute(task) for task in tasks]


ExecutorFactory = Callable[[], RunExecutor]
