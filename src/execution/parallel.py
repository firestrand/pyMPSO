from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..planning.domain import BenchmarkTask
from .protocols import ExecutionResult, RunExecutor, RunExecutorError

TaskRunner = Callable[[BenchmarkTask], ExecutionResult]


class RayExecutor(RunExecutor):
    """Ray-backed execution engine for batched worker execution."""

    def __init__(self, run_fn: TaskRunner, num_cpus: int | None = None):
        self._run_fn = run_fn
        self._num_cpus = num_cpus

    def execute(self, task: BenchmarkTask) -> ExecutionResult:
        # Ray execution is inherently batched in this engine; keep a direct single-task path for the protocol.
        return self._run_fn(task)

    def execute_many(self, tasks: Sequence[BenchmarkTask]) -> list[ExecutionResult]:
        try:
            ray: Any = __import__("ray")
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path
            raise RunExecutorError("Parallel execution requested, but Ray is not installed.") from exc
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RunExecutorError("Parallel execution requested, but Ray could not be imported.") from exc

        remote_runner = ray.remote(self._run_fn)
        ray_initialized = bool(ray.is_initialized())
        try:
            if not ray_initialized:
                if self._num_cpus is not None:
                    ray.init(num_cpus=self._num_cpus, include_dashboard=False, ignore_reinit_error=True)
                else:
                    ray.init(include_dashboard=False, ignore_reinit_error=True)
            futures = [remote_runner.remote(task) for task in tasks]
            return list(ray.get(futures))
        finally:
            if not ray_initialized and ray.is_initialized():
                ray.shutdown()
