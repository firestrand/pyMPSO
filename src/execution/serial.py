from __future__ import annotations

from collections.abc import Callable, Sequence

from ..planning.domain import BenchmarkTask
from .protocols import ExecutionResult, RunExecutor

TaskRunner = Callable[[BenchmarkTask], ExecutionResult]


class SerialExecutor(RunExecutor):
    """Default one-process execution engine."""

    def __init__(self, run_fn: TaskRunner):
        self._run_fn = run_fn

    def execute(self, task: BenchmarkTask) -> ExecutionResult:
        return self._run_fn(task)

    def execute_many(self, tasks: Sequence[BenchmarkTask]) -> list[ExecutionResult]:
        return [self.execute(task) for task in tasks]
