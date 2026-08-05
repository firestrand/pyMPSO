"""Execution strategy and payload abstractions."""

from .protocols import (
    ExecutionResult as ExecutionResult,
)
from .protocols import (
    RunExecutor as RunExecutor,
)
from .protocols import (
    RunExecutorError as RunExecutorError,
)
from .protocols import (
    RunExecutorErrorContext as RunExecutorErrorContext,
)

__all__ = [
    "ExecutionResult",
    "RunExecutor",
    "RunExecutorError",
    "RunExecutorErrorContext",
]
