from typing import Any

from ..particles.swarm_state import SwarmState


class EventSubscriber:
    """Base class for all PSO event subscribers (Observer pattern)."""

    def on_run_start(self, swarm_state: SwarmState, **kwargs: Any) -> None:
        """Called exactly once before the first iteration."""
        pass

    def on_iteration_start(self, iteration: int, swarm_state: SwarmState, **kwargs: Any) -> None:
        """Called at the beginning of each iteration."""
        pass

    def on_iteration_end(self, iteration: int, swarm_state: SwarmState, **kwargs: Any) -> None:
        """Called at the end of each iteration, after all particles are updated."""
        pass

    def on_run_end(self, swarm_state: SwarmState, **kwargs: Any) -> None:
        """Called exactly once after the algorithm terminates."""
        pass


class PSOEventPublisher:
    """Manages event subscribers and broadcasts events to them."""

    def __init__(self) -> None:
        self._subscribers: list[EventSubscriber] = []

    def add_subscriber(self, subscriber: EventSubscriber) -> None:
        """Register a new subscriber."""
        if subscriber not in self._subscribers:
            self._subscribers.append(subscriber)

    def remove_subscriber(self, subscriber: EventSubscriber) -> None:
        """Unregister an existing subscriber."""
        if subscriber in self._subscribers:
            self._subscribers.remove(subscriber)

    def notify_run_start(self, swarm_state: SwarmState, **kwargs: Any) -> None:
        for subscriber in self._subscribers:
            subscriber.on_run_start(swarm_state, **kwargs)

    def notify_iteration_start(self, iteration: int, swarm_state: SwarmState, **kwargs: Any) -> None:
        for subscriber in self._subscribers:
            subscriber.on_iteration_start(iteration, swarm_state, **kwargs)

    def notify_iteration_end(self, iteration: int, swarm_state: SwarmState, **kwargs: Any) -> None:
        for subscriber in self._subscribers:
            subscriber.on_iteration_end(iteration, swarm_state, **kwargs)

    def notify_run_end(self, swarm_state: SwarmState, **kwargs: Any) -> None:
        for subscriber in self._subscribers:
            subscriber.on_run_end(swarm_state, **kwargs)
