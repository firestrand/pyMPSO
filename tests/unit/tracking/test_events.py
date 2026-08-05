from typing import Any

import numpy as np

from src.particles.swarm_state import SwarmState
from src.tracking.events import EventSubscriber, PSOEventPublisher


class MockSubscriber(EventSubscriber):
    def __init__(self):
        self.events_received = []

    def on_run_start(self, swarm_state: SwarmState, **kwargs: Any) -> None:  # noqa: ARG002
        self.events_received.append(("on_run_start", kwargs))

    def on_iteration_start(self, iteration: int, swarm_state: SwarmState, **kwargs: Any) -> None:  # noqa: ARG002
        self.events_received.append(("on_iteration_start", iteration, kwargs))

    def on_iteration_end(self, iteration: int, swarm_state: SwarmState, **kwargs: Any) -> None:  # noqa: ARG002
        self.events_received.append(("on_iteration_end", iteration, kwargs))

    def on_run_end(self, swarm_state: SwarmState, **kwargs: Any) -> None:  # noqa: ARG002
        self.events_received.append(("on_run_end", kwargs))


def test_event_publisher_notifies_subscribers():
    publisher = PSOEventPublisher()
    subscriber1 = MockSubscriber()
    subscriber2 = MockSubscriber()

    publisher.add_subscriber(subscriber1)
    publisher.add_subscriber(subscriber2)

    # Create dummy state
    state = SwarmState(
        positions=np.zeros((2, 2)),
        velocities=np.zeros((2, 2)),
        pbest_positions=np.zeros((2, 2)),
        pbest_fitness=np.zeros(2),
        current_fitness=np.zeros(2),
    )

    # Trigger events
    publisher.notify_run_start(state, extra="data")
    publisher.notify_iteration_start(1, state)
    publisher.notify_iteration_end(1, state, metric=0.5)
    publisher.notify_run_end(state)

    # Verify subscriber 1
    assert len(subscriber1.events_received) == 4
    assert subscriber1.events_received[0] == ("on_run_start", {"extra": "data"})
    assert subscriber1.events_received[1] == ("on_iteration_start", 1, {})
    assert subscriber1.events_received[2] == ("on_iteration_end", 1, {"metric": 0.5})
    assert subscriber1.events_received[3] == ("on_run_end", {})

    # Verify subscriber 2
    assert subscriber1.events_received == subscriber2.events_received


def test_event_publisher_remove_subscriber():
    publisher = PSOEventPublisher()
    subscriber = MockSubscriber()

    publisher.add_subscriber(subscriber)
    publisher.remove_subscriber(subscriber)

    state = SwarmState(
        positions=np.zeros((2, 2)),
        velocities=np.zeros((2, 2)),
        pbest_positions=np.zeros((2, 2)),
        pbest_fitness=np.zeros(2),
        current_fitness=np.zeros(2),
    )

    publisher.notify_run_start(state)
    assert len(subscriber.events_received) == 0


def test_event_subscriber_default_methods():
    """Verify default methods don't raise errors."""
    subscriber = EventSubscriber()
    state = SwarmState(
        positions=np.zeros((2, 2)),
        velocities=np.zeros((2, 2)),
        pbest_positions=np.zeros((2, 2)),
        pbest_fitness=np.zeros(2),
        current_fitness=np.zeros(2),
    )

    # Just calling them to hit the pass statements and ensure no NotImplmentedError
    subscriber.on_run_start(state)
    subscriber.on_iteration_start(1, state)
    subscriber.on_iteration_end(1, state)
    subscriber.on_run_end(state)
