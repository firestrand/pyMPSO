# Component Authoring Guide

This guide is the practical flow for adding new PSO extensions in the current architecture.

## Extension architecture in one paragraph

All pluggable behavior flows through one catalog:

- `src/components/factory.py` owns component registration.
- `run_pso.py` resolves everything through `create_component()`.
- Components are selected in config by `type`.
- No legacy shims are used in execution.

## Supported extension points

| Component key           | Base interface module                   | Config key              | Built-in examples                             |
|-------------------------|----------------------------------------|-------------------------|----------------------------------------------|
| `problem`               | `src/problem/base.py` (`Problem`)       | `problem`               | `sphere`, `ackley`, `rosenbrock`             |
| `topology`              | `src/topology/base.py`                  | `topology`              | `global`, `ring`, `random`                   |
| `influence_strategy`    | `src/influence/base.py`                | `influence_strategy`    | `single_best`, `fips`                        |
| `velocity_strategy`     | `src/velocity/base.py`                 | `velocity_strategy`     | `standard`, `hypersphere`, `fips`            |
| `boundary_handler`      | `src/boundary/base.py`                 | `boundary_handler`      | `clamping`, `damped_reflection`              |
| `initialization_strategy`| `src/initialization/base.py`           | `initialization_strategy`| `random_uniform`, `bounds_aware`             |

## Registration pattern (required)

Add your class with the decorator directly above the class definition:

```python
from src.components.factory import register_component

@register_component("velocity_strategy", "chaos")
class ChaosVelocityUpdate(...):
    ...
```

The registration decorator is the only contract needed for discovery.

## Plugin implementation template

Use the following template as a starting point and adapt the base class and file location to your chosen component type.

```python
from __future__ import annotations

import numpy as np

from src.components.factory import register_component
from src.particles.base import ParticleBase
from src.velocity.base import VelocityUpdateStrategy


@register_component("velocity_strategy", "chaos")
class ChaosVelocityUpdate(VelocityUpdateStrategy):
    """Example velocity strategy plugin for experimentation."""

    def __init__(self, chaos_scale: float = 0.10):
        self.chaos_scale = float(chaos_scale)

    def update(self, particle: ParticleBase, informant_position: np.ndarray, hyperparams: dict) -> np.ndarray:
        rng = hyperparams.get("rng", np.random)
        chaos = rng.uniform(-self.chaos_scale, self.chaos_scale, size=particle.position.shape)
        return particle.velocity + chaos
```

## Wiring into configuration

`run_pso.py` merges `*_params` into the component config, so strategy-specific options are passed through from canonical keys such as
`velocity_strategy_params`, `topology_params`, `influence_strategy_params`, `boundary_handler_params`, and `initialization_strategy_params`.

```yaml
common_pso_params:
  topology: global
  velocity_strategy: chaos
  velocity_strategy_params:
    chaos_scale: 0.15
  initialization_strategy: random_uniform
  boundary_handler: clamping
```

## Minimal validation loop for researchers

1. Add/modify your component implementation and registration.
2. Ensure the module is importable by Python (for in-repo plugins, add `from examples.plugins import chaos_velocity_plugin` or equivalent at startup).
3. Run a config dry-run to validate wiring:

```bash
uv run pso-run --config examples/sphere_config.yaml --dry-run -v
```

4. Run a short single run (`num_runs: 1`) before scaling to benchmark suites.

## Ready-to-run custom component examples

### Example 1: plugin module + YAML config (velocity)

- Component module: `examples/plugins/chaos_velocity_plugin.py`
- Config: `examples/plugins/chaos_velocity_config.yaml`

The config uses `velocity_strategy: chaos_velocity`, wired from the plugin module.

### Example 2: executable demo harness (velocity)

- Script: `examples/plugins/run_custom_velocity_plugin_demo.py`

Run:

```bash
UV_NO_CACHE=1 uv run python examples/plugins/run_custom_velocity_plugin_demo.py
```

That script imports the plugin module first (to register `chaos_velocity`) and then runs the example config.

### Example 3: plugin module + YAML config (topology)

- Component module: `examples/plugins/pair_topology_plugin.py`
- Config: `examples/plugins/pair_topology_config.yaml`

The config uses `topology: paired_ring`, and passes `topology_params.halo` to control neighborhood radius.

### Example 4: executable demo harness (topology)

- Script: `examples/plugins/run_custom_topology_plugin_demo.py`

Run:

```bash
UV_NO_CACHE=1 uv run python examples/plugins/run_custom_topology_plugin_demo.py
```

That script imports the topology plugin first (to register `paired_ring`) and then runs the example config.

## Common extension pitfalls

- New component constructors must accept the arguments they need from config.
- `velocity`/`topology`/`influence` implementations should avoid side effects in constructors.
- Reuse `hyperparams` as the only runtime bridge for per-run shared scalar settings (e.g., RNG, bounds).
- Keep deterministic behavior by using `hyperparams.get("rng")` where randomness is involved.

## High-Performance Vectorized Extensions (New in v0.1)

The framework now supports a high-performance vectorized execution engine (`VectorizedPSOAlgorithm`). When writing components for this engine, you must implement the `Vectorized*` protocols and operate on the `SwarmState` matrix directly.

### The `SwarmState` Object

Instead of iterating over individual `ParticleBase` objects, vectorized components receive a `SwarmState` dataclass containing the entire swarm's data as NumPy matrices of shape `(N_particles, Dimensions)`:

```python
from src.particles.swarm_state import SwarmState

# Example access within a vectorized strategy:
positions = swarm_state.positions        # Shape: (N, D)
velocities = swarm_state.velocities      # Shape: (N, D)
pbest_fitness = swarm_state.pbest_fitness # Shape: (N,)
```

### Writing a Vectorized Component

Vectorized components must be registered using `@register_vectorized_component` and, optionally, `@register_typed_config` if they require specific hyperparameter schemas instead of loose dictionaries.

```python
from src.components.factory import register_vectorized_component, register_typed_config
from src.velocity.vectorized_base import VectorizedVelocityUpdateStrategy
from src.velocity.config import VelocityConfig
from dataclasses import dataclass

@dataclass
class MyVelocityConfig(VelocityConfig):
    my_param: float = 1.0

@register_vectorized_component("velocity_strategy", "my_custom")
@register_typed_config("velocity_strategy", "my_custom")
class VectorizedCustomVelocity(VectorizedVelocityUpdateStrategy):
    def update_velocities(self, swarm_state, informant_matrix, config: MyVelocityConfig, **kwargs):
        # Perform matrix operations
        new_v = swarm_state.velocities * config.my_param
        return new_v
```

## Telemetry & The Observer Pattern

The vectorized engine completely decouples the execution loop from history tracking or printing. If you need to track custom metrics or log specific behaviors, implement an `EventSubscriber` and attach it to the `PSOEventPublisher`.

```python
from src.tracking.events import EventSubscriber

class MyCustomTracker(EventSubscriber):
    def on_iteration_end(self, iteration, swarm_state, **kwargs):
        best_pos, best_fit = swarm_state.get_global_best()
        print(f"Iteration {iteration}: Best is {best_fit}")

# Attachment happens at the orchestrator level, e.g., in run_pso.py:
# pso.event_publisher.add_subscriber(MyCustomTracker())
```
