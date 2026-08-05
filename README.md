# Modular Particle Swarm Optimization (PSO) Framework

A flexible, modular, and extensible Python framework for Particle Swarm Optimization (PSO).

## Features

- **Modular Design:** Swap components like topology, velocity update, boundary handling, and more
- **Flexible Configuration:** Configure via YAML files or programmatically
- **Composable Stopping Criteria:** Create complex stopping conditions with logical operations
- **Problem-Agnostic:** Works with any objective function, with built-in benchmark problems
- **Extensible:** Easy to add new components and strategies
- **Velocity Clamping:** Multiple strategies for controlling particle velocities
- **Benchmarking Support:** Run multiple experiments with different configurations

## Installation

This package requires Python 3.12 or higher.

```bash
# Clone the repository
git clone https://github.com/yourusername/pyMPSO.git
cd pyMPSO

# Install dependencies with uv (dev and quality are default groups)
uv sync
```

## Quick Start

### Run with a Configuration File

The easiest way to use the framework is with a YAML or JSON configuration file:

```bash
uv run pso-run --config examples/sphere_config.yaml -v
```

The CLI also accepts JSON config files:

```bash
uv run pso-run --config examples/ackley_config.json -v
```

This runs a basic PSO on the Sphere function with the configured parameters. For more details, see the [Configuration](#configuration) section.

### Program with the Framework

You can also use the framework programmatically:

```python
from src.run_pso import run_pso_from_config

# Create an in-memory config and run it directly
config_data = {
    "problems_to_benchmark": {
        "sphere": {
            "dimensions": 10,
            "bounds": {"min": -10.0, "max": 10.0},
            "stopping_criteria": {"type": "max_iterations", "max_iterations": 100},
        }
    },
    "common_swarm": {
        "num_particles": 30,
    },
    "common_pso_params": {
        "topology": "global",
        "velocity_strategy": "standard",
        "influence_strategy": "single_best",
        "boundary_handler": "clamping",
        "initialization_strategy": "random_uniform",
    },
    "common_run_config": {
        "num_runs": 1,
        "verbose": False,
    },
}

results = run_pso_from_config(config_path="in-memory", config_data=config_data)
print(f"Best fitness: {results['sphere_10D']['best_fitness']}")
```

## Configuration

The framework consumes benchmark-style YAML or JSON configs with four top-level sections:

1. `problems_to_benchmark`: one or more named problem tasks
2. `common_swarm`: shared swarm-level defaults
3. `common_pso_params`: shared PSO component/config defaults
4. `common_run_config`: shared run behavior (number of runs, seed, verbosity)
5. `pso_variants` (optional): named variant descriptors for reuse across all tasks

`pso_variants` entries are merged as override descriptors (`pso_params`, `swarm`, `run_config`) over common defaults.

Execution flow is now explicitly variant-aware:

1. parse and validate config
2. expand each problem x variant into benchmark tasks
3. apply variant selectors (`--include-variants`, `--exclude-variants`)
4. execute the task matrix with shared orchestration

`common_run_config.random_seed` behavior:
- If `num_runs > 1` and `random_seed` is set, the CLI derives deterministic per-run seeds from the base seed (`seed + task + run_index`) for reproducible 100-run protocol comparisons.
- If `random_seed` is `null`, each run receives a fresh random seed and runs are compared using aggregate statistics rather than per-seed parity.

Example configuration (`examples/variants/variant_matrix.yaml` style):

```yaml
problems_to_benchmark:
  sphere:
    dimensions: 20
    bounds:
      min: -100.0
      max: 100.0
    stopping_criteria:
      - { type: max_iterations, max_iterations: 3000 }

common_swarm:
  num_particles: 30

common_pso_params:
  topology: global
  initialization_strategy: random_uniform
  velocity_strategy: standard
  boundary_handler: clamping
  velocity_strategy_params:
    c1: 1.49445
    c2: 1.49445

common_run_config:
  num_runs: 1
  random_seed: null
  verbose: false

pso_variants:
  default: {}
  constriction:
    pso_params:
      velocity_strategy: constriction
      velocity_strategy_params:
        phi1: 2.05
        phi2: 2.05
  high_population:
    swarm:
      num_particles: 60
  short_runs:
    run_config:
      num_runs: 2
```

Strategy hyperparameters belong in the matching `<component>_params` block, never directly under
`common_pso_params` — the loader rejects unknown top-level keys. Canonical blocks:
`velocity_strategy_params`, `topology_params`, `influence_strategy_params`,
`boundary_handler_params`, `initialization_strategy_params`.

Stopping criteria behavior:
- A list form (multiple `- { ... }` entries) is interpreted as `any` (logical OR) semantics.
- Explicit composite forms are still supported with `any:` and `all:` keys.

Example execution with variant selection:

```bash
uv run pso-run --config examples/variants/variant_matrix.yaml --include-variants default,high_population
uv run pso-run --config examples/variants/variant_matrix.yaml --exclude-variants hypersphere_ring,bounded_constriction
```

Selection is applied before the execution-support check, so narrowing to runnable variants is how you
use a matrix that also contains scalar-only entries.

For cleaner experiment planning, variant matrices are split into:

- `examples/variants/all_known_variants_core.yaml` (canonical algorithm families for quick comparison)
- `examples/variants/all_known_variants_extended.yaml` (parameter ablations and boundary/initialization/topology variants)
- `examples/variants/all_known_variants.yaml` (curated default launch preset aligned with core)
- `examples/variants/sota_variants.yaml` (config-first catalog for implemented SOTA-style families)
- `examples/variants/presets/*.yaml` (one file per variant name; includes all cataloged variants plus `qpso` and `apso`)

An index lives in:

- `examples/variants/README.md`

Examples:

```bash
uv run pso-run --config examples/variants/all_known_variants_core.yaml --include-variants default,qpso,apso
uv run pso-run --config examples/variants/all_known_variants_extended.yaml --include-variants constriction_fixed_chi,qpso_decay,apso_stable
uv run pso-run --config examples/variants/sota_variants.yaml --include-variants standard_global,constriction,qpso,apso
uv run pso-run --config examples/variants/presets/qpso.yaml --include-variants qpso
uv run pso-run --config examples/variants/presets/apso.yaml --include-variants apso
```

The catalogs also contain scalar-only variants (FIPS, CLPSO, bare-bones, DE-hybrid, inertia-weight,
orthogonal learning/mutation). Selecting one of those raises a "not supported by vectorized execution"
error — 33 of the 49 files in `examples/variants/presets/` currently run. Use `--dry-run` to check a
selection cheaply.

See the `examples/` directory for more detailed configuration examples.

## Available PSO Variants

> **Execution support.** Non-vectorized execution has been removed. A component is runnable only if
> it has a **vectorized** implementation registered in `src/components/factory.py`; configs selecting
> anything else are rejected during validation with a "not supported by vectorized execution" error.
> Strategies marked _(scalar only)_ below are implemented and unit-tested but **cannot currently be
> run** — porting them to the vectorized engine is the active workstream. Check any config with
> `--dry-run` before relying on it.

### Velocity Update Strategies
- **Standard** — classic PSO update, `v + c1·r1·(p−x) + c2·r2·(g−x)`. Note this is the 1995 form with
  **no inertia weight**; it is not contractive on its own, so pair it with velocity clamping or prefer
  constriction.
- **Constriction** — constriction coefficient approach (Clerc & Kennedy 2002)
- **Hypersphere** — SPSO 2011 rotation-invariant update
- **QPSO** — quantum-behaved PSO with stochastic attractor updates
- **APSO** — adaptive PSO with time-varying inertia/coefficient scheduling
- **Inertia Weight** _(scalar only)_ — time-varying or fixed inertia weight
- **FIPS** _(scalar only)_ — Fully Informed Particle Swarm
- **Bare Bones** _(scalar only)_ — Gaussian sampling without velocity
- **CLPSO** _(scalar only)_ — Comprehensive Learning PSO
- **Orthogonal Mutation** _(scalar only)_ — standard update plus an orthogonal mutation term (arXiv-inspired)
- **DE Hybrid** _(scalar only)_ — Differential Evolution mutation/crossover layered onto social guidance (arXiv-inspired)

### Influence Strategies
- **Single Best** — classic social learning toward the global/local best exemplar
- **FIPS** _(scalar only)_ — fully informed social model based on neighbor bests
- **Orthogonal Learning** _(scalar only)_ — dimension-wise exemplar selection from particle bests and sampled peers (arXiv-inspired)

### Topologies
- **Global** — all particles are neighbors (fully connected)
- **Ring** — each particle informed by its `k` neighbors on each side plus itself (`topology_params: {k: 2}`)
- **Random** — dynamic SPSO 2011 random topology (`topology_params: {k: 3, rebuild_probability: 1.0}`)

### Boundary Handlers
- **Position Clamping** — hard boundary enforcement (config keys `clamping` and `position_clamping` are aliases)
- **Damped Reflection** — SPSO 2011-style reflection: position clamped to the bound, offending velocity
  component reversed and halved

### Initialization Strategies

Initialization is resolved on the scalar path and is not gated; both options are runnable.

- **Random Uniform** — uniform random initialization within bounds
- **Bounds Aware** — SPSO 2011-style initialization ensuring x+v stays in bounds

### SPSO 2011

`examples/spso2011_benchmark.yaml` composes random topology + hypersphere velocity + damped
reflection over the CEC-2005 problem set (`sphere`, `rastrigin`, `rosenbrock`, `schwefel`, `griewank`,
`ackley`). The full stack runs on the vectorized engine:

```bash
uv run pso-run --config examples/spso2011_benchmark.yaml -v
```

Indicative quality, sphere 10D, 40 particles, 1000 iterations, 10 seeded runs — mean best fitness:

| Stack | sphere 10D | rastrigin 10D |
|---|---|---|
| SPSO 2011 (random + hypersphere + damped reflection) | 2.5e-63 | 8.60 |
| ring k=2 + constriction | 6.2e-31 | 5.17 |
| global + constriction | 2.4e-47 | 6.47 |

Comparison against the C reference is validated statistically rather than by seed-by-seed parity: run
the same configuration and run count (typically 100 runs), then compare mean and standard deviation of
best fitness plus success rate. See `scripts/migration_readiness.md`.

> The helper scripts `scripts/verify_spso2011_parity.py` and
> `scripts/compare_spso2011_c_vs_python.py` still import `src.pso_algorithm`, a module removed in the
> vectorized refactor, and do not run. They need porting to `VectorizedPSOAlgorithm`.

## Benchmarking

The command supports multi-task benchmark runs in one file by listing multiple tasks under `problems_to_benchmark`.

Then run the benchmark:

```bash
uv run pso-run --config examples/benchmark_config.yaml -v
```

You can also pass the config path as a short option:

```bash
uv run pso-run -c examples/benchmark_config.yaml -v
``` 

## Command-Line Interface

```bash
uv run pso-run --help
```

Common options:

- `--config/-c`: path to JSON or YAML config
- `--output/-o`: optional output file path
- `--output-format/-f`: `json` or `yaml` (or inferred from extension)
- `--dry-run`: validate config without executing
- `--no-color`: disable Rich color output
- `--include-variants`: comma-separated variant names to include from `pso_variants`
- `--exclude-variants`: comma-separated variant names to skip from `pso_variants`

## Development and Quality Tooling

Use `uv` with `ruff` and `ty` for formatting, linting, and type checking:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run ty check src tests
uv sync --group dev --group quality
```

## Extending the Framework (Based on Expanded Taxonomy)

The framework is designed for extensibility, aligning with a comprehensive view of PSO components. To add a new PSO variant or technique:

1.  **Identify the Layer:** Determine which algorithmic layer your modification affects (e.g., Update Operator, Learning Pool, Diversity Mechanism) based on the taxonomy below.
2.  **Create New Module/Directory (if needed):** If the layer doesn't have a corresponding directory in `src/` (e.g., `src/diversity/`, `src/parameter_control/`), create it.
3.  **Subclass the Base Interface:** Create a new Python class inheriting from the relevant base class (usually found in `src/<layer>/base.py`). If a base class for a new layer doesn't exist, define a suitable abstract interface.
4.  **Implement the Strategy:** Code the logic for your new component (e.g., a new velocity update rule, a different learning strategy).
5.  **Register the Component:** Add your class with `@register_component("<component_type>", "<component_key>")` in `src/components/factory.py` so it is part of the component catalog.
6.  **Configure:** Use the new key in your YAML configuration file to select your custom component.
7.  **Verify:** Run with `--dry-run` before executing full benchmark suites to validate config and component resolution.

> *Example:* Adding a hypothetical "Chaos-Infused Velocity" strategy.
> 1. Layer: `Update Operator` (maps to `velocity/`)
> 2. Module: Create `src/velocity/chaos_infused.py`
> 3. Subclass: `class ChaosVelocity(VelocityUpdateStrategy): ...`
> 4. Implement: Define the chaotic velocity logic in the `update` method.
> 5. Register: Add `@register_component("velocity_strategy", "chaos")` in `src/components/factory.py`.
> 6. Configure: Use `velocity_strategy: chaos` in your YAML.

See `docs/component-authoring.md` for a fully scaffolded plugin workflow, including:

- base interface mapping by component type,
- a minimal plugin skeleton,
- registration + validation checklist,
- reproducible smoke-test pattern.

## Expanded Component Taxonomy ("Everything is a Module")

This framework aims to support a wide range of PSO variations by modularizing the algorithm according to the following layers. Existing modules in `src/` map to these layers, and new modules can be added to accommodate other layers.

| Layer                     | Coverage & Role                                             | Existing Modules (`src/`)             | Example Plug-ins & Variants (from Literature)                                                                                                |
| :------------------------ | :---------------------------------------------------------- | :------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------- |
| **Representation**        | How particle state (position) is encoded & interpreted.     | `particles/` (Standard vector)      | • *Q-bit amplitudes & observation* (QPSO, QIPSO)                                                                                             |
| **Initialisation**        | Seeding the swarm's initial positions and velocities.       | `initialization/`                   | • **Bounds-aware uniform draw** (SPSO 2011) <br> • *Opposition/divided-opposition seeding* (OBPSO, D-PSO) <br> • *Quasi-random Sobol seeds* |
| **Exemplar / Learning** | Determining the influential points (exemplars) guiding motion. | `topology/`, `influence/`           | • **Ring/Global/etc. + Best Informant** (SPSO 2007/11) <br> • *Comprehensive learning table* (CLPSO) <br> • *Orthogonal learning matrix* (OLPSO) |
| **Update Operator**       | The core motion law (velocity/position update).             | `velocity/`, `position/`            | • **Standard Inertia Weight** <br> • **Constriction Factor** (SPSO 2007) <br> • **Hypersphere Sampling** (SPSO 2011) <br> • *δ-potential well quantum update* (QPSO) <br> • *Gaussian/Cauchy sampling* (BBPSO) <br> • *Lévy flight jump* (LOPSO) <br> • *Gravitational fusion* (PSOGSA) |
| **Parameter Control**     | Adapting algorithm parameters (`w`, `c1`, `c2`, `χ`, etc.).   | *(Implicit via config)*             | • *Linear/Non-linear decay* <br> • *Adaptive-velocity feedback* (APSO-VI) <br> • *Fuzzy/chaotic maps*                                          |
| **Diversity / Mutation**  | Injecting randomness or modifications to escape stagnation.   | *(None currently)*                  | • *Gaussian/Cauchy jumps* (BBPSO) <br> • *Random re-initialisation* <br> • *Lévy flight perturbation* (LOPSO)                          |
| **Niching & Speciation**  | Strategies for finding and maintaining multiple optima.     | *(None currently)*                  | • *Fitness-sharing radius adaptation* <br> • *Equilibrium-factor sharing* (Niche-PSO)                                                   |
| **Multi-Swarm Coord.**    | Managing interactions between multiple sub-swarms.          | *(None currently)*                  | • *Hierarchical regrouping/exclusion* (MSPSO) <br> • *Cooperative sub-swarm decomposition* (CPSO)                                    |
| **Hybridisation Ops**     | Integrating operators from other metaheuristics (DE, GA).   | *(None currently)*                  | • *DE mutation/crossover stack* (DEPSO)                                                                                      |
| **Constraint Handler**    | Managing search space boundaries and constraints.           | `boundary/`, `clamping/` | • **Clamping + Zero Velocity** <br> • *Damped bounce* (SPSO 2011) <br> • *Feasibility-first rules* (MOPSO)                             |
| **Archive & Pareto Layer**| Storing non-dominated solutions (for MOPSO).              | *(None currently)*                  | • *External archive + crowding distance* (MOPSO family)                                                                                    |
| **Random Generator**      | Source and distribution of randomness used.                 | `np.random` (Default)             | • *KISS generator* (SPSO spec) <br> • *Mersenne Twister* <br> • *Specific distributions* (Gaussian, Cauchy, Lévy)                          |

*(Italicised examples represent components not yet implemented but supported by the modular design).* 

**Core Orchestration Files:**

- **`vectorized_pso_algorithm.py`**: The high-performance vectorized engine that runs optimized PSO execution.
- **`components/factory.py`**: Central component registry and decorator-based registration interface for modular PSO layers.
- **`cli.py`**: Typer/Rich command-line entrypoint for running experiments defined in JSON/YAML configuration files.

By thinking in these modular layers, virtually any PSO variant can be implemented or reconstructed within this single framework.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
