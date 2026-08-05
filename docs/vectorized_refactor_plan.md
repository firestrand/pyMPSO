# Development Plan: High-Performance Vectorized PSO Refactor

**Strategy:** Domain-First. Goal is to solidify the business logic (vectorized swarm physics) and decouple telemetry, configuration, and movement semantics to achieve high performance and maintainability.

## GOTCHA Design Specification

- **G — Goals:** Transition the pyMPSO framework from a slow, object-oriented, particle-by-particle loop to a high-performance, NumPy-vectorized matrix engine while maintaining its modular strategy pattern.
- **O — Objectives:** 
  - Achieve at least a 10x-50x execution speedup on standard benchmarks over the current pure-Python loop.
  - Eliminate the `step_hyperparams` "bag of parameters" dictionary in favor of strongly-typed data schemas.
  - Strictly separate execution from orchestration/telemetry by decoupling `ExperimentHistory` via an Observer pattern.
- **T — Tasks:** Triggered by a standard CLI benchmark request. Ends when the benchmark suite results are aggregated and written to disk.
- **C — Capabilities:** The new engine must support existing benchmark functions, global/random topologies, and standard velocity strategies. Complex legacy strategies may temporarily fall back or fail fast until fully migrated.
- **H — Health:** Maintain ≥95% test coverage for the core domain and ≥90% for orchestration.
- **A — Attributes:**
  - *Invariant:* The core `VectorizedPSOAlgorithm` must contain zero I/O operations (no printing, no disk writes).
  - *Invariant:* State changes happen simultaneously across the swarm via matrix operations; sequential particle iteration is forbidden in the hot path.
- **Constraints:** Must remain backward-compatible with existing user YAML/JSON benchmark configuration files.
- **Users:** Researchers and developers using pyMPSO for algorithmic experimentation.
- **Memory:** `SwarmState` represents working memory during a run. `ExperimentHistory` acts as the persistent record.
- **Action space:** Pure functional matrix transformations representing physics updates.
- **Decision loop:** Config Parse -> Task Plan -> Instantiate Swarm -> (Loop: Topology -> Velocity -> Position -> Eval) -> Terminate via Criteria -> Emit Results.
- **Beliefs:** The current architecture suffers from severe Python loop overhead and loose typing that hinders confident refactoring.
- **Intentions:** We commit to defining the `SwarmState` data schema and typed configuration boundaries first (avoiding schema-last anti-patterns) before replacing the orchestration engine.

## Phase 0: Foundation & Interfaces (Coverage ≥95%)
- [x] **Task 0.1:** **Test** `SwarmState` data structure (NumPy matrices for position, velocity, pbest, fitness).
- [x] **Task 0.2:** **Implement** `SwarmState` dataclass to replace lists of `ParticleBase` objects.
- [x] **Task 0.3:** **Test** `PSOEventPublisher` protocol/interface for the Observer pattern.
- [x] **Task 0.4:** **Implement** `PSOEventPublisher` and `EventSubscriber` base classes.
- [x] **Task 0.5:** **Test** Typed configuration dataclasses for specific strategies (e.g., `StandardVelocityConfig`, `GlobalTopologyConfig`).
- [x] **Task 0.6:** **Implement** Typed configuration dataclasses to replace loosely typed `step_hyperparams` dicts.
- [x] **Task 0.7:** **Test** `PositionUpdateStrategy` protocol/interface to unify movement semantics.
- [x] **Task 0.8:** **Implement** `PositionUpdateStrategy` protocol/interface.
- [x] **Task 0.9:** **Test** behavior when `StoppingCriteria` are checked only via `should_stop(swarm_state)` without internal introspection.
- [x] **Task 0.10:** **Refactor** `PSOAlgorithm` to remove `_extract_max_evaluations` and `_resolve_max_iterations`, relying exclusively on the `StoppingCriteria` interface.
→ Stage for human review

## Phase 1: Vectorized Engine & Core Mechanics (Coverage ≥95%)
- [x] **Task 1.1:** **Test** `VectorizedPSOAlgorithm` core loop utilizing `SwarmState`.
- [x] **Task 1.2:** **Implement** `VectorizedPSOAlgorithm` loop (removing historical tracking logic, firing events instead).
- [x] **Task 1.3:** **Test** `StandardPositionUpdate` strategy using NumPy vectorization (`X = X + V`).
- [x] **Task 1.4:** **Implement** `StandardPositionUpdate` strategy.
- [x] **Task 1.5:** **Test** `StandardVelocityUpdate` strategy utilizing `SwarmState` matrices and `StandardVelocityConfig`.
- [x] **Task 1.6:** **Implement** `StandardVelocityUpdate` strategy with NumPy vectorization.
- [x] **Task 1.7:** **Test** `GlobalTopology` strategy returning an informant matrix instead of a per-particle dict.
- [x] **Task 1.8:** **Implement** `GlobalTopology` strategy for vectorized informants.
- [x] **Task 1.9:** **Test** `SingleBestInfluence` strategy utilizing `SwarmState`.
- [x] **Task 1.10:** **Implement** `SingleBestInfluence` strategy.
- [x] **Task 1.11:** **Test** boundary handlers (e.g., `PositionClampingBoundaryHandler`) applied to `SwarmState` matrices.
- [x] **Task 1.12:** **Implement** vectorized boundary handlers.
→ Stage for human review

## Phase 2: Telemetry & Orchestration Integration (Coverage ≥90%)
- [x] **Task 2.1:** **Test** `ExperimentHistory` functioning as an `EventSubscriber` to `VectorizedPSOAlgorithm`.
- [x] **Task 2.2:** **Implement** `ExperimentHistory` subscriber binding.
- [x] **Task 2.3:** **Test** the CLI/Orchestrator instantiating and running `VectorizedPSOAlgorithm` with the new typed configs.
- [x] **Task 2.4:** **Implement** adapter layer in `pso_orchestrator.py` and `factory.py` to map legacy dict configs to the new typed domain models before execution.
- [x] **Task 2.5:** **Test** End-to-end benchmark run using the vectorized engine (happy path for standard components).
- [x] **Task 2.6:** **Implement** wiring to use `VectorizedPSOAlgorithm` as the default execution engine in `run_pso.py` for supported configurations.
→ Stage for human review

## Phase 3: Legacy Migration & Polish (Coverage ≥90%)
- [x] **Task 3.1:** **Test** all legacy strategies (e.g., `RingTopology`, `FIPSVelocityUpdate`) to ensure they are either adapted to vectorized interfaces or gracefully fall back to a compatibility mode.
- [x] **Task 3.2:** **Refactor** remaining strategies to use NumPy vectorization and typed configs.
- [x] **Task 3.3:** **Document** the new `SwarmState` and `EventSubscriber` APIs in `docs/component-authoring.md`.
- [x] **Task 3.4:** **Document** performance improvements and updated benchmark results.
- [x] **Task 3.5:** **Refactor** code to remove the old `PSOAlgorithm` and `ParticleBase` if fully migrated, or explicitly label them as `LegacyPSOAlgorithm`.
→ Stage for human review
