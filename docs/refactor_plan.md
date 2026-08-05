# Refactor Plan: Task-First Modular PSO Runtime (No BC Policy)

## Decision context
- Goal: optimize for modular experimentation across many PSO variants without backward compatibility constraints.
- Priority: one implementation path where architecture, testability, and composability are explicit and enforced.

## Baseline now in place

### Implemented this pass
- `planning/domain.py`
  - `BenchmarkTask` is the single domain model for executable work units.
  - `BenchmarkExecutionPlan` now owns ordered task names, task metadata, and task list.
- `planning/planner.py`
  - `build_benchmark_plan` accepts a `TaskPlanner` callback and returns deterministic typed tasks.
- `execution/protocols.py`
  - `RunExecutor` protocol is backend-agnostic and takes `BenchmarkTask`.
  - `ExecutionResult` is a minimal stable contract: task metadata + outcome/error.
- `execution/serial.py`
  - `SerialExecutor` runs `RunExecutor` against `BenchmarkTask`.
- `execution/parallel.py`
 - `RayExecutor` runs batched `BenchmarkTask` via Ray.
 - Still optional dependency safe via explicit `RunExecutorError`.
- `execution/orchestrator.py`
  - Single orchestration path for plan execution and summary aggregation.
  - Does not assume a concrete optimizer implementation.
- `components/factory.py`
  - Decorator registration added: `@register_component("component_type", "name")`.
  - Component lookups normalized by component type/name and bounded to registered catalog.
- `run_pso.py`
  - `_build_task_plan` now emits `BenchmarkTask` instances.
  - `_run_single_benchmark_execution` consumes `BenchmarkTask` directly.
  - Execution now uses `RunExecutor` strategy implementations.
- `pso_orchestrator.py`
  - Removed legacy callable execution shims.
  - Orchestration now expects protocol-based executors directly.

## Architectural target
1. **Plan construction is deterministic and typed**
   - Inputs: config + selected variants + planners.
   - Output: `BenchmarkExecutionPlan` with immutable task list and stable keys.
2. **Execution is engine-agnostic**
   - `RunExecutor` accepts `BenchmarkTask` and returns `ExecutionResult`.
   - New engines (including vectorized engines) plug in by implementing protocol only.
3. **Component creation is explicit and discoverable**
   - Registry bootstrap + decorator registration support.
   - Invalid component names/types fail quickly before heavy execution.
4. **CLI remains thin**
   - CLI/service handles request parsing, output format, and transport only.
   - Domain and execution logic remain in planning/execution boundaries.

## Phase backlog (remaining)

### Phase A — Variant first-class domain (new)
- Add a dedicated variant domain module:
  - `selected_variants` / `excluded_variants` semantics owned by an explicit model.
- Move variant parsing/filtering out of `run_pso.py` helper functions.
- Keep behavior deterministic and preserve declared-order precedence:
  1) include set
  2) exclude set

### Phase B — Result schema hardening
- Introduce explicit `results` domain models:
  - single run result
  - multi-run summary result
  - suite summary result
- Replace ad-hoc dict handling in summary helpers with schema-backed builders.

### Phase C — Configuration domain and loader split
- Split validation/normalization from execution concerns:
  - config loading
  - structural validation
  - task merge/override policy
- Add explicit error types for:
  - schema errors
  - variant errors
  - planning errors
  - execution/runtime errors

### Phase D — Quality hardening
- Add focused tests for:
  - task creation boundaries
  - plan assembly for multiple variants
  - serial and parallel execution contracts
  - decorator registration flow
- Remove duplicated config checks where planner/orchestrator now owns responsibility.

## Acceptance targets for this slice
- No hidden legacy callable adaptation paths remain in execution flow.
- `RunExecutor` path never accepts raw dict payloads.
- New engines require only `RunExecutor` + `BenchmarkTask` integration.
- Variant materialization and task planning remain deterministic and reproducible.
- Quality checks (format/type/lint) pass for all changed modules.
