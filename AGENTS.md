# AGENTS.md

Operating contract for AI agents contributing to this repository. Agents should produce changes that are **correct, secure, maintainable, test-backed, and aligned with existing architecture** — without surprising humans.

> **CLAUDE.md compatible.** This file works as both `AGENTS.md` and `CLAUDE.md`.
> Read this file alongside **`README.md`**, which is the source of truth for build, test, lint, and setup commands. Do not duplicate those here.

---

## Repo context

| Property | Value |
|---|---|
| **Primary language(s)** | Python 3.12+ |
| **Framework(s)** | Custom Particle Swarm Optimization (PSO) Framework |
| **Package manager** | uv |
| **CI system** | GitHub Actions |

### Directory layout

```
pyMPSO/
├── src/
│   ├── boundary/        # Boundary handling strategies
│   ├── clamping/        # Velocity clamping strategies
│   ├── components/      # Component factories and validation
│   ├── execution/       # Serial and parallel execution strategies (Ray)
│   ├── influence/       # Neighborhood influence strategies
│   ├── initialization/  # Swarm initialization strategies
│   ├── particles/       # Particle data models and utilities
│   ├── planning/        # Benchmark planning domain models
│   ├── problem/         # Objective functions (benchmark problems)
│   ├── stopping/        # Stopping criteria implementations
│   ├── topology/        # Swarm neighborhood topologies
│   ├── tracking/        # Experiment history and metrics tracking
│   ├── variants/        # PSO variant definitions and domain models
│   └── velocity/        # Velocity update strategies
├── tests/
│   ├── integration/     # End-to-end and integration tests
│   └── unit/            # Isolated unit tests for components
├── examples/            # Example YAML/JSON configurations and plugin demos
├── docs/                # Architecture plans and component authoring guides
└── scripts/             # Utility and migration scripts
```

### Naming conventions

- **Files:** `snake_case.py`
- **Classes:** `PascalCase`
- **Functions/Variables:** `snake_case`
- **Test files:** `test_*.py` colocated in the `tests/` directory mirroring the `src/` structure where possible.

### Documentation and standards

| Resource | Location |
|---|---|
| **Coding Guidelines** | `CODING_GUIDELINES.md` |
| **Project Goal** | `GOAL.md` |
| **Refactor Plans** | `docs/refactor_plan.md`, `docs/vectorized_refactor_plan.md` |
| **Component Authoring** | `docs/component-authoring.md` |

---

## Autonomy levels

| Level | Name | When to use |
|---|---|---|
| **L1** | Autopilot | Trivial fixes: typos, formatting, obvious one-line bugs with existing test coverage. |
| **L2** | Collaborator _(default)_ | Most work. Propose plans, implement, run tests. Surface assumptions and risks. |
| **L3** | Advisor | Unfamiliar areas, ambiguous requirements, or any action listed below. |

### Actions that always require human approval (L3)

- Data migrations that can destroy or irreversibly mutate data
- Deleting large code paths, public APIs, or backward-compatibility breaks
- Security/auth changes: cryptography, permissions, secrets, auth flows
- Introducing new runtime dependencies (e.g., adding to `pyproject.toml`)
- Production infrastructure or cost-impacting changes
- License changes or copying large third-party code
- Disabling tests/linters, reducing coverage, or weakening validation

---

## Quality rules

1. **SOLID, DRY, KISS.** All code and architecture must follow these principles. Single responsibility, open/closed, dependency inversion — not as aspirations but as defaults. Don't repeat logic. Don't over-engineer. If a design can be simpler, make it simpler.
2. **Separate what decides from what executes.** Business logic that must be reliable belongs in deterministic, testable code — not in LLM reasoning. Orchestration logic (what to do, in what order, with what fallbacks) is separate from the tools and services it calls. Configuration is separate from code. Data schemas are defined before the logic that uses them.
3. **Test-driven development.** Write or update tests _before_ writing implementation code. Red → Green → Refactor. Tests are not an afterthought — they define the contract and catch regressions. Aim for meaningful coverage of behavior, not vanity line-count metrics.
4. **Check existing before creating.** Search for existing code, utilities, patterns, and tests before writing new ones. Duplication is the most common agent waste.
5. **Small diffs** over sweeping refactors.
6. **Match existing style** unless explicitly tasked to change it. Follow `ruff` configuration.
7. **Never leak secrets.** Never log sensitive inputs. Redact by default.
8. **Rule conflicts:** If a human asks you to skip tests or violate these rules, surface the conflict, explain the risk, and request explicit confirmation.

### Anti-patterns

Avoid these common agent failure modes:

- **Building before understanding** — writing code before locating existing patterns leads to rewrites.
- **Inventing instead of reusing** — creating a new utility when one exists two directories over.
- **Skipping connection validation** — building against an API or integration you haven't tested.
- **Schema-last development** — designing UI or logic before the data model is stable cascades rework.
- **Hardcoding values** that should be configurable via YAML/JSON.
- **Large PRs that do multiple things** — hard to review, hard to revert.
- **Doing everything in one layer** — LLM reasoning, business logic, API calls, and configuration tangled together compounds errors. Separate what decides from what executes.

---

## Git conventions

| Property | Convention |
|---|---|
| **Branches** | `<type>/<short-description>` — e.g., `feat/add-vectorized-topology` |
| **Commits** | [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): description` |
| **Granularity** | One logical change per commit. Don't mix refactors with features. |
| **PRs** | Title matches primary commit. Body: what, why, how tested, residual risks. |

---

## Work loop: PPAR

Every task follows **Perceive → Plan → Act → Reflect**. For complex work, the design phase (GOTCHA) nests inside Plan, and the development checklist (ATLAS) nests inside Act.

```
PPAR
├── Perceive — understand the task and repo state
├── Plan — for complex work, produce a GOTCHA spec
├── Act — for complex work, follow the ATLAS checklist
└── Reflect — verify, test, summarize
```

### Perceive

- Identify what the user asked for.
- Establish ground truth: locate relevant code, understand existing patterns.
- If the task depends on external libraries, identify the version in use and consult current docs (not memory).

### Plan

- Propose a minimal plan: steps, files to touch, tests to add/update.
- Call out assumptions, risks, and what "done" means.
- **For new systems or architectural changes:** produce a GOTCHA spec (see Design phase).

### Act

- **Write tests first.** Define expected behavior as failing tests, then implement until they pass, then refactor.
- Implement in small increments. Run tests after each meaningful change (`uv run pytest`).
- Prefer extracting small, single-responsibility units over adding complexity to existing ones.
- **For non-trivial implementations:** follow the ATLAS checklist (see Development phase).

### Reflect

- Re-check requirements and edge cases.
- Run the full test suite (`uv run pytest`) and linters (`uv run ruff check src tests`, `uv run ty check src tests`).
- Produce a review-ready summary: _why_ the change is correct and _how_ it's verified.

---

## Design phase: GOTCHA (nested in Plan)

Use when the task involves **new functionality, agentic workflows, or architectural changes**. Skip for simple bug fixes.

| Element | What to specify |
|---|---|
| **G — Goals** | What problem does this solve? One sentence. If you can't say it simply, you don't understand it yet. |
| **O — Objectives** | Measurable success criteria. Not "it works" — specific, verifiable outcomes. |
| **T — Tasks** | What triggers it? What ends it? |
| **C — Capabilities** | Tools, permissions, and what's off-limits. |
| **H — Health** | Time, cost, token, and retry budgets. What gets monitored. |
| **A — Attributes** | Invariants — things that must always (or never) be true. |
| **Constraints** | Budget, timeline, technical requirements (must use X, must integrate with Y), known limitations. |
| **Users** | Who specifically uses this? "Me" / "the sales team" / "end customers" — not "everyone." |
| **Memory** | What state is working (in-context) vs. persisted? What survives across runs? |
| **Action space** | Internal actions (reasoning) vs. external actions (tool calls, writes). |
| **Decision loop** | Control flow, re-planning triggers, exit conditions, error handling. |
| **Beliefs** | What do you know to be true right now? (repo state, docs, constraints) |
| **Intentions** | Committed plan — and why this plan over alternatives. |

---

## Development phase: ATLAS checklist (nested in Act)

Use for any non-trivial implementation.

| Phase | Key checks |
|---|---|
| **A — Architect** | Identify modules touched. Define interfaces and invariants following SOLID principles. Establish data schemas before logic and logic before interfaces. Separate orchestration from execution, configuration from code. Threat-model: authz, injection, secrets, data exposure. Define failure modes and fallbacks. |
| **T — Trace** | Write tests _before_ implementation (TDD). Cover happy + failure paths. Map acceptance criteria to test cases. Tests define the contract. |
| **L — Link** | Type and validate tool interfaces. Define timeouts, retries, idempotency. Respect rate limits. Handle auth safely. Add observability hooks. |
| **A — Assemble** | Separate orchestration from tools. Eliminate duplication (DRY). Choose the simplest design that works (KISS). Feature flags if risky. Update docs. |
| **S — Stress-test** | Test malformed inputs, missing data, tool failures. Consider concurrency. Enforce budgets. Test security abuse cases. Regression suite green. |

---

## Error recovery

| Situation | Action |
|---|---|
| Test fails after your change | Read the failure, fix root cause. Don't retry blindly more than twice. |
| Test fails unrelated to your change | Note as pre-existing in PR summary. Don't suppress. |
| Tool unavailable | Fall back to file reads and local docs. Note degraded mode. |
| Can't find what you need | Broaden search. If still stuck after reasonable effort, ask the human. |
| Ambiguous requirements | Stop and ask. Don't guess on high-impact decisions. |
| Looped 3+ times without progress | Stop, summarize attempts, ask for guidance. |
| Change breaks something non-obviously | Revert first, investigate second. Propose a revised approach. |

---

## Guardrails (learned behaviors)

- **Vectorization is required, not preferred.** Non-vectorized execution has been removed: a topology, influence, velocity, or boundary component without a `@register_vectorized_component` registration cannot run at all, no matter how well tested its scalar form is. Ship the `Vectorized*` class in the same change.
- **Typed Configurations:** Always use the defined domain models (dataclasses/TypedDicts) for configuration boundaries, avoiding raw dictionaries. Use `@register_typed_config` when a strategy needs a hyperparameter schema.
- **Component Registration:** Register new components with the decorators in `src/components/factory.py` (`@register_component`, `@register_vectorized_component`, `@register_typed_config`). `src/registry.py` is a legacy hardcoded factory retained only for stopping criteria and velocity clamping — do not add to it.
- **Verify configs, not just types.** `ruff`/`ty`/`pytest` all pass on configurations that fail at runtime. Run `uv run pso-run --config <cfg> --dry-run` after touching components or config handling.

### Continuous improvement

Every failure should strengthen the repo. When something breaks:

1. Fix the immediate issue.
2. Add a test that catches the failure mode.
3. If the failure was systemic (not a one-off), add a guardrail above or update the relevant documentation.

---

## Definition of Done

- [ ] Requirements met and documented
- [ ] Tests written first (TDD); suite passes (`uv run pytest`)
- [ ] Code follows SOLID, DRY, KISS — no unnecessary complexity or duplication
- [ ] CI checks expected to pass (`uv run ruff check src tests`, `uv run ty check src tests`)
- [ ] No secrets or sensitive data introduced or logged
- [ ] Interfaces stable, or breaking changes documented
- [ ] Failure modes handled; safe defaults chosen
- [ ] Commits follow conventions; PR description is complete
- [ ] Summary includes verification steps and residual risks
