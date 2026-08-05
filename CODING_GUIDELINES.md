# Coding Guidelines for Modular PSO Framework

## 1. Purpose

This document outlines the coding standards, principles, and development workflow for the **Modular PSO Framework**. Adhering to these guidelines is crucial for ensuring code quality, consistency, maintainability, and testability. This is a living document and may evolve as the project progresses.

## 2. Core Principles

We strive to follow these established software development principles:

*   **Test-Driven Development (TDD):**
    *   **This project mandates Test-Driven Development.** All new functionality must start with a failing test.
    *   Strictly follow the **Red-Green-Refactor** cycle:
        1.  **Red:** Write a minimal test case for the new functionality. Run it; it *must* fail (or not compile).
        2.  **Green:** Write the simplest possible production code to make the test pass.
        3.  **Refactor:** Improve the implementation code and the test code, ensuring all tests still pass.
    *   Write unit tests *before* or *concurrently with* implementation code.
    *   Ensure tests cover requirements, expected behavior, and edge cases.
*   **SOLID:**
    *   **S**ingle Responsibility Principle: Classes and functions should have one primary responsibility.
    *   **O**pen/Closed Principle: Software entities should be open for extension but closed for modification.
    *   **L**iskov Substitution Principle: Subtypes must be substitutable for their base types.
    *   **I**nterface Segregation Principle: Clients should not be forced to depend on interfaces they do not use.
    *   **D**ependency Inversion Principle: Depend upon abstractions, not concretions.
    *   These principles guide refactoring efforts to improve design.
*   **KISS (Keep It Simple, Stupid):**
    *   Prefer simple, straightforward solutions over complex ones.
    *   Avoid unnecessary abstractions or features.
*   **DRY (Don't Repeat Yourself):**
    *   Avoid duplicating code logic. Use functions, classes, and modules to promote reuse.

## 3. Development Workflow (TDD Focused)

Our development workflow strictly follows TDD principles, often facilitated by the AI assistant:

1.  **Define Task:** Identify the next task (e.g., from a TODO list or issue tracker).
2.  **Plan (if needed):** Discuss the approach, potential challenges, and affected components.
3.  **Write Test(s) (Red):** Create unit or integration tests in the appropriate `tests/` subdirectory that define the expected behavior for the new feature or change.
4.  **Run Tests (Confirm Failure):** Execute the newly written tests *immediately*. They **must fail** (or raise an error if the target code doesn't exist yet), confirming the test setup is correct and the feature isn't already implemented by mistake.
5.  **Implement Code (Green):** Write the *minimum* amount of code in `src/` required to make the new tests pass.
6.  **Run Tests (Confirm Pass):** Execute relevant tests frequently (especially the newly added ones) to ensure correctness and verify the implementation makes the tests pass.
7.  **Refactor:** Improve the code structure, clarity, and adherence to principles (SOLID, KISS, DRY) in both the implementation and test code, while ensuring *all* tests continue to pass.
8.  **Verify:** Run the *entire* test suite (`pytest tests`) to check for regressions or unintended side effects after refactoring or completing a significant feature. **This step should also be performed after completing each distinct task (e.g., a TODO item) before moving to the next.**
9.  **(Optional) Update TODO:** Mark the task as complete if applicable.
10. **Commit (Implicit):** Changes are applied incrementally.

## 4. Coding Standards

*   **Logging Best Practices:**
    *   **Never use print() in production code** - always use the `logging` module instead
    *   Each module should have its own logger: `logger = logging.getLogger(__name__)`
    *   Use appropriate log levels:
        *   `logger.debug()` - Detailed diagnostic information
        *   `logger.info()` - General informational messages
        *   `logger.warning()` - Warning messages for potentially problematic situations
        *   `logger.error()` - Error messages for failures that should be investigated
        *   `logger.critical()` - Critical failures that may cause the program to abort
    *   Avoid logging sensitive information (passwords, API keys, personal data)
    *   Use lazy string formatting: `logger.info("Processing %d items", count)` instead of `logger.info(f"Processing {count} items")`
    *   Configure logging at the application entry point (e.g., in `main()` or `run_pso.py`)
    *   Print statements are acceptable only in:
        *   Demo scripts explicitly meant for user interaction
        *   Test files for debugging purposes
        *   Command-line interface output that is part of the user experience

*   **Naming Conventions:**
    *   Use descriptive names for variables, functions, classes, and modules.
    *   Follow PEP 8: `snake_case` for functions, methods, and variables; `PascalCase` for classes.
*   **Docstrings:**
    *   Write clear docstrings for all public modules, classes, functions, and methods using Google Style format.
    *   Docstrings should explain the *purpose*, *arguments* (`Args:`), *return values* (`Returns:`), and any *exceptions raised* (`Raises:`).
*   **Comments:**
    *   Use comments to explain the *why*, not the *what*. Code should be self-explanatory where possible.
    *   Avoid obvious comments (e.g., `# increment counter`).
    *   Use `# TODO:` for planned future work related to a specific code section.
*   **Imports:**
    *   Follow PEP 8 import ordering: standard library, third-party libraries, local application/library specific imports (from `src`).
    *   Import specific members where appropriate rather than using `*`.
*   **Modularity:**
    *   Adhere to the established project structure (`src`, `tests`, etc.).
    *   Keep classes and functions focused on a single responsibility (SRP).
*   **Error Handling:**
    *   Use specific exception types (`ValueError`, `TypeError`, `FileNotFoundError`, etc.).
    *   Validate inputs where necessary (e.g., positive dimensions, non-empty lists).
*   **Configuration Files:**
    *   Use **YAML (`.yaml`)** for all configuration files.
    *   **Example configurations**, demonstrating specific use cases or component combinations, should reside in the `examples/` directory alongside any corresponding example code or documentation.
    *   General configurations (like a potential future `default.yaml` or user-specific setups) can be placed in a dedicated `configs/` directory.

## 5. Testing Strategy

*   **Unit Tests:** Test individual components (classes, functions) in isolation. Use mocking (`pytest.mock`) extensively to isolate dependencies. Located primarily in `tests/unit/`.
*   **Integration Tests:** Test the interaction between different components (e.g., PSO algorithm coordinating particle updates, neighborhood interactions, boundary handling). Located in `tests/integration/`.
*   **Test Framework:**
    *   All new tests **must** be written using the `pytest` framework.
    *   Utilize `pytest` features such as fixtures (`@pytest.fixture`) for setup/teardown, parametrization (`@pytest.mark.parametrize`), and plain `assert` statements for checks. Avoid `unittest` class-based structures for new tests.
*   **Test Execution:** Run relevant tests frequently during development. Run the *full* test suite (`pytest tests`) after any significant change, refactoring, or before considering a feature complete. This is a critical step in the TDD workflow.
*   **Coverage:** Aim for high test coverage to ensure most code paths are exercised. Use tools like `pytest-cov`.

## 6. AI Assistant Interaction

*   The AI assistant helps implement changes, run commands, and refactor code using available tools.
*   Clearly state the desired action or change.
*   Review the proposed actions and tool outputs carefully to ensure they match the intent and don't introduce errors.
*   Verify test results after changes are applied.

## 7. Consistency

*   When renaming classes, functions, or significant parameters, ensure the change is propagated consistently across *all* relevant files, including:
    *   Source code (`src/**/*.py`)
    *   Unit tests (`tests/unit/**/*.py`)
    *   Integration tests (`tests/integration/**/*.py`)
    *   Configuration files (`examples/**/*.yaml`, `examples/**/*.json`)
    *   Documentation (`README.md`, `GOAL.md`, docstrings, etc.) 