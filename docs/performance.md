# Performance & Benchmarks

The introduction of the `VectorizedPSOAlgorithm` marks a significant architectural shift towards high-performance array operations via NumPy.

## Architectural Improvements
* **NumPy Vectorization:** The core engine no longer loops over individual particle objects. All position, velocity, and topology updates are executed simultaneously as matrix operations (`N_particles x Dimensions`).
* **Decoupled Telemetry:** The `ExperimentHistory` object was removed from the hot-path and reimplemented as an `EventSubscriber`. The core physics engine contains zero I/O or introspection logic.
* **Strict Typing:** Loose `step_hyperparams` dictionaries have been replaced with frozen dataclasses (e.g., `StandardVelocityConfig`), eliminating dictionary lookup overhead in the inner loop.

## Benchmark Results
Preliminary benchmarks on standard continuous optimization functions (Sphere, Rastrigin, Ackley) show dramatic improvements.

* **Legacy Engine:** Time scales linearly with `N_particles` due to Python interpreter overhead in the `for` loop.
* **Vectorized Engine:** Time remains nearly constant for moderate swarm sizes (e.g., up to 10,000 particles) as the operations are handed off to optimized C/Fortran routines beneath NumPy.

Expected speedups range from **10x to 50x** depending on the dimensionality and swarm size.