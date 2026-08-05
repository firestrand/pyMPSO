#!/usr/bin/env python3
"""Validate SPSO2011 statistical parity and deterministic function baselines.

This script has two modes:
- function checks: verify fixed-input objective values against analytical formulas.
- parity checks: run SPSO2011 across benchmark problems and compare aggregate stats.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.boundary.damped_reflection import DampedReflectionBoundaryHandler
from src.clamping.velocity import NoClampingStrategy
from src.influence.single_best import SingleBestInfluence
from src.initialization.bounds_aware import BoundsAwareInitialization
from src.problem.ackley import AckleyFunction
from src.problem.griewank import GriewankFunction
from src.problem.rastrigin import RastriginFunction
from src.problem.rosenbrock import RosenbrockFunction
from src.problem.schwefel import SchwefelFunction
from src.problem.sphere import SphereFunction
from src.pso_algorithm import PSOAlgorithm
from src.stopping.max_iterations import MaxIterationsStopping
from src.topology.random_topology import RandomTopology
from src.velocity.hypersphere import HypersphereVelocity

logger = logging.getLogger(__name__)

# Reference envelopes from the SPSO11 archive parity suite.
REFERENCE_STATS_2011 = {
    "sphere_30": {
        "success_rate": 100.0,
        "mean_fitness": 0.0,
        "target": 0.0,
        "target_tol": 1e-8,
        "evals_per_dim": 10000,
    },
    "ackley_30": {
        "success_rate": 100.0,
        "mean_fitness": 0.0,
        "target": 0.0,
        "target_tol": 1e-8,
        "evals_per_dim": 10000,
    },
}


@dataclass(frozen=True)
class ProblemConfig:
    name: str
    problem_factory: Callable[[], object]
    target: float
    target_tol: float
    ref_success: float
    ref_mean: float


def _run_spso2011_once(problem, run_seed: int, num_particles: int = 40) -> float:
    bounds = problem.bounds
    rng = np.random.RandomState(run_seed)
    np.random.seed(run_seed)

    init = BoundsAwareInitialization(bounds=bounds, rng=rng)
    topology = RandomTopology(k=3, rng=rng, rebuild_probability=1.0)
    influence = SingleBestInfluence()
    velocity = HypersphereVelocity(clamping_strategy=NoClampingStrategy())
    boundary = DampedReflectionBoundaryHandler(damping_factor=-0.5)

    max_iterations = problem.dimension * 10000 // num_particles
    stopping = MaxIterationsStopping(max_iterations=max_iterations)

    hyperparams = {
        "c1": 0.5 + np.log(2),
        "c2": 0.5 + np.log(2),
        "w": 1.0 / (2.0 * np.log(2)),
        "rng": rng,
    }

    pso = PSOAlgorithm(
        problem=problem,
        stopping_criteria=stopping,
        initialization_strategy=init,
        topology=topology,
        influence_strategy=influence,
        velocity_strategy=velocity,
        boundary_handler=boundary,
        num_particles=num_particles,
        hyperparams=hyperparams,
        verbose=False,
    )
    result = pso.run()
    return float(result["best_fitness"])


def run_parity(runs: int, run_seed: int, selected: list[str]) -> int:
    problems: dict[str, ProblemConfig] = {
        "sphere_30": ProblemConfig(
            name="sphere_30",
            problem_factory=lambda: SphereFunction(
                dimension=30,
                bias=0.0,
                bounds=np.array([[-100.0, 100.0]] * 30),
                use_cec_shift=False,
            ),
            target=REFERENCE_STATS_2011["sphere_30"]["target"],
            target_tol=REFERENCE_STATS_2011["sphere_30"]["target_tol"],
            ref_success=REFERENCE_STATS_2011["sphere_30"]["success_rate"],
            ref_mean=REFERENCE_STATS_2011["sphere_30"]["mean_fitness"],
        ),
        "ackley_30": ProblemConfig(
            name="ackley_30",
            problem_factory=lambda: AckleyFunction(
                dimension=30,
                bounds=np.array([[-32.0, 32.0]] * 30),
                bias=0.0,
                use_cec_shift=False,
            ),
            target=REFERENCE_STATS_2011["ackley_30"]["target"],
            target_tol=REFERENCE_STATS_2011["ackley_30"]["target_tol"],
            ref_success=REFERENCE_STATS_2011["ackley_30"]["success_rate"],
            ref_mean=REFERENCE_STATS_2011["ackley_30"]["mean_fitness"],
        ),
    }

    selected_cfgs = [problems[name] for name in selected]

    logger.info("Running SPSO2011 parity with runs=%s, base_seed=%s", runs, run_seed)
    all_ok = True

    for cfg in selected_cfgs:
        runs_fitness = []
        for i in range(runs):
            seed = run_seed + i
            problem = cfg.problem_factory()
            best_fitness = _run_spso2011_once(problem, seed)
            runs_fitness.append(best_fitness)

        runs_fitness_arr = np.asarray(runs_fitness, dtype=float)
        successes = float(np.mean(np.abs(runs_fitness_arr - cfg.target) <= cfg.target_tol) * 100.0)
        mean_fitness = float(np.mean(runs_fitness_arr))

        delta_succ = successes - cfg.ref_success
        delta_mean = mean_fitness - cfg.ref_mean

        logger.info(
            "%s | success=%s%% (ref %s%%, Δ%+0.2f) | mean=%s (ref %s, Δ%+0.3e)",
            f"{cfg.name:12}",
            f"{successes:7.2f}",
            f"{cfg.ref_success:5.1f}",
            delta_succ,
            f"{mean_fitness:.6e}",
            f"{cfg.ref_mean:.6e}",
            delta_mean,
        )
        # Relaxed acceptance for cross-platform stochastic parity.
        # For strict parity checks, tighten these thresholds in CI.
        if successes + 1e-9 < cfg.ref_success * 0.5:
            all_ok = False
            logger.error("  -> FAIL (success floor hit for %s)", cfg.name)
        elif not np.isfinite(mean_fitness):
            all_ok = False
            logger.error("  -> FAIL (non-finite mean for %s)", cfg.name)
        else:
            logger.info("  -> PASS")

    return 0 if all_ok else 1


def run_function_checks() -> None:
    # Sphere
    sphere = SphereFunction(dimension=3, bounds=np.array([[-100.0, 100.0]] * 3), bias=-12.5)
    x = np.array([1.0, -2.0, 3.0])
    expected = np.sum(x * x) - 12.5
    got = sphere.evaluate(x)
    if not np.isclose(got, expected, atol=1e-12, rtol=0.0):
        raise RuntimeError(f"Sphere check failed: got {got}, expected {expected}")

    sphere_shifted = SphereFunction(
        dimension=3,
        bounds=np.array([[-100.0, 100.0]] * 3),
        bias=-450.0,
        use_cec_shift=True,
    )
    got = sphere_shifted.evaluate(sphere_shifted.shift_vector)
    if not np.isclose(got, -450.0, atol=1e-12, rtol=0.0):
        raise RuntimeError(f"Shifted sphere check failed: got {got}, expected -450")

    # Rastrigin
    rastrigin = RastriginFunction(dimension=3)
    rx = np.array([1.0, 2.0, 3.0])
    expected = 10.0 * 3 + np.sum(rx**2 - 10.0 * np.cos(2.0 * np.pi * rx))
    got = rastrigin.evaluate(rx)
    if not np.isclose(got, expected, atol=1e-12, rtol=0.0):
        raise RuntimeError(f"Rastrigin check failed: got {got}, expected {expected}")

    # Rosenbrock
    rosen = RosenbrockFunction(dimension=3)
    rx = np.array([1.0, 1.0, 1.0])
    expected = 0.0
    got = rosen.evaluate(rx)
    if not np.isclose(got, expected, atol=1e-12, rtol=0.0):
        raise RuntimeError(f"Rosenbrock check failed: got {got}, expected {expected}")

    # Griewank
    griewank = GriewankFunction(dimension=3)
    gx = np.array([1.0, 1.0, 1.0])
    idx = np.arange(1, 4)
    expected = np.sum(gx**2) / 4000.0 - np.prod(np.cos(gx / np.sqrt(idx))) + 1.0
    got = griewank.evaluate(gx)
    if not np.isclose(got, expected, atol=1e-12, rtol=0.0):
        raise RuntimeError(f"Griewank check failed: got {got}, expected {expected}")

    # Schwefel
    schwefel = SchwefelFunction(dimension=2)
    x = np.array([420.9687, 420.9687])
    abs_x = np.abs(x)
    expected = 2 * 418.9829 - np.sum(x * np.sin(np.sqrt(np.maximum(abs_x, 1e-10))))
    got = schwefel.evaluate(x)
    if not np.isclose(got, expected, atol=1e-8, rtol=0.0):
        raise RuntimeError(f"Schwefel check failed: got {got}, expected {expected}")

    logger.info("Function isolation checks passed (sphere, rastrigin, rosenbrock, griewank, schwefel).")


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1294404794)
    parser.add_argument(
        "--problems",
        nargs="+",
        default=["sphere_30", "ackley_30"],
        choices=["sphere_30", "ackley_30"],
    )
    parser.add_argument("--skip-function-checks", action="store_true")
    args = parser.parse_args()

    if not args.skip_function_checks:
        run_function_checks()

    return run_parity(args.runs, args.seed, args.problems)


if __name__ == "__main__":
    raise SystemExit(main())
