#!/usr/bin/env python3
"""Compare Python SPSO2011 behavior against the C reference.

Usage examples:
- Function-level parity check:
    python scripts/compare_spso2011_c_vs_python.py --mode function --problems ackley_10 sphere_30 --points 100
- Optimizer-level comparison:
    python scripts/compare_spso2011_c_vs_python.py --mode optimizer --runs 500 --seed 1294404794
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from src.boundary.damped_reflection import DampedReflectionBoundaryHandler
from src.clamping.velocity import NoClampingStrategy
from src.influence.single_best import SingleBestInfluence
from src.initialization.bounds_aware import BoundsAwareInitialization
from src.problem.ackley import AckleyFunction
from src.problem.cec_data import get_cec_shift_vector
from src.problem.griewank import GriewankFunction
from src.problem.rastrigin import RastriginFunction
from src.problem.rosenbrock import RosenbrockFunction
from src.problem.schwefel import SchwefelFunction
from src.problem.sphere import SphereFunction
from src.pso_algorithm import PSOAlgorithm
from src.stopping.composite import AnyStoppingCriteria
from src.stopping.max_evaluations import MaxEvaluationsStopping
from src.stopping.target_fitness import TargetFitnessStopping
from src.topology.random_topology import RandomTopology
from src.velocity.hypersphere import HypersphereVelocity

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CReferenceProblem:
    """Problem descriptor shared by Python and C comparisons."""

    name: str
    c_code: int
    dimension: int
    bounds: np.ndarray
    bias: float
    use_cec_shift: bool
    eval_budget: int
    epsilon: float
    reference_eval: Callable[[np.ndarray, int], float]
    build_problem: Callable[[int], Any]


# SPSO-2011-like settings for comparison.
_NUM_PARTICLES = 40
_CEILING_SWARM = 40
_DAMPING_FACTOR = -0.5
_C1 = 0.5 + np.log(2)
_C2 = 0.5 + np.log(2)
_W = 1.0 / (2.0 * np.log(2))
_WYRAND_MASK64 = (1 << 64) - 1
_WYRAND_MAX = float((1 << 64) - 1)
_WYHASH_WYP0 = 0xA0761D6478BD642F
_WYHASH_WYP1 = 0xE7037ED1A0B428DB


class WyRandGenerator:
    """Python-equivalent implementation of C wyRand (wyhash.h option 6)."""

    def __init__(self, seed: int):
        self._seed = int(seed) & _WYRAND_MASK64

    @staticmethod
    def _mul_xor(a: int, b: int) -> int:
        product = a * b
        lo = product & _WYRAND_MASK64
        hi = (product >> 64) & _WYRAND_MASK64
        return hi ^ lo

    def _next_u64(self) -> int:
        self._seed = (self._seed + _WYHASH_WYP0) & _WYRAND_MASK64
        return self._mul_xor(self._seed ^ _WYHASH_WYP1, self._seed)

    def random(self, size=None):
        if size is None:
            return float(self._next_u64() / _WYRAND_MAX)
        if isinstance(size, int):
            size = (size,)
        out = np.empty(size, dtype=np.float64)
        for idx in np.ndindex(out.shape):
            out[idx] = self.random()
        return out

    def _normal_scalar(self) -> float:
        count = 0
        while True:
            x1 = 2.0 * self.random() - 1.0
            x2 = 2.0 * self.random() - 1.0
            w = x1 * x1 + x2 * x2
            count += 1
            if count > 10000:
                raise RuntimeError("wy_rand_normal convergence failure")
            if w < 1.0:
                break
        w = math.sqrt(-2.0 * math.log(w) / w)
        y1 = x1 * w
        if self.random() < 0.5:
            y1 = -y1
        return float(y1)

    def standard_normal(self, size=None):
        if size is None:
            return self._normal_scalar()
        if isinstance(size, int):
            size = (size,)
        out = np.empty(size, dtype=np.float64)
        for idx in np.ndindex(out.shape):
            out[idx] = self._normal_scalar()
        return out

    def uniform(self, low, high):
        low_arr = np.asarray(low, dtype=np.float64)
        high_arr = np.asarray(high, dtype=np.float64)
        shape = np.broadcast(low_arr, high_arr).shape
        scale = self.random(shape) if shape else self.random()
        return low_arr + scale * (high_arr - low_arr)


def _build_rng(seed: int, rng_mode: str):
    if rng_mode == "c-wy":
        return WyRandGenerator(seed)
    return np.random.RandomState(seed)


def _shifted_point(point: np.ndarray, function_key: str, dim: int) -> np.ndarray:
    shift = get_cec_shift_vector(function_key, dim)
    if shift is None:
        raise RuntimeError(f"Missing CEC shift vector for {function_key} with dim={dim}")
    return point - shift[:dim]


def _sphere_raw(point: np.ndarray, dim: int) -> float:
    shifted = _shifted_point(point, "sphere", dim)
    return float(np.sum(shifted * shifted) - 450.0)


def _ackley_raw(point: np.ndarray, dim: int) -> float:
    shifted = _shifted_point(point, "ackley", dim)
    sum_sq = float(np.sum(shifted * shifted))
    sum_cos = float(np.sum(np.cos(2.0 * np.pi * shifted)))
    return float(-140.0 + 20.0 + np.e - 20.0 * np.exp(-0.2 * np.sqrt(sum_sq / dim)) - np.exp(sum_cos / dim))


def _rosenbrock_raw(point: np.ndarray, dim: int) -> float:
    shifted = _shifted_point(point, "rosenbrock", dim) + 1.0
    value = 390.0
    for i in range(1, dim):
        prev = shifted[i - 1] - 1.0
        value += prev * prev
        diff = shifted[i - 1] * shifted[i - 1] - shifted[i]
        value += 100.0 * diff * diff
    return float(value)


def _rastrigin_raw(point: np.ndarray, dim: int) -> float:
    shifted = _shifted_point(point, "rastrigin", dim)
    return float(-330.0 + 10.0 * dim + np.sum(shifted * shifted - 10.0 * np.cos(2.0 * np.pi * shifted)))


def _schwefel_raw(point: np.ndarray, dim: int) -> float:
    shifted = _shifted_point(point, "schwefel", dim)
    value = 0.0
    for i in range(dim):
        prefix_sum = np.sum(shifted[: i + 1])
        value += prefix_sum * prefix_sum
    return float(value - 450.0)


def _griewank_raw(point: np.ndarray, dim: int) -> float:
    shifted = _shifted_point(point, "griewank", dim)
    quad = np.sum(shifted * shifted) / 4000.0
    cosine = 1.0
    for i in range(dim):
        cosine *= np.cos(shifted[i] / np.sqrt(i + 1))
    return float(-180.0 + 1.0 + quad - cosine)


def _build_problem_specs() -> dict[str, CReferenceProblem]:
    return {
        "sphere_30": CReferenceProblem(
            name="sphere_30",
            c_code=100,
            dimension=30,
            bounds=np.array([[-100.0, 100.0]] * 30),
            bias=-450.0,
            use_cec_shift=True,
            eval_budget=30 * 10000,
            epsilon=1e-6,
            reference_eval=_sphere_raw,
            build_problem=lambda dim=30: SphereFunction(
                dimension=dim,
                bounds=np.array([[-100.0, 100.0]] * dim),
                bias=-450.0,
                use_cec_shift=True,
            ),
        ),
        "ackley_10": CReferenceProblem(
            name="ackley_10",
            c_code=106,
            dimension=10,
            bounds=np.array([[-32.0, 32.0]] * 10),
            bias=-140.0,
            use_cec_shift=True,
            eval_budget=10 * 10000,
            epsilon=1e-4,
            reference_eval=_ackley_raw,
            build_problem=lambda dim=10: AckleyFunction(
                dimension=dim,
                bounds=np.array([[-32.0, 32.0]] * dim),
                bias=-140.0,
                use_cec_shift=True,
            ),
        ),
        "rosenbrock_10": CReferenceProblem(
            name="rosenbrock_10",
            c_code=102,
            dimension=10,
            bounds=np.array([[-100.0, 100.0]] * 10),
            bias=390.0,
            use_cec_shift=True,
            eval_budget=10 * 10000,
            epsilon=1e-2,
            reference_eval=_rosenbrock_raw,
            build_problem=lambda dim=10: RosenbrockFunction(
                dimension=dim,
                bounds=np.array([[-100.0, 100.0]] * dim),
                bias=390.0,
                use_cec_shift=True,
            ),
        ),
        "rastrigin_30": CReferenceProblem(
            name="rastrigin_30",
            c_code=103,
            dimension=30,
            bounds=np.array([[-5.12, 5.12]] * 30),
            bias=-330.0,
            use_cec_shift=True,
            eval_budget=30 * 10000,
            epsilon=1e-2,
            reference_eval=_rastrigin_raw,
            build_problem=lambda dim=30: RastriginFunction(
                dimension=dim,
                bounds=np.array([[-5.12, 5.12]] * dim),
                bias=-330.0,
                use_cec_shift=True,
            ),
        ),
        "schwefel_10": CReferenceProblem(
            name="schwefel_10",
            c_code=104,
            dimension=10,
            bounds=np.array([[-100.0, 100.0]] * 10),
            bias=-450.0,
            use_cec_shift=True,
            eval_budget=10 * 10000,
            epsilon=1e-5,
            reference_eval=_schwefel_raw,
            build_problem=lambda dim=10: SchwefelFunction(
                dimension=dim,
                bounds=np.array([[-100.0, 100.0]] * dim),
                bias=-450.0,
                use_cec_shift=True,
            ),
        ),
        "griewank_10": CReferenceProblem(
            name="griewank_10",
            c_code=105,
            dimension=10,
            bounds=np.array([[-600.0, 600.0]] * 10),
            bias=-180.0,
            use_cec_shift=True,
            eval_budget=10 * 10000,
            epsilon=1e-2,
            reference_eval=_griewank_raw,
            build_problem=lambda dim=10: GriewankFunction(
                dimension=dim,
                bounds=np.array([[-600.0, 600.0]] * dim),
                bias=-180.0,
                use_cec_shift=True,
            ),
        ),
    }


PROBLEM_SPECS = _build_problem_specs()


@dataclass
class RunRecord:
    run: int
    best_fitness: float
    evaluations: int
    trajectory: list[float] | None = None


@dataclass
class TraceDiagnostics:
    run: int
    run_seed: int
    target_evals: int
    best_per_eval: list[float]
    first_reach_target: int | None


def _c_reported_value(raw_value: float, objective: float) -> float:
    return float(abs(raw_value - objective))


def _run_one_python_spso(
    spec: CReferenceProblem,
    seed: int,
    track_history: bool = False,
    rng_mode: str = "numpy",
    rng: Any | None = None,
) -> RunRecord:
    problem = spec.build_problem(spec.dimension)

    if rng is None:
        rng = _build_rng(seed, rng_mode)

    init = BoundsAwareInitialization(bounds=problem.bounds, rng=rng)
    topology = RandomTopology(k=3, rng=rng, rebuild_probability=1.0)
    influence = SingleBestInfluence()
    velocity = HypersphereVelocity(clamping_strategy=NoClampingStrategy())
    boundary = DampedReflectionBoundaryHandler(damping_factor=_DAMPING_FACTOR)

    stopping = AnyStoppingCriteria(
        [
            MaxEvaluationsStopping(max_evaluations=spec.eval_budget),
            TargetFitnessStopping(target=spec.bias, tolerance=spec.epsilon),
        ]
    )

    hyperparams = {
        "c1": _C1,
        "c2": _C2,
        "w": _W,
        "rng": rng,
        "bounds": problem.bounds,
    }

    pso = PSOAlgorithm(
        problem=problem,
        stopping_criteria=stopping,
        initialization_strategy=init,
        topology=topology,
        influence_strategy=influence,
        velocity_strategy=velocity,
        boundary_handler=boundary,
        num_particles=_NUM_PARTICLES,
        hyperparams=hyperparams,
        verbose=False,
    )

    result = pso.run()
    trajectory = None
    history = result.get("history")
    if track_history and history is not None:
        # Backward-compatible path: older algorithm implementations do not expose
        # an iteration history object, so we only build a trajectory when
        # history is actually available.
        # Convert raw objective to C-style reported fitness (fabs(raw - objective)).
        trajectory = [_c_reported_value(m.best_fitness, spec.bias) for m in history.iterations]
    return RunRecord(
        run=seed,
        best_fitness=_c_reported_value(float(result["best_fitness"]), spec.bias),
        evaluations=int(result["evaluations"]),
        trajectory=trajectory,
    )


def run_python_benchmark(
    spec: CReferenceProblem,
    runs: int,
    seed: int,
    track_history: bool = False,
    rng_mode: str = "numpy",
    c_style_stream: bool = False,
) -> list[RunRecord]:
    records: list[RunRecord] = []
    shared_rng = _build_rng(seed, rng_mode) if c_style_stream else None
    for run_idx in range(runs):
        run_seed = seed + run_idx
        records.append(
            _run_one_python_spso(
                spec,
                run_seed,
                track_history=track_history,
                rng_mode=rng_mode,
                rng=shared_rng if c_style_stream else None,
            )
        )
    return records


def _parse_c_run_file(path: Path, c_eval_budget: int) -> list[RunRecord]:
    records: list[RunRecord] = []
    if not path.exists():
        raise FileNotFoundError(f"C run output missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            tokens = line.split()
            # Expected format: run, success(%), eval, error, mean_error, then position coords
            if len(tokens) < 5:
                continue
            run = int(float(tokens[0]))
            evaluations = int(float(tokens[2]))
            error = float(tokens[3]) if 0 <= evaluations <= c_eval_budget else float("nan")
            records.append(RunRecord(run=run, best_fitness=error, evaluations=evaluations))
    return records


def _parse_c_trace_for_run(
    spec: CReferenceProblem,
    seed: int,
    run_index: int,
    c_root: Path,
    c_records: list[RunRecord],
) -> TraceDiagnostics:
    if not (0 <= run_index < len(c_records)):
        raise ValueError(f"Requested run index {run_index} is outside available C records")

    target_record = c_records[run_index]
    run_seed = seed + run_index
    trace_path = c_root / "results" / f"f{spec.c_code}" / f"f_trace_seed{seed}.txt"
    if not trace_path.exists():
        raise FileNotFoundError(f"C trace file missing: {trace_path}")

    skip_lines = 0
    for idx in range(run_index):
        skip_lines += c_records[idx].evaluations

    start_line = skip_lines + 1
    end_line = skip_lines + target_record.evaluations

    raw_fitness: list[float] = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            if line_no < start_line:
                continue
            if line_no > end_line:
                break

            line = raw_line.strip()
            if not line:
                continue

            tokens = line.split()
            expected = 2 + spec.dimension
            if len(tokens) < expected:
                continue

            try:
                raw_fitness.append(float(tokens[1]))
            except ValueError:
                continue

    if len(raw_fitness) < target_record.evaluations:
        raise RuntimeError(
            f"Trace for run {run_index} was incomplete. Expected {target_record.evaluations}"
            f" samples, got {len(raw_fitness)}"
        )

    best_per_eval: list[float] = []
    best_so_far = float("inf")
    first_reach_target = None
    for i, value in enumerate(raw_fitness, start=1):
        if value < best_so_far:
            best_so_far = value
        best_per_eval.append(best_so_far)
        if first_reach_target is None and value <= spec.epsilon:
            first_reach_target = i

    return TraceDiagnostics(
        run=run_index + 1,
        run_seed=run_seed,
        target_evals=target_record.evaluations,
        best_per_eval=best_per_eval,
        first_reach_target=first_reach_target,
    )


def _sample_python_best_by_evaluations(
    py_record: RunRecord,
    target_evals: int,
) -> np.ndarray:
    if py_record.trajectory is None:
        raise ValueError("Python trajectory missing for this run")

    # history tracks iteration-level best values with evaluation counters in
    # the same order as ExperimentHistory.record_iteration.
    iterations = np.arange(len(py_record.trajectory), dtype=int)
    # Reconstruct synthetic evaluation counts: one initial full swarm eval plus
    # 1 per particle per iteration (not exact, but enough to compare coarse traces).
    target_swarm = _NUM_PARTICLES
    evals = iterations * target_swarm + target_swarm

    eval_grid = np.arange(1, target_evals + 1, dtype=int)
    trajectory = np.asarray(py_record.trajectory, dtype=float)
    idx = np.searchsorted(evals, eval_grid, side="right") - 1
    idx = np.clip(idx, 0, len(trajectory) - 1)
    return trajectory[idx]


def _compare_run_diagnostics(
    spec: CReferenceProblem,
    seed: int,
    py_record: RunRecord,
    c_record: RunRecord,
    c_root: Path,
    c_records: list[RunRecord],
) -> None:
    if py_record.trajectory is None:
        raise ValueError("Python trajectory required for diagnostics; rerun with --track-history")

    c_trace = _parse_c_trace_for_run(spec, seed, c_record.run - 1, c_root, c_records)
    py_by_eval = _sample_python_best_by_evaluations(py_record, c_trace.target_evals)
    c_by_eval = np.asarray(c_trace.best_per_eval, dtype=float)

    paired = np.abs(py_by_eval - c_by_eval)
    max_diff_idx = int(np.argmax(paired))
    max_diff = float(paired[max_diff_idx])

    # first index where Python and C differ meaningfully (loose threshold at EPS)
    diff_mask = np.where(paired > max(spec.epsilon, 1e-12))[0]
    first_diff = int(diff_mask[0] + 1) if diff_mask.size > 0 else None

    py_by_iter = np.asarray(py_record.trajectory or [], dtype=float)
    max_iter = min(len(py_by_iter), len(c_by_eval))
    if max_iter > 0:
        c_by_iter = [c_by_eval[min((i + 1) * _NUM_PARTICLES, c_trace.target_evals) - 1] for i in range(max_iter)]
        c_by_iter_arr = np.asarray(c_by_iter, dtype=float)
        iter_diff = np.abs(py_by_iter[:max_iter] - c_by_iter_arr)
        max_iter_idx = int(np.argmax(iter_diff))
    else:
        max_iter_idx = None
        c_by_iter_arr = None

    logger.info(
        "\n[diagnostic] Problem %s, run %s (seed %s)",
        spec.name,
        c_trace.run,
        c_trace.run_seed,
    )
    logger.info("  Python final: %.3e, evals=%s", py_record.best_fitness, py_record.evaluations)
    logger.info("  C final:     %.3e, evals=%s", c_record.best_fitness, c_record.evaluations)
    logger.info(
        "  Trajectory divergence: max_abs diff=%s at eval %s",
        f"{max_diff:.3e}",
        max_diff_idx + 1,
    )
    logger.info(
        "  first meaningful divergence eval: %s",
        first_diff or "n/a",
    )
    logger.info(
        "  C reaches eps=%s at eval: %s",
        spec.epsilon,
        c_trace.first_reach_target or "never",
    )

    if max_iter > 0 and c_by_iter_arr is not None:
        logger.info("  Iteration-level comparison (first 10 iterations):")
        limit = min(10, max_iter)
        for i in range(limit):
            logger.info(
                "    it %s: py=%.3e, c=%.3e, diff=%.3e, eval=%s",
                f"{i:3d}",
                py_by_iter[i],
                c_by_iter_arr[i],
                abs(py_by_iter[i] - c_by_iter_arr[i]),
                min((i + 1) * _NUM_PARTICLES, c_trace.target_evals),
            )

        logger.info(
            "  Max iteration-level diff at it %s: py=%.3e, c=%.3e, diff=%.3e",
            max_iter_idx,
            py_by_iter[max_iter_idx],
            c_by_iter_arr[max_iter_idx],
            abs(py_by_iter[max_iter_idx] - c_by_iter_arr[max_iter_idx]),
        )


def run_c_benchmark(
    spec: CReferenceProblem,
    runs: int,
    seed: int,
    c_binary: Path,
    c_root: Path,
    force: bool,
) -> list[RunRecord]:
    run_file = c_root / "results" / f"f{spec.c_code}" / f"f_run_seed{seed}.txt"
    if not run_file.exists() or force:
        if force and run_file.exists():
            run_file.unlink()
        cmd = [str(c_binary), "-f", str(spec.c_code), "-R", str(runs), "-S", str(_CEILING_SWARM), "-sd", str(seed)]
        subprocess.run(
            cmd,
            cwd=str(c_root),
            check=True,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    records = _parse_c_run_file(run_file, spec.eval_budget)
    if len(records) == 0:
        raise RuntimeError(f"No C runs parsed from {run_file}")
    if len(records) != runs:
        # Keep partial data but warn by returning all parsed lines.
        pass
    return records


def summarize(name: str, records: list[RunRecord], eps: float) -> dict[str, float]:
    values = np.array([rec.best_fitness for rec in records], dtype=float)
    evals = np.array([rec.evaluations for rec in records], dtype=int)
    success = float(np.mean(values <= eps) * 100.0)
    return {
        "name": name,
        "runs": float(len(records)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "success_rate": success,
        "mean_evals": float(np.mean(evals)),
        "mean_log10_error": float(np.mean(np.log10(np.maximum(values, 1e-300)))),
    }


def print_summary(label: str, summary: dict[str, float]) -> None:
    logger.info("%-12s runs=%s", label, summary["runs"])
    logger.info(
        "  mean={mean:.3e}, std={std:.3e}, median={median:.3e}, "
        "min={min:.3e}, max={max:.3e}, success={success:.2f}%".format(
            mean=summary["mean"],
            std=summary["std"],
            median=summary["median"],
            min=summary["min"],
            max=summary["max"],
            success=summary["success_rate"],
        )
    )
    logger.info(
        "  evals_mean=%.1f, log10(mean_error)=%.3f",
        summary["mean_evals"],
        summary["mean_log10_error"],
    )


def compare_optimizer(spec: CReferenceProblem, args: argparse.Namespace) -> None:
    logger.info(
        "\n[optimizer] %s (C function %s, %sD, runs=%s, seed=%s)",
        spec.name,
        spec.c_code,
        spec.dimension,
        args.runs,
        args.seed,
    )
    run_py_track_history = args.track_history
    py_records = run_python_benchmark(
        spec,
        runs=args.runs,
        seed=args.seed,
        track_history=run_py_track_history,
        rng_mode=args.rng_mode,
        c_style_stream=args.c_style_rng_stream,
    )
    c_records = run_c_benchmark(
        spec,
        runs=args.runs,
        seed=args.seed,
        c_binary=Path(args.c_binary).resolve(),
        c_root=Path(args.c_root).resolve(),
        force=args.force_c_run,
    )

    py_summary = summarize("python", py_records, eps=spec.epsilon)
    c_summary = summarize("c", c_records, eps=spec.epsilon)

    print_summary("python", py_summary)
    print_summary("c", c_summary)

    # Paired-by-run difference (same run index only, not guaranteed seed-level equivalence).
    n = min(len(py_records), len(c_records))
    py_vals = np.array([r.best_fitness for r in py_records[:n]], dtype=float)
    c_vals = np.array([r.best_fitness for r in c_records[:n]], dtype=float)
    diff = py_vals - c_vals
    logger.info(
        "paired_diff: mean=%.3e, median=%.3e, abs_max=%.3e",
        float(np.mean(diff)),
        float(np.median(diff)),
        float(np.max(np.abs(diff))),
    )

    if args.top_deltas > 0:
        top_n = min(args.top_deltas, n)
        top_idx = np.argsort(np.abs(diff))[-top_n:][::-1]
        logger.info("top_deltas (run, seed, py, c, diff, evals_py, evals_c):")
        for idx in top_idx:
            run_seed = args.seed + int(idx)
            logger.info(
                "  #%s seed=%s py=%.3e c=%.3e abs_diff=%.3e evals=%s/%s",
                f"{int(idx) + 1:3d}",
                run_seed,
                py_vals[idx],
                c_vals[idx],
                abs(diff[idx]),
                py_records[idx].evaluations,
                c_records[idx].evaluations,
            )

    py_mean = py_summary["mean"]
    c_mean = c_summary["mean"]
    py_std = py_summary["std"]
    c_std = c_summary["std"]
    pooled = 1.0 if py_std == 0 and c_std == 0 else np.sqrt((py_std**2 + c_std**2) / 2)
    cohen_d = (py_mean - c_mean) / pooled if pooled > 0 else float("nan")
    logger.info("effect size (Cohen's d, means): %.3f", cohen_d)

    if args.analyze_run is not None:
        analyze_index = args.analyze_run
        if not (0 <= analyze_index < n):
            raise ValueError(f"Requested analyze-run {analyze_index} is outside parsed range [0, {n - 1}]")

        # If the requested run didn't keep trajectory history, rerun it
        # to enable trajectory-level diagnostics.
        py_target = py_records[analyze_index]
        if py_target.trajectory is None:
            if args.c_style_rng_stream:
                shared_rng = _build_rng(args.seed, args.rng_mode)
                for _ in range(analyze_index):
                    _run_one_python_spso(
                        spec,
                        args.seed + _,
                        track_history=False,
                        rng_mode=args.rng_mode,
                        rng=shared_rng,
                    )
                py_target = _run_one_python_spso(
                    spec,
                    args.seed + analyze_index,
                    track_history=True,
                    rng_mode=args.rng_mode,
                    rng=shared_rng,
                )
            else:
                py_target = _run_one_python_spso(
                    spec,
                    args.seed + analyze_index,
                    track_history=True,
                    rng_mode=args.rng_mode,
                )

        _compare_run_diagnostics(
            spec,
            args.seed,
            py_target,
            c_records[analyze_index],
            Path(args.c_root).resolve(),
            c_records,
        )


def _build_seed_list(args: argparse.Namespace) -> list[int]:
    if args.seeds is not None and len(args.seeds) > 0:
        return [int(seed) for seed in args.seeds]
    if args.seed_count <= 0:
        raise ValueError("--seed-count must be >= 1")
    if args.seed_count == 1:
        return [int(args.seed)]
    return [int(args.seed + args.seed_step * i) for i in range(args.seed_count)]


def _print_seed_summary(
    seed: int,
    py_summary: dict[str, float],
    c_summary: dict[str, float],
) -> None:
    logger.info("seed=%s", seed)
    logger.info(
        "  python: mean={mean:.3e}, std={std:.3e}, median={median:.3e}, "
        "min={min:.3e}, max={max:.3e}, success={success:.2f}%, evals_mean={evals:.1f}".format(
            mean=py_summary["mean"],
            std=py_summary["std"],
            median=py_summary["median"],
            min=py_summary["min"],
            max=py_summary["max"],
            success=py_summary["success_rate"],
            evals=py_summary["mean_evals"],
        )
    )
    logger.info(
        "  c-ref:  mean={mean:.3e}, std={std:.3e}, median={median:.3e}, "
        "min={min:.3e}, max={max:.3e}, success={success:.2f}%, evals_mean={evals:.1f}".format(
            mean=c_summary["mean"],
            std=c_summary["std"],
            median=c_summary["median"],
            min=c_summary["min"],
            max=c_summary["max"],
            success=c_summary["success_rate"],
            evals=c_summary["mean_evals"],
        )
    )


def compare_optimizer_seed_sweep(spec: CReferenceProblem, args: argparse.Namespace) -> None:
    seed_list = _build_seed_list(args)
    logger.info(
        "[seed-sweep] %s (%sD) runs=%s, seeds=%s, first_seed=%s",
        spec.name,
        spec.dimension,
        args.runs,
        len(seed_list),
        seed_list[0],
    )

    all_py_records: list[RunRecord] = []
    all_c_records: list[RunRecord] = []
    per_seed_py_means: list[float] = []
    per_seed_c_means: list[float] = []
    per_seed_py_success: list[float] = []
    per_seed_c_success: list[float] = []
    per_seed_summaries: list[dict[str, Any]] = []

    for seed in seed_list:
        py_records = run_python_benchmark(
            spec,
            runs=args.runs,
            seed=seed,
            track_history=False,
            rng_mode=args.rng_mode,
            c_style_stream=args.c_style_rng_stream,
        )
        c_records = run_c_benchmark(
            spec,
            runs=args.runs,
            seed=seed,
            c_binary=Path(args.c_binary).resolve(),
            c_root=Path(args.c_root).resolve(),
            force=args.force_c_run,
        )

        py_summary = summarize("python", py_records, eps=spec.epsilon)
        c_summary = summarize("c", c_records, eps=spec.epsilon)
        _print_seed_summary(seed, py_summary, c_summary)
        per_seed_summaries.append(
            {
                "seed": seed,
                "python": py_summary,
                "c": c_summary,
            }
        )

        all_py_records.extend(py_records)
        all_c_records.extend(c_records)
        per_seed_py_means.append(py_summary["mean"])
        per_seed_c_means.append(c_summary["mean"])
        per_seed_py_success.append(py_summary["success_rate"])
        per_seed_c_success.append(c_summary["success_rate"])

    logger.info(
        "[seed-sweep aggregate] seed_count=%s, total_runs=%s",
        len(seed_list),
        len(all_py_records),
    )
    agg_py = summarize("python", all_py_records, eps=spec.epsilon)
    agg_c = summarize("c", all_c_records, eps=spec.epsilon)
    print_summary("python", agg_py)
    print_summary("c-ref", agg_c)

    all_py_vals = np.array([r.best_fitness for r in all_py_records], dtype=float)
    all_c_vals = np.array([r.best_fitness for r in all_c_records], dtype=float)
    diff = all_py_vals - all_c_vals
    logger.info(
        "paired_by_run_diff: mean=%.3e, median=%.3e, abs_max=%.3e",
        float(np.mean(diff)),
        float(np.median(diff)),
        float(np.max(np.abs(diff))),
    )

    py_pool_std = agg_py["std"]
    c_pool_std = agg_c["std"]
    pooled = np.sqrt((py_pool_std**2 + c_pool_std**2) / 2.0) if (py_pool_std or c_pool_std) else 1.0
    cohen_d = (agg_py["mean"] - agg_c["mean"]) / pooled if pooled > 0 else float("nan")
    logger.info("effect size (Cohen's d over pooled run set): %.3f", cohen_d)

    seed_means_py = np.array(per_seed_py_means, dtype=float)
    seed_means_c = np.array(per_seed_c_means, dtype=float)
    logger.info(
        "seed-mean summary: python=%.3e±%.3e (mean±std), c-ref=%.3e±%.3e",
        float(np.mean(seed_means_py)),
        float(np.std(seed_means_py)),
        float(np.mean(seed_means_c)),
        float(np.std(seed_means_c)),
    )
    logger.info(
        "seed-success summary: python=%.3f±%.3f%%, c-ref=%.3f±%.3f%%",
        float(np.mean(per_seed_py_success)),
        float(np.std(per_seed_py_success)),
        float(np.mean(per_seed_c_success)),
        float(np.std(per_seed_c_success)),
    )

    return {
        "problem": spec.name,
        "dimension": spec.dimension,
        "c_code": spec.c_code,
        "runs": args.runs,
        "seed_count": len(seed_list),
        "seed_list": seed_list,
        "seed_summaries": per_seed_summaries,
        "aggregate": {
            "python": agg_py,
            "c": agg_c,
        },
        "paired_diff": {
            "mean": float(np.mean(all_py_vals - all_c_vals)),
            "median": float(np.median(all_py_vals - all_c_vals)),
            "abs_max": float(np.max(np.abs(all_py_vals - all_c_vals))),
        },
        "cohen_d": float(cohen_d),
        "seed_mean_summary": {
            "python_mean": float(np.mean(seed_means_py)),
            "python_std": float(np.std(seed_means_py)),
            "c_mean": float(np.mean(seed_means_c)),
            "c_std": float(np.std(seed_means_c)),
        },
        "seed_success_summary": {
            "python_mean": float(np.mean(per_seed_py_success)),
            "python_std": float(np.std(per_seed_py_success)),
            "c_mean": float(np.mean(per_seed_c_success)),
            "c_std": float(np.std(per_seed_c_success)),
        },
    }


def compare_function_output(spec: CReferenceProblem, args: argparse.Namespace) -> None:
    logger.info(
        "[function] %s (C function %s, %sD, points=%s, seed=%s)",
        spec.name,
        spec.c_code,
        spec.dimension,
        args.points,
        args.seed,
    )
    rng = np.random.RandomState(args.seed)
    samples = rng.uniform(spec.bounds[:, 0], spec.bounds[:, 1], size=(args.points, spec.dimension))
    problem = spec.build_problem(spec.dimension)

    diffs: list[float] = []
    for point in samples:
        py = _c_reported_value(float(problem.evaluate(point)), spec.bias)
        c_reference = _c_reported_value(float(spec.reference_eval(point, spec.dimension)), spec.bias)
        diffs.append(py - c_reference)

    max_abs = np.max(np.abs(diffs)) if diffs else 0.0
    mean_abs = float(np.mean(np.abs(diffs))) if diffs else 0.0
    logger.info("max_abs_diff=%.3e (using shared C-style definition)", max_abs)
    logger.info("mean_raw_error=%.3e mean_abs=%.3e max_error=%.3e", mean(diffs), mean_abs, max_abs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Python SPSO2011 against C reference parity outputs")
    parser.add_argument(
        "--mode",
        choices=["function", "optimizer", "seed-sweep", "both", "diagnostic"],
        default="both",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=["sphere_30", "ackley_10"],
        choices=sorted(PROBLEM_SPECS.keys()),
        help="Problem keys to compare.",
    )
    parser.add_argument("--seed", type=int, default=1294404794)
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Optional explicit list of base seeds for seed-sweep mode.",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=1,
        help="Number of sequential seeds for seed-sweep mode (default 1).",
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=1,
        help="Step size between sequential seeds for seed-sweep mode.",
    )
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--points", type=int, default=100)
    parser.add_argument(
        "--c-binary",
        default="../c-version/build/standard_pso",
        help="Path to C binary (defaults to repository sibling ../c-version/build/standard_pso)",
    )
    parser.add_argument(
        "--c-root",
        default="../c-version",
        help="Directory containing c-version/Results and where C binary writes outputs.",
    )
    parser.add_argument(
        "--force-c-run",
        action="store_true",
        help="Re-run the C binary even if matching f_run_seed file exists.",
    )
    parser.add_argument(
        "--track-history",
        action="store_true",
        help="Capture Python best-fitness trajectory per run for diagnostic output.",
    )
    parser.add_argument(
        "--top-deltas",
        type=int,
        default=10,
        help="Number of largest run-level paired deltas to display (optimizer/diagnostic mode).",
    )
    parser.add_argument(
        "--analyze-run",
        type=int,
        default=None,
        help="Run index (0-based) to perform full trajectory diagnostics against C trace.",
    )
    parser.add_argument(
        "--rng-mode",
        choices=["numpy", "c-wy"],
        default="numpy",
        help="Python RNG backend for comparison runs.",
    )
    parser.add_argument(
        "--c-style-rng-stream",
        action="store_true",
        help="Use a single shared RNG stream across all runs (like C `-R` loop).",
    )
    parser.add_argument(
        "--export-json",
        help="Path to write JSON summary (seed-sweep mode currently).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    if args.mode in {"function", "both"}:
        for key in args.problems:
            compare_function_output(PROBLEM_SPECS[key], args)

    if args.mode == "diagnostic" and args.analyze_run is None:
        raise ValueError("--analyze-run is required when --mode diagnostic")

    if args.export_json is not None and args.mode != "seed-sweep":
        raise ValueError("--export-json is currently supported only for --mode seed-sweep")

    seed_sweep_reports: list[dict[str, Any]] = []
    if args.mode in {"optimizer", "both", "diagnostic", "seed-sweep"}:
        for key in args.problems:
            if args.mode == "seed-sweep":
                seed_sweep_reports.append(compare_optimizer_seed_sweep(PROBLEM_SPECS[key], args))
            else:
                compare_optimizer(PROBLEM_SPECS[key], args)

    if args.mode == "seed-sweep" and args.export_json is not None:
        payload = {
            "mode": "seed-sweep",
            "problems": args.problems,
            "seed": args.seed,
            "seed_count": args.seed_count,
            "seed_step": args.seed_step,
            "runs": args.runs,
            "problems_run": seed_sweep_reports,
            "rng_mode": args.rng_mode,
            "c_style_rng_stream": args.c_style_rng_stream,
        }
        with Path(args.export_json).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        logger.info("Saved seed-sweep report to: %s", args.export_json)


if __name__ == "__main__":
    main()
