# SPSO2011 Migration Readiness Checklist (Python vs C)

Purpose:
Validate that the Python SPSO2011 implementation is stable enough to replace or ship alongside the C reference.

## 1) Function-level parity baseline

Run fixed-point checks first (shared benchmark-space formula, including bias convention):

```bash
./.venv/bin/python scripts/verify_spso2011_parity.py --problems sphere_30 ackley_30 --runs 100 --seed 1294404794
```

Set `--skip-function-checks` when you have already confirmed functions for the current change set.

## 2) Optimizer parity across a single seed stream

Quick smoke run with C-style RNG stream semantics:

```bash
./.venv/bin/python scripts/compare_spso2011_c_vs_python.py \
  --mode optimizer \
  --problems ackley_10 \
  --runs 50 \
  --seed 1294404794 \
  --rng-mode c-wy \
  --c-style-rng-stream \
  --c-root /tmp/spso2011_runs \
  --c-binary ../c-version/build/standard_pso
```

## 3) Seed-sweep migration evaluation (preferred)

Run multiple base seeds and export machine-readable aggregates:

```bash
./.venv/bin/python scripts/compare_spso2011_c_vs_python.py \
  --mode seed-sweep \
  --problems ackley_10 \
  --runs 500 \
  --seed 1294404794 \
  --seed-count 20 \
  --seed-step 1 \
  --rng-mode c-wy \
  --c-style-rng-stream \
  --c-root /tmp/spso2011_runs \
  --c-binary ../c-version/build/standard_pso \
  --force-c-run \
  --export-json /tmp/ackley_seed_sweep.json
```

The JSON schema includes:
- per-seed summaries (`seed_summaries`)
- aggregate pool summaries (`aggregate`)
- paired run-level delta metrics (`paired_diff`)
- `seed_mean_summary`
- `seed_success_summary`
- `cohen_d`

## 4) One-command migration run

Use the helper script to run the baseline checks above in one sequence:

```bash
bash scripts/run_spso2011_migration_check.sh \
  1294404794 \
  500 \
  3 \
  ../c-version/build/standard_pso \
  /tmp/spso2011_runs \
  /tmp/ackley_seed_sweep.json
```

## Pass criteria (recommended)

- Function checks close to numerical precision (`max_abs_diff` low single-digit e-14 for Ackley).
- No persistent directional effect size (`cohen_d`) in seed-sweep.
- Rare large outliers should be quantified (track by `seed_success_summary` and `seed-mean` spread).
- Document rationale for any consistent outperformance/underperformance before declaring migration-ready.
