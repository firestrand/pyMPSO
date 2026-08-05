#!/usr/bin/env bash

set -euo pipefail

SEED="${1:-1294404794}"
RUNS="${2:-500}"
SEED_COUNT="${3:-3}"
C_BIN="${4:-../c-version/build/standard_pso}"
C_ROOT="${5:-/tmp/spso2011_runs}"
OUT_JSON="${6:-/tmp/spso2011_seed_sweep.json}"

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"

echo "[migration] Function-level checks (seed=${SEED})"
"${PYTHON_BIN}" scripts/verify_spso2011_parity.py \
  --runs 100 \
  --seed "${SEED}" \
  --problems sphere_30 ackley_30

echo "[migration] Seed-sweep parity check for Ackley (runs=${RUNS}, seeds=${SEED_COUNT})"
"${PYTHON_BIN}" scripts/compare_spso2011_c_vs_python.py \
  --mode seed-sweep \
  --problems ackley_10 \
  --seed "${SEED}" \
  --seed-count "${SEED_COUNT}" \
  --seed-step 1 \
  --runs "${RUNS}" \
  --rng-mode c-wy \
  --c-style-rng-stream \
  --c-root "${C_ROOT}" \
  --c-binary "${C_BIN}" \
  --force-c-run \
  --export-json "${OUT_JSON}"

echo "[migration] Report saved to ${OUT_JSON}"
