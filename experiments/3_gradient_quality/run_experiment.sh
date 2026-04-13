#!/usr/bin/env bash
# Run gradient-quality optimization for all simulators.
#
# Each script loads the real-robot ground truth, uses calibrated params from
# Experiment 1, and optimizes a K-knot spline to match the real trajectory.
#
# Usage:
#   ./run_all.sh                             # run all
#   ./run_all.sh --axion                     # run only Axion
#   ./run_all.sh --axion --mjx               # run Axion and MJX
#   ./run_all.sh --iterations 100 --K 15     # forward extra args to every script
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$DIR/../data"
RESULTS="$DIR/results"
mkdir -p "$RESULTS"

GT="$DATA/right_turn_b.json"

RUN_AXION=false; RUN_MJX=false; RUN_SEMI=false; RUN_TINY=false
RUN_ALL=true
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --axion)          RUN_AXION=true; RUN_ALL=false; shift;;
        --mjx)            RUN_MJX=true;   RUN_ALL=false; shift;;
        --semi-implicit)  RUN_SEMI=true;  RUN_ALL=false; shift;;
        --tinydiffsim)    RUN_TINY=true;  RUN_ALL=false; shift;;
        *)                EXTRA_ARGS+=("$1"); shift;;
    esac
done

if $RUN_ALL || $RUN_AXION; then
    echo "=== Axion (adjoint) ==="
    python "$DIR/optimize_axion.py" \
        --ground-truth "$GT" \
        --save "$RESULTS/axion.json" \
        "${EXTRA_ARGS[@]}"
    echo ""
fi

if $RUN_ALL || $RUN_MJX; then
    echo "=== MJX (jax.grad / BPTT) ==="
    python "$DIR/optimize_mjx.py" \
        --ground-truth "$GT" \
        --save "$RESULTS/mjx.json" \
        "${EXTRA_ARGS[@]}"
    echo ""
fi

if $RUN_ALL || $RUN_SEMI; then
    echo "=== Semi-Implicit (warp tape / BPTT) ==="
    python "$DIR/optimize_semi_implicit.py" \
        --ground-truth "$GT" \
        --save "$RESULTS/semi_implicit.json" \
        "${EXTRA_ARGS[@]}"
    echo ""
fi

if $RUN_ALL || $RUN_TINY; then
    echo "=== TinyDiffSim (CppAD) ==="
    python "$DIR/optimize_tinydiffsim.py" \
        --ground-truth "$GT" \
        --save "$RESULTS/tinydiffsim.json" \
        "${EXTRA_ARGS[@]}"
    echo ""
fi

echo "Done. Results in $RESULTS/"
