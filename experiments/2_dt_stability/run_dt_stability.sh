#!/usr/bin/env bash
# Run dt stability sweeps for all simulators on the obstacle scene.
# Uses calibrated params from Experiment 1 (sim_to_real).
#
# Usage:
#   ./run_dt_stability.sh                   # run all
#   ./run_dt_stability.sh --axion           # run only Axion
#   ./run_dt_stability.sh --axion --mujoco  # run Axion and MuJoCo
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="$DIR/results"
mkdir -p "$RESULTS"

RUN_AXION=false; RUN_MUJOCO=false; RUN_SEMI=false
RUN_ALL=true
for arg in "$@"; do
    case $arg in
        --axion) RUN_AXION=true; RUN_ALL=false;;
        --mujoco) RUN_MUJOCO=true; RUN_ALL=false;;
        --semi-implicit) RUN_SEMI=true; RUN_ALL=false;;
    esac
done

if $RUN_ALL || $RUN_AXION; then
    echo "=== Axion ==="
    python "$DIR/sweep_axion.py" --save "$RESULTS/sweep_axion.json"
    echo ""
fi

if $RUN_ALL || $RUN_MUJOCO; then
    echo "=== MuJoCo ==="
    python "$DIR/sweep_mujoco.py" --save "$RESULTS/sweep_mujoco.json"
    echo ""
fi

if $RUN_ALL || $RUN_SEMI; then
    echo "=== Semi-Implicit ==="
    python "$DIR/sweep_semi_implicit.py" --save "$RESULTS/sweep_semi_implicit.json"
    echo ""
fi

echo "Done. Results in $RESULTS/"
