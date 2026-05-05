#!/bin/bash
# Run the full adjoint intervention ablation study.
# Saves JSON results to results/ablation/ for table generation.
#
# Usage:
#   bash tests/differentiable_simulator/run_ablation.sh
#   bash tests/differentiable_simulator/run_ablation.sh --scene helhest-hard

set -e

OUTDIR="results/ablation"
mkdir -p "$OUTDIR"

SCENE_ARGS="${@:---scene helhest-straight helhest-gentle helhest-hard pendulum box}"
HORIZONS="--horizons 1 5 10 20 40"
DECAY="--decay-steps 40"

echo "============================================"
echo "  Adjoint Intervention Ablation Study"
echo "  Output: $OUTDIR/"
echo "============================================"

echo ""
echo ">>> 1/4: Baseline (hard threshold)"
python -m tests.differentiable_simulator.benchmark_gradient_quality \
    $SCENE_ARGS $HORIZONS $DECAY \
    --save "$OUTDIR/baseline.json"

echo ""
echo ">>> 2/4: Soft blending only"
python -m tests.differentiable_simulator.benchmark_gradient_quality \
    $SCENE_ARGS $HORIZONS $DECAY \
    --soft-blending \
    --save "$OUTDIR/soft_blending.json"

echo ""
echo ">>> 3/4: Normalization only"
python -m tests.differentiable_simulator.benchmark_gradient_quality \
    $SCENE_ARGS $HORIZONS $DECAY \
    --normalize \
    --save "$OUTDIR/normalization.json"

echo ""
echo ">>> 4/4: Soft blending + normalization"
python -m tests.differentiable_simulator.benchmark_gradient_quality \
    $SCENE_ARGS $HORIZONS $DECAY \
    --soft-blending --normalize \
    --save "$OUTDIR/soft_blending_normalize.json"

echo ""
echo "============================================"
echo "  All done. Results in $OUTDIR/"
echo "============================================"
ls -la "$OUTDIR/"
