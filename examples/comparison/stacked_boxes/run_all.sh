#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DIR/results"

python "$DIR/axion_sim.py"        --save "$DIR/results/axion.json"
python "$DIR/mujoco_sim.py"       --save "$DIR/results/mujoco.json"
python "$DIR/genesis_sim.py"      --save "$DIR/results/genesis.json"
python "$DIR/mjx.py"          --save "$DIR/results/mjx.json"
python "$DIR/xpbd.py"         --save "$DIR/results/xpbd.json"
python "$DIR/featherstone.py" --save "$DIR/results/featherstone.json"

python "$DIR/plot_results.py"
