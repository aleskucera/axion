#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$DIR/stacked_boxes/run_all.sh"
bash "$DIR/control_stability/run_all.sh"
bash "$DIR/ball_throw/run_all.sh"
bash "$DIR/curling_box/run_all.sh"
bash "$DIR/helhest/run_all.sh"
