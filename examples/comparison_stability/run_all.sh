#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$DIR/stacked_boxes/run_all.sh"
bash "$DIR/pendulum_control/run_all.sh"
