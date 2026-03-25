#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$DIR/helhest_batch/run_all.sh"
bash "$DIR/helhest_scalability/run_sweep.sh"
