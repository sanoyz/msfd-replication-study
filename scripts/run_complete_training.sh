#!/usr/bin/env bash
#
# Complete training pipeline (Pipeline 1):
#   teacher -> cache -> MSFD (5 seeds) -> KD (5 seeds) -> baseline (5 seeds)
#   -> deliberate practice -> progressive distillation -> final evaluation
#
# Expected runtime: ~18 hours on a free-tier T4 GPU.
# Safe to restart: uses per-epoch resume checkpoints.
#
set -euo pipefail

python code/run_all.py --config configs/default.yaml