#!/usr/bin/env bash
set -euo pipefail

# Edit this section for a different checkpoint or inference setup.
SERVER_ROOT="${SERVER_ROOT:-/mnt/cfs/usr/jixiang/vla_learning/Isaac-GR00T}"
PYTHON_BIN="${PYTHON_BIN:-/root/gr00t-venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-${SERVER_ROOT}/seer_robot_gr00t_finetune/checkpoints/seer_lerobot_ee_v21_full/seer_lerobot_ee_v21_full/checkpoint-10000}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18081}"
DEVICE="${DEVICE:-cuda}"
DENOISING_STEPS="${DENOISING_STEPS:-2}"

# Optional extra arguments, for example:
#   EXTRA_ARGS=(--some-future-option value)
EXTRA_ARGS=()

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/gr00t_inference_server.py" \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --device "${DEVICE}" \
  --denoising-steps "${DENOISING_STEPS}" \
  "${EXTRA_ARGS[@]}"
