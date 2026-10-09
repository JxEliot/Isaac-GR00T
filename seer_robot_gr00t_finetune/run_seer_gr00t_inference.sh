#!/usr/bin/env bash
set -Eeuo pipefail

# Offline PyTorch inference check for the SEER GR00T checkpoint.
# Override any setting with an environment variable, for example:
#   CHECKPOINT=checkpoint-8000 TRAJ_IDS="0 1" STEPS=20 ./run_seer_gr00t_inference.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/root/gr00t-venv/bin/python}"
MODEL_ROOT="${MODEL_ROOT:-${SCRIPT_DIR}/checkpoints/seer_lerobot_ee_v21_full/seer_lerobot_ee_v21_full}"
CHECKPOINT="${CHECKPOINT:-checkpoint-10000}"
DATASET_PATH="${DATASET_PATH:-/mnt/cfs/usr/jixiang/train_datasets/lerobot_ee_v21}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"
TRAJ_IDS="${TRAJ_IDS:-0 1 2 3 4}"
STEPS="${STEPS:-0}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-8}"
DENOISING_STEPS="${DENOISING_STEPS:-2}"
INFERENCE_MODE="${INFERENCE_MODE:-pytorch}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/inference_results}"
PLOT_PATH="${PLOT_PATH:-${OUTPUT_DIR}/trajectory.png}"
SKIP_TIMING_STEPS="${SKIP_TIMING_STEPS:-1}"

usage() {
  cat <<'EOF'
Usage:
  ./run_seer_gr00t_inference.sh

Environment overrides:
  CHECKPOINT=checkpoint-10000   checkpoint directory name
  MODEL_ROOT=...                 parent directory containing checkpoint-*
  DATASET_PATH=...               LeRobot dataset path
  TRAJ_IDS="0 1 2 3 4"           trajectory IDs
  STEPS=0                        full episodes; positive values limit steps
  EXECUTION_HORIZON=8            executed actions per prediction
  DENOISING_STEPS=2              diffusion denoising steps
  OUTPUT_DIR=/tmp/...            result directory
  PLOT_PATH=/tmp/...png          plot path (episode suffix added for multiple IDs)
  PYTHON_BIN=/root/gr00t-venv/bin/python

Example:
  TRAJ_IDS="0 1" STEPS=20 ./run_seer_gr00t_inference.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODEL_PATH="${MODEL_ROOT}/${CHECKPOINT}"
SCRIPT_PATH="${REPO_ROOT}/scripts/deployment/standalone_inference_script.py"
mkdir -p "${OUTPUT_DIR}"

[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: Python not found: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${SCRIPT_PATH}" ]] || { echo "ERROR: GR00T inference script not found: ${SCRIPT_PATH}" >&2; exit 1; }
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "ERROR: checkpoint not found: ${MODEL_PATH}" >&2; exit 1; }
[[ -f "${DATASET_PATH}/meta/info.json" ]] || { echo "ERROR: dataset not found: ${DATASET_PATH}" >&2; exit 1; }

read -r -a TRAJ_ARGS <<< "${TRAJ_IDS}"
echo "GR00T SEER inference"
echo "  model:       ${MODEL_PATH}"
echo "  dataset:     ${DATASET_PATH}"
echo "  trajectories: ${TRAJ_IDS}"
echo "  steps:       ${STEPS}"
echo "  horizon:     ${EXECUTION_HORIZON}"
echo "  denoising:   ${DENOISING_STEPS}"
echo "  output plot: ${PLOT_PATH}"

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" "${SCRIPT_PATH}" \
  --model-path "${MODEL_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --embodiment-tag "${EMBODIMENT_TAG}" \
  --traj-ids "${TRAJ_ARGS[@]}" \
  --steps "${STEPS}" \
  --execution-horizon "${EXECUTION_HORIZON}" \
  --denoising-steps "${DENOISING_STEPS}" \
  --inference-mode "${INFERENCE_MODE}" \
  --skip-timing-steps "${SKIP_TIMING_STEPS}" \
  --save-plot-path "${PLOT_PATH}" \
  --save-results-dir "${OUTPUT_DIR}"
