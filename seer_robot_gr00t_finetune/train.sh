#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

GR00T_VENV="${GR00T_VENV:-/root/gr00t-venv}"
if [[ -x "$GR00T_VENV/bin/python" ]]; then
    export PATH="$GR00T_VENV/bin:$PATH"
fi

WANDB_ENV_FILE="${WANDB_ENV_FILE:-/mnt/cfs/usr/jixiang/wandb_key/wandb.env}"
if [[ -r "$WANDB_ENV_FILE" ]]; then
    set -a
    # Keep the API key outside the repository while exporting it to the trainer.
    source "$WANDB_ENV_FILE"
    set +a
fi

DATASET_PATH="${DATASET_PATH:-/mnt/cfs/usr/jixiang/train_datasets/parts_box_packing_lerobot_retimed_clean/parts_box_packing}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/mnt/cfs/usr/jixiang/hf_models/GR00T-N1.7-3B}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/checkpoints/parts_box_packing}"

NUM_GPUS="${NUM_GPUS:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-gr00t-parts-box-packing}"

MODE="${1:-train}"
case "$MODE" in
    train)
        ;;
    smoke)
        MAX_STEPS="${SMOKE_MAX_STEPS:-50}"
        SAVE_STEPS="$MAX_STEPS"
        ;;
    resume)
        ;;
    *)
        echo "Usage: bash $0 [train|smoke|resume]" >&2
        exit 2
        ;;
esac

python3 "$SCRIPT_DIR/validate_dataset.py" "$DATASET_PATH"
mkdir -p "$OUTPUT_DIR"

export NUM_GPUS GLOBAL_BATCH_SIZE MAX_STEPS SAVE_STEPS
export DATALOADER_NUM_WORKERS USE_WANDB
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

args=(
    --base-model-path "$BASE_MODEL_PATH"
    --dataset-path "$DATASET_PATH"
    --embodiment-tag new_embodiment
    --modality-config-path "$SCRIPT_DIR/modality_config.py"
    --output-dir "$OUTPUT_DIR"
    --experiment-name parts_box_packing
    --wandb-project "$WANDB_PROJECT"
)

if [[ "$MODE" == "resume" ]]; then
    args+=(--resume-from-checkpoint)
fi

cd "$REPO_ROOT"
exec bash examples/finetune.sh "${args[@]}"
