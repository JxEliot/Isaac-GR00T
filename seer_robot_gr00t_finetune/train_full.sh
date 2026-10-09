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
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/checkpoints/parts_box_packing_full}"

NUM_GPUS="${NUM_GPUS:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-gr00t-parts-box-packing-full}"
MASTER_PORT="${MASTER_PORT:-29500}"

MODE="${1:-train}"
case "$MODE" in
    train)
        ;;
    smoke)
        MAX_STEPS="${SMOKE_MAX_STEPS:-10}"
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

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

launch_args=(
    gr00t/experiment/launch_finetune.py
    --base_model_path "$BASE_MODEL_PATH"
    --dataset_path "$DATASET_PATH"
    --embodiment_tag new_embodiment
    --modality_config_path "$SCRIPT_DIR/modality_config.py"
    --tune-llm
    --tune-visual
    --num_gpus "$NUM_GPUS"
    --output_dir "$OUTPUT_DIR"
    --experiment_name parts_box_packing_full
    --wandb_project "$WANDB_PROJECT"
    --global_batch_size "$GLOBAL_BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning_rate "$LEARNING_RATE"
    --weight_decay "$WEIGHT_DECAY"
    --warmup_ratio "$WARMUP_RATIO"
    --max_steps "$MAX_STEPS"
    --save_steps "$SAVE_STEPS"
    --save_total_limit 5
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
    --shard_size 1024
    --num_shards_per_epoch 100000
    --episode_sampling_rate 0.1
)

if [[ "$USE_WANDB" == "1" ]]; then
    launch_args+=(--use_wandb)
fi
if [[ "$MODE" == "resume" ]]; then
    launch_args+=(--resume_from_checkpoint)
fi

cd "$REPO_ROOT"
if [[ "$NUM_GPUS" == "1" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    exec python "${launch_args[@]}"
fi

exec torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" \
    "${launch_args[@]}"
