# SEER Robot GR00T Fine-tuning

This directory contains only the GR00T N1.7 fine-tuning flow for the SEER dual-arm robot and
the parts-box-packing dataset. It does not include environment installation, raw-data
conversion, simulation, deployment, PI0.5, or value-model training.

## Fixed inputs

- Dataset: `/mnt/cfs/usr/jixiang/train_datasets/parts_box_packing_lerobot_retimed_clean/parts_box_packing`
- Base model: `/mnt/cfs/usr/jixiang/hf_models/GR00T-N1.7-3B`
- Embodiment tag: `new_embodiment`
- Default output: `seer_robot_gr00t_finetune/checkpoints/parts_box_packing`

The robot data used by this recipe is:

- State, 30 dimensions: two 7-DoF arms, waist, lift, two grippers, and two 6D end-effector poses.
- Action, 14 dimensions: two 6D end-effector targets and two gripper commands.
- Robot cameras: head, left wrist, and right wrist.
- Language: task text selected by `task_index`.

The end-effector targets are stored as absolute poses. The modality config registers them as
relative EEF actions so GR00T computes relative actions against the corresponding current EEF
pose. Gripper commands remain absolute.

## Training flow

Run every command from the repository root:

```bash
cd /mnt/cfs/usr/jixiang/vla_learning/Isaac-GR00T
```

1. Validate the dataset without starting training:

```bash
python3 seer_robot_gr00t_finetune/validate_dataset.py \
  /mnt/cfs/usr/jixiang/train_datasets/parts_box_packing_lerobot_retimed_clean/parts_box_packing
```

2. Run a short end-to-end smoke test:

```bash
NUM_GPUS=1 GLOBAL_BATCH_SIZE=64 \
  bash seer_robot_gr00t_finetune/train.sh smoke
```

3. Start the full training run:

```bash
NUM_GPUS=2 GLOBAL_BATCH_SIZE=64 MAX_STEPS=10000 SAVE_STEPS=2000 \
  bash seer_robot_gr00t_finetune/train.sh train
```

4. Resume the latest checkpoint in the output directory:

```bash
NUM_GPUS=2 GLOBAL_BATCH_SIZE=64 MAX_STEPS=10000 SAVE_STEPS=2000 \
  bash seer_robot_gr00t_finetune/train.sh resume
```

All settings can be overridden with environment variables:

| Variable | Default |
| --- | --- |
| `DATASET_PATH` | fixed dataset path above |
| `BASE_MODEL_PATH` | `/mnt/cfs/usr/jixiang/hf_models/GR00T-N1.7-3B` |
| `OUTPUT_DIR` | `seer_robot_gr00t_finetune/checkpoints/parts_box_packing` |
| `NUM_GPUS` | `1` |
| `GLOBAL_BATCH_SIZE` | `64` |
| `MAX_STEPS` | `10000` |
| `SAVE_STEPS` | `2000` |
| `DATALOADER_NUM_WORKERS` | `4` |
| `USE_WANDB` | `0` |
| `WANDB_PROJECT` | `gr00t-parts-box-packing` |

If GPU memory is insufficient, reduce `GLOBAL_BATCH_SIZE` to `2` or `1`. Set `USE_WANDB=1`
only when experiment tracking is needed.

## Files

- `modality_config.py`: SEER dual-arm state, action, camera, and language definitions.
- `validate_dataset.py`: read-only validation before training.
- `train.sh`: smoke test, full training, and resume entry point.
- `train_full.sh`: full-parameter fine-tuning of the LLM, visual encoder, and action model.

## Full-parameter fine-tuning

`train.sh` uses the repository defaults, which freeze the LLM and visual encoder. Use
`train_full.sh` when every model component must be trained. It enables `tune_llm` and
`tune_visual`; the projector, diffusion action model, and VLLN are trainable by default.

The default `GLOBAL_BATCH_SIZE` is `64`. This is the total batch across all GPUs before
gradient accumulation. Full-parameter fine-tuning uses a separate output directory and a
lower default learning rate of `1e-5`.

Short smoke test:

```bash
NUM_GPUS=2 GLOBAL_BATCH_SIZE=64 \
  bash seer_robot_gr00t_finetune/train_full.sh smoke
```

Full run:

```bash
NUM_GPUS=2 GLOBAL_BATCH_SIZE=64 MAX_STEPS=10000 SAVE_STEPS=2000 \
  bash seer_robot_gr00t_finetune/train_full.sh train
```

Resume:

```bash
NUM_GPUS=2 GLOBAL_BATCH_SIZE=64 MAX_STEPS=10000 SAVE_STEPS=2000 \
  bash seer_robot_gr00t_finetune/train_full.sh resume
```

Full-finetune-specific defaults:

| Variable | Default |
| --- | --- |
| `GLOBAL_BATCH_SIZE` | `64` |
| `GRADIENT_ACCUMULATION_STEPS` | `1` |
| `LEARNING_RATE` | `1e-5` |
| `OUTPUT_DIR` | `seer_robot_gr00t_finetune/checkpoints/parts_box_packing_full` |
