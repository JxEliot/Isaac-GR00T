#!/usr/bin/env bash
set -euo pipefail

export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-http://mirrors.baidubce.com/pypi/simple/}"

# Convert the uploaded SEER LeRobot v3 dataset to the legacy LeRobot v2.1
# layout expected by the GR00T data loader. The v3 source is never modified.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GR00T_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONVERTER_DIR="$GR00T_ROOT/scripts/lerobot_conversion"
LEROBOT_SRC="$GR00T_ROOT/../lerobot/src"
PURE_CONVERTER="$GR00T_ROOT/seer_robot_gr00t_finetune/data_process/convert_v21_pure.py"
PYTHON_BIN="${PYTHON_BIN:-/root/gr00t-venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/cfs/usr/jixiang/train_datasets}"
DATASET_NAME="${DATASET_NAME:-lerobot_ee}"
DATASET_PATH="$DATASET_ROOT/$DATASET_NAME"

if [[ ! -d "$DATASET_PATH" ]]; then
  echo "[ERROR] Dataset does not exist: $DATASET_PATH" >&2
  exit 1
fi

cd "$GR00T_ROOT"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Expected conversion environment is missing: $PYTHON_BIN" >&2
  exit 1
fi
if ! "$PYTHON_BIN" -c 'import pyarrow' >/dev/null 2>&1; then
  echo "[INFO] Installing pyarrow with Baidu PyPI mirror"
  uv pip install --python "$PYTHON_BIN" pyarrow
fi
if [[ ! -x "$PURE_CONVERTER" ]]; then
  echo "[ERROR] Pure converter is missing: $PURE_CONVERTER" >&2
  exit 1
fi

echo "[INFO] Source v3 dataset: $DATASET_PATH"
echo "[INFO] Output v2.1 dataset: ${DATASET_PATH}_v21"

# The upstream converter replaces DATASET_PATH with v2.1 and keeps the original
# as DATASET_PATH_v30. Use a staging copy so the original uploaded dataset is
# preserved exactly and the final v2.1 dataset is placed beside it.
STAGING_ROOT="$(mktemp -d "$DATASET_ROOT/.lerobot_v21_convert.XXXXXX")"
cleanup() { rm -rf "$STAGING_ROOT"; }
trap cleanup EXIT
cp -a "$DATASET_PATH" "$STAGING_ROOT/$DATASET_NAME"
CONVERTED="$STAGING_ROOT/${DATASET_NAME}_v21"
"$PYTHON_BIN" "$PURE_CONVERTER" "$STAGING_ROOT/$DATASET_NAME" "$CONVERTED"

TARGET="${DATASET_PATH}_v21"
rm -rf "$TARGET"
mv "$CONVERTED" "$TARGET"

"$PYTHON_BIN" "$GR00T_ROOT/seer_robot_gr00t_finetune/data_process/validate_v21_dataset.py" "$TARGET"
echo "[OK] v2.1 dataset created: $TARGET"
