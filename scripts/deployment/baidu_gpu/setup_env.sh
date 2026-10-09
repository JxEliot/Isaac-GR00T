#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

if [[ "$(id -u)" -ne 0 ]]; then
    echo "Run this script as root because the environment is installed under /root." >&2
    exit 1
fi

# shellcheck source=baidu_uv_env.sh
source "$SCRIPT_DIR/baidu_uv_env.sh"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found on PATH." >&2
    exit 1
fi
if [[ ! -x /usr/bin/python3.12 ]]; then
    echo "Python 3.12 is required at /usr/bin/python3.12." >&2
    exit 1
fi

# TorchCodec requires FFmpeg shared libraries, which uv cannot install.
if ! command -v ffmpeg >/dev/null 2>&1; then
    apt-get update
    apt-get install -y --no-install-recommends ffmpeg
fi

mkdir -p "$UV_CACHE_DIR" /root/.triton/autotune
cd "$REPO_ROOT"

echo "Repository: $REPO_ROOT"
echo "Python environment: $UV_PROJECT_ENVIRONMENT"
echo "uv cache: $UV_CACHE_DIR"
echo "default Python index: $UV_DEFAULT_INDEX"
echo "GPU availability is not required while resolving and installing CUDA wheels."

# Keep the committed resolution unchanged. Explicit sources in pyproject.toml and uv.lock
# still provide PyTorch CUDA, NVIDIA TensorRT, and FlashAttention artifacts.
uv sync \
    --frozen \
    --python /usr/bin/python3.12 \
    --no-dev \
    --no-install-package flash-attn \
    --no-install-project

bash "$SCRIPT_DIR/install_flash_attn.sh"

uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps -e "$REPO_ROOT"

"$UV_PROJECT_ENVIRONMENT/bin/python" - <<'PY'
import sys

import torch
import transformers
import tyro
import flash_attn
import torchcodec
from torchcodec.decoders import VideoDecoder

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"torch_cuda_build={torch.version.cuda}")
print(f"cuda_available_now={torch.cuda.is_available()}")
print(f"transformers={transformers.__version__}")
print(f"tyro={tyro.__version__}")
print(f"flash_attn={flash_attn.__version__}")
print(f"torchcodec={torchcodec.__version__}; VideoDecoder ready")
PY

cat <<EOF

Environment ready. Activate it with:
  source $SCRIPT_DIR/activate.sh
EOF
