#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "This Baidu deployment wheel is currently defined for x86_64 only." >&2
    exit 1
fi

UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/root/gr00t-venv}"
WHEEL_DIR="${FLASH_ATTN_WHEEL_DIR:-/mnt/cfs/usr/jixiang/.cache/gr00t-wheels}"
WHEEL_NAME="flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
WHEEL_PATH="$WHEEL_DIR/$WHEEL_NAME"
EXPECTED_SHA256="4e2f9e39313266b1544b68138b15b91ee6221eccf14f7902b7c6620351340810"
OFFICIAL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/$WHEEL_NAME"
PROXY_PREFIX="${FLASH_ATTN_PROXY_PREFIX:-https://gh-proxy.com/}"

mkdir -p "$WHEEL_DIR"

verify_wheel() {
    [[ -f "$WHEEL_PATH" ]] || return 1
    local actual
    actual="$(sha256sum "$WHEEL_PATH" | cut -d" " -f1)"
    [[ "$actual" == "$EXPECTED_SHA256" ]]
}

if ! verify_wheel; then
    rm -f "$WHEEL_PATH"
    echo "Downloading FlashAttention wheel through: $PROXY_PREFIX"
    wget \
        --tries=5 \
        --timeout=60 \
        --continue \
        --output-document="$WHEEL_PATH" \
        "${PROXY_PREFIX}${OFFICIAL_URL}"
fi

if ! verify_wheel; then
    actual="$(sha256sum "$WHEEL_PATH" 2>/dev/null | cut -d" " -f1 || true)"
    echo "FlashAttention SHA256 mismatch." >&2
    echo "expected: $EXPECTED_SHA256" >&2
    echo "actual:   ${actual:-missing}" >&2
    exit 1
fi

uv pip install \
    --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
    --no-deps \
    --reinstall \
    "$WHEEL_PATH"
