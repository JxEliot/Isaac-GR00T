# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

SHELL ["/bin/bash", "-c"]

ARG UV_VERSION=0.10.12
ARG UV_DEFAULT_INDEX=http://mirrors.baidubce.com/pypi/simple/
ARG UV_INDEX_URL=http://mirrors.baidubce.com/pypi/simple/

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX} \
    UV_INDEX_URL=${UV_INDEX_URL} \
    UV_PROJECT_ENVIRONMENT=/root/gr00t-venv \
    UV_CACHE_DIR=/tmp/gr00t-uv-cache \
    UV_LINK_MODE=hardlink \
    UV_HTTP_TIMEOUT=300 \
    UV_HTTP_RETRIES=5 \
    UV_CONCURRENT_DOWNLOADS=4 \
    VIRTUAL_ENV=/root/gr00t-venv \
    PATH=/root/gr00t-venv/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked <<'EOF'
set -euo pipefail
apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    git-lfs \
    libaio-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libswscale-dev \
    ninja-build \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python-is-python3
rm -rf /var/lib/apt/lists/*
curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | env UV_INSTALL_DIR=/usr/local/bin sh
git lfs install --system
EOF

WORKDIR /workspace/Isaac-GR00T
COPY . .

# The environment lives under /root. BuildKit cache is separate from the final image.
RUN --mount=type=cache,target=/root/.cache/uv <<'EOF'
set -euo pipefail
export UV_CACHE_DIR=/root/.cache/uv
export UV_LINK_MODE=hardlink
export FLASH_ATTN_WHEEL_DIR=/root/.cache/uv/gr00t-wheels
bash scripts/deployment/baidu_gpu/setup_env.sh
EOF

CMD ["/bin/bash"]
