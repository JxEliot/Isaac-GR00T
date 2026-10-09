# Baidu GPU deployment

This deployment targets Baidu AI Cloud on Ubuntu 24.04 x86_64. The current container may have
no visible GPU; the resolved environment still installs the CUDA 12.8 builds required by the
future two-GPU RTX PRO 6000 image.

## Index policy

The Baidu mirror is the default Python package index:

```bash
export UV_DEFAULT_INDEX=http://mirrors.baidubce.com/pypi/simple/
export UV_INDEX_URL=http://mirrors.baidubce.com/pypi/simple/
```

Do not replace registry URLs inside the committed `uv.lock`. The lock records exact package
versions, hashes, and source identities. This project also needs explicit non-PyPI sources:

- PyTorch CUDA 12.8: `https://download.pytorch.org/whl/cu128`
- TensorRT packages: `https://pypi.nvidia.com`
- FlashAttention 2.8.3: the release wheel declared in `pyproject.toml`

`UV_DEFAULT_INDEX` affects normal PyPI packages while those explicit sources remain intact.
`uv sync --frozen` guarantees that setup does not rewrite `uv.lock`.

## Current container setup

The Python environment is installed under `/root/gr00t-venv`. No `.venv` is created inside the
repository. The temporary uv cache is `/tmp/gr00t-uv-cache`, on the same filesystem as `/root`,
so uv can hardlink packages into the environment instead of slowly unpacking them on CFS. The
verified FlashAttention wheel is retained on CFS for reuse.

```bash
cd /mnt/cfs/usr/jixiang/vla_learning/Isaac-GR00T
bash scripts/deployment/baidu_gpu/setup_env.sh
source scripts/deployment/baidu_gpu/activate.sh
```

The setup can run without a GPU. Verify the environment with:

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

`torch.cuda.is_available()` is expected to be `False` on the current CPU-only container and
`True` after the image is started with NVIDIA runtime and the two GPUs attached.

## Build the GPU image

Run from the repository root:

```bash
docker build \
  -f scripts/deployment/baidu_gpu/Dockerfile \
  -t gr00t-n1d7-baidu:cu128 .
```

Run on the future GPU worker:

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --shm-size=32g \
  -v /mnt/cfs:/mnt/cfs \
  gr00t-n1d7-baidu:cu128
```

For full fine-tuning, the training command remains:

```bash
cd /workspace/Isaac-GR00T
NUM_GPUS=2 GLOBAL_BATCH_SIZE=64 \
  bash seer_robot_gr00t_finetune/train_full.sh train
```

## FlashAttention download

The x86_64 FlashAttention wheel is hosted on GitHub, which can be unstable from the Baidu
network. `setup_env.sh` therefore skips that one URL during `uv sync`, downloads the same wheel
through `https://gh-proxy.com/`, verifies it against the SHA256 stored in `uv.lock`, and installs
it with `uv pip`. Override `FLASH_ATTN_PROXY_PREFIX` to use an internal mirror. A previously
verified wheel is reused from `/mnt/cfs/usr/jixiang/.cache/gr00t-wheels`.
