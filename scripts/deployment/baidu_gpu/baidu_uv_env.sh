#!/usr/bin/env bash

# Source this file before running uv manually on Baidu AI Cloud.
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-http://mirrors.baidubce.com/pypi/simple/}"
export UV_INDEX_URL="${UV_INDEX_URL:-$UV_DEFAULT_INDEX}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/root/gr00t-venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/gr00t-uv-cache}"
export UV_LINK_MODE="${UV_LINK_MODE:-hardlink}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-5}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"

export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
