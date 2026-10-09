#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=baidu_uv_env.sh
source "$SCRIPT_DIR/baidu_uv_env.sh"

if [[ ! -f "$UV_PROJECT_ENVIRONMENT/bin/activate" ]]; then
    echo "Environment not found: $UV_PROJECT_ENVIRONMENT" >&2
    echo "Run: bash $SCRIPT_DIR/setup_env.sh" >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
