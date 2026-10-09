#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
MODEL_REQUEST_CONFIG="${MODEL_REQUEST_CONFIG:-$PROJECT_ROOT/model_request/config/model_request.json}"
BACKEND="${MODEL_BACKEND:-}" # pi05 or gr00t; empty means config value
LIVE="${MODEL_LIVE:-0}"
PUBLISH_ACTION="${MODEL_PUBLISH_ACTION:-0}"
FAKE_DATA="${MODEL_FAKE_DATA:-0}"
MODEL_URL="${MODEL_URL:-}"

args=( -m model_request --config "$MODEL_REQUEST_CONFIG" )
[[ -n "$BACKEND" ]] && args+=( --backend "$BACKEND" )
[[ "$LIVE" == "1" ]] && args+=( --live )
[[ "$PUBLISH_ACTION" == "1" ]] && args+=( --publish-action )
[[ "$FAKE_DATA" == "1" ]] && args+=( --fake-data )
[[ -n "$MODEL_URL" ]] && args+=( --url "$MODEL_URL" )

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" "${args[@]}"
