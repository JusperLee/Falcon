#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
APP_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$("$SCRIPT_DIR/bootstrap.sh")"

# Weights come from the Hugging Face short link JusperLee/Falcon; download once before starting.
if [ "$(cd -- "$APP_DIR" && "$PYTHON_BIN" -c 'from falcon_runtime import weights_source; print(weights_source("small"))')" != "cached" ]; then
    printf '\nFirst run: downloading JusperLee/Falcon from Hugging Face (about 45 MB)…\n' >&2
    if ! (cd -- "$APP_DIR" && "$PYTHON_BIN" -c 'from falcon_runtime import resolve_checkpoint; resolve_checkpoint("small")'); then
        printf '\nDownload failed. Retry through a mirror: HF_ENDPOINT=https://hf-mirror.com ./scripts/launch.sh\n' >&2
        exit 1
    fi
fi

printf '\nFalcon Studio · 本地音视频分离\n' >&2
printf '浏览器将自动打开；结束时在此窗口按 Control-C。\n\n' >&2
cd -- "$APP_DIR"
exec "$PYTHON_BIN" "$APP_DIR/server.py" "$@"
