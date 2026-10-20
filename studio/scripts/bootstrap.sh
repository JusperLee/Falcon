#!/bin/bash
# stdout contains only the selected Python path; messages go to stderr.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
APP_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"

valid_python() {
    [ -x "$1" ] && "$1" -c 'import sys; raise SystemExit(not ((3, 10) <= sys.version_info[:2] < (3, 13)))' >/dev/null 2>&1
}

if [ -n "${FALCON_VENV:-}" ]; then
    ENV_DIR="$FALCON_VENV"
elif valid_python "$APP_DIR/.venv/bin/python"; then
    ENV_DIR="$APP_DIR/.venv"
else
    ENV_DIR="$HOME/Library/Caches/FalconStudio/venv"
fi
case "$ENV_DIR" in
    /*) ;;
    *) ENV_DIR="$PWD/$ENV_DIR" ;;
esac

if valid_python "$ENV_DIR/bin/python" && "$ENV_DIR/bin/python" "$SCRIPT_DIR/check_environment.py" --quiet; then
    printf '%s\n' "$ENV_DIR/bin/python"
    exit 0
fi

if ! valid_python "$ENV_DIR/bin/python"; then
    if [ -e "$ENV_DIR" ]; then
        printf '已有环境无法运行或 Python 版本不受支持：%s\n请设置 FALCON_VENV 指向新的环境目录，再重试。\n' "$ENV_DIR" >&2
        exit 1
    fi
    SOURCE_PYTHON="${FALCON_PYTHON:-}"
    if [ -n "$SOURCE_PYTHON" ] && ! valid_python "$SOURCE_PYTHON"; then
        printf 'FALCON_PYTHON 需要指向可运行的 Python 3.10–3.12：%s\n' "$SOURCE_PYTHON" >&2
        exit 1
    fi
    if [ -z "$SOURCE_PYTHON" ]; then
        for name in python3.11 python3.12 python3.10 python3; do
            candidate="$(command -v "$name" 2>/dev/null || true)"
            if [ -n "$candidate" ] && valid_python "$candidate"; then
                SOURCE_PYTHON="$candidate"
                break
            fi
        done
    fi
    if [ -z "$SOURCE_PYTHON" ]; then
        # Finder does not inherit a terminal's PATH; check common installations.
        for candidate in \
            /opt/homebrew/bin/python3.11 \
            /usr/local/bin/python3.11 \
            /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
            "$HOME"/.local/share/uv/python/cpython-3.11.*/bin/python3.11; do
            if valid_python "$candidate"; then
                SOURCE_PYTHON="$candidate"
                break
            fi
        done
    fi
    if [ -z "$SOURCE_PYTHON" ]; then
        printf '\n没有找到 Python 3.10–3.12。请先安装 Python 3.11，或设置：\n  export FALCON_PYTHON="/完整路径/python3.11"\n然后重新启动。详见 README.md。\n' >&2
        exit 1
    fi
    printf '\n正在创建运行环境：%s\n' "$ENV_DIR" >&2
    mkdir -p "$(dirname -- "$ENV_DIR")"
    "$SOURCE_PYTHON" -m venv "$ENV_DIR" >&2
fi

printf '\n正在安装 Falcon 依赖，首次运行需要联网，请稍候…\n' >&2
if [ -f "$APP_DIR/requirements-lock.txt" ]; then
    REQUIREMENTS_FILE="$APP_DIR/requirements-lock.txt"
else
    REQUIREMENTS_FILE="$APP_DIR/requirements.txt"
fi
if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$ENV_DIR/bin/python" -r "$REQUIREMENTS_FILE" >&2
else
    if ! "$ENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
        "$ENV_DIR/bin/python" -m ensurepip >&2
    fi
    "$ENV_DIR/bin/python" -m pip install -r "$REQUIREMENTS_FILE" >&2
fi
"$ENV_DIR/bin/python" "$SCRIPT_DIR/check_environment.py" >&2
printf '%s\n' "$ENV_DIR/bin/python"
