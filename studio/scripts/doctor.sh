#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
APP_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$("$SCRIPT_DIR/bootstrap.sh")"
"$PYTHON_BIN" "$SCRIPT_DIR/check_environment.py"
"$PYTHON_BIN" - "$APP_DIR" <<'PY'
from pathlib import Path
import platform
import sys
import torch

app = Path(sys.argv[1])
sys.path.insert(0, str(app))
from falcon_runtime import VARIANTS, weights_source
print(f"系统：{platform.platform()} / {platform.machine()}")
print(f"PyTorch：{torch.__version__}")
print(f"MPS 已构建：{torch.backends.mps.is_built()}")
print(f"MPS 当前可用：{torch.backends.mps.is_available()}")
print(f"CUDA 可用：{torch.cuda.is_available()}" + (f"（{torch.cuda.get_device_name(0)}）" if torch.cuda.is_available() else ""))
for variant, info in VARIANTS.items():
    source = weights_source(variant)
    if source == "cached":
        print(f"模型 {info['repo']}：已下载（Hugging Face 缓存）")
    elif source == "huggingface":
        print(f"模型 {info['repo']}：首次加载时从 Hugging Face 下载")
    else:
        print(f"模型 {info['repo']}：未安装 huggingface_hub，无法下载", file=sys.stderr)
        if variant == "small":
            raise SystemExit(1)
print("这里只检查环境和模型文件；实际加载与分离结果请在应用中确认。")
PY
