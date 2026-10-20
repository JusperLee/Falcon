#!/usr/bin/env python3
"""Verify supported package versions and their native imports without pip."""
from __future__ import annotations

import argparse
import importlib
from importlib import metadata
from pathlib import Path
import re
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    errors: list[str] = []
    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        errors.append("请使用 Python 3.10、3.11 或 3.12；建议 Python 3.11。")
    requirements = Path(__file__).resolve().parent.parent / "requirements.txt"
    for line in requirements.read_text(encoding="utf-8").splitlines():
        requirement = line.split("#", 1)[0].strip()
        if not requirement:
            continue
        package_spec, bounds = requirement.split(">=", 1)
        package = package_spec.split("[", 1)[0]
        minimum, maximum = bounds.split(",<", 1)
        try:
            actual = metadata.version(package)
            # Requirements intentionally use only stable numeric release bounds.
            def parts(version: str) -> tuple[int, ...]:
                match = re.match(r"^(\d+(?:\.\d+)*)", version)
                if not match:
                    raise ValueError(f"无法识别版本：{version}")
                return tuple(int(part) for part in match.group(1).split("."))
            if not parts(minimum) <= parts(actual) < parts(maximum):
                errors.append(f"{package}: 已安装 {actual}，应用需要 >={minimum},<{maximum}")
        except metadata.PackageNotFoundError:
            errors.append(f"缺少 {package}")
    for module in ("torch", "numpy", "yaml", "einops", "fastapi", "uvicorn", "PIL", "websockets"):
        try:
            importlib.import_module(module)
        except Exception as exc:
            errors.append(f"{module} 导入失败：{exc}")
    if not args.quiet:
        if errors:
            print("\n".join(errors), file=sys.stderr)
        else:
            print(f"依赖检查通过：{sys.executable}")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
