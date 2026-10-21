"""Assemble the Hugging Face Space from studio/ and space/, and optionally push it.

    python space/build.py --out /tmp/falcon-space                      # assemble only
    python space/build.py --push JusperLee/Falcon-Studio [--private]   # assemble and upload

The Space reuses Falcon Studio's runtime, model code, configs and web modules unchanged.
space/ adds the Gradio/ZeroGPU server (app.py, clip.py), the Space card and a forked
index.html/app.js; space/web/i18n-space.js is merged into Studio's i18n.js.
Pushing replaces the Space's files with this build and requests ZeroGPU hardware.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile

SPACE = Path(__file__).resolve().parent
REPO = SPACE.parent
STUDIO = REPO / "studio"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", "*.test.cjs")
I18N_IMPORT = "import audioMessages from './i18n-audio.js';\n"
I18N_SPREAD = "...audioMessages };"


def assemble(out: Path) -> Path:
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(STUDIO / "falcon_runtime.py", out)
    shutil.copytree(STUDIO / "falcon_model", out / "falcon_model", ignore=IGNORE)
    shutil.copytree(STUDIO / "configs", out / "configs", ignore=IGNORE)
    shutil.copytree(STUDIO / "web", out / "web", ignore=IGNORE)
    for name in ("app.py", "clip.py", "requirements.txt", "README.md", "README_zh.md"):
        shutil.copy2(SPACE / name, out)
    shutil.copy2(REPO / "LICENSE", out)
    for item in (SPACE / "web").iterdir():
        shutil.copy2(item, out / "web" / item.name)
    i18n = out / "web" / "i18n.js"
    text = i18n.read_text()
    if text.count(I18N_IMPORT) != 1 or text.count(I18N_SPREAD) != 1:
        raise SystemExit("studio/web/i18n.js changed shape; update I18N_IMPORT/I18N_SPREAD in space/build.py")
    text = text.replace(I18N_IMPORT, I18N_IMPORT + "import spaceMessages from './i18n-space.js';\n")
    i18n.write_text(text.replace(I18N_SPREAD, "...audioMessages, ...spaceMessages };"))
    return out


def push(build: Path, repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=build, delete_patterns="*",
                      commit_message="Update Falcon Studio Space")
    runtime = api.get_space_runtime(repo_id)
    if runtime.requested_hardware != "zero-a10g":
        api.request_space_hardware(repo_id, "zero-a10g")
    print(f"https://huggingface.co/spaces/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="assemble into this (new or empty) directory")
    parser.add_argument("--push", metavar="REPO_ID", help="upload to this Space, e.g. JusperLee/Falcon-Studio")
    parser.add_argument("--private", action="store_true", help="create the Space as private")
    args = parser.parse_args()
    if not args.out and not args.push:
        parser.error("give --out and/or --push")
    if args.out:
        assemble(args.out)
        print(f"Assembled {args.out}")
        if args.push:
            push(args.out, args.push, args.private)
    elif args.push:
        with tempfile.TemporaryDirectory() as tmp:
            push(assemble(Path(tmp) / "space"), args.push, args.private)


if __name__ == "__main__":
    main()
