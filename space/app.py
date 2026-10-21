"""Falcon Studio on a Hugging Face ZeroGPU Space.

ZeroGPU attaches a GPU per call, so a 0.5 s live loop does not fit. The browser records
a clip (camera, microphone and the same in-browser mouth tracking as Falcon Studio) and
sends it once recording stops; the clip is separated on the GPU on the live Studio's
window schedule (clip.py). Nothing is written to disk.
"""
from __future__ import annotations

import spaces  # noqa: F401 -- must be imported before torch on ZeroGPU

import base64
import logging
import os
from pathlib import Path
import time

import gradio as gr
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import clip
from falcon_runtime import VARIANTS, FalconRuntime

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
ZERO_GPU = os.environ.get("SPACES_ZERO_GPU", "").lower() in ("1", "true")
log = logging.getLogger("falcon-space")


def load_models() -> tuple[dict, str]:
    """Both variants, warmed up on the CPU, then placed on CUDA at import time as ZeroGPU expects."""
    device = "cuda" if ZERO_GPU else os.environ.get("FALCON_DEVICE", "cpu")
    runtimes = {}
    for variant in VARIANTS:
        runtime = FalconRuntime(variant).load("cpu")
        for model in (runtime.audio_model, runtime.video_model):
            model.to(device)
        runtimes[variant] = runtime
    return runtimes, device


RUNTIMES, DEVICE = load_models()


def gpu_seconds(variant, audio, crops, times):
    # Requested time must be left in the visitor's quota (x1.5 on RTX PRO 6000); actual use is
    # what counts. Measured on ZeroGPU: 8 s Small 0.2 s warm / 1.2 s cold, 60 s Big 2.5 s.
    return min(30, 10 + clip.plan(len(audio)) // 8)


@spaces.GPU(duration=gpu_seconds)
def separate_on_gpu(variant: str, audio, crops, times):
    runtime = RUNTIMES[variant]
    started = time.perf_counter()
    target, windows, clipped = clip.separate(runtime.audio_model, runtime.video_model, audio, crops, times, DEVICE)
    return target, windows, clipped, (time.perf_counter() - started) * 1000


server = gr.Server(docs_url=None, redoc_url=None, openapi_url=None)


@server.api(name="separate", concurrency_limit=2 if ZERO_GPU else 1)
def separate(request: dict) -> dict:
    try:
        variant, audio, crops, times = clip.decode_request(request, RUNTIMES)
    except ValueError as exc:
        raise gr.Error(str(exc), print_exception=False) from exc
    try:
        target, windows, clipped, gpu_ms = separate_on_gpu(variant, audio, crops, times)
    except gr.Error:
        raise  # ZeroGPU quota and queue messages are meant for the visitor
    except Exception as exc:
        log.exception("Separation failed")
        raise gr.Error("分离失败，请稍后重试。", print_exception=False) from exc
    seconds = len(audio) / clip.SR
    return {"audio": base64.b64encode(target.astype("<f4").tobytes()).decode("ascii"),
            "sample_rate": clip.SR, "windows": windows, "gpu_ms": round(gpu_ms, 1),
            "rtf": round(gpu_ms / 1000 / seconds, 4), "device": "zerogpu" if ZERO_GPU else DEVICE,
            "clipped_samples": clipped, "variant": variant}


@server.get("/api/status")
def status():
    return {"app": "falcon-studio-space", "ready": True, "mode": "clip", "device": "zerogpu" if ZERO_GPU else DEVICE,
            "max_seconds": clip.MAX_SECONDS, "sample_rate": clip.SR, "video_fps": clip.FPS,
            "window_seconds": clip.WINDOW / clip.SR, "hop_seconds": clip.HOP / clip.SR,
            "variants": [{"id": name, "name": f"Falcon {name.title()}", "available": True, "repo": info["repo"],
                          "parameters": RUNTIMES[name].parameter_count} for name, info in VARIANTS.items()]}


@server.get("/")
def index():
    return FileResponse(WEB / "index.html")


server.mount("/static", StaticFiles(directory=WEB), name="static")

CSP = ("default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; "
       "img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; worker-src 'self' blob:; "
       "frame-ancestors 'self' https://huggingface.co")


class SecurityHeaders:
    """Studio's headers, except that huggingface.co may embed the page (the Space iframe)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"cache-control"]
                cache = "public, max-age=86400" if path.startswith("/static/vendor/") else "no-cache"
                headers += [(b"x-content-type-options", b"nosniff"), (b"referrer-policy", b"no-referrer"),
                            (b"permissions-policy", b"camera=(self), microphone=(self)"),
                            (b"content-security-policy", CSP.encode()), (b"cache-control", cache.encode())]
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


server.add_middleware(SecurityHeaders)

if __name__ == "__main__":
    # On Spaces GRADIO_SERVER_NAME/PORT come from the environment; locally this is 127.0.0.1:7860.
    server.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"), show_error=True,
                  footer_links=[], run_history=False, ssr_mode=False, mcp_server=False)
