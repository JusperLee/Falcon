"""Loopback-only Falcon Studio server. No audio/video is written to disk."""
from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import io
import json
import logging
import os
import socket
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from starlette.middleware.trustedhost import TrustedHostMiddleware

ROOT = Path(__file__).resolve().parent
SR, FPS, WINDOW, HOP = 16000, 25, 32000, 8000
GUARD, FADE = 1280, 320  # 80 ms right context, 20 ms prediction crossfade
MAX_MESSAGE_BYTES = 2_000_000
MAX_SAFE_INTEGER = 2**53 - 1  # JavaScript's exact integer timeline limit
log = logging.getLogger("falcon")


def decode_request(data: dict) -> tuple[np.ndarray, np.ndarray, int]:
    if data.get("type") != "infer":
        raise ValueError("未知请求类型。")
    if data.get("window_seconds", 2) != 2 or data.get("hop_seconds", 0.5) != 0.5:
        raise ValueError("当前模型使用 2 秒窗口、0.5 秒步长。")
    if not isinstance(data.get("reset", False), bool):
        raise ValueError("重置标记必须为布尔值。")
    audio64, frames64 = data.get("audio"), data.get("frames")
    if not isinstance(audio64, str) or len(audio64) != 4 * ((WINDOW * 4 + 2) // 3):
        raise ValueError("音频必须是 32000 个 16 kHz float32 采样。")
    try:
        audio = np.frombuffer(base64.b64decode(audio64, validate=True), dtype="<f4").copy()
    except (ValueError, binascii.Error) as exc:
        raise ValueError("音频编码无效。") from exc
    if audio.shape != (WINDOW,) or not np.isfinite(audio).all() or np.max(np.abs(audio)) > 4:
        raise ValueError("音频数据无效或超出允许范围。")
    if not isinstance(frames64, list) or len(frames64) != 50:
        raise ValueError("每个窗口需要 50 帧同步嘴部图像。")
    frames = []
    for frame64 in frames64:
        if not isinstance(frame64, str) or len(frame64) > 40000:
            raise ValueError("嘴部图像大小无效。")
        try:
            with Image.open(io.BytesIO(base64.b64decode(frame64, validate=True))) as im:
                if im.size != (96, 96) or im.format not in ("JPEG", "PNG"):
                    raise ValueError("嘴部图像必须为 96×96 的 JPEG 或 PNG。")
                frames.append(np.asarray(im.convert("L"), dtype=np.uint8))
        except (UnidentifiedImageError, OSError, binascii.Error) as exc:
            raise ValueError("无法读取嘴部图像。") from exc
    endpoint = data.get("end_sample", WINDOW)
    if isinstance(endpoint, bool) or not isinstance(endpoint, int) or not WINDOW <= endpoint <= MAX_SAFE_INTEGER:
        raise ValueError("音频时间戳无效。")
    return audio, np.stack(frames), endpoint


class OutputSlicer:
    """Crossfade aligned alternate predictions, never duplicate emitted samples."""
    def __init__(self):
        self.future = None
        self.last_endpoint = None

    def take(self, prediction: np.ndarray, endpoint: int, reset: bool = False):
        pred = np.asarray(prediction, dtype=np.float32).reshape(-1)
        if pred.size != WINDOW or not np.isfinite(pred).all():
            raise ValueError("模型输出长度错误或包含非有限数值。")
        start, stop = WINDOW - GUARD - HOP, WINDOW - GUARD
        chunk = pred[start:stop].copy()
        contiguous = not reset and self.last_endpoint is not None and endpoint - self.last_endpoint == HOP
        if contiguous and self.future is not None:
            ramp = np.linspace(0, 1, FADE, dtype=np.float32)
            chunk[:FADE] = self.future * (1 - ramp) + chunk[:FADE] * ramp
        self.future = pred[stop:stop + FADE].copy()
        self.last_endpoint = endpoint
        clipped = int(np.count_nonzero(np.abs(chunk) > 1))
        return np.clip(chunk, -1, 1).astype("<f4"), contiguous, clipped


class Engine:
    def __init__(self, device="auto", variant="small"):
        self.device = device
        self.variant = variant
        self.active_connections = 0
        self.runtime = None
        self.loading = False
        self.error = None
        self.busy = False
        self.load_lock = threading.Lock()
        self.infer_lock = threading.Lock()
        self.load_thread = None

    def status(self):
        from falcon_runtime import VARIANTS
        status = {"ready": False, "loading": self.loading, "error": self.error,
                  "device": self.device, "model": f"Falcon · {self.variant.title()}", "checkpoint": VARIANTS[self.variant]["weights"],
                  "variant": self.variant, "variants": self.variants(),
                  "sample_rate": SR, "video_fps": FPS, "window_seconds": 2,
                  "hop_seconds": .5, "output_delay_ms": 80, "busy": self.busy,
                  "mode": "sliding-window", "app": "falcon-studio"}
        if self.runtime is not None:
            status.update(self.runtime.status())
            status.update(loading=self.loading, error=self.error, busy=self.busy, variant=self.variant, ready=not self.loading and bool(status["ready"]))
        return status

    @staticmethod
    def variants():
        # "huggingface": not in models/ yet, downloaded from the variant's repo on first load.
        from falcon_runtime import VARIANTS, weights_source
        return [{"id": name, "name": f"Falcon {name.title()}", "available": weights_source(name) is not None,
                 "source": weights_source(name), "checkpoint": info["weights"], "repo": info["repo"]}
                for name, info in VARIANTS.items()]

    def start_load(self, variant=None):
        variant = variant or self.variant
        if variant not in ("small", "big"):
            raise ValueError("未知模型版本。")
        with self.load_lock:
            if self.loading:
                if variant != self.variant:
                    raise RuntimeError("模型加载中，请等待完成后再切换。")
                return
            if variant == self.variant and self.runtime is not None:
                return
            if self.active_connections:
                raise RuntimeError("请先停止所有分离会话，再切换模型。")
            selected = next(v for v in self.variants() if v["id"] == variant)
            if not selected["available"]:
                raise ValueError(f"尚未安装 {selected['name']} 权重：models/{selected['checkpoint']}")
            self.variant = variant
            self.runtime = None
            self.loading, self.error = True, None
            self.load_thread = threading.Thread(target=self._load, daemon=True, name="falcon-load")
            self.load_thread.start()

    def _load(self):
        try:
            from falcon_runtime import FalconRuntime
            runtime = FalconRuntime(variant=self.variant)
            runtime.load(device=self.device)
            self.runtime = runtime
        except Exception as exc:
            log.exception("Falcon initialization failed")
            self.error = str(exc)
        finally:
            self.loading = False

    def infer(self, audio, frames):
        if self.runtime is None:
            raise RuntimeError(self.error or "模型仍在加载，请稍后重试。")
        if not self.infer_lock.acquire(blocking=False):
            raise RuntimeError("另一个窗口正在使用模型，请关闭重复会话。")
        self.busy = True
        try:
            started = time.perf_counter()
            prediction = self.runtime.infer(audio, frames)
            return prediction, (time.perf_counter() - started) * 1000
        finally:
            self.busy = False
            self.infer_lock.release()


def same_origin(origin: str | None, host: str | None) -> bool:
    if not origin:  # CLI/testing clients do not have Origin
        return True
    parsed = urlparse(origin)
    return parsed.scheme in ("http", "https") and parsed.netloc == host and parsed.hostname in ("localhost", "127.0.0.1", "::1")


def create_app(engine=None, autoload=True):
    engine = engine or Engine(os.getenv("FALCON_DEVICE", "auto"))

    @asynccontextmanager
    async def lifespan(app):
        if autoload:
            try:
                engine.start_load()
            except (ValueError, RuntimeError) as exc:
                engine.error = str(exc)
        yield

    app = FastAPI(title="Falcon Studio", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.engine = engine
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])

    @app.middleware("http")
    async def local_headers(request: Request, call_next):
        if not same_origin(request.headers.get("origin"), request.headers.get("host")):
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Only same-origin local requests are allowed"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(self), microphone=(self)"
        response.headers["Cache-Control"] = "no-store"
        # MediaPipe runs local WebAssembly in the video worker. JavaScript eval
        # and remote scripts remain disallowed.
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self' ws://127.0.0.1:* ws://localhost:*; worker-src 'self' blob:; frame-ancestors 'none'"
        return response

    @app.get("/api/status")
    async def status():
        return engine.status()

    @app.post("/api/load")
    async def load(request: Request):
        raw = await request.body()
        if len(raw) > 1024:
            raise HTTPException(status_code=413, detail="加载请求过大。")
        try:
            payload = json.loads(raw) if raw else {}
            if not isinstance(payload, dict):
                raise ValueError("加载请求必须为 JSON 对象。")
            engine.start_load(payload.get("variant"))
        except (ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return engine.status()

    @app.websocket("/ws")
    async def websocket(ws: WebSocket):
        if not same_origin(ws.headers.get("origin"), ws.headers.get("host")):
            await ws.close(code=1008)
            return
        await ws.accept()
        engine.active_connections = getattr(engine, "active_connections", 0) + 1
        slicer = OutputSlicer()
        try:
            while True:
                raw = await ws.receive_text()
                if len(raw) > MAX_MESSAGE_BYTES:
                    await ws.close(code=1009)
                    return
                request_id = None
                try:
                    data = json.loads(raw)
                    if not isinstance(data, dict):
                        raise ValueError("请求必须是 JSON 对象。")
                    request_id = data.get("id")
                    if isinstance(request_id, bool) or not isinstance(request_id, (str, int)) or len(str(request_id)) > 80:
                        raise ValueError("请求 ID 无效。")
                    if isinstance(request_id, int) and abs(request_id) > MAX_SAFE_INTEGER:
                        raise ValueError("请求 ID 超出精确整数范围。")
                    audio, frames, endpoint = await asyncio.to_thread(decode_request, data)
                    if engine.busy:
                        raise RuntimeError("模型忙，请跳过此窗口后重试。")
                    prediction, inference_ms = await asyncio.to_thread(engine.infer, audio, frames)
                    output, contiguous, clipped = slicer.take(prediction, endpoint, data.get("reset", False))
                    await ws.send_json({"type": "result", "id": request_id,
                        "audio": base64.b64encode(output.tobytes()).decode("ascii"), "sample_rate": SR,
                        "inference_ms": round(inference_ms, 1), "rtf": round(inference_ms / 500, 3),
                        "window_rtf": round(inference_ms / 2000, 3), "device": engine.status().get("device"),
                        "start_sample": endpoint - GUARD - HOP, "end_sample": endpoint - GUARD,
                        "output_delay_ms": 80, "contiguous": contiguous, "clipped_samples": clipped})
                except (ValueError, RuntimeError) as exc:
                    await ws.send_json({"type": "error", "id": request_id, "message": str(exc)})
                except Exception:
                    log.exception("Inference failed")
                    await ws.send_json({"type": "error", "id": request_id, "message": "推理失败，请查看启动终端中的错误信息。"})
        except WebSocketDisconnect:
            pass
        finally:
            engine.active_connections -= 1

    @app.get("/")
    async def index():
        return FileResponse(ROOT / "web" / "index.html")

    app.mount("/static", StaticFiles(directory=ROOT / "web", check_dir=False), name="static")
    app.mount("/", StaticFiles(directory=ROOT / "web", check_dir=False), name="web")
    return app


app = create_app()


def main():
    parser = argparse.ArgumentParser(description="Falcon Studio · local audio-visual extraction")
    parser.add_argument("--host", choices=["127.0.0.1", "localhost"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                        help="auto prefers CUDA, then Apple Metal (mps), then the CPU")
    parser.add_argument("--model", default="JusperLee/Falcon", help="JusperLee/Falcon (default) or JusperLee/Falcon-Big")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    from falcon_runtime import variant_for
    try:
        variant = variant_for(args.model)
    except ValueError as exc:
        parser.error(str(exc))
    if args.port is None:
        for candidate in range(7860, 7900):
            with socket.socket() as sock:
                try:
                    sock.bind((args.host, candidate))
                except OSError:
                    continue
                args.port = candidate
                break
        else:
            parser.error("No free local port between 7860 and 7899")
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    import uvicorn
    url = f"http://{args.host}:{args.port}"
    print(f"\n  FALCON STUDIO   {url}\n  按 Control-C 退出。摄像头与麦克风只在点击开启后使用。\n", flush=True)
    if not args.no_browser:
        def open_when_ready():
            import urllib.request
            for _ in range(60):
                try:
                    with urllib.request.urlopen(url + "/api/status", timeout=1) as response:
                        if json.load(response).get("app") == "falcon-studio":
                            webbrowser.open(url)
                            return
                except Exception:
                    time.sleep(.5)
        threading.Thread(target=open_when_ready, daemon=True).start()
    uvicorn.run(create_app(Engine(args.device, variant)), host=args.host, port=args.port,
                ws_max_size=MAX_MESSAGE_BYTES, ws_max_queue=1, log_level="info")


if __name__ == "__main__":
    main()
