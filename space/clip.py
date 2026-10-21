"""Whole-clip Falcon separation on the live Studio's window schedule.

The browser records one clip: 16 kHz audio plus the 96×96 mouth crops captured during it,
each with its capture time. After recording stops the server cuts the clip into the
2 s windows with a 0.5 s hop that Falcon Studio sends live: zero history before the clip,
zero padding and the last crop held after it, and for each video slot the crop nearest
to it in time, like Studio's frameWindow. It keeps the same 0.5 s of each prediction
80 ms before the window end, crossfades neighbours over 20 ms (Studio's OutputSlicer) and
runs all windows in batches. The output equals the live Studio when no window is skipped;
where Studio would skip a window for stale video, the clip holds the nearest crop instead.
Pure NumPy apart from separate().
"""
from __future__ import annotations

import base64
import binascii
import io

import numpy as np
from PIL import Image, UnidentifiedImageError

SR, FPS = 16_000, 25
WINDOW, HOP, GUARD, FADE = 32_000, 8_000, 1_280, 320  # 2 s, 0.5 s, 80 ms, 20 ms
FRAME = SR // FPS  # 640 samples per video frame
WINDOW_FRAMES = WINDOW // FRAME
LEFT = WINDOW - HOP - GUARD  # zero history before the first output chunk
MAX_SECONDS = 60
MAX_FRAMES = 2 * FPS * MAX_SECONDS
MAX_FRAME_BYTES = 40_000


def plan(samples: int) -> int:
    """Number of windows: one per 0.5 s output chunk, the last one padded past the end."""
    return -(-samples // HOP)


def window_audio(audio: np.ndarray, w: int) -> np.ndarray:
    """Window w covers clip samples [w*HOP - LEFT, w*HOP + HOP + GUARD); zeros outside the clip."""
    start = w * HOP - LEFT
    out = np.zeros(WINDOW, np.float32)
    lo, hi = max(0, start), min(len(audio), start + WINDOW)
    if hi > lo:
        out[lo - start:hi - start] = audio[lo:hi]
    return out


def window_frames(times: np.ndarray, w: int) -> np.ndarray:
    """Crop index for each of window w's 50 video slots; times: sorted capture times (>= 0) in clip samples.

    As Studio's frameWindow: the crop nearest to each slot centre, ties to the earlier one.
    Before the clip that is the first crop (Studio's synthetic history), after the last
    crop the last one (Studio's drain).
    """
    centres = w * HOP - LEFT + (np.arange(WINDOW_FRAMES) + 0.5) * FRAME
    after = np.minimum(np.searchsorted(times, centres, side="left"), len(times) - 1)
    before = np.maximum(after - 1, 0)
    return np.where(np.abs(centres - times[before]) <= np.abs(times[after] - centres), before, after)


def stitch(predictions: np.ndarray, samples: int) -> tuple[np.ndarray, int]:
    """Studio's OutputSlicer over consecutive windows, trimmed to the clip length."""
    start, stop = WINDOW - GUARD - HOP, WINDOW - GUARD
    out = np.empty(len(predictions) * HOP, np.float32)
    ramp = np.linspace(0, 1, FADE, dtype=np.float32)
    future = None
    for w, prediction in enumerate(predictions):
        chunk = prediction[start:stop].astype(np.float32, copy=True)
        if future is not None:
            chunk[:FADE] = future * (1 - ramp) + chunk[:FADE] * ramp
        future = prediction[stop:stop + FADE].astype(np.float32, copy=True)
        out[w * HOP:(w + 1) * HOP] = chunk
    out = out[:samples]
    clipped = int(np.count_nonzero(np.abs(out) > 1))
    return np.clip(out, -1, 1).astype(np.float32), clipped


def normalize_mouth(crops: np.ndarray) -> np.ndarray:
    """falcon_runtime.preprocess_mouth for many crops: CenterCrop 88, /255, (x - .421) / .165."""
    x = crops[:, 4:92, 4:92].astype(np.float32) / 255.0
    return np.ascontiguousarray((x - 0.421) / 0.165)


def separate(audio_model, video_model, audio: np.ndarray, crops: np.ndarray, times: np.ndarray,
             device: str = "cpu", batch: int = 32) -> tuple[np.ndarray, int, int]:
    """Separate a whole clip. Returns (target, windows, clipped samples)."""
    import torch

    windows = plan(len(audio))
    mouth = normalize_mouth(crops)
    predictions = np.empty((windows, WINDOW), np.float32)
    with torch.inference_mode():
        for first in range(0, windows, batch):
            ids = range(first, min(windows, first + batch))
            wave = np.stack([window_audio(audio, w) for w in ids])
            lips = np.stack([mouth[window_frames(times, w)] for w in ids])[:, None]
            embedding = video_model(torch.from_numpy(lips).to(device))
            output = audio_model(torch.from_numpy(wave).to(device), embedding)
            predictions[first:first + len(ids)] = output[:, 0].float().cpu().numpy()
    if not np.isfinite(predictions).all():
        raise RuntimeError("Falcon produced non-finite output.")
    target, clipped = stitch(predictions, len(audio))
    return target, windows, clipped


def decode_request(data, variants) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """Validate one clip: {variant, audio: base64 float32 LE, frames: [base64 JPEG/PNG],
    times: [capture time of each crop in clip samples, non-decreasing]}."""
    if not isinstance(data, dict):
        raise ValueError("请求必须是 JSON 对象。")
    variant = data.get("variant", "small")
    if variant not in variants:
        raise ValueError("未知模型版本。")
    audio64, frames64, times = data.get("audio"), data.get("frames"), data.get("times")
    if not isinstance(audio64, str) or len(audio64) > 4 * (MAX_SECONDS * SR * 4 + 2) // 3:
        raise ValueError("录音需要 1–{seconds} 秒的同步音视频。".replace("{seconds}", str(MAX_SECONDS)))
    try:
        raw = base64.b64decode(audio64, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("音频编码无效。") from exc
    audio = np.frombuffer(raw[:len(raw) // 4 * 4], dtype="<f4").astype(np.float32)
    if len(raw) % 4 or not SR <= len(audio) <= MAX_SECONDS * SR:
        raise ValueError("录音需要 1–{seconds} 秒的同步音视频。".replace("{seconds}", str(MAX_SECONDS)))
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) > 4:
        raise ValueError("音频数据无效或超出允许范围。")
    if not isinstance(frames64, list) or not isinstance(times, list) or len(frames64) != len(times) \
            or not len(audio) // (4 * FRAME) <= len(frames64) <= MAX_FRAMES or not frames64:
        raise ValueError("嘴部画面数量与录音时长不符。")
    if not all(isinstance(t, (int, float)) and not isinstance(t, bool) for t in times):
        raise ValueError("嘴部画面时间戳无效。")
    stamps = np.asarray(times, dtype=np.float64)
    if not np.isfinite(stamps).all() or np.any(np.diff(stamps) < 0) \
            or stamps[0] < 0 or stamps[-1] > len(audio) + SR:
        raise ValueError("嘴部画面时间戳无效。")
    decoded: dict[str, np.ndarray] = {}  # crops held while the face is lost repeat one image
    crops = np.empty((len(frames64), 96, 96), np.uint8)
    for i, frame64 in enumerate(frames64):
        if not isinstance(frame64, str) or len(frame64) > MAX_FRAME_BYTES:
            raise ValueError("嘴部图像大小无效。")
        if frame64 not in decoded:
            try:
                with Image.open(io.BytesIO(base64.b64decode(frame64, validate=True))) as im:
                    if im.size != (96, 96) or im.format not in ("JPEG", "PNG"):
                        raise ValueError("嘴部图像必须为 96×96 的 JPEG 或 PNG。")
                    decoded[frame64] = np.asarray(im.convert("L"), dtype=np.uint8)
            except (UnidentifiedImageError, OSError, binascii.Error) as exc:
                raise ValueError("无法读取嘴部图像。") from exc
        crops[i] = decoded[frame64]
    return variant, audio, crops, stamps
