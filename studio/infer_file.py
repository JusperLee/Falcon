"""Offline Falcon inference on one clip: mixture WAV + mouth crops -> extracted target WAV.

Usage:
    python infer_file.py --audio mix.wav --mouth mouth.npy --out target.wav [--model JusperLee/Falcon-Big] [--device auto|cuda|mps|cpu]

Weights are downloaded from the model's Hugging Face short link on first use and cached.

Audio must be mono 16 kHz WAV (16-bit PCM or 32-bit float), e.g.
    ffmpeg -i input.mp4 -ac 1 -ar 16000 mix.wav
Mouth crops are grayscale, 25 fps, shape (T, H, W) with H, W >= 88 (96x96 as produced
by Falcon Studio and the training data), stored as .npy, .npz (first array) or .pt.
The whole clip is processed in one pass, matching the evaluation protocol.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import wave

import numpy as np
import torch

from falcon_runtime import SAMPLE_RATE, VIDEO_FPS, FalconRuntime, variant_for

CROP = 88


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as f:
        rate, channels, width, frames = f.getframerate(), f.getnchannels(), f.getsampwidth(), f.readframes(f.getnframes())
    if rate != SAMPLE_RATE or channels != 1:
        raise SystemExit(f"{path}: need mono {SAMPLE_RATE} Hz, got {channels} ch {rate} Hz. "
                         f"Convert with: ffmpeg -i in -ac 1 -ar {SAMPLE_RATE} out.wav")
    if width == 2:
        return np.frombuffer(frames, "<i2").astype(np.float32) / 32768.0
    if width == 4:
        return np.frombuffer(frames, "<f4").astype(np.float32)
    raise SystemExit(f"{path}: unsupported sample width {width * 8} bit; use 16-bit PCM or 32-bit float.")


def write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).round().astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(SAMPLE_RATE); f.writeframes(pcm.tobytes())


def read_mouth(path: Path) -> np.ndarray:
    if path.suffix == ".pt":
        data = torch.load(path, map_location="cpu", weights_only=True)
        data = data.numpy() if isinstance(data, torch.Tensor) else np.asarray(data)
    elif path.suffix == ".npz":
        with np.load(path) as archive:
            data = archive[archive.files[0]]
    else:
        data = np.load(path)
    data = np.squeeze(np.asarray(data, dtype=np.float32))
    if data.ndim != 3 or data.shape[1] < CROP or data.shape[2] < CROP:
        raise SystemExit(f"{path}: expected grayscale mouth crops (T, H, W) with H, W >= {CROP}; got {data.shape}.")
    return data


def preprocess(mouth: np.ndarray) -> torch.Tensor:
    """Validation pipeline of the training code: /255, center crop 88x88, normalize 0.421/0.165."""
    top, left = (mouth.shape[1] - CROP) // 2, (mouth.shape[2] - CROP) // 2
    x = mouth[:, top:top + CROP, left:left + CROP] / 255.0
    return torch.from_numpy(np.ascontiguousarray((x - 0.421) / 0.165, dtype=np.float32))[None, None]


def extract(runtime: FalconRuntime, audio: np.ndarray, mouth: np.ndarray) -> np.ndarray:
    frames = min(len(mouth), int(len(audio) * VIDEO_FPS // SAMPLE_RATE))
    if frames < VIDEO_FPS:
        raise SystemExit("Need at least 1 s of synchronized audio and mouth frames.")
    samples = frames * SAMPLE_RATE // VIDEO_FPS
    with runtime._lock, torch.inference_mode():
        if not runtime.ready:
            raise SystemExit(runtime.error or "Falcon model failed to load.")
        waveform = torch.from_numpy(np.ascontiguousarray(audio[:samples])).unsqueeze(0).to(runtime.device)
        embedding = runtime.video_model(preprocess(mouth[:frames]).to(runtime.device))
        output = runtime.audio_model(waveform, embedding)[0, 0].float().cpu().numpy()
    if not np.isfinite(output).all():
        raise SystemExit("Falcon produced non-finite output.")
    return output[:samples]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--mouth", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="JusperLee/Falcon", help="JusperLee/Falcon (default) or JusperLee/Falcon-Big")
    parser.add_argument("--checkpoint", default=None, help="local weights file or hf://<org>/<repo>/<file> instead of the model's own")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    audio, mouth = read_wav(args.audio), read_mouth(args.mouth)
    trimmed = abs(len(audio) / SAMPLE_RATE - len(mouth) / VIDEO_FPS)
    if trimmed > 1 / VIDEO_FPS:
        print(f"Note: audio and mouth durations differ by {trimmed:.2f} s; the longer one is trimmed.")
    runtime = FalconRuntime(variant_for(args.model), args.checkpoint).load(args.device)
    output = extract(runtime, audio, mouth)
    write_wav(args.out, output)
    print(f"Wrote {args.out} ({len(output) / SAMPLE_RATE:.2f} s, {runtime.variant}, {runtime.device}).")


if __name__ == "__main__":
    main()
