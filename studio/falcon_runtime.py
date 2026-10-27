"""Strict checkpoint-backed Falcon inference for synchronized two-second windows.

This is windowed inference, not a stateful streaming architecture. The checkpoint
expects mono 16 kHz audio and 25 fps, aligned grayscale mouth crops of 96×96.
The browser/server is responsible for lip tracking and capture synchronization.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import threading
import time

import numpy as np
import torch
import yaml

from falcon_model.models.Falcon_Contrastive import FalconSpeakerContrastive
from falcon_model.models.Speaker_Contrastive_v2 import SpeakerContrastiveV2
from falcon_model.videomodels.frcnn_videomodel import FRCNNVideoModel


SAMPLE_RATE = 16_000
VIDEO_FPS = 25
WINDOW_SAMPLES = 32_000
WINDOW_FRAMES = 50
ROOT = Path(__file__).resolve().parent
# Weights come from these Hugging Face short links: downloaded on first use and cached in
# ~/.cache/huggingface. A local file or hf://<org>/<repo>/<file> can be passed instead.
VARIANTS = {
    "small": {"config": "Falcon.yml", "weights": "falcon.pt", "repo": "JusperLee/Falcon",
              "architecture": "FalconSpeakerContrastive"},
    "big": {"config": "FalconBig.yml", "weights": "falcon-big.pt", "repo": "JusperLee/Falcon-Big",
            "architecture": "SpeakerContrastiveV2"},
}
MODEL_CLASSES = {"FalconSpeakerContrastive": FalconSpeakerContrastive, "SpeakerContrastiveV2": SpeakerContrastiveV2}
# The Hub counts a model download when config.yaml is requested; requests for the .pt weights are not counted.
HUB_COUNTED_FILE = "config.yaml"


def variant_for(name: str) -> str:
    """Variant id for a short link ("JusperLee/Falcon", "JusperLee/Falcon-Big") or "small"/"big"."""
    for variant, info in VARIANTS.items():
        if name in (variant, info["repo"]):
            return variant
    raise ValueError(f"Unknown Falcon model {name!r}; use {' or '.join(v['repo'] for v in VARIANTS.values())}")


def weights_source(variant: str) -> str | None:
    """'cached' (already downloaded), 'huggingface' (downloads on first load) or None."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    cached = try_to_load_from_cache(VARIANTS[variant]["repo"], VARIANTS[variant]["weights"])
    return "cached" if isinstance(cached, str) else "huggingface"


def resolve_checkpoint(variant: str, checkpoint: str | Path | None = None) -> Path:
    """The variant's Hugging Face weights, or a local file / hf://<org>/<repo>/<file> override."""
    spec = str(checkpoint) if checkpoint else None
    if spec and not spec.startswith("hf://"):
        return Path(spec)
    if spec:
        repo_id, _, filename = spec[len("hf://"):].rpartition("/")
    else:
        repo_id, filename = VARIANTS[variant]["repo"], VARIANTS[variant]["weights"]
    if not repo_id or not filename:
        raise ValueError(f"Expected hf://<org>/<repo>/<file>, got {spec}")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise FileNotFoundError("huggingface_hub is not installed (pip install huggingface_hub); "
                                "or pass a local weights file.") from exc
    try:
        path = Path(hf_hub_download(repo_id=repo_id, filename=filename))
    except Exception as exc:
        raise FileNotFoundError(f"Could not download {filename} from Hugging Face repo {repo_id}: {exc}. "
                                "Set HF_ENDPOINT to a mirror, or pass a local weights file.") from exc
    count_download(repo_id)
    return path


def count_download(repo_id: str) -> threading.Thread:
    """HEAD the repo's config.yaml in the background so this load shows in the Hub's download stats.

    Best effort and never delays loading: offline, a mirror or a repo without the file only skips
    the count. HF_HUB_DISABLE_TELEMETRY=1 turns it off.
    """
    def head():
        try:
            from huggingface_hub import constants, get_hf_file_metadata, hf_hub_url
            if not constants.HF_HUB_DISABLE_TELEMETRY:
                get_hf_file_metadata(hf_hub_url(repo_id, HUB_COUNTED_FILE))
        except Exception:
            pass

    thread = threading.Thread(target=head, name="hub-download-count", daemon=True)
    thread.start()
    return thread


def pick_device(requested: str = "auto") -> str:
    """'auto' prefers CUDA, then Apple Metal (MPS), then the CPU."""
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be 'auto', 'cpu', 'cuda' or 'mps'")
    cuda, mps = torch.cuda.is_available(), torch.backends.mps.is_available()
    if requested == "cuda" and not cuda:
        raise RuntimeError("CUDA is unavailable in this Python process; install a CUDA build of PyTorch or choose CPU.")
    if requested == "mps" and not mps:
        raise RuntimeError("Apple Metal/MPS is unavailable in this Python process; choose CPU.")
    if requested == "auto":
        return "cuda" if cuda else "mps" if mps else "cpu"
    return requested


def preprocess_mouth(frames: np.ndarray) -> torch.Tensor:
    """Match get_preprocessing_pipelines()['val'] from the training repository."""
    if not isinstance(frames, np.ndarray) or frames.dtype != np.uint8:
        raise ValueError("Mouth frames must be a uint8 NumPy array.")
    if frames.shape != (WINDOW_FRAMES, 96, 96):
        raise ValueError(f"Expected {WINDOW_FRAMES} grayscale 96×96 mouth crops; got {frames.shape}.")
    # Validation CenterCrop(88, 88), Normalize(0, 255), Normalize(.421, .165).
    x = frames[:, 4:92, 4:92].astype(np.float32) / 255.0
    x = (x - 0.421) / 0.165
    return torch.from_numpy(np.ascontiguousarray(x))[None, None]


class FalconRuntime:
    """One model pair, serialized across requests; never returns the input as output."""

    def __init__(self, variant: str = "small", checkpoint_path: str | Path | None = None):
        if variant not in VARIANTS:
            raise ValueError("variant must be 'small' or 'big'")
        self.variant = variant
        self.config_path = ROOT / "configs" / VARIANTS[variant]["config"]
        self.checkpoint_spec = checkpoint_path
        # Resolved (downloaded if needed) in load(); until then only the name is reported.
        local = checkpoint_path and not str(checkpoint_path).startswith("hf://")
        self.checkpoint_path = Path(checkpoint_path if local else VARIANTS[variant]["weights"])
        self._lock = threading.RLock()
        self.audio_model = None
        self.video_model = None
        self.device = "unloaded"
        self.requested_device = "auto"
        self.ready = False
        self.error: str | None = None
        self.warning: str | None = None
        self.last_inference_ms: float | None = None
        self.warmup_ms: float | None = None
        self.audio_keys = 0
        self.video_keys = 0
        self.parameter_count = 0

    def load(self, device: str = "auto") -> "FalconRuntime":
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be 'auto', 'cpu', 'cuda' or 'mps'")
        with self._lock:
            self.ready = False
            self.error = None
            self.warning = None
            self.requested_device = device
            try:
                self.checkpoint_path = resolve_checkpoint(self.variant, self.checkpoint_spec)
                if not self.checkpoint_path.is_file():
                    raise FileNotFoundError(f"Falcon checkpoint not found: {self.checkpoint_path}")
                # All repository modules are loaded locally; there are no network downloads.
                config = yaml.safe_load(self.config_path.read_text())
                if "train_conf" in config:
                    config = config["train_conf"]
                audio_config = copy.deepcopy(config["audionet"]["audionet_config"])
                video_config = copy.deepcopy(config["videonet"]["videonet_config"])
                video_config["pretrain"] = None
                architecture = config["audionet"]["audionet_name"]
                if architecture != VARIANTS[self.variant]["architecture"]:
                    raise ValueError(f"Unexpected architecture for {self.variant}: {architecture}")
                if config["videonet"]["videonet_name"] != "FRCNNVideoModel":
                    raise ValueError(f"Unsupported video architecture: {config['videonet']['videonet_name']}")
                self.audio_model = MODEL_CLASSES[architecture](sample_rate=SAMPLE_RATE, **audio_config)
                self.video_model = FRCNNVideoModel(**video_config)
                # weights_only avoids executing arbitrary checkpoint Python objects.
                ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
                state = ckpt.get("state_dict", ckpt)
                audio_state = {k[len("audio_model."):]: v for k, v in state.items() if k.startswith("audio_model.")}
                video_state = {k[len("video_model."):]: v for k, v in state.items() if k.startswith("video_model.")}
                if not audio_state or not video_state:
                    raise ValueError("Checkpoint must contain both audio_model and video_model weights.")
                # Missing shared/repeats config keys can silently disable almost all layers.
                # Strict loading rejects that failure rather than silently using partial weights.
                self.audio_model.load_state_dict(audio_state, strict=True)
                self.video_model.load_state_dict(video_state, strict=True)
                self.audio_keys, self.video_keys = len(audio_state), len(video_state)
                self.parameter_count = sum(p.numel() for m in (self.audio_model, self.video_model) for p in m.parameters())
                del ckpt, state, audio_state, video_state

                # Small recurrent kernels generally slow down with excessive CPU threads.
                threads = max(1, min(8, int(os.environ.get("FALCON_CPU_THREADS", "4"))))
                torch.set_num_threads(threads)
                for model in (self.audio_model, self.video_model):
                    # The original video model's train() override does not return self.
                    model.eval()
                    model.requires_grad_(False)
                self.device = pick_device(device)

                # Warm up real model paths before reporting ready; in auto mode a GPU that
                # cannot run the model (driver, memory, unsupported ops) falls back to the CPU.
                started = time.perf_counter()
                try:
                    self._warmup()
                except Exception as exc:
                    if self.device == "cpu" or device != "auto":
                        raise
                    name = {"cuda": "CUDA", "mps": "Metal"}[self.device]
                    self.warning = f"{name} warmup failed; using CPU: {str(exc).splitlines()[0][:180]}"
                    self.device = "cpu"
                    self._warmup()
                self.warmup_ms = round((time.perf_counter() - started) * 1000, 1)
                self.ready = True
                return self
            except Exception as exc:
                self.error = str(exc)
                raise

    def _warmup(self) -> None:
        for model in (self.audio_model, self.video_model):
            model.to(self.device)
        self._forward(np.zeros(WINDOW_SAMPLES, dtype=np.float32), np.full((WINDOW_FRAMES, 96, 96), 107, np.uint8))

    @classmethod
    def from_pretrained(cls, name: str = "JusperLee/Falcon", device: str = "auto") -> "FalconRuntime":
        """Load by Hugging Face short link ("JusperLee/Falcon", "JusperLee/Falcon-Big") or "small"/"big".

        device: "auto" (CUDA, then Metal, then CPU), "cuda", "mps" or "cpu".
        """
        return cls(variant_for(name)).load(device)

    def status(self) -> dict:
        # Do not acquire the inference lock: status remains responsive during processing.
        return {
            "ready": self.ready,
            "variant": self.variant,
            "architecture": VARIANTS[self.variant]["architecture"],
            "device": self.device,
            "requested_device": self.requested_device,
            "checkpoint": self.checkpoint_path.name,
            "sample_rate": SAMPLE_RATE,
            "video_fps": VIDEO_FPS,
            "window_seconds": 2.0,
            "audio_weight_keys": self.audio_keys,
            "video_weight_keys": self.video_keys,
            "parameters": self.parameter_count,
            "last_inference_ms": self.last_inference_ms,
            "warmup_ms": self.warmup_ms,
            "error": self.error,
            "warning": self.warning,
            "mode": "windowed_audio_visual",
        }

    @torch.inference_mode()
    def _forward(self, audio: np.ndarray, frames: np.ndarray) -> np.ndarray:
        mouth = preprocess_mouth(frames).to(self.device)
        waveform = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0).to(self.device)
        mouth_embedding = self.video_model(mouth)
        separated = self.audio_model(waveform, mouth_embedding)
        output = separated[0, 0].float().cpu().numpy().copy()
        if output.shape != (WINDOW_SAMPLES,) or not np.isfinite(output).all():
            raise RuntimeError("Falcon produced invalid output; this window was not played.")
        return output

    def infer(self, audio: np.ndarray, frames: np.ndarray) -> np.ndarray:
        if not isinstance(audio, np.ndarray) or audio.dtype != np.float32 or audio.shape != (WINDOW_SAMPLES,):
            raise ValueError(f"Expected float32 mono audio shape ({WINDOW_SAMPLES},) at 16 kHz.")
        if not np.isfinite(audio).all():
            raise ValueError("Audio contains non-finite samples.")
        with self._lock:
            if not self.ready:
                raise RuntimeError(self.error or "Falcon model has not finished loading.")
            started = time.perf_counter()
            output = self._forward(audio, frames)
            self.last_inference_ms = round((time.perf_counter() - started) * 1000, 1)
            return output
