"""Device selection: auto prefers CUDA, then Apple Metal, then the CPU; a failing GPU falls back in auto mode."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from falcon_runtime import FalconRuntime, pick_device, weights_source


def gpus(monkeypatch, cuda, mps):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


@pytest.mark.parametrize("cuda,mps,expected", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")])
def test_auto_prefers_cuda_then_metal_then_cpu(monkeypatch, cuda, mps, expected):
    gpus(monkeypatch, cuda, mps)
    assert pick_device("auto") == expected


def test_explicit_devices(monkeypatch):
    gpus(monkeypatch, True, False)
    assert pick_device("cuda") == "cuda" and pick_device("cpu") == "cpu"
    with pytest.raises(RuntimeError, match="Metal"):
        pick_device("mps")
    gpus(monkeypatch, False, True)
    with pytest.raises(RuntimeError, match="CUDA"):
        pick_device("cuda")
    with pytest.raises(ValueError):
        pick_device("tpu")


@pytest.mark.skipif(torch.cuda.is_available() or weights_source("small") != "cached",
                    reason="needs a machine without CUDA and the cached JusperLee/Falcon weights")
def test_auto_falls_back_to_cpu_when_the_gpu_cannot_run(monkeypatch):
    # Reported as available, but this PyTorch build has no CUDA: moving the model fails.
    gpus(monkeypatch, True, False)
    runtime = FalconRuntime("small").load("auto")
    assert runtime.ready and runtime.device == "cpu" and runtime.warning.startswith("CUDA warmup failed")


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the error without a GPU")
def test_explicit_cuda_without_gpu_fails_clearly():
    runtime = FalconRuntime("small")
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        runtime.load("cuda")
    assert not runtime.ready and "CUDA" in runtime.error
