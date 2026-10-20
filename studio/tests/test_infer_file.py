"""Offline clip inference matches the windowed runtime and handles arbitrary lengths."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import infer_file
from falcon_runtime import WINDOW_FRAMES, WINDOW_SAMPLES, FalconRuntime, weights_source

pytestmark = pytest.mark.skipif(weights_source("small") != "cached", reason="JusperLee/Falcon not downloaded yet")


@pytest.fixture(scope="module")
def runtime():
    return FalconRuntime("small").load("cpu")


def test_two_second_clip_matches_windowed_runtime(runtime):
    rng = np.random.default_rng(1)
    audio = (rng.standard_normal(WINDOW_SAMPLES) * .05).astype(np.float32)
    mouth = rng.integers(0, 256, (WINDOW_FRAMES, 96, 96), dtype=np.uint8)
    np.testing.assert_allclose(infer_file.extract(runtime, audio, mouth.astype(np.float32)),
                               runtime.infer(audio, mouth), rtol=1e-5, atol=1e-7)


def test_arbitrary_length_and_file_round_trip(runtime, tmp_path):
    rng = np.random.default_rng(2)
    audio = (rng.standard_normal(59200) * .05).astype(np.float32)  # 3.7 s
    mouth = rng.integers(0, 256, (95, 96, 96), dtype=np.uint8)  # 3.8 s, trimmed to 92 frames
    np.save(tmp_path / "mouth.npy", mouth)
    infer_file.write_wav(tmp_path / "mix.wav", audio)
    out = infer_file.extract(runtime, infer_file.read_wav(tmp_path / "mix.wav"), infer_file.read_mouth(tmp_path / "mouth.npy"))
    assert out.shape == (92 * 640,)
    assert np.isfinite(out).all()
