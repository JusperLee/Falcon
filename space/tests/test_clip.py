"""clip.py against Falcon Studio's live path: the client's window/frame choice (a Python
transcription of studio/web/app.js maybeInfer, frameWindow and the Stop drain) and the
server's OutputSlicer and preprocess_mouth, imported from studio/."""
import base64
import io
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "space"), str(REPO / "studio")]
import clip  # noqa: E402
from falcon_runtime import preprocess_mouth  # noqa: E402
from server import OutputSlicer  # noqa: E402

SR, FPS, HOP, GUARD, WINDOW = 16000, 25, 8000, 1280, 32000
LEFT = WINDOW - HOP - GUARD


def studio_frame_window(times_s, start_time, end_time):
    """app.js frameWindow with sessionEpoch = audioOrigin = 0; times in seconds."""
    padded = start_time < 0
    index = next((i for i, t in enumerate(times_s) if t >= 0), -1) if padded else 0
    if index < 0 or len(times_s) - index < 2 or times_s[index] > max(start_time, 0) + .12 or times_s[-1] < end_time - .12:
        return None
    first, picks = index, []
    for n in range(FPS * 2):
        target = start_time + (n + .5) / FPS
        if padded and target < 0:
            picks.append(first)
            continue
        while index + 1 < len(times_s) and abs(times_s[index + 1] - target) < abs(times_s[index] - target):
            index += 1
        if abs(times_s[index] - target) > .16:
            return None
        picks.append(index)
    return picks


def studio_windows(samples, times_s):
    """The windows live Studio sends for one target, then Stop: zero-padded ring, last crop held."""
    target = GUARD + math.ceil(samples / HOP) * HOP
    times = list(times_s)
    t = times[-1] + 1 / FPS
    while t < target / SR + 1 / FPS:  # requestStop appends copies of the last crop
        times.append(t)
        t += 1 / FPS
    out, end = [], HOP + GUARD
    while end - GUARD <= target - GUARD:
        picks = studio_frame_window(times, (end - WINDOW) / SR, end / SR)
        assert picks is not None, f"Studio would skip the window ending at {end}"
        out.append((end, [min(p, len(times_s) - 1) for p in picks]))  # appended crops are the last crop
        if end >= target:
            break
        end += HOP
    return out


def capture_times(seconds, seed):
    rng = np.random.default_rng(seed)
    times = (np.arange(int(seconds * FPS)) + .5) / FPS + rng.uniform(-.008, .008, int(seconds * FPS)) + rng.uniform(0, .01)
    return np.sort(np.clip(times, 0, None))


class MockVideo(torch.nn.Module):
    def forward(self, lips):  # [B, 1, 50, 88, 88] -> [B, 50]
        return lips.mean(dim=(1, 3, 4))


class MockAudio(torch.nn.Module):
    def forward(self, wave, embedding):  # each output sample depends on its audio and its video slot
        slots = embedding.repeat_interleave(WINDOW // embedding.shape[1], dim=1)
        return torch.tanh(wave * 3 + .01 * slots)[:, None]


@pytest.mark.parametrize("seconds,seed", [(1.0, 0), (1.3, 1), (2.0, 2), (3.77, 3), (6.5, 4)])
def test_windows_and_crops_match_studio(seconds, seed):
    samples = int(seconds * SR)
    times_s = capture_times(seconds, seed)
    times = times_s * SR
    expected = studio_windows(samples, times_s)
    assert clip.plan(samples) == len(expected)
    rng = np.random.default_rng(seed)
    audio = rng.uniform(-.5, .5, samples).astype(np.float32)
    for w, (end, picks) in enumerate(expected):
        studio_audio = np.array([audio[s] if 0 <= s < samples else 0 for s in range(end - WINDOW, end)], np.float32)
        np.testing.assert_array_equal(clip.window_audio(audio, w), studio_audio)
        assert clip.window_frames(times, w).tolist() == picks, f"window {w}"


def studio_separate(audio_model, video_model, audio, crops, times_s):
    """Studio's live path for one target: per-window preprocess_mouth + model + OutputSlicer."""
    slicer, chunks = OutputSlicer(), []
    for end, picks in studio_windows(len(audio), times_s):
        wave = np.array([audio[s] if 0 <= s < len(audio) else 0 for s in range(end - WINDOW, end)], np.float32)
        with torch.inference_mode():
            embedding = video_model(preprocess_mouth(crops[picks]))
            prediction = audio_model(torch.from_numpy(wave)[None], embedding)[0, 0].numpy()
        chunk, _, _ = slicer.take(prediction, end + LEFT, reset=not chunks)
        chunks.append(chunk)
    return np.concatenate(chunks)[:len(audio)]


def test_separate_matches_studio_live_path():
    seconds, rng = 4.3, np.random.default_rng(7)
    times_s = capture_times(seconds, 7)
    audio = rng.uniform(-.6, .6, int(seconds * SR)).astype(np.float32)
    crops = rng.integers(0, 256, (len(times_s), 96, 96), dtype=np.uint8)
    target, windows, clipped = clip.separate(MockAudio(), MockVideo(), audio, crops, times_s * SR, batch=3)
    assert windows == clip.plan(len(audio)) and clipped == 0
    np.testing.assert_allclose(target, studio_separate(MockAudio(), MockVideo(), audio, crops, times_s), atol=1e-6)


def test_window_frames_breaks_ties_to_the_earlier_crop():
    times = (np.arange(200) + .5) * clip.FRAME  # crops exactly on the 25 fps grid
    for w in (1, 2, 3):
        centres = w * HOP - LEFT + (np.arange(50) + .5) * clip.FRAME
        distance = np.abs(times[None, :] - centres[:, None])
        expected = np.argmin(distance, axis=1)  # first minimum = earlier crop
        assert clip.window_frames(times, w).tolist() == expected.tolist()
    assert (np.abs(times[None, :] - (2 * HOP - LEFT + (np.arange(50) + .5) * clip.FRAME)[:, None]) == clip.FRAME / 2).any()


def test_plan_and_stitch_lengths():
    assert [clip.plan(n) for n in (1, 8000, 8001, 16000, 960000)] == [1, 1, 2, 2, 120]
    target, clipped = clip.stitch(np.full((3, WINDOW), 2.0, np.float32), 20000)
    assert target.shape == (20000,) and clipped == 20000 and target.max() == 1.0


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["small", "big"])
def test_real_model_matches_studio_live_path(variant):
    from falcon_runtime import FalconRuntime
    runtime = FalconRuntime(variant).load("cpu")
    seconds, rng = 3.3, np.random.default_rng(3)
    times_s = capture_times(seconds, 3)
    audio = (rng.standard_normal(int(seconds * SR)) * .1).astype(np.float32)
    crops = rng.integers(60, 200, (len(times_s), 96, 96), dtype=np.uint8)
    reference = studio_separate(runtime.audio_model, runtime.video_model, audio, crops, times_s)
    single, _, _ = clip.separate(runtime.audio_model, runtime.video_model, audio, crops, times_s * SR, batch=1)
    np.testing.assert_array_equal(single, reference)
    # Batches only change float32 rounding: on the CPU a batch of 4 matches a float64 run to
    # ~1e-8, while one window at a time is off by up to ~2e-4 on these near-silent outputs.
    batched, _, _ = clip.separate(runtime.audio_model, runtime.video_model, audio, crops, times_s * SR, batch=4)
    assert np.max(np.abs(batched - reference)) < 1e-3


def encode_request(seconds=1.5, variant="small", **overrides):
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(int(seconds * SR)) * .1).astype("<f4")
    jpeg = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (96, 96), dtype=np.uint8)).save(jpeg, "JPEG")
    frame = base64.b64encode(jpeg.getvalue()).decode()
    count = int(seconds * FPS)
    request = {"variant": variant, "audio": base64.b64encode(audio.tobytes()).decode(),
               "frames": [frame] * count, "times": ((np.arange(count) + .5) * clip.FRAME).tolist()}
    request.update(overrides)
    return request


def test_decode_request_accepts_a_clip():
    variant, audio, crops, times = clip.decode_request(encode_request(), {"small": 1, "big": 1})
    assert variant == "small" and audio.shape == (24000,) and crops.shape == (37, 96, 96) and times.shape == (37,)


@pytest.mark.parametrize("change", [
    {"variant": "huge"},
    {"audio": base64.b64encode(np.zeros(8000, "<f4").tobytes()).decode()},  # 0.5 s
    {"audio": base64.b64encode(np.zeros(16001 * 4 + 1, np.uint8).tobytes()).decode()},  # not float32
    {"audio": base64.b64encode(np.full(24000, np.nan, "<f4").tobytes()).decode()},
    {"audio": "not base64!"},
    {"times": [0.0] * 36},
    {"times": list(range(37, 0, -1))},
    {"times": [-1.0] + [float(t) for t in range(36)]},
    {"frames": ["x" * 40001] * 37},
    {"frames": [base64.b64encode(b"not an image").decode()] * 37},
])
def test_decode_request_rejects(change):
    with pytest.raises(ValueError):
        clip.decode_request(encode_request(**change), {"small": 1, "big": 1})


def test_decode_request_rejects_wrong_crop_size():
    jpeg = io.BytesIO()
    Image.new("L", (64, 64)).save(jpeg, "JPEG")
    frames = [base64.b64encode(jpeg.getvalue()).decode()] * 37
    with pytest.raises(ValueError, match="96×96"):
        clip.decode_request(encode_request(frames=frames), {"small": 1})
