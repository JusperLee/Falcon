"""The assembled Space served by its own app.py on the CPU (real weights, cached from Hugging Face)."""
import base64
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from test_clip import capture_times, encode_request

SPACE = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    out = tmp_path_factory.mktemp("space") / "build"
    subprocess.run([sys.executable, str(SPACE / "build.py"), "--out", str(out)], check=True)
    return out


@pytest.fixture(scope="module")
def base(build):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "GRADIO_SERVER_PORT": str(port), "FALCON_DEVICE": "cpu", "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("SPACES_ZERO_GPU", None)
    process = subprocess.Popen([sys.executable, "app.py"], cwd=build, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(240):
            try:
                with urllib.request.urlopen(url + "/api/status", timeout=1):
                    break
            except OSError:
                if process.poll() is not None:
                    raise RuntimeError(process.stdout.read().decode())
                time.sleep(.5)
        else:
            raise RuntimeError("Space did not start")
        yield url
    finally:
        process.terminate()
        process.wait(10)


def get(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.status, dict(response.headers), response.read()


def call(base, payload):
    request = urllib.request.Request(base + "/gradio_api/call/separate", data=json.dumps({"data": [payload]}).encode(),
                                     headers={"Content-Type": "application/json", "x-gradio-user": "app"})
    with urllib.request.urlopen(request, timeout=30) as response:
        event_id = json.load(response)["event_id"]
    with urllib.request.urlopen(f"{base}/gradio_api/call/separate/{event_id}", timeout=300) as response:
        text = response.read().decode()
    events = [block for block in text.split("\n\n") if block.startswith("event:")]
    event = next(block for block in events if not block.startswith("event: heartbeat"))
    name, data = event.split("\n", 1)
    return name.split(": ", 1)[1], json.loads(data.split(": ", 1)[1])


def test_build_merges_space_files(build):
    assert (build / "web" / "vendor" / "mediapipe" / "face_landmarker.task").is_file()
    assert "import spaceMessages from './i18n-space.js'" in (build / "web" / "i18n.js").read_text()
    assert (build / "web" / "app.js").read_text().startswith("// Falcon Studio on a ZeroGPU Space")
    assert not list(build.rglob("__pycache__")) and not list(build.rglob("*.test.cjs"))
    assert (build / "README.md").read_text().startswith("---\ntitle: Falcon Studio")


def test_page_status_and_headers(base):
    status, headers, body = get(base + "/")
    assert status == 200 and b"/static/app.js" in body
    assert "frame-ancestors 'self' https://huggingface.co" in headers["content-security-policy"]
    assert headers["permissions-policy"] == "camera=(self), microphone=(self)"
    _, headers, _ = get(base + "/static/vendor/mediapipe/vision_bundle.mjs")
    assert headers["cache-control"] == "public, max-age=86400"
    _, _, body = get(base + "/api/status")
    status = json.loads(body)
    assert status["ready"] and status["mode"] == "clip" and status["max_seconds"] == 60
    assert [v["id"] for v in status["variants"]] == ["small", "big"]


@pytest.mark.parametrize("variant", ["small", "big"])
def test_separate_round_trip(base, variant):
    seconds = 3.7
    request = encode_request(seconds, variant)
    request["times"] = (capture_times(seconds, 1)[:len(request["frames"])] * 16000).tolist()
    event, data = call(base, request)
    assert event == "complete", data
    result = data[0]
    target = np.frombuffer(base64.b64decode(result["audio"]), "<f4")
    assert target.shape == (int(seconds * 16000),) and np.isfinite(target).all() and np.abs(target).max() <= 1
    assert result["windows"] == 8 and result["variant"] == variant and result["sample_rate"] == 16000
    assert result["gpu_ms"] > 0 and result["device"] == "cpu"


def test_separate_reports_invalid_clip(base):
    event, data = call(base, encode_request(times=[0.0]))
    assert event == "error" and data["error"] == "嘴部画面数量与录音时长不符。"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_every_chinese_string_has_an_english_translation(build):
    subprocess.run(["node", str(SPACE / "tests" / "check_i18n.mjs"), str(build)], check=True)
