import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import decode_request, OutputSlicer, create_app, WINDOW, HOP, GUARD, same_origin


def request_payload():
    image = io.BytesIO()
    Image.new("L", (96, 96), 107).save(image, format="JPEG")
    return {"type": "infer", "id": 1, "audio": base64.b64encode(np.zeros(WINDOW, "<f4").tobytes()).decode(),
            "frames": [base64.b64encode(image.getvalue()).decode()] * 50,
            "end_sample": WINDOW, "reset": True}


class StubEngine:
    """Only protocol tests use this stub. Production always uses FalconRuntime."""
    busy = False
    def status(self):
        return {"ready": True, "device": "test", "app": "falcon-studio"}
    def start_load(self, variant=None):
        pass
    def infer(self, audio, frames):
        return np.linspace(-.2, .2, WINDOW, dtype=np.float32), 25


def test_wire_roundtrip_and_timestamp():
    data = request_payload()
    audio, frames, endpoint = decode_request(data)
    assert audio.shape == (WINDOW,) and frames.shape == (50, 96, 96) and endpoint == WINDOW
    with TestClient(create_app(StubEngine(), autoload=False)) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json(data)
            result = ws.receive_json()
            output = np.frombuffer(base64.b64decode(result["audio"]), "<f4")
            assert result["type"] == "result" and len(output) == HOP
            assert result["start_sample"] == WINDOW - GUARD - HOP
            assert result["end_sample"] == WINDOW - GUARD
            assert result["rtf"] == .05 and abs(result["window_rtf"] - .0125) <= .00051
            assert not result["contiguous"]
            data.update(end_sample=WINDOW + HOP, reset=False, id=2)
            ws.send_json(data)
            assert ws.receive_json()["contiguous"]


@pytest.mark.parametrize("mutator", [
    lambda p: p.update(audio="malformed"),
    lambda p: p.update(frames=p["frames"][:-1]),
    lambda p: p.update(audio=base64.b64encode(np.full(WINDOW, np.nan, "<f4").tobytes()).decode()),
    lambda p: p.update(end_sample=-1),
    lambda p: p.update(end_sample=True),
    lambda p: p.update(end_sample=32000.5),
    lambda p: p.update(end_sample=2**53),
    lambda p: p.update(reset="false"),
    lambda p: p.update(reset=1),
    lambda p: p.update(reset=None),
    lambda p: p.update(window_seconds=3),
])
def test_invalid_payload_rejected(mutator):
    data = request_payload()
    mutator(data)
    with pytest.raises(ValueError):
        decode_request(data)


def test_crossfade_uses_same_time_and_resets_after_gap():
    slicer = OutputSlicer()
    previous = np.arange(WINDOW, dtype=np.float32) / 100000
    first, _, _ = slicer.take(previous, WINDOW)
    next_window = np.arange(HOP, HOP + WINDOW, dtype=np.float32) / 100000
    second, contiguous, _ = slicer.take(next_window, WINDOW + HOP)
    assert contiguous
    assert np.allclose(np.r_[first, second], np.arange(WINDOW - GUARD - HOP, WINDOW - GUARD + HOP) / 100000)
    jumped, contiguous, _ = slicer.take(np.zeros(WINDOW, np.float32), WINDOW + 3 * HOP)
    assert not contiguous and np.count_nonzero(jumped) == 0


def test_http_and_websocket_reject_external_origins():
    with TestClient(create_app(StubEngine(), autoload=False)) as client:
        assert client.get("/api/status").status_code == 200
        assert client.post("/api/load", headers={"Origin": "https://malicious.example"}).status_code == 403
        with pytest.raises(Exception):
            with client.websocket_connect("/ws", headers={"Origin": "https://malicious.example"}):
                pass
    assert same_origin("http://localhost:7860", "localhost:7860")
    assert not same_origin("http://localhost:9999", "localhost:7860")


def test_websocket_validation_error_does_not_kill_session():
    with TestClient(create_app(StubEngine(), autoload=False)) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_text("not json")
            assert ws.receive_json()["type"] == "error"
            ws.send_json(request_payload())
            assert ws.receive_json()["type"] == "result"


@pytest.mark.parametrize("invalid_id", [True, False, 2**53, -(2**53)])
def test_websocket_rejects_boolean_and_imprecise_ids(invalid_id):
    with TestClient(create_app(StubEngine(), autoload=False)) as client:
        with client.websocket_connect("/ws") as ws:
            invalid = request_payload()
            invalid["id"] = invalid_id
            ws.send_json(invalid)
            assert ws.receive_json()["type"] == "error"
            ws.send_json(request_payload())
            assert ws.receive_json()["type"] == "result"


def test_unknown_or_missing_variant_rejected_without_disrupting_small():
    from server import Engine
    engine = Engine()
    with pytest.raises(ValueError):
        engine.start_load("bogus")
    assert engine.variant == "small" and not engine.loading
    engine.active_connections = 1
    with pytest.raises(RuntimeError):
        engine.start_load("big")
    assert not engine.loading


def test_bad_load_payload_rejected():
    with TestClient(create_app(StubEngine(), autoload=False)) as client:
        assert client.post("/api/load", content="bad").status_code == 400
        assert client.post("/api/load", json=[]).status_code == 400
        assert client.post("/api/load", json={}).status_code == 200


def test_local_tracking_assets_and_wasm_policy():
    with TestClient(create_app(StubEngine(), autoload=False)) as client:
        response = client.get("/")
        policy = response.headers["content-security-policy"]
        assert "script-src 'self' 'wasm-unsafe-eval';" in policy
        assert "'unsafe-eval'" not in policy
        for asset in ("vision_bundle.mjs", "face_landmarker.task",
                      "wasm/vision_wasm_module_internal.js",
                      "wasm/vision_wasm_module_internal.wasm"):
            response = client.get(f"/static/vendor/mediapipe/{asset}")
            assert response.status_code == 200 and len(response.content) > 1000
            if asset.endswith(".wasm"):
                assert response.headers["content-type"] == "application/wasm"
