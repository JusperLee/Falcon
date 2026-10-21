<p align="center">
  <a href="https://github.com/JusperLee/Falcon">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="../assets/logo-tile.png">
      <img src="../assets/logo.png" alt="Falcon logo" width="120">
    </picture>
  </a>
</p>

# Falcon Studio

**English** | [中文](README_zh.md)

> The real-time demo of the [Falcon](../README.md) repository. The paper, training and evaluation code are described in the root README.

> **🎙️ Try it online:** <https://u464727-dmja-7896a455.bjb3.seetacloud.com:8443/> (Chrome). It runs on a GPU server, so camera and microphone data are sent to that server. The rest of this guide covers the local install.
>
> **🤗 Hugging Face Space:** <https://huggingface.co/spaces/JusperLee/Falcon-Studio>. The same interface records a clip of up to 60 s and separates it afterwards on ZeroGPU, without headphone monitoring or meeting output. Audio and mouth crops are sent to the Space. See [`space/`](../space/README.md).

Falcon Studio runs Falcon audio-visual target speaker extraction live on a Mac, or on Linux with an NVIDIA GPU. The browser captures the microphone and camera and tracks the speaker's mouth; a local Python service runs the model and sends the separated voice back for headphone monitoring, recording, or a video meeting (Zoom / Tencent Meeting).

Everything runs on your machine: the interface, fonts, scripts and the MediaPipe face tracker ship with the project, with no CDN. The app opens at `http://127.0.0.1:7860`, or the next free port in 7860–7899. Camera and microphone data only go to the local service and are not written to disk. When you enable meeting output, the separated voice goes to the virtual audio device you choose and from there to your meeting app.

## Quick start

1. Double-click **Start Falcon.command** in Finder. The first run checks the environment, installs missing dependencies and downloads the weights from Hugging Face.
2. The browser opens; wait until the model has loaded and warmed up.
3. Click **Enable devices** and allow camera and microphone access.
4. **Auto tracking** is the default. Face the camera with your mouth visible, wait for the green lip outline and the **AUTO** crop to follow you, then click **Start separation**.
5. To hear the result, put on headphones first, then turn on **Headphone monitoring** (off by default).
6. After stopping you can **Play** the recording or **Export WAV**. Switch between **Separated** and **Mixture** to hear or save the separated voice or the microphone input the model received, aligned to the same moments. The browser keeps up to 5 minutes of exportable audio; live separation and meeting output keep running after that limit.
7. **Stop** first separates the last second of speech, then releases the devices (click again to stop at once). To quit the app, press **Control-C** in the terminal window.

The interface starts in English; switch between **English / 中文** at the top right at any time. Switching keeps the capture session, model and recording.

## Models

| Model | Hugging Face | Notes |
|---|---|---|
| Falcon (default) | [`JusperLee/Falcon`](https://huggingface.co/JusperLee/Falcon) | The paper model; fastest |
| Falcon-Big | [`JusperLee/Falcon-Big`](https://huggingface.co/JusperLee/Falcon-Big) | Wider variant, higher capacity |

Choose the model under **Separation engine** (Falcon Small / Falcon Big in the menu); stop separation before switching. Weights are downloaded from the Hugging Face short link the first time a model is loaded and cached in `~/.cache/huggingface`; after that the app runs offline. If huggingface.co is slow or blocked, use a mirror:

```bash
HF_ENDPOINT=https://hf-mirror.com ./scripts/launch.sh
```

Weights are loaded strictly against the model configuration. A mismatch is reported as an error; the app never plays the unprocessed mixture as if it were separated.

## Mouth tracking

**Auto tracking** runs the bundled MediaPipe Face Landmarker in a worker and keeps the mouth crop on the speaker; no camera frame leaves the machine.

- **Tracking**: face the camera in good light with the mouth fully visible; natural head movement is fine.
- **Face lost**: separation does not pause. The crop stays where the mouth was last seen and the status shows *No face detected · Using the last crop*. The model cannot see lip movement meanwhile, so quality can drop. When the face returns near the same place, tracking re-locks after 4 consistent frames, with no click needed.
- **Target moved far**: a face elsewhere in the frame must stay stable for about 2.5 s before it becomes the new target; until then the crop and output stay as they were. If several unconfirmed faces are visible, the app asks you to click **Reacquire**. **Reacquire** also selects the target immediately at any time.
- **Manual override**: choose **Manual crop** or drag the crop box. Drag the corner to resize, use the arrow keys to move, `+` / `-` to resize. Manual mode does not check whether a face is in the box; choose **Auto tracking** to return.

Tracking follows geometry, not identity. Keep one target speaker in view. If someone else takes the same seat, or the target turns to profile or is occluded for a long time, the crop can follow the wrong mouth; after **Reacquire**, check that the box is on the right speaker.

## Meetings: Zoom / Tencent Meeting

Meeting mode sends the separated voice to a virtual microphone. Recommended: **Chrome + BlackHole 2ch + headphones**.

```text
physical mic + camera → Falcon → BlackHole 2ch → meeting app microphone
                                ↘ optional headphone monitoring
other participants ─────────────────────────────→ headphones
```

[BlackHole](https://github.com/ExistentialAudio/BlackHole) is a macOS audio driver that turns one app's output into another app's input. Install it once by double-clicking **Setup Meeting Audio.command**, or run `./scripts/install-blackhole.sh` (`--check` only reports the status). It uses the Homebrew cask `blackhole-2ch`, may ask for an administrator password and needs a **macOS restart**. The normal launcher never installs drivers.

Before each meeting:

| Setting | Device |
|---|---|
| Falcon input microphone | your physical microphone |
| Falcon **Meeting output device** | BlackHole 2ch, then **Enable meeting output** |
| Meeting app microphone | BlackHole 2ch |
| Meeting app speaker | headphones |
| macOS sound output | headphones |

Never set the macOS output, the meeting speaker or Falcon's input to BlackHole, or the other side will hear themselves. **Headphone monitoring** only controls what you hear; meeting output works with it off. Check the meeting app's microphone test first. If the voice sounds processed, turn off the meeting app's own noise suppression and automatic microphone gain, so it does not process the separated voice a second time.

Chrome keeps capturing camera frames when you switch to the meeting window; other browsers capture in the foreground only, so keep the Falcon tab visible there. Studio routes audio only; it has no virtual camera, and the meeting video is not delayed to match the separated audio.

## How the live pipeline works

| Stage | Implementation |
|---|---|
| Audio | microphone resampled to 16 kHz mono |
| Video | 25 fps; the tracked or manual mouth crop scaled to 96×96 grayscale, center-cropped to 88×88 by the model |
| Window | the latest 2 s of audio and video |
| Hop | 0.5 s; each result plays the newest 0.5 s, keeping 80 ms of right context |
| Joins | short crossfade between consecutive outputs; band-limited resampling to the device rate |
| Overload | bounded queue; stale windows are skipped instead of building up latency |

The first output needs about 0.58 s of valid input plus computation. Each fresh target starts with 1.42 s of silent left context and its first valid mouth frame, so the first output starts at the real capture boundary instead of dropping the opening words. The synthetic history is excluded from both recordings. Stopping also drains clips shorter than the first hop; the saved audio ends at the instant Stop was clicked. Initial separation quality can differ from a full real-context window. In steady state the designed delay is about 0.7–0.8 s (0.5 s hop + 80 ms context + inference + playback buffer). Every hop recomputes the full 2 s window, because the model exposes no streaming state; this is windowed inference, not a native streaming model. On an Apple M5 Pro, a 2 s window takes about 0.12 s on Metal and about 0.65 s on the CPU, so the CPU may skip windows.

Auto tracking produces a landmark-guided square mouth crop, not the full face alignment used to build the training data, so live input can differ from training conditions. Large head turns, occlusion and lighting changes can lower quality.

The input and output panels show scrolling spectrograms of the last ~8 s (0–8 kHz, −90 to 0 dBFS). After stopping, click the waveform, spectrogram or progress bar to play from any point. Recordings and spectrograms stay in page memory until the page is closed; use **Export WAV** to keep audio.

## Command line

Run from `studio/`:

```bash
./scripts/launch.sh                              # pick Metal or CPU automatically, open the browser
./scripts/launch.sh --device cuda                # or --device mps / --device cpu
./scripts/launch.sh --model JusperLee/Falcon-Big # start with Falcon-Big loaded
./scripts/launch.sh --port 7861 --no-browser
./scripts/doctor.sh                              # check dependencies, Metal / CUDA and model downloads
```

`auto` tries CUDA, then Metal, and falls back to the CPU if the GPU warm-up fails. On Linux, install a CUDA build of PyTorch (see pytorch.org) and the requirements, then run `python server.py --device cuda` and open the printed address in Chrome; the `.command` launchers and BlackHole are macOS only. The launcher reuses a working `.venv` in this folder, otherwise installs into `~/Library/Caches/FalconStudio/venv`, outside iCloud. It prefers the pinned `requirements-lock.txt`. Python 3.10–3.12 is supported (3.11 recommended); use `FALCON_PYTHON=/path/to/python3.11` to pick an interpreter, `FALCON_VENV=/path` for another environment, and `./scripts/bootstrap.sh` to install without starting.

### Offline inference on a clip

```bash
ffmpeg -i input.mp4 -ac 1 -ar 16000 mix.wav
python infer_file.py --audio mix.wav --mouth mouth.npy --out target.wav --model JusperLee/Falcon
```

`mouth.npy` holds the target speaker's grayscale 96×96 mouth crops at 25 fps, shape `(T, 96, 96)` (`.npz` and `.pt` also work). `--checkpoint` takes a local weights file or `hf://<org>/<repo>/<file>` instead of the model's own weights.

### Python API

```python
import numpy as np
from falcon_runtime import FalconRuntime

model = FalconRuntime.from_pretrained("JusperLee/Falcon")  # or "JusperLee/Falcon-Big"; device="auto" | "cuda" | "mps" | "cpu"
audio = np.zeros(32000, dtype=np.float32)                  # 2 s mono 16 kHz mixture
mouth = np.zeros((50, 96, 96), dtype=np.uint8)             # 50 mouth crops, 25 fps
target = model.infer(audio, mouth)                         # (32000,) target speech
```

## Troubleshooting

**Double-clicking does nothing.** Run `./scripts/launch.sh` in Terminal to see the error. If the scripts lost their execute bit: `chmod +x "Start Falcon.command" "Setup Meeting Audio.command" scripts/*.sh`.

**The weights do not download.** Check the network and retry with `HF_ENDPOINT=https://hf-mirror.com ./scripts/launch.sh`, or pass a downloaded file with `--checkpoint`.

**No camera or microphone.** Open the `127.0.0.1` address printed in the terminal (the port may be 7861 or higher) in a normal browser window. Allow the browser in *System Settings → Privacy & Security → Camera / Microphone*, then restart the browser. Close meeting apps that may hold the devices.

**Video works but there is no sound.** Wait for the model to be ready and the first window to finish. Auto mode needs a face once at the start. Monitoring is off by default; check that the system output is your headphones.

**Tracking does not lock, or follows the wrong person.** Keep one unobstructed face in view, avoid backlight and profile views, and click **Reacquire**. Use **Manual crop** if it stays unstable.

**Echo or howling.** Turn monitoring off and use headphones; speakers feed the separated voice back into the microphone.

**strict / missing / unexpected keys.** The weights do not match the selected model's configuration. Use the matching Falcon / Falcon-Big weights; do not disable strict loading.

**The meeting app hears nothing.** Check that **Enable meeting output** is on with BlackHole 2ch, and that the meeting app's microphone is BlackHole 2ch. After installing BlackHole, restart macOS, the browser and the meeting app. If meeting output is unavailable, use Chrome.

**GPU (Metal / CUDA) errors, stutter or constant overload.** Restart with `--device cpu`, close other heavy apps and keep a single Falcon tab open. If inference stays above the 500 ms budget, stale windows are dropped and output can have gaps.

**Port in use.** Without `--port`, the app picks the next free port in 7860–7899; with `--port` it uses exactly that port. Closing the browser tab does not stop the service; press Control-C in the terminal.

**Broken dependencies.** Rerun `./scripts/bootstrap.sh`, or install into a fresh environment with `FALCON_VENV=/new/path`.

## Project layout

```text
Start Falcon.command         Finder launcher
Setup Meeting Audio.command  optional BlackHole installer
server.py                    local FastAPI + WebSocket service, window joining
falcon_runtime.py            weight download, strict loading, preprocessing, inference
infer_file.py                offline clip inference
falcon_model/                inference copy of the model code (see PROVENANCE.md)
configs/                     Falcon.yml and FalconBig.yml
web/                         interface: capture, mouth tracking, playback, meeting output
scripts/                     launcher, environment setup, BlackHole installer
tests/                       Python, Node and browser tests
```

The service exposes `GET /api/status`, `POST /api/load` and `/ws`, restricted to same-origin local pages.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/
for t in tests/*.mjs tests/*.cjs web/*.test.cjs; do node "$t"; done
```

The tracking test with the real MediaPipe model runs in a browser: start `FALCON_QA_PORT=8766 python tests/serve_browser.py`, open `http://127.0.0.1:8766/__test/` in Chrome and click the button. It uses a public test portrait and no camera.

### Device startup or capture stalls

Startup shows its current stage and releases devices on preview/audio initialization timeouts so it can be retried. Device enumeration does not block Start. Runtime hints distinguish stalled microphone capture, near-silent input, missing mouth frames, alignment waits and slow inference responses.

For troubleshooting, run `JSON.stringify(window.falconDiagnostics(), null, 2)` in the browser console and include the browser/device version. This exposes aggregate counters, input/output energy and state only, with no audio, images or cookies. It does not evaluate separation quality.
