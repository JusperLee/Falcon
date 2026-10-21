---
title: Falcon Studio
emoji: 🦅
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12.12"
app_file: app.py
license: apache-2.0
short_description: Audio-visual target speaker extraction in your browser
models:
  - JusperLee/Falcon
  - JusperLee/Falcon-Big
preload_from_hub:
  - JusperLee/Falcon falcon.pt
  - JusperLee/Falcon-Big falcon-big.pt
---

# Falcon Studio · Hugging Face Space

**English** | [中文](README_zh.md)

The [Falcon Studio](https://github.com/JusperLee/Falcon/tree/main/studio) interface on a ZeroGPU Space. Record a clip with your camera and microphone; Falcon extracts the voice of the speaker whose mouth is in the frame.

ZeroGPU attaches a GPU for each call and counts it against a daily quota, so the live Studio's loop of one call every 0.5 s does not fit. This Space records first and separates afterwards. For live separation, headphone monitoring and meeting output, run Falcon Studio locally.

## How it works

1. The browser captures audio at 16 kHz and tracks the mouth with MediaPipe Face Landmarker, as in Falcon Studio. Tracking runs in the browser; video frames are not uploaded.
2. A recording is up to 60 s. When it stops, the audio and the 96×96 grayscale mouth crops, each with its capture time, are sent to the Space. The start and end of the recording without any mouth crop are trimmed.
3. The server cuts the clip into the windows Falcon Studio sends live: 2 s of audio and 50 mouth slots, a 0.5 s hop, the 0.5 s chunk 80 ms before each window end, 20 ms crossfades, zero history before the clip and zero padding after it. Each slot takes the crop nearest in time, as Studio does. All windows run in batches inside one GPU call. With the same crops, the output equals the live Studio when it skips no window; where Studio would pause for stale video, the Space holds the nearest crop.
4. The page shows the GPU time, and you can play, compare with the mixture and export both as WAV.

Both models load when the Space starts: [`JusperLee/Falcon`](https://huggingface.co/JusperLee/Falcon) (the paper model) and [`JusperLee/Falcon-Big`](https://huggingface.co/JusperLee/Falcon-Big).

## Privacy and quota

Audio and mouth crops go to this Space only for the separation and are not written to disk; results stay in your browser. Opened from the Space page, a separation counts against your own ZeroGPU quota. Anonymous visitors get the smallest quota and can be refused ("ZeroGPU runs limit"); signing in to Hugging Face gives more quota and a higher queue priority. A separation takes about 0.2–2.5 s of GPU time (8 s Small to 60 s Big, measured), plus upload and queueing.

## Development

From the GitHub repository root, with Python 3.12:

```bash
pip install -r space/requirements.txt gradio==6.29.1
python space/build.py --out /tmp/falcon-space    # copies studio/ runtime and web files + space/
cd /tmp/falcon-space && python app.py            # http://127.0.0.1:7860, CPU unless FALCON_DEVICE is set
python -m pytest space/tests                     # from the repository root
python space/build.py --push JusperLee/Falcon-Studio
```

`space/` holds only what differs from Falcon Studio: `app.py` (Gradio server mode, `@spaces.GPU`), `clip.py` (window schedule and request checks), `web/index.html` and `web/app.js` (record, then separate), and `web/i18n-space.js`. Everything else is copied from `studio/` by `build.py`.
