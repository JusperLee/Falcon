<div align="center">

<a href="https://github.com/JusperLee/Falcon">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-tile.png">
    <img src="assets/logo.png" alt="Falcon logo: a falcon head with sound waves" width="160">
  </picture>
</a>

<h1>Falcon<br>Efficient Streaming Audio-Visual Target Speaker Extraction<br>for Real-World Acoustic Scenes</h1>

**English** | [中文](README_zh.md)

**Wendi Sang**<sup>\*</sup>, **Kai Li**<sup>\*</sup>, **Yifan Li**, **Jianqiang Huang**, **Xiaolin Hu**

<sup>\*</sup>Equal contribution <!-- TODO: affiliations and corresponding author -->

**NeurIPS 2026**

<!-- TODO: fill in the links once the paper, weights and dataset are public. -->
[**Paper (TODO)**](#) | [**🎙️ Live demo**](https://u464727-dmja-7896a455.bjb3.seetacloud.com:8443/) | [**🤗 Space**](https://huggingface.co/spaces/JusperLee/Falcon-Studio) | [**Project page**](https://cslikai.cn/Falcon/) | [**🤗 Falcon**](https://huggingface.co/JusperLee/Falcon) | [**🤗 Falcon-Big**](https://huggingface.co/JusperLee/Falcon-Big) | [**RealSSA (TODO)**](#-realssa-benchmark) | [**Falcon Studio**](studio/README.md)

<p>
  <a href="https://visitor-badge.laobi.icu/"><img src="https://visitor-badge.laobi.icu/badge?page_id=JusperLee.Falcon&amp;left_text=Visitors&amp;left_color=%23555555&amp;right_color=%23007ec6&amp;radius=0" alt="Visitors (README views)" title="Counts README badge views, not unique visitors"></a>
  <img src="https://img.shields.io/badge/NeurIPS-2026-8c1b13?style=flat-square" alt="NeurIPS 2026">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square" alt="License: Apache 2.0"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python 3.10 or newer">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&amp;logo=pytorch&amp;logoColor=white" alt="PyTorch 2.x">
  <a href="studio/README.md"><img src="https://img.shields.io/badge/Falcon%20Studio-macOS%20%7C%20Linux%20CUDA%20%7C%20Chrome-lightgrey?style=flat-square" alt="Falcon Studio: macOS, Linux with CUDA, Chrome"></a>
</p>

</div>

> Falcon is a causal audio-visual target speaker extraction (AVTSE) model that separates in a single encoder–decoder pass. It reaches state-of-the-art quality on RealSSA, LRS2-2Mix, LRS3-2Mix and VoxCeleb2-2Mix with 0.5 M trainable parameters, and runs live in video meetings on a laptop.

## 💥 News

- **[2026-10-21]** **Falcon Studio** on a [Hugging Face ZeroGPU Space](https://huggingface.co/spaces/JusperLee/Falcon-Studio): record a clip in the browser, then separate it on a GPU.
- **[2026-10-20]** Training, evaluation and inference code released, together with **Falcon Studio**, a real-time demo that plugs into Zoom / Tencent Meeting.
- **[2026-09-24]** Falcon is accepted to **NeurIPS 2026**.

## 🧩 Overview

Streaming AVTSE is needed in latency-sensitive applications such as calls, meetings and hearing assistance. Current causal systems face two coupled gaps. Standard benchmarks fix the number of speakers at two and ignore near-/far-field interference, reverberation and sudden noise. Strong causal architectures are also too heavy for edge devices, and lightweight alternatives regain capacity by invoking a small separator many times, which offsets the savings.

- **RealSSA** is a realistic simulated-scene AVTSE benchmark with 2–6 dynamic near-/far-field speakers in four scene types, plus a real-recorded evaluation split.
- **Falcon** is a causal AVTSE framework. Its AudioBlock performs multi-scale time-frequency separation in a single encoder–decoder pass.
- An **encoder-space speaker contrastive loss** suppresses near-field leakage at zero inference cost and transfers across backbones.
- Compared with the causal AV-TFGridNet baseline, Falcon cuts parameters by 91.9 %, computation by 20.2× and GPU latency by 11.1×.

<p align="center">
  <img src="assets/Overall_Pipeline.png" alt="Overall pipeline of Falcon: an audio encoder and a frozen video encoder feed the audio and video blocks, whose fused features drive the separator; its mask is applied to the encoder output before the audio decoder." width="100%">
  <br><em>Overall pipeline of Falcon for audio-visual target speaker extraction.</em>
</p>

<p align="center">
  <img src="assets/separator.png" alt="Single-pass multi-scale separator: four encoder layers, a TFRB at the coarsest scale, GCI fusion at every scale and a decoder of GCI and TFRB blocks; phi denotes upsampling." width="80%">
  <br><em>Single-pass multi-scale separator. φ denotes the upsampling operation.</em>
</p>

<p align="center">
  <img src="assets/TFRB.png" alt="Time-frequency refinement block: a bidirectional LSTM along frequency, a unidirectional LSTM along time, then masked 2D self-attention." width="100%">
  <br><em>Time-frequency refinement block (TFRB).</em>
</p>

## 🌆 RealSSA benchmark

Each RealSSA mixture has 2–6 dynamic speakers: a near-field target plus near- and far-field interferers, together with environmental noise and sudden events. The four scene types are small meetings, medium indoor spaces, large public spaces and outdoor streets, with room geometry, RT60, speaker distance and SIR sampled per scene. Room impulse responses are simulated with gpuRIR. Targets are normalized to −23 LUFS, mixtures are peak-limited to −1 dBFS, and all audio is mono 16 kHz. A real-world evaluation split was recorded in eight environments; faces in it are pixelated.

<p align="center">
  <img src="assets/realssa_scenes.jpg" alt="Real-world AVTSE scenarios motivating RealSSA: small rooms, coffee shops, shopping malls and outdoor streets." width="100%">
  <br><em>Real-world scenarios motivating RealSSA. From left: small rooms, coffee shops, shopping malls, outdoor streets.</em>
</p>

<!-- TODO: add the RealSSA download link and license once it is public. -->

## ⚡ Installation

The repository has two parts with separate environments:

| Part | Folder | Purpose |
|---|---|---|
| Training & evaluation | repository root | Train Falcon and evaluate on LRS2/LRS3/VoxCeleb2/RealSSA-style data (`look2hear/`) |
| Falcon Studio | [`studio/`](studio/README.md) | Live separation in the browser and in video meetings, offline clip inference, Python API |

```bash
git clone https://github.com/JusperLee/Falcon.git
cd Falcon

# Training & evaluation (PyTorch >= 1.13, CUDA recommended)
pip install -r requirements.txt

# Falcon Studio only (Python 3.10-3.12; runs on Apple Metal, NVIDIA CUDA or the CPU)
pip install -r studio/requirements.txt
```

## 📦 Model weights

| Model | Hugging Face | File | Config | Audio network | Total params* | Used for |
|---|---|---|---|---|---|---|
| **Falcon** (paper) | [`JusperLee/Falcon`](https://huggingface.co/JusperLee/Falcon) | `falcon.pt` | `configs/Falcon.yml` | `FalconSpeakerContrastive` | 11.72 M | paper results, Studio default |
| **Falcon-Big** | [`JusperLee/Falcon-Big`](https://huggingface.co/JusperLee/Falcon-Big) | `falcon-big.pt` | `studio/configs/FalconBig.yml` | `SpeakerContrastiveV2` (wider) | 12.35 M | Studio "Big" option, inference only |

\*Including the frozen 11.2 M-parameter FRCNN lip-reading encoder. The paper counts trainable parameters only (0.5 M for Falcon).

Nothing needs to be downloaded by hand. Falcon Studio, `infer_file.py` and `FalconRuntime.from_pretrained` load the weights from these Hugging Face short links on first use and cache them in `~/.cache/huggingface`. If huggingface.co is slow or blocked, set a mirror, e.g. `export HF_ENDPOINT=https://hf-mirror.com`. A local file can be passed with `--checkpoint`, for example one fetched with `hf download JusperLee/Falcon falcon.pt --local-dir weights`.

The files contain only the `audio_model.*` and `video_model.*` weights. Training additionally needs the pretrained visual backbone `frcnn_128_512.backbone.pth.tar` in `pretrain_zoo/` (see Training and evaluation below).

## 🎧 Inference

### Falcon Studio: live separation and meetings

Two online instances are linked at the top of this page: the live demo runs Falcon Studio on a GPU server; the [Hugging Face Space](https://huggingface.co/spaces/JusperLee/Falcon-Studio) records a clip of up to 60 s and separates it afterwards on ZeroGPU, with the live Studio's windows. Its code is in [`space/`](space/README.md).

Falcon Studio runs the model locally, tracks the speaker's mouth with MediaPipe in the browser, and plays the separated voice on headphones or sends it to Zoom / Tencent Meeting through a virtual audio device (BlackHole). On macOS, double-click `studio/Start Falcon.command`, or run:

```bash
cd studio
./scripts/launch.sh
```

Setup, meeting routing and troubleshooting are described in [`studio/README.md`](studio/README.md) ([中文](studio/README_zh.md)). The model processes 2 s audio-visual windows and outputs a new 0.5 s every hop; on an Apple M5 Pro a window takes about 0.12 s on Metal. On Linux with an NVIDIA GPU, use `--device cuda` (`auto` picks CUDA first).

### Offline: one clip

```bash
cd studio
ffmpeg -i input.mp4 -ac 1 -ar 16000 mix.wav
python infer_file.py --audio mix.wav --mouth mouth.npy --out target.wav --model JusperLee/Falcon
```

`mouth.npy` holds grayscale 96×96 mouth crops of the target speaker at 25 fps, shape `(T, 96, 96)`, in the same format as the training data (`.npz` and `.pt` also work). The whole clip is processed in one pass.

### Python API

```python
import numpy as np
from falcon_runtime import FalconRuntime  # run from studio/

model = FalconRuntime.from_pretrained("JusperLee/Falcon")  # or "JusperLee/Falcon-Big"; device="auto", "cuda", "mps" or "cpu"
audio = np.zeros(32000, dtype=np.float32)              # 2 s mono 16 kHz mixture
mouth = np.zeros((50, 96, 96), dtype=np.uint8)         # 50 grayscale mouth crops, 25 fps
target = model.infer(audio, mouth)                     # (32000,) float32 target speech
```

## 🏋️ Training and evaluation

Data follow the LRS2/LRS3 layout: `tr/`, `cv/` and `tt/`, each with `mix/` and `target/` WAV folders (16 kHz) and the target speakers' mouth crops. Build the JSON indices:

```bash
DATA=/path/to/data      # contains tr/ cv/ tt/
INDEX=/path/to/index    # output folder for the JSON files

python DataPreProcess/process_lrs23.py --in_audio_dir $DATA --out_dir $INDEX
python DataPreProcess/generate_near_json.py --data_root $DATA --out_root $INDEX --sample_rate 16000
```

`process_lrs23.py` writes `mix.json` and `target.json` for each split. `generate_near_json.py` writes `near.json`, which indexes the near-field interferers that the speaker contrastive loss uses as negatives. Then edit `configs/Falcon.yml`: set `datamodule.data_config.train_dir`, `valid_dir` and `test_dir` to `$INDEX/tr`, `$INDEX/cv` and `$INDEX/tt` (the scripts do not expand environment variables). Put the lip-reading-pretrained visual backbone `frcnn_128_512.backbone.pth.tar` in `pretrain_zoo/` at the repository root. <!-- TODO: link to the backbone weights. -->

```bash
# Train (DDP, AdamW lr 1e-3, ReduceLROnPlateau, early stopping 30, grad clip 5)
python audio_train.py --conf_dir configs/Falcon.yml

# Evaluate SI-SDR / SDR on the test split (one or more GPUs)
python audio_visual_test.py --conf_dir Experiments/checkpoint/Falcon/conf.yml --gpus 0

# Evaluate the released weights directly from Hugging Face
python audio_visual_test.py --conf_dir Experiments/checkpoint/Falcon/conf.yml --ckpt_path hf://JusperLee/Falcon/falcon.pt --gpus 0

# Parameters and MACs
python evaluated_mac_params.py
```

Evaluation reads the data paths from that `conf.yml` (edit them the same way) and, unless `--ckpt_path` is given, loads the best checkpoint in `Experiments/checkpoint/<exp_name>/`: the lowest entry of `best_k_models.json` if it exists, else `last.ckpt`, else the last `epoch=*.ckpt`. Results go to `Experiments/checkpoint/Falcon/results/metrics_combined.csv`. The training code (`look2hear/`) and the Studio runtime (`studio/falcon_model/`) produce identical output for `falcon.pt`; the Studio copy only drops training-time dependencies and adds a CPU FFT bridge for Apple Metal (see [`PROVENANCE.md`](studio/falcon_model/PROVENANCE.md)).

## 📊 Results

Causal AVTSE methods (SI-SNRi / SDRi in dB; DNSMOS on the real-recorded split). **Bold** marks the best result per column.

| Method | RealSSA SI-SNRi | SDRi | LRS2-2Mix SI-SNRi | SDRi | LRS3-2Mix SI-SNRi | SDRi | VoxCeleb2-2Mix SI-SNRi | SDRi | Real-world DNSMOS |
|---|---|---|---|---|---|---|---|---|---|
| AV-ConvTasNet | 1.2 | 2.6 | 9.8 | 10.3 | 12.5 | 12.8 | 6.9 | 8.2 | 2.580 |
| AV-Sepformer | 5.5 | 6.3 | 10.6 | 11.1 | 13.4 | 13.7 | 6.9 | 8.3 | 2.813 |
| AV-TF-GridNet | 7.6 | 8.1 | 12.7 | 13.1 | 14.3 | 14.8 | 5.8 | 7.7 | 3.030 |
| RTFSNet-12 | 4.9 | 5.7 | 12.4 | 12.8 | 14.0 | 14.4 | 8.4 | 10.7 | 3.066 |
| CTCNet | 5.4 | 6.2 | 8.0 | 8.8 | 9.4 | 10.2 | 2.5 | 4.7 | 2.840 |
| AV-DPRNN | 5.4 | 6.2 | 10.2 | 10.7 | 11.9 | 12.3 | 7.8 | 9.0 | 2.901 |
| AV-Mossformer2 | 6.0 | 6.7 | 13.2 | 13.9 | 16.2 | 16.6 | 7.7 | 9.1 | 2.912 |
| Swift-Net-12 | 7.3 | 7.8 | 13.9 | 14.1 | 16.0 | 16.3 | 9.4 | 10.8 | **3.276** |
| Dolphin | 6.7 | 7.3 | 14.0 | 14.6 | 14.3 | 14.9 | 9.2 | 11.1 | 2.960 |
| **Falcon (ours)** | **7.7** | **8.2** | **14.1** | **14.7** | **16.4** | **16.7** | **10.5** | **11.8** | **3.276** |

Efficiency on a 2 s, 16 kHz clip with 50 mouth frames (RTX 3090 / Intel Xeon, batch size 1; trainable parameters only):

| Method | Params (M) | MACs (G) | CPU latency (ms) | GPU latency (ms) | RAM (MB) | GPU memory (MB) |
|---|---|---|---|---|---|---|
| AV-ConvTasNet | 11.0 | 15.0 | 674.4 | 13.3 | 100.0 | 241.8 |
| AV-Sepformer | 39.7 | 226.5 | 5143.0 | 53.9 | 178.0 | 369.7 |
| AV-TF-GridNet | 6.2 | 213.8 | 5646.3 | 567.8 | 136.3 | 1015.8 |
| RTFSNet-12 | 0.6 | 67.0 | 4149.3 | 164.7 | 457.9 | 411.7 |
| CTCNet | 6.8 | 150.1 | 3267.4 | 162.8 | 233.2 | 216.5 |
| AV-DPRNN | 1.3 | **4.5** | **271.0** | **5.8** | **73.0** | **192.4** |
| AV-Mossformer2 | 56.7 | 224.2 | 16670.7 | 114.9 | 330.9 | 539.8 |
| Swift-Net-12 | **0.5** | 45.3 | 4163.8 | 172.1 | 482.3 | 411.7 |
| Dolphin | 5.6 | 18.1 | 2411.0 | 87.9 | 560.2 | 390.3 |
| **Falcon (ours)** | **0.5** | 10.6 | 1361.2 | 51.3 | 125.3 | 364.7 |

Dolphin and AV-Mossformer2 were re-implemented as causal variants under our training protocol; the other baselines follow the values reported by Swift-Net. Ablations and real-world NISQA results are in the paper.

## 🗂️ Repository layout

```
Falcon/
├── audio_train.py            # training entry point (PyTorch Lightning, DDP)
├── audio_visual_test.py      # SI-SDR / SDR evaluation, single or multi-GPU
├── evaluated_mac_params.py   # parameters and MACs
├── configs/Falcon.yml        # paper configuration
├── DataPreProcess/           # mix/target/near JSON index generation
├── look2hear/                # models, video models, layers, losses, datasets, Lightning systems, metrics
├── Experiments/checkpoint/   # config snapshot and best-checkpoint metadata (weights downloaded separately)
├── assets/                   # README figures
├── project_page/             # project page with demo samples (https://cslikai.cn/Falcon/; media on oss.cslikai.cn)
├── space/                    # Hugging Face ZeroGPU Space: record, then separate (build.py assembles it with studio/)
└── studio/                   # Falcon Studio
    ├── server.py             # local FastAPI + WebSocket inference server
    ├── falcon_runtime.py     # Hugging Face weight download, strict loading, 2 s windowed inference
    ├── infer_file.py         # offline clip inference CLI
    ├── falcon_model/         # inference copy of the model code (see PROVENANCE.md)
    ├── configs/              # Falcon.yml (Small) and FalconBig.yml (Big)
    ├── web/                  # browser app: capture, mouth tracking, playback, meeting output
    ├── scripts/              # launcher, environment bootstrap, BlackHole installer
    └── tests/                # Python, Node and browser checks
```

## 📝 Citation

```bibtex
@inproceedings{sang2026falcon,
  title     = {Efficient Streaming Audio-Visual Target Speaker Extraction for Real-World Acoustic Scenes},
  author    = {Sang, Wendi and Li, Kai and Li, Yifan and Huang, Jianqiang and Hu, Xiaolin},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## 📄 License

The code and the released weights are licensed under the [Apache License 2.0](LICENSE). Falcon Studio bundles MediaPipe Tasks Vision and its face landmarker model (`studio/web/vendor/mediapipe`), distributed by Google under the Apache License 2.0 with their own notices. LRS2, LRS3 and VoxCeleb2 are subject to their own licenses and are not redistributed here.

## 🙏 Acknowledgements

The frozen visual encoder follows [CTCNet](https://github.com/JusperLee/CTCNet). Baseline numbers follow [Swift-Net](https://github.com/JusperLee/Swift-Net) except for the causal re-implementations of [Dolphin](https://github.com/JusperLee/Dolphin) and [AV-MossFormer2](https://github.com/modelscope/ClearerVoice-Studio). Live mouth tracking in Falcon Studio uses [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker).
