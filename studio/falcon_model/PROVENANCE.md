# Falcon model code in Falcon Studio

`falcon_model/` is an inference-only copy of the model code in the training code
(`look2hear/` at the repository root):

| Studio file | Source |
|---|---|
| `models/Falcon_Contrastive.py`, `models/Falcon.py` | `FalconSpeakerContrastive` (Falcon, the paper model, `configs/Falcon.yml`) |
| `models/Speaker_Contrastive_v2.py`, `models/SISNRRepel.py` | `SpeakerContrastiveV2` (Falcon-Big, `configs/FalconBig.yml`) |
| `videomodels/frcnn_videomodel.py`, `resnet.py`, `shufflenetv2.py` | `FRCNNVideoModel`, the frozen lip-reading encoder |
| `layers/normalizations.py`, `layers/activations.py` | layer helpers |

Inference-only changes; no learned layers are removed:

- `timm`'s `DropPath` is replaced by `drop_path.py`. DropPath is the identity in evaluation mode.
- The base class keeps only the parameter-free constructor and the sample-rate accessor, so
  PyTorch Lightning is not needed.
- Unused imports (`ResNet1D`, a `sys.path` change) and package-level imports of training-only
  datasets, losses and metrics are removed.
- On Apple Metal (MPS), STFT and inverse STFT run on the CPU because complex tensors are not fully
  supported there. FFT parameters, windows and output length are unchanged; everything else runs
  on the selected device.

`falcon_runtime.py` builds each model from its configuration (deep-copied, because the original
`Separator` consumes `repeats` and `shared` with `dict.pop()`) and loads both `audio_model.*` and
`video_model.*` with `strict=True`, so a mismatched checkpoint fails instead of running a partial
network.

Mouth crops follow the validation transform of the training code: uint8 grayscale at 25 fps,
divided by 255, center crop 96×96 → 88×88, normalized with mean 0.421 and std 0.165. Audio is mono
16 kHz without peak or RMS normalization.

On the CPU, this copy and the training code give bit-identical output for `falcon.pt`
(11,724,385 parameters, maximum absolute difference 0).

The released files `falcon.pt` (Hugging Face `JusperLee/Falcon`) and `falcon-big.pt`
(`JusperLee/Falcon-Big`) were exported from the training checkpoints of epochs 714 and 715. They hold
`state_dict`, the architecture name and the epoch; the tensors are unchanged. Optimizer and
Lightning state and the training hyper-parameters are not included.

Inference uses rolling two-second windows. The model exposes no streaming state, so this is
windowed inference, not sample-by-sample streaming.
