"""
audio_visual_test_contrastive_v2.py – Test script for SpeakerContrastiveV2 models.

Key fix: bypasses best_model.pth / from_pretrain entirely.
Loads audio_model + video_model weights directly from a Lightning .ckpt file,
which is exactly how validation works during training.

Background:
    Separator.__init__ calls dict.pop("repeats") and dict.pop("shared"),
    mutating the original config dict in-place.  When audio_train_nass_v2.py
    later saves the config with yaml.safe_dump, those keys are GONE.
    from_pretrain then builds a model with an empty Separator (0 layers),
    and load_state_dict(strict=False) silently ignores 97% of the checkpoint.
    Loading from .ckpt avoids this entirely.

Usage:
    python audio_visual_test_contrastive_v2.py \\
        --conf_dir Experiments/checkpoint/<exp_name>/conf.yml \\
        --gpus 0

    # multi-GPU
    python audio_visual_test_contrastive_v2.py \\
        --conf_dir Experiments/checkpoint/<exp_name>/conf.yml \\
        --gpus 0,1,2,3
"""

import os
import random
import argparse
from typing import Union
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
import yaml
import json
import warnings
import torchaudio

import torch
import torch.multiprocessing as mp

import look2hear.models
import look2hear.videomodels
import look2hear.datas
from look2hear.datas.avspeech_dataset_near import AVSpeechDataModuleWithNear
from look2hear.metrics import ALLMetricsTracker
from look2hear.utils import tensors_to_device, RichProgressBarTheme, MyMetricsTextColumn, BatchesProcessedColumn

from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

warnings.filterwarnings("ignore")

# ==========================================
# 1. Argument Parsing
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--conf_dir",
                    default="Experiments/checkpoint/NASSv2_lrs3_complex_nonmoving_img89/conf.yml",
                    help="Full path to save best validation model")
parser.add_argument("--ckpt_path",
                    default="",
                    help="Checkpoint to evaluate (local file or hf://<org>/<repo>/<file>); default: best in exp_dir")
parser.add_argument("--gpus",
                    type=str,
                    default="0",
                    help="Comma-separated list of GPU ids to use, e.g., '0,1,2,3'")

compute_metrics = ["si_sdr", "sdr"]


def _find_best_ckpt(exp_dir):
    """Find the best checkpoint in the experiment directory.

    Priority: best_k_models.json (lowest val_loss) → last.ckpt → any epoch=*.ckpt
    """
    # 1) Find best from best_k_models.json
    best_k_path = os.path.join(exp_dir, "best_k_models.json")
    if os.path.exists(best_k_path):
        with open(best_k_path) as f:
            best_k = json.load(f)
        if best_k:
            # val_loss is negative; lower is better
            best_path = min(best_k, key=best_k.get)
            if os.path.exists(best_path):
                print(f"   Using best ckpt: {os.path.basename(best_path)} (val_loss={best_k[best_path]:.4f})")
                return best_path

    # 2) last.ckpt
    last_path = os.path.join(exp_dir, "last.ckpt")
    if os.path.exists(last_path):
        print(f"   Using last.ckpt")
        return last_path

    # 3) Any epoch ckpt (pick the latest)
    import glob
    ckpts = sorted(glob.glob(os.path.join(exp_dir, "epoch=*.ckpt")))
    if ckpts:
        print(f"   Using latest ckpt: {os.path.basename(ckpts[-1])}")
        return ckpts[-1]

    raise FileNotFoundError(f"No checkpoint found in {exp_dir}")


def _load_models_from_ckpt(ckpt_path, config, device):
    """Load audio_model and video_model from a Lightning .ckpt file.

    This mirrors exactly how Lightning restores the system during
    training-time validation — no from_pretrain, no config re-parsing.
    """
    import copy

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    system_state = ckpt["state_dict"]

    # Rebuild model from original config YAML (not the pop-polluted conf.yml)
    # We don't need config to build the model; we infer from ckpt state_dict
    # Strategy: build model skeleton from config, then load full weights from ckpt

    # --- Audio model ---
    # Use deepcopy to prevent Separator.__init__'s dict.pop from polluting config
    audionet_cfg = copy.deepcopy(config["train_conf"]["audionet"]["audionet_config"])
    model = getattr(look2hear.models, config["train_conf"]["audionet"]["audionet_name"])(
        sample_rate=config["train_conf"]["datamodule"]["data_config"]["sample_rate"],
        **audionet_cfg,
    )

    # Extract audio_model weights from ckpt (strip "audio_model." prefix)
    audio_state = {k.replace("audio_model.", ""): v
                   for k, v in system_state.items()
                   if k.startswith("audio_model.")}

    missing, unexpected = model.load_state_dict(audio_state, strict=False)
    if missing:
        print(f"   ⚠ audio_model missing keys: {len(missing)}")
    if unexpected:
        print(f"   ⚠ audio_model unexpected keys: {len(unexpected)}")
    print(f"   ✓ audio_model loaded: {len(audio_state)} keys from ckpt")

    # --- Video model ---
    videonet_cfg = copy.deepcopy(config["train_conf"]["videonet"]["videonet_config"])
    # Skip pretrain loading (avoid overwriting trained weights)
    pretrain_path = videonet_cfg.pop("pretrain", None)
    video_model = getattr(look2hear.videomodels, config["train_conf"]["videonet"]["videonet_name"])(
        **videonet_cfg, pretrain=None,
    )

    # Extract video_model weights from ckpt
    video_state = {k.replace("video_model.", ""): v
                   for k, v in system_state.items()
                   if k.startswith("video_model.")}

    if video_state:
        video_model.load_state_dict(video_state, strict=True)
        print(f"   ✓ video_model loaded: {len(video_state)} keys from ckpt")
    elif pretrain_path:
        # Fallback: if ckpt has no video_model (unlikely), use pretrain
        video_model = getattr(look2hear.videomodels, config["train_conf"]["videonet"]["videonet_name"])(
            **config["train_conf"]["videonet"]["videonet_config"],
        )
        print(f"   ✓ video_model loaded from pretrain: {pretrain_path}")

    model.to(device).eval()
    video_model.to(device).eval()

    return model, video_model


# ==========================================
# 2. Multi-process Worker Function
# ==========================================
def process_worker(rank, world_size, config):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Initialize DataLoader / Dataset
    data_cfg = {
        k: v for k, v in config["train_conf"]["datamodule"]["data_config"].items()
        if k not in ("use_near", "train_samples")
    }
    datamodule = AVSpeechDataModuleWithNear(**data_cfg, use_near=False)
    datamodule.setup()
    _, _, test_set = datamodule.make_sets

    # Split test set across processes (sliced)
    indices = list(range(len(test_set)))
    local_indices = indices[rank::world_size]

    # Model path setup
    exp_dir = os.path.join(
        os.getcwd(), "Experiments", "checkpoint", config["train_conf"]["exp"]["exp_name"]
    )

    # Find best ckpt and load audio_model + video_model from it
    ckpt_path = config.get("ckpt_path") or _find_best_ckpt(exp_dir)
    model, video_model = _load_models_from_ckpt(ckpt_path, config, device)

    # Each process uses its own metrics file to avoid write conflicts
    ex_save_dir = os.path.join(exp_dir, "results/")
    os.makedirs(ex_save_dir, exist_ok=True)
    metrics_file = os.path.join(ex_save_dir, f"metrics_rank{rank}.csv")
    metrics = ALLMetricsTracker(save_file=metrics_file)

    # Only show progress bar on Rank 0 to avoid multi-process terminal clutter
    if rank == 0:
        metricscolumn = MyMetricsTextColumn(style=RichProgressBarTheme.metrics)
        progress = Progress(
            TextColumn("[bold blue]Testing (Rank 0)", justify="right"),
            BarColumn(bar_width=None),
            "•",
            BatchesProcessedColumn(style=RichProgressBarTheme.batch_progress),
            "•",
            TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
            "•",
            metricscolumn
        )

        with progress:
            with torch.no_grad():
                for idx_count, idx in enumerate(progress.track(local_indices)):
                    mix, sources, mouth, key, _ = tensors_to_device(test_set[idx], device=device)
                    mouth_emb = video_model(torch.from_numpy(mouth[None, None]).float().to(device))
                    est_sources = model(mix[None], mouth_emb)

                    try:
                        metrics(mix=mix,
                                clean=sources.unsqueeze(0),
                                estimate=est_sources.squeeze(0),
                                key=key)
                    except ValueError as e:
                        if "all zeros" in str(e).lower() or "nan" in str(e).lower():
                            pass
                        else:
                            raise e

                    if idx_count % 50 == 0:
                        metricscolumn.update(metrics.get_mean())
    else:
        # Non-Rank-0 processes compute silently
        with torch.no_grad():
            for idx in local_indices:
                mix, sources, mouth, key, _ = tensors_to_device(test_set[idx], device=device)
                mouth_emb = video_model(torch.from_numpy(mouth[None, None]).float().to(device))
                est_sources = model(mix[None], mouth_emb)

                try:
                    metrics(mix=mix,
                            clean=sources.unsqueeze(0),
                            estimate=est_sources.squeeze(0),
                            key=key)
                except ValueError as e:
                    if "all zeros" in str(e).lower() or "nan" in str(e).lower():
                        pass
                    else:
                        raise e

    metrics.final()

# ==========================================
# 3. Main Function and Result Merging
# ==========================================
def main(config):
    # Get GPU count from the passed string and set CUDA_VISIBLE_DEVICES
    os.environ['CUDA_VISIBLE_DEVICES'] = config["gpus"]
    gpu_list = config["gpus"].split(",")
    world_size = len(gpu_list)

    print(f"🚀 [Info] Starting evaluation on {world_size} GPU(s): {config['gpus']}")

    # Spawn multi-process
    if world_size > 1:
        mp.spawn(process_worker, args=(world_size, config), nprocs=world_size, join=True)
    else:
        process_worker(0, 1, config)

    # Merge metrics.csv from all processes
    exp_dir = os.path.join(
        os.getcwd(), "Experiments", "checkpoint", config["train_conf"]["exp"]["exp_name"]
    )
    ex_save_dir = os.path.join(exp_dir, "results/")
    all_metrics = []

    for i in range(world_size):
        metric_file = os.path.join(ex_save_dir, f"metrics_rank{i}.csv")
        if os.path.exists(metric_file):
            all_metrics.append(pd.read_csv(metric_file))

    if all_metrics:
        combined_metrics = pd.concat(all_metrics, ignore_index=True)
        final_csv_path = os.path.join(ex_save_dir, "metrics_combined.csv")
        combined_metrics.to_csv(final_csv_path, index=False)
        print(f"\n✅ [Success] Combined metrics saved to: {final_csv_path}")

        # Print global final average
        numeric_cols = combined_metrics.select_dtypes(include=[np.number])
        print("-" * 30)
        print("Final Average Metrics (All GPUs):")
        print(numeric_cols.mean())
        print("-" * 30)

        # Clean up per-rank shard files
        for i in range(world_size):
            metric_file = os.path.join(ex_save_dir, f"metrics_rank{i}.csv")
            if os.path.exists(metric_file):
                os.remove(metric_file)

if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    with open(args.conf_dir, "rb") as f:
        train_conf = yaml.safe_load(f)
    arg_dic["train_conf"] = train_conf
    if args.ckpt_path.startswith("hf://"):
        # Download once before spawning workers; cached in ~/.cache/huggingface.
        from huggingface_hub import hf_hub_download
        repo_id, _, filename = args.ckpt_path[len("hf://"):].rpartition("/")
        arg_dic["ckpt_path"] = hf_hub_download(repo_id=repo_id, filename=filename)
        try:  # The Hub counts downloads by config.yaml, not by the weights file; best effort.
            from huggingface_hub import get_hf_file_metadata, hf_hub_url
            get_hf_file_metadata(hf_hub_url(repo_id, "config.yaml"))
        except Exception:
            pass

    main(arg_dic)
