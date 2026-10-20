"""Generate near.json index files for tr/cv/tt splits.

Each mix wav has a corresponding *_meta.json that records ALL near-speaker paths.
This script reads those meta files and writes near.json with format:
    [
      [[near_path_1, n_samples_1], [near_path_2, n_samples_2], ...],  # sample 0
      [[near_path_1, n_samples_1]],                                    # sample 1
      ...
    ]
indexed in the same order as mix.json.  Each sample entry is a list of
[path, n_samples] pairs, one per near speaker found in the meta JSON.

Usage:
    python DataPreProcess/generate_near_json.py \
        --data_root -- \
        --out_root   -- \
        --sample_rate 16000
"""

import argparse
import json
import os
import soundfile as sf
from tqdm import tqdm


def get_all_near_paths(meta_path: str):
    """Return a list of all near-speaker paths in the meta JSON.
    Each element corresponds to one component with type == 'near'."""
    with open(meta_path, "r") as f:
        meta = json.load(f)
    paths = []
    for comp in meta["components"]:
        if comp["type"] == "near":
            paths.append(comp["path"])
    return paths  # may be empty or have multiple entries


def process_split(mix_json_path: str, mix_dir: str, out_json_path: str, sample_rate: int):
    with open(mix_json_path, "r") as f:
        mix_infos = json.load(f)

    near_infos = []
    missing = 0

    for mix_path, mix_n_samples in tqdm(mix_infos, desc=os.path.basename(out_json_path)):
        basename = os.path.splitext(os.path.basename(mix_path))[0]
        meta_path = os.path.join(mix_dir, basename + "_meta.json")

        sample_nears = []  # list of [path, n_samples] for this mix

        if os.path.exists(meta_path):
            near_paths = get_all_near_paths(meta_path)
            for near_path in near_paths:
                if os.path.exists(near_path):
                    try:
                        info = sf.info(near_path)
                        sample_nears.append([near_path, info.frames])
                    except Exception:
                        pass  # skip unreadable file

        if len(sample_nears) == 0:
            # fallback: use the mix itself as a single dummy entry
            sample_nears = [[mix_path, mix_n_samples]]
            missing += 1

        near_infos.append(sample_nears)

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(near_infos, f, indent=2)

    near_counts = [len(s) for s in near_infos]
    max_k = max(near_counts)
    min_k = min(near_counts)
    avg_k = sum(near_counts) / len(near_counts)
    print(f"  Wrote {len(near_infos)} entries to {out_json_path}  "
          f"(missing/fallback: {missing}, near_per_sample min={min_k} max={max_k} avg={avg_k:.2f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        default="--",
                        help="Root dir containing tr/cv/tt sub-dirs with mix/ inside")
    parser.add_argument("--out_root", type=str,
                        default="--",
                        help="Root dir for output JSON files (same structure as input index)")
    parser.add_argument("--sample_rate", type=int, default=16000)
    args = parser.parse_args()

    for split in ["tr", "cv", "tt"]:
        mix_json = os.path.join(args.out_root, split, "mix.json")
        mix_dir  = os.path.join(args.data_root, split, "mix")
        out_json = os.path.join(args.out_root, split, "near.json")

        if not os.path.exists(mix_json):
            print(f"Skipping {split}: {mix_json} not found")
            continue

        process_split(mix_json, mix_dir, out_json, args.sample_rate)


if __name__ == "__main__":
    main()
