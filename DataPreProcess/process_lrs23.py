import argparse
import json
import os
import soundfile as sf
from tqdm import tqdm


def preprocess_one_dir(in_audio_dir, out_dir, out_filename, data_type):
    """Create .json file for one condition."""
    file_infos = []
    in_dir = os.path.abspath(in_audio_dir)
    wav_list = os.listdir(in_dir)
    wav_list.sort()
    for wav_file in tqdm(wav_list):
        if not wav_file.endswith(".wav"):
            continue
        wav_path = os.path.join(in_dir, wav_file)
        samples = sf.SoundFile(wav_path)
        if out_filename == "mix":
            file_infos.append((wav_path, len(samples)))
        else:
            file_infos.append(
                (
                    wav_path,
                    wav_path.replace(out_filename, "mouths").replace(".wav", ".pt"),
                    len(samples),
                )
            )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, out_filename + ".json"), "w") as f:
        json.dump(file_infos, f, indent=4)


def get_all_near_paths(meta_path: str):
    """Return a list of all near-speaker paths in the meta JSON."""
    with open(meta_path, "r") as f:
        meta = json.load(f)
    paths = []
    for comp in meta["components"]:
        if comp["type"] == "near":
            paths.append(comp["path"])
    return paths


def generate_near_json(mix_json_path: str, mix_dir: str, out_json_path: str):
    with open(mix_json_path, "r") as f:
        mix_infos = json.load(f)

    near_infos = []
    missing = 0

    for mix_path, mix_n_samples in tqdm(mix_infos, desc=os.path.basename(out_json_path)):
        basename = os.path.splitext(os.path.basename(mix_path))[0]
        meta_path = os.path.join(mix_dir, basename + "_meta.json")

        sample_nears = []

        if os.path.exists(meta_path):
            near_paths = get_all_near_paths(meta_path)
            for near_path in near_paths:
                if os.path.exists(near_path):
                    try:
                        info = sf.info(near_path)
                        sample_nears.append([near_path, info.frames])
                    except Exception:
                        pass

        if len(sample_nears) == 0:
            sample_nears = [[mix_path, mix_n_samples]]
            missing += 1

        near_infos.append(sample_nears)

    with open(out_json_path, "w") as f:
        json.dump(near_infos, f, indent=2)

    near_counts = [len(s) for s in near_infos]
    print(f"  Wrote {len(near_infos)} entries to {out_json_path}  "
          f"(missing/fallback: {missing}, near_per_sample min={min(near_counts)} "
          f"max={max(near_counts)} avg={sum(near_counts)/len(near_counts):.2f})")


def preprocess(inp_args):
    """Create .json files for all conditions."""
    speaker_list = ["mix", "target"]
    for data_type in ["tr", "cv", "tt"]:
        for spk in speaker_list:
            preprocess_one_dir(
                os.path.join(inp_args.in_audio_dir, data_type, spk),
                os.path.join(inp_args.out_dir, data_type),
                spk,
                data_type,
            )

        # Generate near.json after mix.json is ready
        mix_json = os.path.join(inp_args.out_dir, data_type, "mix.json")
        mix_dir  = os.path.join(inp_args.in_audio_dir, data_type, "mix")
        out_json = os.path.join(inp_args.out_dir, data_type, "near.json")

        if not os.path.exists(mix_json):
            print(f"Skipping near.json for {data_type}: {mix_json} not found")
            continue

        generate_near_json(mix_json, mix_dir, out_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("WHAM data preprocessing")
    parser.add_argument(
        "--in_audio_dir",
        type=str,
        default="--",
        help="Directory path of audio including tr, cv and tt",
    )
    parser.add_argument(
        "--out_dir", type=str, default="LRS3_multi", help="Directory path to put output files"
    )
    args = parser.parse_args()
    print(args)
    preprocess(args)
