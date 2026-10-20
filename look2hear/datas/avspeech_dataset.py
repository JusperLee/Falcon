import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import hub
from torch.utils.data import Dataset, DataLoader
import random as random
import os
import shutil
import zipfile
import json
from typing import Dict, Iterable, List, Iterator
from .transform import get_preprocessing_pipelines
from torchvision import transforms
import matplotlib.pyplot as plt

def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)

class AVSpeechTrainDataset(Dataset):
    def __init__(
        self,
        json_dir: str = None,
        sample_rate: int = 8000,
        segment: float = 4.0,
        normalize_audio: bool = False,
        samples: int = None,
    ):
        super().__init__()
        if json_dir == None:
            raise ValueError("JSON DIR is None!")
        self.json_dir = json_dir
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        self.lipreading_preprocessing_func = get_preprocessing_pipelines()["train"]
        self.transform = transforms.Grayscale(num_output_channels=1)
        print("lipreading_preprocessing_func: ", self.lipreading_preprocessing_func)

        self.seg_len = int(segment * sample_rate)
        self.fps_len = int(segment * 25)

        self.test = self.seg_len is None
        mix_json = os.path.join(json_dir, "mix.json")
        sources_json = [
            os.path.join(json_dir, source + ".json") for source in ["target"]
        ]

        with open(mix_json, "r") as f:
            mix_infos = json.load(f)
        sources_infos = []
        for src_json in sources_json:
            with open(src_json, "r") as f:
                sources_infos.append(json.load(f))

        self.mix = []
        self.sources = []

        orig_len = len(mix_infos)
        drop_utt, drop_len = 0, 0
        print(len(mix_infos), len(sources_infos[0]))

        for i in range(len(mix_infos) - 1, -1, -1):
            if mix_infos[i][1] < self.seg_len:
                drop_utt = drop_utt + 1
                drop_len = drop_len + mix_infos[i][1]
                del mix_infos[i]
                for src_inf in sources_infos:
                    del src_inf[i]
            else:
                for src_inf in sources_infos:
                    self.mix.append(mix_infos[i])
                    self.sources.append(src_inf[i])

        print(
            "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                drop_utt, drop_len / sample_rate / 3600, orig_len, self.seg_len
            )
        )
        self.length = samples

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        self.EPS = 1e-8
        # Randomly select an index from the actual data
        mix_index = random.randint(0, len(self.mix) - 1)

        if self.test:
            rand_start = 0
            stop = None
        else:
            # Use mix_index instead of idx
            rand_start = random.randint(0, self.mix[mix_index][1] - self.seg_len)
            stop = rand_start + self.seg_len

        # Use mix_index for all data access
        mix_source, _ = sf.read(
            self.mix[mix_index][0], start=rand_start, stop=stop, dtype="float32"
        )
        source = sf.read(
            self.sources[mix_index][0], start=rand_start, stop=stop, dtype="float32"
        )[0]

        mouth_start = int((rand_start / 16000) * 25)

        source_mouth = self.lipreading_preprocessing_func(
            self.transform(torch.load(self.sources[mix_index][1], weights_only=True).permute(0,3,1,2)).squeeze(1).numpy()
        )[mouth_start : self.fps_len + mouth_start]

        source = torch.from_numpy(source).mean(-1)
        mixture = torch.from_numpy(mix_source).mean(-1)

        if self.normalize_audio:
            m_std = mixture.std(-1, keepdim=True)
            mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
            source = normalize_tensor_wav(source, eps=self.EPS, std=m_std)

        return mixture, source, source_mouth, self.mix[mix_index][0].split("/")[-1]


class AVSpeechEvalDataset(Dataset):
    def __init__(
        self,
        json_dir: str = None,
        sample_rate: int = 8000,
        normalize_audio: bool = False,
    ):
        super().__init__()
        if json_dir == None:
            raise ValueError("JSON DIR is None!")
        self.json_dir = json_dir
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        self.lipreading_preprocessing_func = get_preprocessing_pipelines()["val"]
        self.transform = transforms.Grayscale(num_output_channels=1)
        print("lipreading_preprocessing_func: ", self.lipreading_preprocessing_func)

        mix_json = os.path.join(json_dir, "mix.json")
        sources_json = [
            os.path.join(json_dir, source + ".json") for source in ["target"]
        ]

        with open(mix_json, "r") as f:
            mix_infos = json.load(f)
        sources_infos = []
        for src_json in sources_json:
            with open(src_json, "r") as f:
                sources_infos.append(json.load(f))

        self.mix = []
        self.sources = []
        
        orig_len = len(mix_infos)
        drop_utt, drop_len = 0, 0
        print(len(mix_infos), len(sources_infos[0]))
        
        for i in range(len(mix_infos)):
            for src_inf in sources_infos:
                self.mix.append(mix_infos[i])
                self.sources.append(src_inf[i])

        print(
            "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                drop_utt, drop_len / sample_rate / 3600, orig_len, None
            )
        )
        self.length = orig_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        self.EPS = 1e-8
        
        rand_start = 0
        stop = None

        mix_source, _ = sf.read(
            self.mix[idx][0], start=rand_start, stop=stop, dtype="float32"
        )
        source = sf.read(
            self.sources[idx][0], start=rand_start, stop=stop, dtype="float32"
        )[0]

        source_mouth = self.lipreading_preprocessing_func(
            self.transform(torch.load(self.sources[idx][1], weights_only=True).permute(0,3,1,2)).squeeze(1).numpy()
        )
        
        source = torch.from_numpy(source).mean(-1)
        mixture = torch.from_numpy(mix_source).mean(-1)

        if self.normalize_audio:
            m_std = mixture.std(-1, keepdim=True)
            mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
            source = normalize_tensor_wav(source, eps=self.EPS, std=m_std)

        # Read near-speaker count from _meta.json (same stem as mix wav)
        n_near = 0
        mix_path = self.mix[idx][0]
        meta_path = mix_path.replace(".wav", "_meta.json")
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            n_near = sum(1 for c in meta.get("components", []) if c.get("type") == "near")
        except Exception:
            pass

        return mixture, source, source_mouth, self.mix[idx][0].split("/")[-1], n_near

class AVSpeechDataModule(object):
    def __init__(
        self,
        train_dir: str,
        valid_dir: str,
        test_dir: str,
        n_src: int = 2,
        sample_rate: int = 8000,
        segment: float = 4.0,
        normalize_audio: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        train_samples: int = 10000,
    ) -> None:
        super().__init__()
        if train_dir == None or valid_dir == None or test_dir == None:
            raise ValueError("JSON DIR is None!")
        if n_src not in [1, 2]:
            raise ValueError("{} is not in [1, 2]".format(n_src))

        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        self.n_src = n_src
        self.sample_rate = sample_rate
        self.segment = segment
        self.normalize_audio = normalize_audio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.train_samples = train_samples

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

    def setup(self) -> None:
        self.data_train = AVSpeechTrainDataset(
            json_dir=self.train_dir,
            sample_rate=self.sample_rate,
            segment=self.segment,
            normalize_audio=self.normalize_audio,
            samples=self.train_samples,
        )
        self.data_val = AVSpeechEvalDataset(
            json_dir=self.valid_dir,
            sample_rate=self.sample_rate,
            normalize_audio=self.normalize_audio,
        )
        self.data_test = AVSpeechEvalDataset(
            json_dir=self.test_dir,
            sample_rate=self.sample_rate,
            normalize_audio=self.normalize_audio,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            shuffle=False,
            #batch_size=self.batch_size,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            shuffle=False,
            #batch_size=self.batch_size,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    @property
    def make_loader(self):
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()

    @property
    def make_sets(self):
        return self.data_train, self.data_val, self.data_test

# if __name__ == "__main__":
#     from tqdm import tqdm
#     val_set = AVSpeechTrainDataset(
#         "${DATA_ROOT}/av-tiger-lrs3/DataPreProcess/LRS3_multi/tr",
#         n_src=2,
#         sample_rate=16000,
#         # segment=conf["data"]["segment"],
#         segment=2.0,
#         normalize_audio=False
#     )
#     for idx in tqdm(range(len(val_set))):
#         mixture, sources, sources_mouths, _ = val_set[idx]
#         import pdb; pdb.set_trace()
