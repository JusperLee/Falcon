"""Inference-only BaseModel API; no Lightning or training dependencies."""
from torch import nn


class BaseModel(nn.Module):
    def __init__(self, sample_rate, in_chan=1):
        super().__init__()
        self._sample_rate = sample_rate
        self._in_chan = in_chan

    def sample_rate(self):
        return self._sample_rate
