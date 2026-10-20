import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from typing import List
from torch.nn.modules.batchnorm import _BatchNorm
from collections.abc import Iterable

EPS = 1e-5

def norm(x, dims: List[int], EPS: float = 1e-8):
    mean = x.mean(dim=dims, keepdim=True)
    var2 = torch.var(x, dim=dims, keepdim=True, unbiased=False)
    value = (x - mean) / torch.sqrt(var2 + EPS)
    return value


def glob_norm(x, ESP: float = 1e-8):
    dims: List[int] = torch.arange(1, len(x.shape)).tolist()
    return norm(x, dims, ESP)


class MLayerNorm(nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(channel_size), requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """Assumes input of size `[batch, chanel, *]`."""
        return (self.gamma * normed_x.transpose(1, -1) + self.beta).transpose(1, -1)

    def forward(self, x, EPS: float = 1e-8):
        pass


class GlobalLN(MLayerNorm):
    def forward(self, x, EPS: float = 1e-8):
        value = glob_norm(x, EPS)
        return self.apply_gain_and_bias(value)


class ChannelLN(MLayerNorm):
    def forward(self, x, EPS: float = 1e-8):
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        return self.apply_gain_and_bias((x - mean) / (var + EPS).sqrt())


class CumulateLN(MLayerNorm):
    def forward(self, x, EPS: float = 1e-8):
        batch, channels, time = x.size()
        cum_sum = torch.cumsum(x.sum(1, keepdim=True), dim=1)
        cum_pow_sum = torch.cumsum(x.pow(2).sum(1, keepdim=True), dim=1)
        cnt = torch.arange(
            start=channels, end=channels * (time + 1), step=channels, dtype=x.dtype, device=x.device
        ).view(1, 1, -1)
        cum_mean = cum_sum / cnt
        cum_var = (cum_pow_sum / cnt) - cum_mean.pow(2)
        return self.apply_gain_and_bias((x - cum_mean) / (cum_var + EPS).sqrt())


class BatchNorm(_BatchNorm):
    """Wrapper class for pytorch BatchNorm1D and BatchNorm2D"""

    def _check_input_dim(self, input):
        if input.dim() < 2 or input.dim() > 4:
            raise ValueError(
                "expected 4D or 3D input (got {}D input)".format(input.dim())
            )


class CumulativeLayerNorm(nn.LayerNorm):
    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine
        )

    def forward(self, x):
        # x: N x C x L
        # N x L x C
        x = torch.transpose(x, 1, 2)
        # N x L x C == only channel norm
        x = super().forward(x)
        # N x C x L
        x = torch.transpose(x, 1, 2)
        return x


class CumulateLN(nn.Module):
    def __init__(self, dimension, eps=1e-8, trainable=True):
        super(CumulateLN, self).__init__()

        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            self.gain = Variable(torch.ones(1, dimension, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, dimension, 1), requires_grad=False)

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step

        batch_size = input.size(0)
        channel = input.size(1)
        time_step = input.size(2)

        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T

        entry_cnt = np.arange(channel, channel * (time_step + 1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(
            2
        )  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T

        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)

        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(
            x.type()
        )

class GlobalLayerNorm(nn.Module):
    def __init__(self, num_channels: int = 1, eps: float = EPS):
        super(GlobalLayerNorm, self).__init__()
        self.num_channels = num_channels
        self.eps = eps

        self.norm = nn.GroupNorm(num_groups=1, num_channels=self.num_channels, eps=self.eps)

    def forward(self, x: torch.Tensor):
        return self.norm(x)


class LayerNormalization4D(nn.Module):
    def __init__(self, input_dimension: Iterable, eps: float = EPS):
        super(LayerNormalization4D, self).__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]

        self.dim = (1, 3) if param_size[-1] > 1 else (1,)
        self.gamma = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x: torch.Tensor):
        mu_ = x.mean(dim=self.dim, keepdim=True)
        std_ = torch.sqrt(x.var(dim=self.dim, unbiased=False, keepdim=True) + self.eps)
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat
    
class CumulateLN4D(nn.Module):
    def __init__(self, dimension, eps=1e-8, trainable=True):
        super(CumulateLN4D, self).__init__()
        self.eps = eps
        if trainable:
            # gain and bias dimensions match input, controlling (C, W) respectively
            self.gain = nn.Parameter(torch.ones(1, dimension, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1, 1))
        else:
            self.gain = torch.ones(1, dimension, 1, 1, requires_grad=False)
            self.bias = torch.zeros(1, dimension, 1, 1, requires_grad=False)

    def forward(self, input):
        # input size: (Batch, Channel, Height, Width)
        B, C, H, W = input.size()
        
        # Cumulative normalization along the Height dimension (H corresponds to time steps)
        step_sum = input.sum(dim=1, keepdim=True)  # (B, 1, H, W)
        step_pow_sum = input.pow(2).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        
        cum_sum = torch.cumsum(step_sum, dim=2)  # (B, 1, H, W)
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=2)  # (B, 1, H, W)

        # Cumulative mean and variance for each H step
        entry_cnt = torch.arange(C, C * (H + 1), C, dtype=input.dtype, device=input.device)
        entry_cnt = entry_cnt.view(1, 1, -1, 1).expand_as(cum_sum)  # (B, 1, H, W)

        cum_mean = cum_sum / entry_cnt  # (B, 1, H, W)
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(2)  # (B, 1, H, W)
        cum_std = (cum_var + self.eps).sqrt()  # (B, 1, H, W)

        # Broadcast cumulative mean and std to match input shape
        cum_mean = cum_mean.expand(-1, C, -1, -1)
        cum_std = cum_std.expand(-1, C, -1, -1)

        # Normalize
        x = (input - cum_mean) / cum_std
        return x * self.gain + self.bias

class LayerNormalizationHW(nn.Module):
    def __init__(self, num_channels, eps=1e-8):
        super(LayerNormalizationHW, self).__init__()
        # GlobalLayerNorm for W dimension
        self.global_layer_norm = GlobalLayerNorm(num_channels=num_channels, eps=eps)
        # CumulateLN for H dimension
        self.cumulate_ln = CumulateLN4D(dimension=num_channels, eps=eps)

    def forward(self, x):
        # Input shape: (B, C, H, W)

        # GlobalLayerNorm on W dimension
        x = self.global_layer_norm(x)  # Normalizes over C and W dimensions

        # CumulateLN on H dimension
        x = self.cumulate_ln(x)  # (B, C, H, W) cumulative normalization on H
        
        return x

# Aliases.
gLN = GlobalLayerNorm
cLN = CumulateLN
LN = CumulativeLayerNorm
bN = BatchNorm
LN4d = LayerNormalization4D
cLNhw = LayerNormalizationHW

def get(identifier):
    if identifier is None:
        return nn.Identity
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        if hasattr(nn, identifier):
            cls = getattr(nn, identifier)
        else:
            cls = globals().get(identifier)
        if cls is None:
            raise ValueError("Could not interpret normalization identifier: " + str(identifier))
        return cls
    else:
        raise ValueError("Could not interpret normalization identifier: " + str(identifier))
