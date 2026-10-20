import torch
from torch.nn.modules.loss import _Loss
import torch.nn as nn
from dataclasses import dataclass, field, fields
from math import ceil

def l2norm(mat, keepdim=False):
    return torch.norm(mat, dim=-1, keepdim=keepdim)

def l1norm(mat, keepdim=False):
    return torch.norm(mat, dim=-1, keepdim=keepdim, p=1)

class STFTBase(torch.nn.Module):
    """
    Base layer for (i)STFT
    NOTE:
        1) Recommend sqrt_hann window with 2**N frame length, because it 
           could achieve perfect reconstruction after overlap-add
        2) Now haven't consider padding problems yet
    """
    def __init__(self,
                device,
                frame_length: int,
                frame_shift: int,
                window: str):
        super(STFTBase, self).__init__()  # Initialize the torch.nn.Module base class

        self.frame_length=frame_length
        self.frame_shift=frame_shift
        self.window=window
        K = self._init_kernel(self.frame_length, self.frame_shift).to(device)
        self.K = torch.nn.Parameter(K, requires_grad=False).to(device)
        self.num_bins = self.K.shape[0] // 2
    
    def _init_kernel(self, frame_len, frame_hop):
        # FFT points
        N = frame_len
        # window
        if self.window == 'hann':
            W = torch.hann_window(frame_len)
        if N//4 == frame_hop:
            const = (2/3)**0.5       
            W = const*W
        elif N//2 == frame_hop:
            W = W**0.5
        S = 0.5 * (N * N / frame_hop)**0.5
        
        # Updated FFT calculation for efficiency
        K = torch.fft.rfft(torch.eye(N) / S, dim=1)[:frame_len]
        K = torch.stack((torch.real(K), torch.imag(K)), dim=2)
        K = torch.transpose(K, 0, 2) * W # 2 x N/2+1 x F
        K = torch.reshape(K, (N + 2, 1, frame_len)) # N+2 x 1 x F
        return K

    def extra_repr(self):
        return (f"window={self.window}, stride={self.frame_shift}, " +
                f"kernel_size={self.K.shape[0]}x{self.K.shape[2]}")

class STFT(STFTBase):
    """
    Short-time Fourier Transform as a Layer
    """

    def forward(self, x, cplx=False):
        """
        Accept (single or multiple channel) raw waveform and output magnitude and phase
        args
            x: input signal, N x C x S or N x S
        return
            m: magnitude, N x C x F x T or N x F x T
            p: phase, N x C x F x T or N x F x T
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError(
                "{} expect 2D/3D tensor, but got {:d}D signal".format(
                    self.__name__, x.dim()))
        # if N x S, reshape N x 1 x S
        N_frame = ceil(x.shape[-1] / self.frame_shift)
        len_padded = N_frame * self.frame_shift
        if x.dim() == 2:
            
            x = torch.cat((x, torch.zeros(x.shape[0], len_padded-x.shape[-1], device=x.device)), dim=-1)
            x = torch.unsqueeze(x, 1)
            # N x 2F x T
            c = torch.nn.functional.conv1d(x, self.K.to(x.device), stride=self.frame_shift, padding=0)
            # N x F x T
            r, i = torch.chunk(c, 2, dim=1)
        else:        
            x = torch.cat((x, torch.zeros(x.shape[0], x.shape[1], len_padded-x.shape[-1], device=x.device)), dim=-1)
            N, C, S = x.shape
            x = x.reshape(N * C, 1, S)
            # NC x 2F x T
            c = torch.nn.functional.conv1d(x, self.K.to(x.device), stride=self.frame_shift, padding=0)
            # N x C x 2F x T
            c = c.reshape(N, C, -1, c.shape[-1])
            # N x C x F x T
            r, i = torch.chunk(c, 2, dim=2)

        if cplx:
            return r, i
        m = (r**2 + i**2 + 1.0e-10)**0.5
        p = torch.atan2(i, r)
        return m, p
    
class PIT_SISNR_mag_loss(_Loss):
    def __init__(self, frame_length, frame_shift, window, num_stages, scale_inv, mel_opt, device):
        super().__init__()
        self.frame_length=frame_length
        self.frame_shift=frame_shift
        self.window=window
        self.num_stages=num_stages
        self.scale_inv=scale_inv
        self.mel_opt=mel_opt
        self.stft = [STFT(device, self.frame_length, self.frame_shift, self.window) for _ in range(self.num_stages)]
    
    def forward(self,mix,src,idx):
        eps=1e-12
        mix_zm = mix - torch.mean(mix, dim=-1, keepdim=True)
        src_zm = src - torch.mean(src, dim=-1, keepdim=True)
        if self.scale_inv:
            scale = torch.sum(mix_zm * src_zm, dim=-1, keepdim=True) / (l2norm(src_zm, keepdim=True)**2 + eps)
            src_zm = torch.clamp(scale, min=1e-2) * src_zm
        mix_zm = self.stft[idx](mix_zm)[0]
        src_zm = self.stft[idx](src_zm)[0]
        if self.mel_opt:    
            mix_zm = self.mel_fb(mix_zm)
            src_zm = self.mel_fb(src_zm)
        utt_loss = -20 * torch.log10(eps + l2norm(l2norm((src_zm))) / (l2norm(l2norm(mix_zm - src_zm)) + eps))
        # print("utt_mag:",utt_loss.shape)
        return utt_loss
    
    
class PIT_SISNR_time_loss(_Loss):
    def __init__(self,scale_inv):
        super().__init__()
        self.scale_inv=scale_inv
    
    def forward(self,mix,src):
        eps=1e-12
        mix_zm = mix - torch.mean(input=mix, dim=-1, keepdim=True)
        src_zm = src - torch.mean(input=src, dim=-1, keepdim=True)
        src_zm_scale=src_zm
        if self.scale_inv:
            scale_factor = torch.sum(mix_zm * src_zm, dim=-1, keepdim=True) / (l2norm(src_zm, keepdim=True)**2 + eps)
            src_zm_scale = scale_factor * src_zm
        utt_loss = - 20 * torch.log10(eps + l2norm(src_zm_scale) / (l2norm(mix_zm - src_zm_scale) + eps))
        utt_loss = torch.clamp(utt_loss, min=-30)
        # print("utt_time:",utt_loss.shape)
        return utt_loss
    

class MultistageLoss(_Loss):
    def __init__(self, frame_length, frame_shift, window, num_stages, scale_inv, mel_opt, device):
        super().__init__()
        self.PIT_SISNR_mag_loss=PIT_SISNR_mag_loss(frame_length, frame_shift, window, num_stages, scale_inv, mel_opt, device)
        self.PIT_SISNR_time_loss=PIT_SISNR_time_loss(scale_inv)

    def forward(self, estim_src, estim_src_bn, src, epoch=None):
        B=estim_src.shape[0]
        loss=[]
        alpha = 0.4 * 0.8**(1+(epoch-81)//5) if epoch > 80 else 0.4
        for i in range(B):
            cur_loss_s_bn = []
            for idx, estim_src_value in enumerate(estim_src_bn[i]):
                cur_loss_s_bn.append(self.PIT_SISNR_mag_loss(mix=estim_src_value.unsqueeze(0), src=src[i].unsqueeze(0), idx=idx))
            cur_loss_s = self.PIT_SISNR_time_loss(mix=estim_src[i].unsqueeze(0), src=src[i].unsqueeze(0))

            cur_loss = (1-alpha) * cur_loss_s + alpha * (sum(cur_loss_s_bn) / len(cur_loss_s_bn))
            loss.append(cur_loss)
        # print(cur_loss.shape)
        return sum(loss)/len(loss)

class MultistageLoss_val(_Loss):
    def __init__(self, frame_length, frame_shift, window, num_stages, scale_inv, mel_opt, device):
        super().__init__()
        self.PIT_SISNR_mag_loss=PIT_SISNR_mag_loss(frame_length, frame_shift, window, num_stages, scale_inv, mel_opt, device)
        self.PIT_SISNR_time_loss=PIT_SISNR_time_loss(scale_inv)

    def forward(self, estim_src, estim_src_bn, src, epoch=None):
        cur_loss_s_bn = 0
        cur_loss_s_bn = []
        for idx, estim_src_value in enumerate(estim_src_bn):
            cur_loss_s_bn.append(self.PIT_SISNR_mag_loss(mix=estim_src_value, src=src, idx=idx))
        cur_loss_s = self.PIT_SISNR_time_loss(mix=estim_src, src=src)

        return cur_loss_s.mean()
    
class PairwiseNegSDR(_Loss):
    def __init__(self, sdr_type, zero_mean=True, take_log=True, EPS=1e-8):
        super().__init__()
        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = EPS

    def forward(self, ests, targets):
        if targets.size() != ests.size() or targets.ndim != 3:
            raise TypeError(
                f"Inputs must be of shape [batch, n_src, time], got {targets.size()} and {ests.size()} instead"
            )
        assert targets.size() == ests.size()
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=2, keepdim=True)
            mean_estimate = torch.mean(ests, dim=2, keepdim=True)
            targets = targets - mean_source
            ests = ests - mean_estimate
        # Step 2. Pair-wise SI-SDR. (Reshape to use broadcast)
        s_target = torch.unsqueeze(targets, dim=1)
        s_estimate = torch.unsqueeze(ests, dim=2)
        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, n_src, n_src, 1]
            pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)
            # [batch, 1, n_src, 1]
            s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + self.EPS
            # [batch, n_src, n_src, time]
            pair_wise_proj = pair_wise_dot * s_target / s_target_energy
        else:
            # [batch, n_src, n_src, time]
            pair_wise_proj = s_target.repeat(1, s_target.shape[2], 1, 1)
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = s_estimate - s_target
        else:
            e_noise = s_estimate - pair_wise_proj
        # [batch, n_src, n_src]
        pair_wise_sdr = torch.sum(pair_wise_proj ** 2, dim=3) / (
            torch.sum(e_noise ** 2, dim=3) + self.EPS
        )
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + self.EPS)
        return -pair_wise_sdr


class SingleSrcNegSDR(_Loss):
    def __init__(
        self, sdr_type, zero_mean=True, take_log=True, reduction="none", EPS=1e-8
    ):
        assert reduction != "sum", NotImplementedError
        super().__init__(reduction=reduction)

        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = 1e-8

    def forward(self, ests, targets):
        if targets.size() != ests.size() or targets.ndim != 2:
            raise TypeError(
                f"Inputs must be of shape [batch, time], got {targets.size()} and {ests.size()} instead"
            )
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=1, keepdim=True)
            mean_estimate = torch.mean(ests, dim=1, keepdim=True)
            targets = targets - mean_source
            ests = ests - mean_estimate
        # Step 2. Pair-wise SI-SDR.
        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, 1]
            dot = torch.sum(ests * targets, dim=1, keepdim=True)
            # [batch, 1]
            s_target_energy = torch.sum(targets ** 2, dim=1, keepdim=True) + self.EPS
            # [batch, time]
            scaled_target = dot * targets / s_target_energy
        else:
            # [batch, time]
            scaled_target = targets
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = ests - targets
        else:
            e_noise = ests - scaled_target
        # [batch]
        losses = torch.sum(scaled_target ** 2, dim=1) / (
            torch.sum(e_noise ** 2, dim=1) + self.EPS
        )
        if self.take_log:
            losses = 10 * torch.log10(losses + self.EPS)
        losses = losses.mean() if self.reduction == "mean" else losses
        return -losses


class MultiSrcNegSDR(_Loss):
    def __init__(self, sdr_type, zero_mean=True, take_log=True, EPS=1e-8):
        super().__init__()

        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = 1e-8

    def forward(self, ests, targets):
        if targets.size() != ests.size() or targets.ndim != 3:
            raise TypeError(
                f"Inputs must be of shape [batch, n_src, time], got {targets.size()} and {ests.size()} instead"
            )
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=2, keepdim=True)
            mean_est = torch.mean(ests, dim=2, keepdim=True)
            targets = targets - mean_source
            ests = ests - mean_est
        # Step 2. Pair-wise SI-SDR.
        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, n_src]
            pair_wise_dot = torch.sum(ests * targets, dim=2, keepdim=True)
            # [batch, n_src]
            s_target_energy = torch.sum(targets ** 2, dim=2, keepdim=True) + self.EPS
            # [batch, n_src, time]
            scaled_targets = pair_wise_dot * targets / s_target_energy
        else:
            # [batch, n_src, time]
            scaled_targets = targets
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = ests - targets
        else:
            e_noise = ests - scaled_targets
        # [batch, n_src]
        pair_wise_sdr = torch.sum(scaled_targets ** 2, dim=2) / (
            torch.sum(e_noise ** 2, dim=2) + self.EPS
        )
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + self.EPS)
        return -torch.mean(pair_wise_sdr, dim=-1)

class freq_MAE_WavL1Loss(_Loss):
    def __init__(self, win=2048, stride=512):
        super().__init__()
        self.EPS = 1e-8
        self.win = win
        self.stride = stride

    def forward(self, ests, targets):
        B, nsrc, _ = ests.shape
        est_spec = torch.stft(ests.view(-1, ests.shape[-1]), n_fft=self.win, hop_length=self.stride, 
                          window=torch.hann_window(self.win).to(ests.device).float(),
                          return_complex=True)
        est_target = torch.stft(targets.view(-1, targets.shape[-1]), n_fft=self.win, hop_length=self.stride, 
                                window=torch.hann_window(self.win).to(ests.device).float(),
                                return_complex=True)
        freq_L1 = (est_spec.real - est_target.real).abs().mean((1,2)) + (est_spec.imag - est_target.imag).abs().mean((1,2))
        freq_L1 = freq_L1.view(B, nsrc).mean(-1)
        
        wave_l1 = (ests - targets).abs().mean(-1)
        # print(freq_L1.shape, wave_l1.shape)
        wave_l1 = wave_l1.view(B, nsrc).mean(-1)
        return freq_L1 + wave_l1

class freq_MSE_Loss(_Loss):
    def __init__(self, win=640, stride=160):
        super().__init__()
        self.EPS = 1e-8
        self.win = win
        self.stride = stride

    def forward(self, ests, targets):
        B, nsrc, _ = ests.shape
        est_spec = torch.stft(ests.view(-1, ests.shape[-1]), n_fft=self.win, hop_length=self.stride, 
                          window=torch.hann_window(self.win).to(ests.device).float(),
                          return_complex=True)
        est_target = torch.stft(targets.view(-1, targets.shape[-1]), n_fft=self.win, hop_length=self.stride, 
                                window=torch.hann_window(self.win).to(ests.device).float(),
                                return_complex=True)
        freq_mse = (est_spec.real - est_target.real).square().mean((1,2)) + (est_spec.imag - est_target.imag).square().mean((1,2))
        freq_mse = freq_mse.view(B, nsrc).mean(-1)
        
        return freq_mse

# aliases
pairwise_neg_sisdr = PairwiseNegSDR("sisdr")
pairwise_neg_sdsdr = PairwiseNegSDR("sdsdr")
pairwise_neg_snr = PairwiseNegSDR("snr")
singlesrc_neg_sisdr = SingleSrcNegSDR("sisdr")
singlesrc_neg_sdsdr = SingleSrcNegSDR("sdsdr")
singlesrc_neg_snr = SingleSrcNegSDR("snr")
multisrc_neg_sisdr = MultiSrcNegSDR("sisdr")
multisrc_neg_sdsdr = MultiSrcNegSDR("sdsdr")
multisrc_neg_snr = MultiSrcNegSDR("snr")
freq_mae_wavl1loss = freq_MAE_WavL1Loss()
pairwise_neg_sisdr_freq_mse = (PairwiseNegSDR("sisdr"), freq_MSE_Loss())
pairwise_neg_snr_multidecoder = (PairwiseNegSDR("snr"), MultiSrcNegSDR("snr"))