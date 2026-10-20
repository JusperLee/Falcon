"""
SpeakerContrastiveV2 — NASS-style same-space speaker contrastive model.

Extends SISNRRepel with a lightweight speaker embedding head (spk_pool_proj)
that pools 4-D encoder feature maps into L2-normalized speaker embeddings.
Query / positive / negative all live in the same STFTEncoder feature space.

Pooling strategy: AttentionSpeakerPool
  - AdaptiveAvgPool2d(1) collapses ALL time-frequency info into a single point,
    making embeddings insensitive to speaker identity (near speakers share the
    same room acoustics → avg looks almost identical).
  - AttentionSpeakerPool instead:
      1. Mean over frequency (F) — reduces spectral redundancy, keeps time axis
      2. Learnable attention over time (T') — upweights speaker-informative frames
         (voiced, stable phonemes) and downweights silence/noise
      3. Linear projection → spk_embed_dim → L2-normalize
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .SISNRRepel import SISNRRepel


class AttentionSpeakerPool(nn.Module):
    """Temporal-attention speaker pooling.

    Input : [B, N, T', F]   (encoder feature map)
    Output: [B, spk_embed_dim]   (before L2 normalization)

    Steps
    -----
    1. Mean over F  →  [B, N, T']      # collapse frequency, keep time
    2. Permute      →  [B, T', N]
    3. Attn score   →  [B, T', 1]      # MLP: N → N//4 → 1
    4. Softmax      →  weights over T'
    5. Weighted sum →  [B, N]          # content-aware temporal pool
    6. Project      →  [B, D]          # N → spk_embed_dim
    """

    def __init__(self, encoder_out_chan: int, spk_embed_dim: int):
        super().__init__()

        # Temporal attention: score each frame based on its content
        self.temporal_attn = nn.Sequential(
            nn.Linear(encoder_out_chan, encoder_out_chan // 4),
            nn.Tanh(),
            nn.Linear(encoder_out_chan // 4, 1, bias=False),
        )

        # Final projection to speaker embedding space
        self.proj = nn.Sequential(
            nn.Linear(encoder_out_chan, spk_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(spk_embed_dim, spk_embed_dim),
        )

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        # feat_map: [B, N, T', F]

        # Step 1: collapse frequency dimension
        x = feat_map.mean(dim=-1)          # [B, N, T']

        # Step 2: put time axis first for attention
        x = x.permute(0, 2, 1)            # [B, T', N]

        # Step 3-4: compute normalised attention weights over time
        attn_weights = self.temporal_attn(x).softmax(dim=1)  # [B, T', 1]

        # Step 5: weighted temporal pooling
        pooled = (attn_weights * x).sum(dim=1)               # [B, N]

        # Step 6: project to embedding space
        return self.proj(pooled)                              # [B, D]


class SpeakerContrastiveV2(SISNRRepel):
    """Audio-visual speech separation with same-space speaker contrastive head."""

    def __init__(self, *args, spk_embed_dim: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        self.spk_embed_dim = spk_embed_dim

        encoder_out_chan = self.encoder.get_out_chan()  # e.g. 256

        self.spk_pool_proj = AttentionSpeakerPool(encoder_out_chan, spk_embed_dim)

    # ------------------------------------------------------------------
    def encode_stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode waveform through STFTEncoder only.

        Args:
            waveform: [B, T]
        Returns:
            Feature map [B, N, T', F]
        """
        return self.encoder(waveform)

    def pool_spk_embed(self, feat_map: torch.Tensor) -> torch.Tensor:
        """Pool 4-D feature map to L2-normalized speaker embedding.

        Args:
            feat_map: [B, N, T', F]
        Returns:
            [B, spk_embed_dim] L2-normalized
        """
        emb = self.spk_pool_proj(feat_map)       # [B, D]
        return F.normalize(emb, p=2, dim=-1)

    # ------------------------------------------------------------------
    def forward(self, audio_mixture: torch.Tensor,
                mouth_embedding: torch.Tensor = None, **kwargs):
        """
        Forward pass.

        During training returns ``(separated_audios, separated_audio_embeddings)``
        so the lightning module can compute the contrastive loss.
        During eval returns only ``separated_audios``.
        """
        audio_mixture_embedding = self.encoder(audio_mixture)          # [B, N, T', F]
        audio = self.audio_bottleneck(audio_mixture_embedding)         # [B, C, T', F]
        video = self.video_bottleneck(mouth_embedding)                 # [B, C_v, T_v]
        refined_features = self.separator(audio, video)                # [B, C, T', F]

        separated_audio_embeddings = self.mask_generator(
            refined_features, audio_mixture_embedding
        )  # [B, n_src, N, T', F]

        separated_audios = self.decoder(
            separated_audio_embeddings, audio_mixture.shape
        )  # [B, n_src, T]

        if self.training:
            return separated_audios, separated_audio_embeddings
        return separated_audios

    def get_model_args(self):
        model_args = {"n_src": self.n_src}
        return model_args
