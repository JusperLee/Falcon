###
# AudioVisualLightningModuleNASSv2
#
# NASS-style same-space speaker contrastive training:
#   1. Forward through SpeakerContrastiveV2 → (est_sources, sep_audio_embs)
#   2. Query  = pool_spk_embed(sep_audio_embs[:, 0])       — has gradient
#   3. Positive = encode_stft(target_wav) → pool_spk_embed  — detached
#   4. Negative = encode_stft(near_wav)  → pool_spk_embed   — detached
#   5. NASSContrastiveLoss with β annealing
#   6. total_loss = pit_loss + contrastive_weight * cl_loss
###

import torch
import torch.nn.functional as F

from .av_litmodule import AudioVisualLightningModule
from ..losses.nass_contrastive_loss import NASSContrastiveLoss

_NEAR_BUCKETS = [0, 1, 2, 3, 4]
_BUCKET_LABEL = {
    0: "train_cl_0near",
    1: "train_cl_1near",
    2: "train_cl_2near",
    3: "train_cl_3near",
    4: "train_cl_4plus_near",
}
_BUCKET_N_LABEL = {
    0: "train_n_samples_0near",
    1: "train_n_samples_1near",
    2: "train_n_samples_2near",
    3: "train_n_samples_3near",
    4: "train_n_samples_4plus_near",
}


class AudioVisualLightningModuleNASSv2(AudioVisualLightningModule):
    """Adds NASS-style same-space speaker contrastive loss."""

    def __init__(
        self,
        *args,
        contrastive_weight: float = 1.0,
        margin: float = 0.0,
        alpha_pos: float = 1.0,
        alpha_neg: float = 1.0,
        beta_init: float = 0.0,
        beta_final: float = 1.5,
        beta_anneal_epochs: int = 100,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.contrastive_weight = contrastive_weight
        self.contrastive_loss = NASSContrastiveLoss(margin, alpha_pos, alpha_neg)
        self.beta_init = beta_init
        self.beta_final = beta_final
        self.beta_anneal_epochs = beta_anneal_epochs

    # ------------------------------------------------------------------
    @staticmethod
    def _trim_or_pad(wav: torch.Tensor, length: int) -> torch.Tensor:
        T = wav.shape[-1]
        if T >= length:
            return wav[..., :length]
        return torch.cat([wav, wav.new_zeros(*wav.shape[:-1], length - T)], dim=-1)

    def _get_beta(self) -> float:
        if self.beta_anneal_epochs <= 0:
            return self.beta_final
        progress = min(self.current_epoch / self.beta_anneal_epochs, 1.0)
        return self.beta_init + (self.beta_final - self.beta_init) * progress

    # ------------------------------------------------------------------
    def training_step(self, batch, batch_nb):
        mixtures, targets, mouth, _, near_audios = batch

        if targets.ndim == 2:
            targets = targets.unsqueeze(1)

        seg_len = mixtures.shape[-1]

        # ── Video embedding (frozen) ─────────────────────────────────
        if mouth.ndim == 4:
            mouth_in = mouth.unsqueeze(1)
        else:
            mouth_in = mouth
        with torch.no_grad():
            mouth_emb = self.video_model(mouth_in.type_as(mixtures))

        # ── Main forward ─────────────────────────────────────────────
        est_sources, sep_audio_embs = self.audio_model(mixtures, mouth_emb)
        # est_sources:     [B, n_src, T]
        # sep_audio_embs:  [B, n_src, N, T', F]

        # ── PIT loss ─────────────────────────────────────────────────
        pit_loss = self.loss_func["train"](est_sources, targets)

        # ── Speaker contrastive loss ─────────────────────────────────
        B = mixtures.size(0)
        beta = self._get_beta()

        # Query: pool separated embedding for source 0 (has gradient)
        query_feat = sep_audio_embs[:, 0]  # [B, N, T', F]
        embed_query = self.audio_model.pool_spk_embed(query_feat)  # [B, D]

        # Positive: encode target waveform (detached)
        with torch.no_grad():
            target_wav = targets[:, 0]  # [B, T]
            pos_feat = self.audio_model.encode_stft(target_wav)  # [B, N, T', F]
            embed_pos = self.audio_model.pool_spk_embed(pos_feat)  # [B, D]

        # Negatives: encode each near speaker (detached)
        embed_negs_list = []
        near_count_n = {k: 0 for k in _NEAR_BUCKETS}

        for b in range(B):
            n_near = len(near_audios[b])
            bucket = min(n_near, 4)
            near_count_n[bucket] += 1

            if n_near == 0:
                embed_negs_list.append(None)
                continue

            near_wavs = torch.stack([
                self._trim_or_pad(nw.to(mixtures.device), seg_len)
                for nw in near_audios[b]
            ])  # [K, T]

            with torch.no_grad():
                neg_feat = self.audio_model.encode_stft(near_wavs)  # [K, N, T', F]
                neg_embs = self.audio_model.pool_spk_embed(neg_feat)  # [K, D]
            embed_negs_list.append(neg_embs)

        cl_loss = self.contrastive_loss(embed_query, embed_pos, embed_negs_list, beta)

        # ── Aggregate ────────────────────────────────────────────────
        weighted_cl = self.contrastive_weight * cl_loss
        total_loss = pit_loss + weighted_cl

        # ── Embedding similarity monitoring ────────────────────────
        with torch.no_grad():
            sim_pos_avg = F.cosine_similarity(embed_query, embed_pos).mean()
            sim_neg_vals = []
            sim_neg_max_vals = []
            neg_above_margin_count = 0
            neg_total_count = 0
            for b in range(B):
                if embed_negs_list[b] is not None:
                    sn = F.cosine_similarity(
                        embed_query[b].unsqueeze(0).expand(
                            embed_negs_list[b].size(0), -1
                        ),
                        embed_negs_list[b],
                        dim=1,
                    )
                    sim_neg_vals.append(sn.mean())
                    sim_neg_max_vals.append(sn.max())
                    neg_above_margin_count += (sn > self.contrastive_loss.margin).sum().item()
                    neg_total_count += sn.numel()
            if sim_neg_vals:
                sim_neg_avg = torch.stack(sim_neg_vals).mean()
                sim_neg_max = torch.stack(sim_neg_max_vals).mean()
                neg_active_ratio = neg_above_margin_count / max(neg_total_count, 1)
            else:
                sim_neg_avg = torch.tensor(0.0, device=mixtures.device)
                sim_neg_max = torch.tensor(0.0, device=mixtures.device)
                neg_active_ratio = 0.0

        # ── Logging ──────────────────────────────────────────────────
        log_kw = dict(on_step=True, on_epoch=True, prog_bar=False,
                      sync_dist=True, logger=True)
        self.log("train_loss", total_loss, on_epoch=True, prog_bar=True,
                 sync_dist=True, logger=True)
        self.log("train_pit_loss", pit_loss, **log_kw)
        self.log("train_spk_contrast_loss", weighted_cl, **log_kw)
        self.log("train_beta", float(beta), **log_kw)
        self.log("train_cos_sim_pos", sim_pos_avg, **log_kw)
        self.log("train_cos_sim_neg_avg", sim_neg_avg, **log_kw)
        self.log("train_cos_sim_neg_max", sim_neg_max, **log_kw)
        self.log("train_neg_active_ratio", neg_active_ratio, **log_kw)

        for bucket in _NEAR_BUCKETS:
            self.log(_BUCKET_N_LABEL[bucket], float(near_count_n[bucket]), **log_kw)

        return {"loss": total_loss}
