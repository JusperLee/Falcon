###
# AudioVisualLightningModuleWithMDCL
#
# Overrides training_step to add Hard Negative MDCL:
#   1. Unpack near_audios from the batch
#   2. Compute waveform-domain Hard MDCL repulsion loss via _mdcl_loss_hard:
#        - softmax-weighted over K near speakers by SI-SNR similarity
#        - β anneals linearly from beta_init → beta_final over beta_anneal_epochs
#        - K=0: skip; K=1: identical to uniform MDCL; β=0: uniform weights
#   3. Log PIT / MDCL / beta and per-bucket counts (DDP-safe fixed key set)
#
# DDP safety: self.log() always uses the fixed _NEAR_BUCKETS key set
# (every GPU, every step), so sync_dist=True is safe.
###

import torch

from .av_litmodule import AudioVisualLightningModule

_NEAR_BUCKETS = [0, 1, 2, 3, 4]   # 4 = "4 or more"
_BUCKET_LABEL = {
    0: "train_mdcl_0near",
    1: "train_mdcl_1near",
    2: "train_mdcl_2near",
    3: "train_mdcl_3near",
    4: "train_mdcl_4plus_near",
}
_BUCKET_N_LABEL = {
    0: "train_n_samples_0near",
    1: "train_n_samples_1near",
    2: "train_n_samples_2near",
    3: "train_n_samples_3near",
    4: "train_n_samples_4plus_near",
}


class AudioVisualLightningModuleWithMDCL(AudioVisualLightningModule):
    """
    Adds Hard Negative MDCL to the base AudioVisual module.

    β anneals from beta_init → beta_final over beta_anneal_epochs,
    concentrating gradient on the near speaker the model currently
    confuses most with the target (highest SI-SNR).
    """

    def __init__(
        self,
        *args,
        beta_init: float = 0.0,
        beta_final: float = 1.5,
        beta_anneal_epochs: int = 200,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.beta_init          = beta_init
        self.beta_final         = beta_final
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

        # ── Video embedding (no grad) ────────────────────────────────────────
        if mouth.ndim == 4:
            mouth_in = mouth.unsqueeze(1)
        else:
            mouth_in = mouth
        with torch.no_grad():
            mouth_emb = self.video_model(mouth_in.type_as(mixtures))

        # ── Main forward ─────────────────────────────────────────────────────
        est_sources = self.audio_model(mixtures, mouth_emb)   # [B, n_src, T]

        # ── PIT loss ─────────────────────────────────────────────────────────
        pit_loss = self.loss_func["train"](est_sources, targets)

        # ── Hard MDCL ────────────────────────────────────────────────────────
        B    = mixtures.size(0)
        beta = self._get_beta()

        mdcl_total     = torch.tensor(0.0, device=mixtures.device)
        near_loss_sums = {k: torch.tensor(0.0, device=mixtures.device)
                          for k in _NEAR_BUCKETS}
        near_count_n   = {k: 0 for k in _NEAR_BUCKETS}

        # Print diagnostic info for first 5 steps on rank0
        do_debug = (batch_nb < 5 and self.global_rank == 0)
        debug_lines = []

        for b in range(B):
            n_near = len(near_audios[b])
            bucket = min(n_near, 4)
            near_count_n[bucket] += 1

            if n_near == 0:
                continue

            near_b_wavs = [
                self._trim_or_pad(nw.to(mixtures.device), seg_len).unsqueeze(0)
                for nw in near_audios[b]
            ]

            est_b      = est_sources[b : b + 1, 0, :]   # [1, T]
            raw_mdcl_b = self.audio_model._mdcl_loss_hard(
                est_b, near_b_wavs, beta=beta
            )
            mdcl_total = mdcl_total + raw_mdcl_b
            near_loss_sums[bucket] = near_loss_sums[bucket] + raw_mdcl_b.detach()

            if do_debug:
                # Print mag_sim for each near speaker to help judge if margin is appropriate
                with torch.no_grad():
                    sim_vals = [
                        f"{self.audio_model._mag_sim(est_b.detach(), nw).item():.1f}dB"
                        for nw in near_b_wavs
                    ]
                debug_lines.append(
                    f"  b={b} n_near={n_near} "
                    f"mag_sim={sim_vals} "
                    f"raw_mdcl={raw_mdcl_b.item():.4f} "
                    f"margin={self.audio_model.mdcl_margin:.1f}dB"
                )

        if do_debug:
            import sys
            print(
                f"[MDCL DEBUG] step={batch_nb} beta={beta:.3f} "
                f"pit={pit_loss.item():.4f}",
                file=sys.stderr, flush=True,
            )
            for line in debug_lines:
                print(line, file=sys.stderr, flush=True)

        # ── Aggregate ────────────────────────────────────────────────────────
        weighted_mdcl = self.audio_model.mdcl_alpha * (mdcl_total / B)
        total_loss    = pit_loss + weighted_mdcl

        # ── Logging ──────────────────────────────────────────────────────────
        log_kw = dict(on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, logger=True)
        self.log("train_loss",      total_loss,    on_epoch=True, prog_bar=True,
                 sync_dist=True, logger=True)
        self.log("train_pit_loss",  pit_loss,      **log_kw)
        self.log("train_mdcl_loss", weighted_mdcl, **log_kw)
        self.log("train_beta",      float(beta),   **log_kw)

        for bucket in _NEAR_BUCKETS:
            n   = near_count_n[bucket]
            avg = (near_loss_sums[bucket] / n
                   if n > 0
                   else torch.tensor(0.0, device=mixtures.device))
            self.log(_BUCKET_LABEL[bucket],   avg,      **log_kw)
            self.log(_BUCKET_N_LABEL[bucket], float(n), **log_kw)

        return {"loss": total_loss}
