"""
NASS-style same-space contrastive loss.

Triplet + Hard Negative weighting:
  L_pos = alpha_pos * (1 - cos_sim(q, p))
  L_neg = alpha_neg * Σ_k q_β(k) * relu(cos_sim(q, n_k) - margin)
    where q_β(k) = softmax_k(β · cos_sim(q, n_k))
  L = L_pos + L_neg

All embeddings are assumed L2-normalized before entering this loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NASSContrastiveLoss(nn.Module):

    def __init__(
        self,
        margin: float = 0.0,
        alpha_pos: float = 1.0,
        alpha_neg: float = 1.0,
    ):
        super().__init__()
        self.margin = margin
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg

    def forward(
        self,
        embed_query: torch.Tensor,
        embed_pos: torch.Tensor,
        embed_negs_list: list,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            embed_query:    [B, D]  — L2-normalized, has gradient
            embed_pos:      [B, D]  — L2-normalized, detached
            embed_negs_list: list of length B; each element is [K_b, D] or None
            beta:           concentration parameter for hard-negative weighting

        Returns:
            Scalar loss averaged over the batch.
        """
        B = embed_query.size(0)
        loss_total = embed_query.new_zeros(1)
        n_valid = 0

        for b in range(B):
            q = embed_query[b]  # [D]
            p = embed_pos[b]    # [D]

            # Positive term
            sim_pos = F.cosine_similarity(q.unsqueeze(0), p.unsqueeze(0))  # [1]
            l_pos = self.alpha_pos * (1.0 - sim_pos)

            # Negative term
            negs = embed_negs_list[b]
            if negs is not None and negs.size(0) > 0:
                K = negs.size(0)
                sim_neg = F.cosine_similarity(
                    q.unsqueeze(0).expand(K, -1), negs, dim=1
                )  # [K]

                if K == 1:
                    l_neg = self.alpha_neg * F.relu(sim_neg - self.margin).sum()
                else:
                    weights = torch.softmax(beta * sim_neg, dim=0)  # [K]
                    l_neg = self.alpha_neg * (weights * F.relu(sim_neg - self.margin)).sum()
            else:
                l_neg = embed_query.new_zeros(1)

            loss_total = loss_total + l_pos + l_neg
            n_valid += 1

        if n_valid == 0:
            return embed_query.new_zeros(1)
        return loss_total / n_valid
