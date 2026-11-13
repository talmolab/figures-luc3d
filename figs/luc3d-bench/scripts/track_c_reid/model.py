"""Pose-history ReID encoder, inspired by PrecisionTrack's MART pose encoder.

Input: (B, T, K, 3)   -- B windows of T frames x K keypoints x 3 dims
Output: (B, d_embed)  -- L2-normalised embeddings
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderConfig:
    window_len: int = 10
    n_keypoints: int = 15
    in_dim: int = 3
    d_embed: int = 128
    n_block: int = 2
    n_head: int = 4
    dropout: float = 0.1
    use_velocity_channel: bool = True   # include per-frame finite-difference velocity as features


class ProjLN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x):
        return self.ln(self.lin(x))


class TransformerBlock(nn.Module):
    def __init__(self, d: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        h, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False, attn_mask=attn_mask)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x


def normalise_window(win: torch.Tensor) -> torch.Tensor:
    """Subtract per-window centroid (mean across T and K) and scale by mean distance.

    Input: (B, T, K, 3). Output: (B, T, K, 3), centroid-removed, scale-normalised so
    inter-keypoint distances are O(1).
    """
    centroid = win.mean(dim=(1, 2), keepdim=True)         # (B, 1, 1, 3)
    win_c = win - centroid
    # robust scale: median distance from centroid (taken across T*K points)
    dist = win_c.norm(dim=-1)                              # (B, T, K)
    scale = dist.mean(dim=(1, 2), keepdim=True).clamp_min(1e-3)  # (B, 1, 1)
    scale = scale.unsqueeze(-1)                            # (B, 1, 1, 1)
    return win_c / scale


class PoseHistoryEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        T = cfg.window_len
        K = cfg.n_keypoints
        D = cfg.in_dim

        # per-frame input dim: flatten K*3 + optional K*3 velocity
        per_frame_dim = K * D
        if cfg.use_velocity_channel:
            per_frame_dim += K * D

        self.in_proj = ProjLN(per_frame_dim, cfg.d_embed)
        self.pos_embed = nn.Embedding(T, cfg.d_embed)
        self.dropout = nn.Dropout(cfg.dropout)

        # causal mask
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

        self.blocks = nn.ModuleList(
            TransformerBlock(cfg.d_embed, cfg.n_head, cfg.dropout) for _ in range(cfg.n_block)
        )
        self.ln_f = nn.LayerNorm(cfg.d_embed)
        self.proj_out = nn.Linear(cfg.d_embed, cfg.d_embed, bias=False)

    def forward(self, win: torch.Tensor) -> torch.Tensor:
        """win: (B, T, K, 3) -> (B, d_embed) L2-normalised."""
        B, T, K, D = win.shape
        w = normalise_window(win)
        # velocity: zero-pad first frame
        if self.cfg.use_velocity_channel:
            vel = torch.zeros_like(w)
            vel[:, 1:] = w[:, 1:] - w[:, :-1]
            feats = torch.cat([w, vel], dim=-1)  # (B, T, K, 2D)
        else:
            feats = w
        x = feats.reshape(B, T, -1)                # (B, T, in_dim)
        x = self.in_proj(x)                        # (B, T, d_embed)
        pos = self.pos_embed(torch.arange(T, device=x.device))
        x = self.dropout(x + pos)
        for blk in self.blocks:
            x = blk(x, attn_mask=self.causal_mask[:T, :T])
        x = self.ln_f(x[:, -1])                    # (B, d_embed)
        x = self.proj_out(x)
        return F.normalize(x, p=2, dim=-1)


# ---------------- losses ----------------

def supcon_loss(
    embeddings: torch.Tensor,   # (B, d) L2-normalised
    labels: torch.Tensor,       # (B,) int64 global id
    temperature: float = 0.1,
) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al. 2020), aka generalised NT-Xent.

    For each anchor i, positives are samples with the same label, negatives all others.
    """
    device = embeddings.device
    B = embeddings.shape[0]
    # similarity matrix
    sim = embeddings @ embeddings.t() / temperature  # (B, B)
    # mask out self
    self_mask = torch.eye(B, dtype=torch.bool, device=device)
    sim.masked_fill_(self_mask, -1e9)

    labels = labels.view(-1, 1)
    pos_mask = labels.eq(labels.t())              # (B, B) bool
    pos_mask.masked_fill_(self_mask, False)

    # for each i, compute log p(positive)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)  # (B, B)
    # if no positives, skip
    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0
    if not valid.any():
        return torch.tensor(0.0, device=device, requires_grad=True)
    pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1)
    loss_per = -pos_log_prob[valid] / pos_count[valid].float()
    return loss_per.mean()
