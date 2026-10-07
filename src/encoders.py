"""Dual encoder used for the embedding-similarity analysis (scripts/train_embedding_model.py).

Module names match the released checkpoint models/embedding_model.pt.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpressionEncoder(nn.Module):
    """978 -> 512 -> 256 with GELU, LayerNorm and dropout."""

    def __init__(self, input_dim: int = 978, hidden_dim: int = 512,
                 output_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


class DrugEncoder(nn.Module):
    """Morgan fingerprint -> 256, concatenated with cell/dose/time embeddings -> 512 -> 256."""

    def __init__(self, n_cells: int, n_doses: int, n_times: int, fingerprint_dim: int = 2048,
                 cell_embed_dim: int = 64, dose_embed_dim: int = 16, time_embed_dim: int = 16,
                 hidden_dim: int = 512, output_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.output_dim = output_dim
        self.fingerprint_dim = fingerprint_dim

        self.cell_embed = nn.Embedding(n_cells, cell_embed_dim)
        self.dose_embed = nn.Embedding(n_doses, dose_embed_dim)
        self.time_embed = nn.Embedding(n_times, time_embed_dim)

        self.fp_proj = nn.Sequential(
            nn.Linear(fingerprint_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
        )
        total_dim = 256 + cell_embed_dim + dose_embed_dim + time_embed_dim
        self.proj = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, drug_fp, cell_idx, dose_idx, time_idx):
        h = torch.cat([self.fp_proj(drug_fp), self.cell_embed(cell_idx),
                       self.dose_embed(dose_idx), self.time_embed(time_idx)], dim=-1)
        return F.normalize(self.proj(h), p=2, dim=-1)


class DualEncoder(nn.Module):
    """Expression and drug encoders trained with a bidirectional InfoNCE loss."""

    def __init__(self, expr_input_dim: int = 978, fingerprint_dim: int = 2048,
                 n_cells: int = 50, n_doses: int = 6, n_times: int = 10,
                 embed_dim: int = 256, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.expr_encoder = ExpressionEncoder(expr_input_dim, hidden_dim, embed_dim, dropout)
        self.drug_encoder = DrugEncoder(n_cells=n_cells, n_doses=n_doses, n_times=n_times,
                                        fingerprint_dim=fingerprint_dim, hidden_dim=hidden_dim,
                                        output_dim=embed_dim, dropout=dropout)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, expr, drug_fp, cell_idx, dose_idx, time_idx):
        return self.expr_encoder(expr), self.drug_encoder(drug_fp, cell_idx, dose_idx, time_idx)

    def compute_loss(self, z_expr, z_drug):
        logits = (z_expr @ z_drug.T) / torch.clamp(self.temperature, min=1e-3)
        labels = torch.arange(z_expr.size(0), device=z_expr.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
