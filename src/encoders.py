"""Neural network encoders for expression and drug data."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpressionEncoder(nn.Module):
    """Encoder for L1000 gene expression profiles."""

    def __init__(
        self,
        input_dim: int = 978,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.1
    ):
        """Initialize expression encoder.

        Args:
            input_dim: Input dimension (number of genes, default: 978)
            hidden_dim: Hidden layer dimension (default: 512)
            output_dim: Output embedding dimension (default: 256)
            dropout: Dropout rate (default: 0.1)
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with L2 normalization.

        Args:
            x: Expression tensor [batch_size, input_dim]

        Returns:
            L2-normalized embeddings [batch_size, output_dim]
        """
        h = self.net(x)
        return F.normalize(h, p=2, dim=-1)


class DrugEncoder(nn.Module):
    """Encoder for drug with condition context using Morgan Fingerprints."""

    def __init__(
        self,
        n_cells: int,
        n_doses: int,
        n_times: int,
        fingerprint_dim: int = 2048,
        cell_embed_dim: int = 64,
        dose_embed_dim: int = 16,
        time_embed_dim: int = 16,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.1
    ):
        """Initialize drug encoder with Morgan Fingerprints.

        Args:
            n_cells: Number of unique cell lines
            n_doses: Number of dose bins
            n_times: Number of time points
            fingerprint_dim: Morgan fingerprint dimension (default: 2048)
            cell_embed_dim: Cell embedding dimension (default: 64)
            dose_embed_dim: Dose embedding dimension (default: 16)
            time_embed_dim: Time embedding dimension (default: 16)
            hidden_dim: Hidden layer dimension (default: 512)
            output_dim: Output embedding dimension (default: 256)
            dropout: Dropout rate (default: 0.1)
        """
        super().__init__()

        self.output_dim = output_dim
        self.fingerprint_dim = fingerprint_dim

        # Embedding layers for context (no drug embedding - uses fingerprints)
        self.cell_embed = nn.Embedding(n_cells, cell_embed_dim)
        self.dose_embed = nn.Embedding(n_doses, dose_embed_dim)
        self.time_embed = nn.Embedding(n_times, time_embed_dim)

        # Fingerprint projection
        self.fp_proj = nn.Sequential(
            nn.Linear(fingerprint_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout)
        )

        # Combined projection network
        total_dim = 256 + cell_embed_dim + dose_embed_dim + time_embed_dim
        self.proj = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(
        self,
        drug_fp: torch.Tensor,
        cell_idx: torch.Tensor,
        dose_idx: torch.Tensor,
        time_idx: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with L2 normalization.

        Args:
            drug_fp: Morgan fingerprint tensor [batch_size, fingerprint_dim]
            cell_idx: Cell line indices [batch_size]
            dose_idx: Dose bin indices [batch_size]
            time_idx: Time point indices [batch_size]

        Returns:
            L2-normalized embeddings [batch_size, output_dim]
        """
        # Project fingerprint
        h_drug = self.fp_proj(drug_fp)

        # Embed context
        h_cell = self.cell_embed(cell_idx)
        h_dose = self.dose_embed(dose_idx)
        h_time = self.time_embed(time_idx)

        # Concatenate and project
        h = torch.cat([h_drug, h_cell, h_dose, h_time], dim=-1)
        h = self.proj(h)

        return F.normalize(h, p=2, dim=-1)


class DualEncoder(nn.Module):
    """Combined expression and drug encoder for contrastive learning."""

    def __init__(
        self,
        expr_input_dim: int = 978,
        fingerprint_dim: int = 2048,
        n_cells: int = 50,
        n_doses: int = 6,
        n_times: int = 10,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        """Initialize dual encoder.

        Args:
            expr_input_dim: Expression input dimension (default: 978)
            fingerprint_dim: Morgan fingerprint dimension (default: 2048)
            n_cells: Number of unique cell lines
            n_doses: Number of dose bins
            n_times: Number of time points
            embed_dim: Shared embedding dimension (default: 256)
            hidden_dim: Hidden layer dimension (default: 512)
            dropout: Dropout rate (default: 0.1)
        """
        super().__init__()

        self.expr_encoder = ExpressionEncoder(
            input_dim=expr_input_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            dropout=dropout
        )

        self.drug_encoder = DrugEncoder(
            n_cells=n_cells,
            n_doses=n_doses,
            n_times=n_times,
            fingerprint_dim=fingerprint_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            dropout=dropout
        )

        # Learnable temperature parameter for InfoNCE
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(
        self,
        expr: torch.Tensor,
        drug_fp: torch.Tensor,
        cell_idx: torch.Tensor,
        dose_idx: torch.Tensor,
        time_idx: torch.Tensor
    ):
        """Forward pass through both encoders.

        Args:
            expr: Expression tensor [batch_size, expr_input_dim]
            drug_fp: Morgan fingerprint tensor [batch_size, fingerprint_dim]
            cell_idx: Cell indices [batch_size]
            dose_idx: Dose indices [batch_size]
            time_idx: Time indices [batch_size]

        Returns:
            Tuple of (expression embeddings, drug embeddings)
        """
        z_expr = self.expr_encoder(expr)
        z_drug = self.drug_encoder(drug_fp, cell_idx, dose_idx, time_idx)

        return z_expr, z_drug

    def compute_loss(
        self,
        z_expr: torch.Tensor,
        z_drug: torch.Tensor
    ) -> torch.Tensor:
        """Compute bidirectional InfoNCE loss.

        Args:
            z_expr: Expression embeddings [batch_size, embed_dim]
            z_drug: Drug embeddings [batch_size, embed_dim]

        Returns:
            Contrastive loss scalar
        """
        batch_size = z_expr.size(0)

        # Compute similarity matrix
        # [batch_size, batch_size]
        logits = (z_expr @ z_drug.T) / torch.clamp(self.temperature, min=1e-3)

        # Labels are diagonal (matching pairs)
        labels = torch.arange(batch_size, device=z_expr.device)

        # Bidirectional cross-entropy
        loss_expr_to_drug = F.cross_entropy(logits, labels)
        loss_drug_to_expr = F.cross_entropy(logits.T, labels)

        return (loss_expr_to_drug + loss_drug_to_expr) / 2

    def get_temperature(self) -> float:
        """Get current temperature value."""
        return self.temperature.item()
