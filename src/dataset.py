"""PyTorch dataset for L1000 data."""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Dict, Tuple


class L1000Dataset(Dataset):
    """Dataset for L1000 expression-drug pairs with Morgan Fingerprints."""

    def __init__(
        self,
        expr_matrix: np.ndarray,
        metadata: pd.DataFrame,
        drug_fingerprints: Dict[str, np.ndarray],
        vocab: Dict[str, list]
    ):
        """Initialize dataset.

        Args:
            expr_matrix: Expression matrix [N, n_genes]
            metadata: Metadata with columns: pert_id_idx, cell_id_idx, dose_bin_idx, pert_time_idx
            drug_fingerprints: Dictionary mapping pert_id to fingerprint array
            vocab: Vocabulary mapping with pert_id list
        """
        self.expr = torch.from_numpy(expr_matrix).float()

        # Build fingerprint matrix aligned with pert_id_idx
        pert_id_vocab = vocab['pert_id']
        fp_list = []
        for pert_id in pert_id_vocab:
            if pert_id in drug_fingerprints:
                fp_list.append(drug_fingerprints[pert_id])
            else:
                # Use zero vector if fingerprint not found
                fp_dim = next(iter(drug_fingerprints.values())).shape[0]
                fp_list.append(np.zeros(fp_dim, dtype=np.float32))

        self.drug_fps = torch.from_numpy(np.array(fp_list)).float()

        # Extract indices
        self.drug_idx = torch.from_numpy(metadata['pert_id_idx'].values).long()
        self.cell_idx = torch.from_numpy(metadata['cell_id_idx'].values).long()
        self.dose_idx = torch.from_numpy(metadata['dose_bin_idx'].values).long()
        self.time_idx = torch.from_numpy(metadata['pert_time_idx'].values).long()

        assert len(self.expr) == len(self.drug_idx), "Size mismatch"

    def __len__(self) -> int:
        return len(self.expr)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Get item by index.

        Returns:
            Tuple of (expr, drug_fp, cell_idx, dose_idx, time_idx)
        """
        drug_id = self.drug_idx[idx]
        return (
            self.expr[idx],
            self.drug_fps[drug_id],  # Look up fingerprint by drug index
            self.cell_idx[idx],
            self.dose_idx[idx],
            self.time_idx[idx]
        )


def create_dataloaders(
    expr_matrix: np.ndarray,
    metadata: pd.DataFrame,
    drug_fingerprints: Dict[str, np.ndarray],
    vocab: Dict[str, list],
    batch_size: int = 512,
    train_split: float = 0.9,
    num_workers: int = 4,
    seed: int = 42
):
    """Create train and validation dataloaders.

    Args:
        expr_matrix: Expression matrix
        metadata: Metadata dataframe
        drug_fingerprints: Dictionary mapping pert_id to fingerprint array
        vocab: Vocabulary mapping
        batch_size: Batch size (default: 512)
        train_split: Fraction for training (default: 0.9)
        num_workers: Number of dataloader workers (default: 4)
        seed: Random seed (default: 42)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create dataset
    dataset = L1000Dataset(expr_matrix, metadata, drug_fingerprints, vocab)

    # Split indices
    n_samples = len(dataset)
    n_train = int(n_samples * train_split)

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader
