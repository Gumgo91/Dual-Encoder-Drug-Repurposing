"""PyTorch dataset and loaders for scripts/train_embedding_model.py."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class L1000Dataset(Dataset):
    """(expression, fingerprint, cell, dose, time) tuples, one per signature."""

    def __init__(self, expr_matrix: np.ndarray, metadata: pd.DataFrame,
                 drug_fingerprints: Dict[str, np.ndarray], vocab: Dict[str, list]):
        self.expr = torch.from_numpy(expr_matrix).float()
        fp_dim = next(iter(drug_fingerprints.values())).shape[0]
        fp_list = [drug_fingerprints.get(p, np.zeros(fp_dim, dtype=np.float32)) for p in vocab["pert_id"]]
        self.drug_fps = torch.from_numpy(np.array(fp_list)).float()
        self.drug_idx = torch.from_numpy(metadata["pert_id_idx"].values).long()
        self.cell_idx = torch.from_numpy(metadata["cell_id_idx"].values).long()
        self.dose_idx = torch.from_numpy(metadata["dose_bin_idx"].values).long()
        self.time_idx = torch.from_numpy(metadata["pert_time_idx"].values).long()

    def __len__(self) -> int:
        return len(self.expr)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return (self.expr[idx], self.drug_fps[self.drug_idx[idx]],
                self.cell_idx[idx], self.dose_idx[idx], self.time_idx[idx])


def create_dataloaders(expr_matrix, metadata, drug_fingerprints, vocab, batch_size: int = 512,
                       train_split: float = 0.9, num_workers: int = 0, seed: int = 42):
    """Signature-level random split (same permutation as src.data.get_random_split)."""
    dataset = L1000Dataset(expr_matrix, metadata, drug_fingerprints, vocab)
    n_train = int(len(dataset) * train_split)
    indices = np.random.RandomState(seed).permutation(len(dataset))
    train_set = torch.utils.data.Subset(dataset, indices[:n_train])
    val_set = torch.utils.data.Subset(dataset, indices[n_train:])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers, pin_memory=True, drop_last=False)
    return train_loader, val_loader
