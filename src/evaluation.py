"""Advanced evaluation metrics for drug repurposing."""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr
from collections import defaultdict


def compute_retrieval_metrics(
    similarities: np.ndarray,
    k_values: List[int] = [1, 5, 10, 20, 50]
) -> Dict[str, float]:
    """Compute comprehensive retrieval metrics.

    Args:
        similarities: Similarity matrix [N, N] where diagonal are true pairs
        k_values: List of k for top-k metrics

    Returns:
        Dictionary of metrics
    """
    n = similarities.shape[0]
    metrics = {}

    # Get rankings (argsort descending)
    rankings = np.argsort(-similarities, axis=1)

    # True labels are diagonal (each sample matches itself)
    true_labels = np.arange(n)

    # Compute metrics for each sample
    reciprocal_ranks = []
    hits_at_k = {k: [] for k in k_values}

    for i in range(n):
        # Find where true label appears in ranking
        true_pos = np.where(rankings[i] == true_labels[i])[0][0]
        rank = true_pos + 1  # 1-indexed

        # Reciprocal Rank
        reciprocal_ranks.append(1.0 / rank)

        # Hit@k
        for k in k_values:
            hits_at_k[k].append(1.0 if true_pos < k else 0.0)

    # Mean Reciprocal Rank (MRR)
    metrics['mrr'] = np.mean(reciprocal_ranks)

    # Hit@k (same as Top-k accuracy)
    for k in k_values:
        metrics[f'hit@{k}'] = np.mean(hits_at_k[k]) * 100  # Percentage

    # nDCG@k
    for k in k_values:
        dcg_scores = []
        for i in range(n):
            true_pos = np.where(rankings[i] == true_labels[i])[0][0]
            if true_pos < k:
                # DCG = 1 / log2(rank + 1)
                dcg = 1.0 / np.log2(true_pos + 2)
            else:
                dcg = 0.0
            # IDCG = 1 / log2(2) = 1.0 (perfect ranking)
            idcg = 1.0
            ndcg = dcg / idcg
            dcg_scores.append(ndcg)
        metrics[f'ndcg@{k}'] = np.mean(dcg_scores)

    return metrics


def compute_moa_hit_at_k(
    rankings: np.ndarray,
    moa_map: Dict[str, set],
    drug_vocab: List[str],
    k_values: List[int] = [1, 5, 10, 20, 50]
) -> Dict[str, float]:
    """Compute MOA-level hit@k.

    If any drug in top-k shares MOA with true drug, count as hit.

    Args:
        rankings: Drug rankings [N, N] (indices into drug_vocab)
        moa_map: Mapping from pert_id to set of MOA terms
        drug_vocab: List of drug IDs (pert_id)
        k_values: List of k values

    Returns:
        Dictionary of MOA-level hit@k metrics
    """
    metrics = {}
    n = rankings.shape[0]

    hits_at_k = {k: [] for k in k_values}

    for i in range(n):
        true_drug = drug_vocab[i]
        true_moas = moa_map.get(true_drug, set())

        if len(true_moas) == 0:
            continue  # Skip drugs without MOA annotation

        for k in k_values:
            hit = False
            for rank_idx in rankings[i, :k]:
                pred_drug = drug_vocab[rank_idx]
                pred_moas = moa_map.get(pred_drug, set())

                # Check MOA overlap
                if len(true_moas & pred_moas) > 0:
                    hit = True
                    break

            hits_at_k[k].append(1.0 if hit else 0.0)

    for k in k_values:
        if len(hits_at_k[k]) > 0:
            metrics[f'moa_hit@{k}'] = np.mean(hits_at_k[k]) * 100
        else:
            metrics[f'moa_hit@{k}'] = 0.0

    return metrics


def compute_moa_hit_at_k_with_ids(
    similarities: np.ndarray,
    sample_drug_ids: np.ndarray,
    moa_map: Dict[str, set],
    k_values: List[int] = [1, 5, 10, 20, 50]
) -> Dict[str, float]:
    """Compute MOA-level hit@k using actual drug IDs.

    Args:
        similarities: Similarity matrix [N, N]
        sample_drug_ids: Drug IDs for samples [N] (pert_id strings)
        moa_map: Mapping from pert_id to set of MOA terms
        k_values: List of k values

    Returns:
        Dictionary of MOA-level hit@k metrics
    """
    metrics = {}
    n = similarities.shape[0]

    # Get rankings
    rankings = np.argsort(-similarities, axis=1)

    hits_at_k = {k: [] for k in k_values}
    n_evaluated = 0

    for i in range(n):
        true_drug = sample_drug_ids[i]
        true_moas = moa_map.get(true_drug, set())

        if len(true_moas) == 0:
            continue  # Skip drugs without MOA annotation

        n_evaluated += 1

        for k in k_values:
            hit = False
            for rank_idx in rankings[i, :k]:
                pred_drug = sample_drug_ids[rank_idx]
                pred_moas = moa_map.get(pred_drug, set())

                # Check MOA overlap
                if len(true_moas & pred_moas) > 0:
                    hit = True
                    break

            hits_at_k[k].append(1.0 if hit else 0.0)

    for k in k_values:
        if len(hits_at_k[k]) > 0:
            metrics[f'moa_hit@{k}'] = np.mean(hits_at_k[k]) * 100
        else:
            metrics[f'moa_hit@{k}'] = 0.0

    # Add metadata
    metrics['n_evaluated'] = n_evaluated
    metrics['n_total'] = n

    return metrics


def compute_chemical_embedding_correlation(
    drug_fingerprints: Dict[str, np.ndarray],
    drug_embeddings: np.ndarray,
    drug_vocab: List[str],
    n_samples: int = 1000
) -> Tuple[float, float]:
    """Compute correlation between chemical similarity and embedding similarity.

    Args:
        drug_fingerprints: Mapping from pert_id to fingerprint
        drug_embeddings: Learned embeddings [N_drugs, embed_dim]
        drug_vocab: List of drug IDs
        n_samples: Number of pairs to sample

    Returns:
        (spearman_r, p_value)
    """
    # Sample random pairs
    n_drugs = len(drug_vocab)
    pairs = np.random.choice(n_drugs, size=(n_samples, 2), replace=True)

    chem_sims = []
    embed_sims = []

    for i, j in pairs:
        if i == j:
            continue

        drug_i = drug_vocab[i]
        drug_j = drug_vocab[j]

        # Chemical similarity (Tanimoto)
        fp_i = drug_fingerprints.get(drug_i)
        fp_j = drug_fingerprints.get(drug_j)

        if fp_i is None or fp_j is None:
            continue

        tanimoto = np.sum(fp_i * fp_j) / (np.sum(fp_i) + np.sum(fp_j) - np.sum(fp_i * fp_j) + 1e-8)

        # Embedding similarity (cosine)
        emb_i = drug_embeddings[i]
        emb_j = drug_embeddings[j]
        cos_sim = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j) + 1e-8)

        chem_sims.append(tanimoto)
        embed_sims.append(cos_sim)

    # Spearman correlation
    r, p = spearmanr(chem_sims, embed_sims)

    return r, p


def permutation_test(
    model,
    disease_signature: np.ndarray,
    drug_embeddings: np.ndarray,
    n_permutations: int = 1000,
    device: str = 'cuda'
) -> Tuple[np.ndarray, float]:
    """Permutation test for disease signature ranking.

    Args:
        model: Expression encoder
        disease_signature: Disease expression profile [n_genes]
        drug_embeddings: Drug embeddings [N_drugs, embed_dim]
        n_permutations: Number of permutations
        device: Device for computation

    Returns:
        (null_distribution, p_value)
    """
    # True ranking
    disease_tensor = torch.from_numpy(disease_signature).float().to(device).unsqueeze(0)
    with torch.no_grad():
        true_emb = model(disease_tensor).cpu().numpy()[0]

    true_sims = drug_embeddings @ true_emb
    true_max_sim = np.max(true_sims)

    # Null distribution
    null_max_sims = []

    for _ in range(n_permutations):
        # Permute genes
        perm_sig = np.random.permutation(disease_signature)
        perm_tensor = torch.from_numpy(perm_sig).float().to(device).unsqueeze(0)

        with torch.no_grad():
            perm_emb = model(perm_tensor).cpu().numpy()[0]

        perm_sims = drug_embeddings @ perm_emb
        null_max_sims.append(np.max(perm_sims))

    null_distribution = np.array(null_max_sims)

    # P-value: fraction of null max >= true max
    p_value = np.mean(null_distribution >= true_max_sim)

    return null_distribution, p_value


def fingerprint_shuffle_test(
    drug_encoder,
    drug_fingerprints: Dict[str, np.ndarray],
    drug_vocab: List[str],
    cell_idx: torch.Tensor,
    dose_idx: torch.Tensor,
    time_idx: torch.Tensor,
    n_shuffles: int = 100,
    device: str = 'cuda'
) -> Tuple[float, float]:
    """Test if fingerprint structure matters by shuffling bits.

    Args:
        drug_encoder: Drug encoder model
        drug_fingerprints: Original fingerprints
        drug_vocab: Drug vocabulary
        cell_idx, dose_idx, time_idx: Context tensors
        n_shuffles: Number of shuffles
        device: Device

    Returns:
        (true_variance, mean_null_variance)
    """
    # True embeddings
    fp_list = [drug_fingerprints[d] for d in drug_vocab]
    fp_tensor = torch.from_numpy(np.array(fp_list)).float().to(device)

    with torch.no_grad():
        true_embs = drug_encoder(fp_tensor, cell_idx, dose_idx, time_idx).cpu().numpy()

    true_variance = np.var(true_embs)

    # Null embeddings (shuffled bits)
    null_variances = []

    for _ in range(n_shuffles):
        # Shuffle each fingerprint's bits independently
        shuffled_fps = []
        for fp in fp_list:
            shuffled_fp = np.random.permutation(fp)
            shuffled_fps.append(shuffled_fp)

        shuffled_tensor = torch.from_numpy(np.array(shuffled_fps)).float().to(device)

        with torch.no_grad():
            null_embs = drug_encoder(shuffled_tensor, cell_idx, dose_idx, time_idx).cpu().numpy()

        null_variances.append(np.var(null_embs))

    mean_null_variance = np.mean(null_variances)

    return true_variance, mean_null_variance


class GroupKFold:
    """Group-based K-Fold for preventing data leakage.

    Ensures (drug, cell, dose, time) combinations don't appear in both splits.
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def split(self, metadata, group_cols=['pert_id', 'cell_id', 'dose_bin', 'pert_time']):
        """Generate train/val splits.

        Args:
            metadata: DataFrame with group columns
            group_cols: Columns defining groups

        Yields:
            (train_indices, val_indices)
        """
        # Create group identifier
        metadata = metadata.copy()
        metadata['group'] = metadata[group_cols].apply(
            lambda x: '|'.join(x.astype(str)), axis=1
        )

        unique_groups = metadata['group'].unique()
        np.random.shuffle(unique_groups)

        fold_size = len(unique_groups) // self.n_splits

        for i in range(self.n_splits):
            val_groups = unique_groups[i * fold_size:(i + 1) * fold_size]
            train_groups = np.setdiff1d(unique_groups, val_groups)

            train_idx = metadata[metadata['group'].isin(train_groups)].index.values
            val_idx = metadata[metadata['group'].isin(val_groups)].index.values

            yield train_idx, val_idx


def cold_drug_split(metadata, test_ratio: float = 0.2):
    """Split by drugs: test drugs never seen in training.

    Args:
        metadata: DataFrame with 'pert_id' column
        test_ratio: Fraction of drugs for testing

    Returns:
        (train_indices, test_indices)
    """
    unique_drugs = metadata['pert_id'].unique()
    np.random.shuffle(unique_drugs)

    n_test = int(len(unique_drugs) * test_ratio)
    test_drugs = unique_drugs[:n_test]
    train_drugs = unique_drugs[n_test:]

    train_idx = metadata[metadata['pert_id'].isin(train_drugs)].index.values
    test_idx = metadata[metadata['pert_id'].isin(test_drugs)].index.values

    return train_idx, test_idx


def cold_cell_split(metadata, test_ratio: float = 0.2):
    """Split by cell lines: test cells never seen in training.

    Args:
        metadata: DataFrame with 'cell_id' column
        test_ratio: Fraction of cells for testing

    Returns:
        (train_indices, test_indices)
    """
    unique_cells = metadata['cell_id'].unique()
    np.random.shuffle(unique_cells)

    n_test = int(len(unique_cells) * test_ratio)
    test_cells = unique_cells[:n_test]
    train_cells = unique_cells[n_test:]

    train_idx = metadata[metadata['cell_id'].isin(train_cells)].index.values
    test_idx = metadata[metadata['cell_id'].isin(test_cells)].index.values

    return train_idx, test_idx
