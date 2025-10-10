"""
Zero-shot evaluation: Train on seen drugs, test on unseen drugs (cold-drug split).
"""
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import L1000DataLoader
from src.dataset import L1000Dataset
from src.encoders import DualEncoder
from src.evaluation import cold_drug_split, compute_retrieval_metrics


def train_model(model, train_loader, device, n_epochs=10):
    """Train model on training set."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            expr, drug_fp, cell_idx, dose_idx, time_idx = [x.to(device) for x in batch]

            # Forward
            z_expr, z_drug = model(expr, drug_fp, cell_idx, dose_idx, time_idx)
            loss = model.compute_loss(z_expr, z_drug)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.4f}")

    return model


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()

    all_z_expr = []
    all_z_drug = []

    with torch.no_grad():
        for batch in test_loader:
            expr, drug_fp, cell_idx, dose_idx, time_idx = [x.to(device) for x in batch]
            z_expr, z_drug = model(expr, drug_fp, cell_idx, dose_idx, time_idx)

            all_z_expr.append(z_expr.cpu())
            all_z_drug.append(z_drug.cpu())

    z_expr = torch.cat(all_z_expr, dim=0)
    z_drug = torch.cat(all_z_drug, dim=0)

    # Compute similarity matrix
    similarities = z_expr @ z_drug.T  # [N, N]

    # Compute metrics
    metrics = compute_retrieval_metrics(similarities.numpy(), k_values=[1, 5, 10, 20, 50])

    return metrics


def run_zero_shot_experiment(data_dir="data", device="cuda", n_epochs=10):
    """
    Run zero-shot experiment with cold-drug split.

    Train on 80% of drugs, test on remaining 20% (unseen drugs).
    """
    print("=" * 80)
    print("Zero-Shot Evaluation (Cold-Drug Split)")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    loader = L1000DataLoader(data_dir)
    expr_matrix, metadata, vocab = loader.prepare_training_data()
    drug_fingerprints = loader.get_fingerprints()

    # Create dataset
    dataset = L1000Dataset(expr_matrix, metadata, drug_fingerprints, vocab)

    # Cold-drug split: test drugs never seen in training
    print("\nCreating cold-drug split (20% unseen drugs)...")
    train_idx, test_idx = cold_drug_split(metadata, test_ratio=0.2)

    # Check unique drugs
    train_drugs = set(metadata.iloc[train_idx]['pert_id'].unique())
    test_drugs = set(metadata.iloc[test_idx]['pert_id'].unique())
    overlap = train_drugs & test_drugs

    print(f"\nSplit statistics:")
    print(f"  Train: {len(train_idx)} samples, {len(train_drugs)} unique drugs")
    print(f"  Test:  {len(test_idx)} samples, {len(test_drugs)} unique drugs")
    print(f"  Overlap: {len(overlap)} drugs (should be 0 for true zero-shot)")

    # Create data loaders
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=512,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=512,
        shuffle=False,
        num_workers=0
    )

    # Create model
    print(f"\nCreating model...")
    model = DualEncoder(
        expr_input_dim=978,
        fingerprint_dim=2048,
        n_cells=len(vocab['cell_id']),
        n_doses=len(vocab['dose_bin']),
        n_times=len(vocab['pert_time']),
        embed_dim=256,
        hidden_dim=512,
        dropout=0.1
    ).to(device)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print(f"\nTraining on {len(train_drugs)} seen drugs...")
    model = train_model(model, train_loader, device, n_epochs=n_epochs)

    # Evaluate on unseen drugs (zero-shot)
    print(f"\nEvaluating on {len(test_drugs)} unseen drugs (zero-shot)...")
    metrics = evaluate_model(model, test_loader, device)

    # Print results
    print("\n" + "=" * 80)
    print("Zero-Shot Results (Unseen Drugs)")
    print("=" * 80)

    # Extract Hit@k values
    for key in sorted([k for k in metrics.keys() if k.startswith('hit@')]):
        k_val = int(key.split('@')[1])
        print(f"  Hit@{k_val:3d}: {metrics[key]:6.2f}%")

    print(f"\n  MRR:       {metrics['mrr']:.4f}")

    # Extract nDCG@k values
    for key in sorted([k for k in metrics.keys() if k.startswith('ndcg@')]):
        k_val = int(key.split('@')[1])
        print(f"  nDCG@{k_val:2d}:   {metrics[key]:.4f}")

    # Save results
    output_file = "results/evaluation/zero_shot_results.pkl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    results = {
        'metrics': metrics,
        'train_drugs': len(train_drugs),
        'test_drugs': len(test_drugs),
        'train_samples': len(train_idx),
        'test_samples': len(test_idx),
        'overlap': len(overlap)
    }

    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Zero-shot evaluation")
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()

    run_zero_shot_experiment(
        data_dir=args.data_dir,
        device=args.device,
        n_epochs=args.epochs
    )
