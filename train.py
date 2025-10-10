"""Quick training script with smaller dataset for testing."""

import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from src.data_loader import L1000DataLoader
from src.dataset import create_dataloaders
from src.encoders import DualEncoder
from tqdm import tqdm
from src.utils import parse_dose_um, bin_dose, create_categorical_indices

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train DaCoCo model")
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Number of samples to use (default: 10000, use -1 for all)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size (default: 512)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs (default: 5)')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory (default: data)')
    parser.add_argument('--output', type=str, default='results/models/dacoco_model.pt',
                        help='Output model path')
    args = parser.parse_args()

    print("=" * 60)
    print("DaCoCo Training")
    print("=" * 60)

    # Settings
    DATA_DIR = args.data_dir
    N_SAMPLES = args.n_samples
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nDevice: {DEVICE}")
    print(f"Samples: {'ALL' if N_SAMPLES == -1 else N_SAMPLES}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")

    # Load data
    print("\n" + "=" * 60)
    print("Loading Data")
    print("=" * 60)

    loader = L1000DataLoader(DATA_DIR)
    expr_matrix, metadata, vocab = loader.prepare_training_data(
        pert_type="trt_cp",
        time_hours=24,
        min_replicates=2,
        use_landmark_only=True
    )

    # Sample randomly or use all
    if N_SAMPLES != -1 and N_SAMPLES < len(metadata):
        print(f"\nSampling {N_SAMPLES} signatures...")
        sample_indices = np.random.choice(len(metadata), N_SAMPLES, replace=False)
        expr_matrix = expr_matrix[sample_indices]
        metadata = metadata.iloc[sample_indices].reset_index(drop=True)
        print(f"Sampled {len(metadata)} signatures")

    # Get fingerprints
    drug_fingerprints = loader.get_fingerprints()

    print(f"\nFinal dataset:")
    print(f"  Expression: {expr_matrix.shape}")
    print(f"  Drugs: {len(vocab['pert_id'])}")
    print(f"  Cells: {len(vocab['cell_id'])}")
    print(f"  Fingerprints: {len(drug_fingerprints)}")

    # Create dataloaders with num_workers=0 for Windows
    print("\n" + "=" * 60)
    print("Creating Dataloaders")
    print("=" * 60)

    train_loader, val_loader = create_dataloaders(
        expr_matrix, metadata, drug_fingerprints, vocab,
        batch_size=BATCH_SIZE,
        train_split=0.9,
        num_workers=0,  # Windows compatibility
        seed=42
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print("\n" + "=" * 60)
    print("Creating Model")
    print("=" * 60)

    model = DualEncoder(
        expr_input_dim=978,
        fingerprint_dim=2048,
        n_cells=len(vocab['cell_id']),
        n_doses=len(vocab['dose_bin']),
        n_times=len(vocab['pert_time']),
        embed_dim=256,
        hidden_dim=512,
        dropout=0.1
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Training
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch in pbar:
            expr, drug_fp, cell_idx, dose_idx, time_idx = [x.to(DEVICE) for x in batch]

            z_expr, z_drug = model(expr, drug_fp, cell_idx, dose_idx, time_idx)
            loss = model.compute_loss(z_expr, z_drug)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                expr, drug_fp, cell_idx, dose_idx, time_idx = [x.to(DEVICE) for x in batch]
                z_expr, z_drug = model(expr, drug_fp, cell_idx, dose_idx, time_idx)

                loss = model.compute_loss(z_expr, z_drug)
                val_loss += loss.item()

                # Top-1 accuracy
                logits = z_expr @ z_drug.T
                preds = logits.argmax(dim=1)
                labels = torch.arange(len(expr), device=DEVICE)
                correct += (preds == labels).sum().item()
                total += len(expr)

        val_loss /= len(val_loader)
        val_acc = correct / total * 100

        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Acc: {val_acc:.2f}%")

    # Save model
    print("\n" + "=" * 60)
    print("Saving Model")
    print("=" * 60)

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'n_samples': len(metadata),
        'epochs': EPOCHS,
    }, args.output)

    print(f"Saved to: {args.output}")
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == '__main__':
    main()
