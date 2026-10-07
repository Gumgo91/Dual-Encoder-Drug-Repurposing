"""Train the dual encoder used for the embedding-similarity analysis (Figure 5).

The model uses the 90% training split, AdamW (learning rate 1e-3, weight decay 1e-4),
batch size 512 and a constant learning rate for 10 epochs. The checkpoint analysed in
the paper is provided as models/embedding_model.pt; it was trained without a fixed
seed, so a new run gives a close but not identical model.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_all  # noqa: E402
from src.dataset import create_dataloaders  # noqa: E402
from src.encoders import DualEncoder  # noqa: E402
from src.paths import MODELS_DIR  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=str(MODELS_DIR / "embedding_model_retrained.pt"))
    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_all()
    vocab = data["vocab"]
    train_loader, val_loader = create_dataloaders(data["expr"], data["meta"], data["morgan_fps"], vocab,
                                                  batch_size=args.batch_size, train_split=0.9, seed=42)
    model = DualEncoder(expr_input_dim=978, fingerprint_dim=2048, n_cells=len(vocab["cell_id"]),
                        n_doses=len(vocab["dose_bin"]), n_times=len(vocab["pert_time"]),
                        embed_dim=256, hidden_dim=512, dropout=0.1).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            expr, drug_fp, cell_idx, dose_idx, time_idx = [x.to(device) for x in batch]
            loss = model.compute_loss(*model(expr, drug_fp, cell_idx, dose_idx, time_idx))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                expr, drug_fp, cell_idx, dose_idx, time_idx = [x.to(device) for x in batch]
                val_loss += model.compute_loss(*model(expr, drug_fp, cell_idx, dose_idx, time_idx)).item()
        print(f"Epoch {epoch}: train loss {train_loss / len(train_loader):.4f}, "
              f"validation loss {val_loss / len(val_loader):.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "vocab": vocab,
                "n_samples": len(data["meta"]), "epochs": args.epochs}, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
