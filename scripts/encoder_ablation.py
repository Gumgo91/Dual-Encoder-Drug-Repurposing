"""Encoder comparison on 10,000 held-out signatures with three training seeds.

Produces results/encoder_ablation.pkl (Table 1, Table 3, Supplementary Table S1, Figure 2).
The ChemBERTa encoder needs cache/chemberta_embeddings.npy (scripts/chemberta_embeddings.py).
"""

import argparse
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_all  # noqa: E402
from src.paths import RESULTS_DIR  # noqa: E402
from src.training import ENCODERS, build_drug_features, train_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoders", nargs="+", default=ENCODERS, choices=ENCODERS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 11])
    parser.add_argument("--out", default=str(RESULTS_DIR / "encoder_ablation.pkl"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_all()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    for name in args.encoders:
        table, extras = build_drug_features(name, data, device)
        for seed in args.seeds:
            metrics, _ = train_model(name, data, device=device, seed=seed,
                                     drug_feat_table=table, extras=extras)
            results[f"{name}_seed{seed}"] = metrics
            with open(out, "wb") as f:
                pickle.dump(results, f)
    print(f"Saved {out}. Summarize with: python scripts/summarize_results.py")


if __name__ == "__main__":
    main()
