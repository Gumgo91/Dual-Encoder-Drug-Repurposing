"""Canonical-SMILES-disjoint retrieval on split seed 0.

Produces results/zero_shot.pkl (Table 2, Figures 3-4, Supplementary Table S3) for the
2,048-bit Morgan fingerprint, the radius 0-3 concatenation and the 8,192-bit variant.
"""

import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_all  # noqa: E402
from src.paths import RESULTS_DIR  # noqa: E402
from src.training import build_drug_features, evaluate_zero_shot, train_on_cold_drug_split  # noqa: E402

MODELS = ["morgan_v1", "multiradius_concat", "morgan_r2_wide_8192"]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_all()
    results = {}
    for name in MODELS:
        print(f"\n=== {name} ===", flush=True)
        table, _ = build_drug_features(name, data, device)
        model = train_on_cold_drug_split(name, data, table, device, split_seed=0, seed=42)
        results[name] = evaluate_zero_shot(model, data, device, table, split_seed=0)
        print(f"  Hit@10 = {results[name]['zs_hit10']:.2f}% (n = {results[name]['zs_n']})", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "zero_shot.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"Saved {RESULTS_DIR / 'zero_shot.pkl'}")


if __name__ == "__main__":
    main()
