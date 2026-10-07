"""2,048-bit versus 8,192-bit Morgan fingerprints on three canonical-SMILES splits.

Produces results/zero_shot_multisplit.pkl (Supplementary Table S4).
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_all  # noqa: E402
from src.paths import RESULTS_DIR  # noqa: E402
from src.training import (TANIMOTO_BINS, build_drug_features, evaluate_zero_shot,  # noqa: E402
                          train_on_cold_drug_split)

SPLIT_SEEDS = [0, 1, 2]
MODELS = {"morgan_2048": "morgan_v1", "morgan_8192": "morgan_r2_wide_8192"}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_all()
    tables = {key: build_drug_features(name, data, device)[0] for key, name in MODELS.items()}

    per_split = {}
    for split_seed in SPLIT_SEEDS:
        per_split[split_seed] = {}
        for key, name in MODELS.items():
            print(f"\n=== split {split_seed}: {key} ===", flush=True)
            model = train_on_cold_drug_split(name, data, tables[key], device, split_seed=split_seed, seed=42)
            zs = evaluate_zero_shot(model, data, device, tables[key], split_seed=split_seed)
            per_split[split_seed][key] = dict(
                hit10_all=zs["zs_hit10"], n_all=zs["zs_n"],
                strata={b: {k: zs["strata"][b][k] for k in ("n", "hit1", "hit10", "hit50")}
                        for b, _ in TANIMOTO_BINS})
            print(f"  Hit@10 = {zs['zs_hit10']:.2f}%", flush=True)

    aggregate = {}
    for b, _ in TANIMOTO_BINS:
        aggregate[b] = {"n_samples": [per_split[s]["morgan_2048"]["strata"][b]["n"] for s in SPLIT_SEEDS]}
        for key in MODELS:
            values = [per_split[s][key]["strata"][b]["hit10"] for s in SPLIT_SEEDS]
            aggregate[b][key] = dict(mean=np.mean(values), se=np.std(values, ddof=1) / np.sqrt(len(values)),
                                     values=values)

    out = dict(split_seeds=SPLIT_SEEDS, results_per_split=per_split, aggregate=aggregate)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "zero_shot_multisplit.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"Saved {RESULTS_DIR / 'zero_shot_multisplit.pkl'}")


if __name__ == "__main__":
    main()
