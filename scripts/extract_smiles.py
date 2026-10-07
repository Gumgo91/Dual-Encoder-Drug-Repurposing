"""Write data/drug_smiles.pkl (pert_id -> SMILES) from the GSE92742 compound metadata."""

import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import DATA_DIR  # noqa: E402


def main():
    pert_info = pd.read_csv(DATA_DIR / "GSE92742_Broad_LINCS_pert_info.txt.gz", sep="\t", low_memory=False)
    compounds = pert_info[pert_info["pert_type"] == "trt_cp"]
    drug_smiles = {}
    for pert_id, smiles in zip(compounds["pert_id"], compounds["canonical_smiles"]):
        if pd.notna(smiles) and isinstance(smiles, str) and len(smiles) > 0:
            drug_smiles[pert_id] = smiles
    out = DATA_DIR / "drug_smiles.pkl"
    with open(out, "wb") as f:
        pickle.dump(drug_smiles, f)
    print(f"Saved SMILES for {len(drug_smiles)} compounds to {out}")


if __name__ == "__main__":
    main()
