"""Chemical-space description of the L1000 compounds (Section 2.1, Supplementary Table S2)."""

import pickle
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import get_cold_drug_split, get_random_split, load_all  # noqa: E402
from src.paths import RESULTS_DIR  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def tanimoto(fp_a, fp_b):
    return np.bitwise_and(fp_a, fp_b).sum() / max(np.bitwise_or(fp_a, fp_b).sum(), 1)


def main():
    data = load_all()
    pert_ids, smiles = data["vocab"]["pert_id"], data["smiles_raw"]

    fps = {}
    for pid in tqdm(pert_ids, desc="Morgan fingerprints"):
        mol = Chem.MolFromSmiles(smiles[pid]) if smiles.get(pid) is not None else None
        if mol is None:
            continue
        arr = np.zeros(2048, dtype=np.uint8)
        AllChem.DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), arr)
        fps[pid] = arr
    valid_ids = list(fps.keys())
    fp_mat = np.stack([fps[p] for p in valid_ids], axis=0)

    rng = np.random.RandomState(42)
    idx_a = rng.randint(0, len(valid_ids), 50_000)
    idx_b = rng.randint(0, len(valid_ids), 50_000)
    keep = idx_a != idx_b
    tani = np.array([tanimoto(fp_mat[a], fp_mat[b]) for a, b in zip(idx_a[keep], idx_b[keep])])

    scaffold_counts = {}
    for pid in tqdm(valid_ids, desc="Scaffolds"):
        try:
            scaf = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(smiles[pid])))
        except Exception:
            continue
        scaf = scaf or "<acyclic>"
        scaffold_counts[scaf] = scaffold_counts.get(scaf, 0) + 1
    counts = np.array(sorted(scaffold_counts.values(), reverse=True))
    cum = np.cumsum(counts) / counts.sum()

    canon = data["pid_to_canon"]
    canon_to_pids = {}
    for p, c in canon.items():
        if c is not None:
            canon_to_pids.setdefault(c, []).append(p)
    shared = sum(len(v) for v in canon_to_pids.values() if len(v) > 1)
    _, test_idx, _, test_pids, _, test_canon = get_cold_drug_split(data["meta"], canon, seed=0)

    # Overlap between the 10,000-signature evaluation pool (Table 1) and the training split
    meta = data["meta"]
    train_idx, val_idx = get_random_split(len(meta), seed=42)
    pool = val_idx[np.random.RandomState(7).choice(len(val_idx), 10000, replace=False)]
    cols = ["pert_id", "cell_id", "dose_bin", "pert_time"]
    train_pids = set(meta["pert_id"].values[train_idx])
    train_conditions = set(map(tuple, meta[cols].values[train_idx]))
    pool_rows = meta[cols].values[pool]
    pool_overlap = dict(compound=float(np.mean([r[0] in train_pids for r in pool_rows])),
                        condition=float(np.mean([tuple(r) in train_conditions for r in pool_rows])))

    out = dict(n_drugs=len(pert_ids), n_valid_fps=len(valid_ids), avg_bits_on=float(fp_mat.sum(axis=1).mean()),
               pairwise_tanimoto=tani, n_unique_scaffolds=len(scaffold_counts),
               scaffold_singletons=int((counts == 1).sum()), scaffold_top10_coverage=float(cum[9]),
               scaffold_top50_coverage=float(cum[49]), scaffold_top100_coverage=float(cum[99]),
               n_unique_canonical=len(canon_to_pids), n_pids_sharing_canonical=shared,
               zero_shot_test=dict(n_canonical=len(test_canon), n_pids=len(test_pids), n_signatures=len(test_idx)),
               pool_overlap=pool_overlap)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "chemical_space.pkl", "wb") as f:
        pickle.dump(out, f)

    print(f"pert_ids: {len(pert_ids)}; valid SMILES: {len(valid_ids)}; unique canonical SMILES: "
          f"{len(canon_to_pids)}; pert_ids sharing a canonical SMILES: {shared}")
    print(f"Mean on-bits: {out['avg_bits_on']:.1f}")
    print(f"Pairwise Tanimoto (n = {len(tani)}): mean {tani.mean():.3f}, median {np.median(tani):.3f}, "
          f"P95 {np.percentile(tani, 95):.3f}, P99 {np.percentile(tani, 99):.3f}")
    print(f"Scaffolds: {len(scaffold_counts)} ({out['scaffold_singletons']} singletons); top-10/50/100 coverage "
          f"{cum[9]:.1%} / {cum[49]:.1%} / {cum[99]:.1%}")
    print(f"Zero-shot test set: {len(test_canon)} canonical compounds, {len(test_pids)} pert_ids, "
          f"{len(test_idx)} signatures")
    print(f"Evaluation pool: compound in training {pool_overlap['compound']:.1%}, "
          f"same drug-cell-dose-time combination in training {pool_overlap['condition']:.1%}")


if __name__ == "__main__":
    main()
