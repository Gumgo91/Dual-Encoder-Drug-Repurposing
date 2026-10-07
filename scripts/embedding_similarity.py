"""Tanimoto similarity versus learned drug-embedding similarity (Section 3.3, Figure 5).

Drug-level embeddings are the mean drug-encoder output over all signatures of a compound.
50,000 random compound pairs are drawn (self-pairs removed), and the Spearman correlation
between Tanimoto similarity and embedding cosine similarity is reported with a 1,000-sample
bootstrap CI. Random Gaussian unit vectors serve as a negative control.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_all  # noqa: E402
from src.encoders import DualEncoder  # noqa: E402
from src.paths import MODELS_DIR, RESULTS_DIR  # noqa: E402


def tanimoto_pair(a, b):
    return np.logical_and(a, b).sum() / max(np.logical_or(a, b).sum(), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(MODELS_DIR / "embedding_model.pt"))
    parser.add_argument("--out", default=str(RESULTS_DIR / "embedding_similarity.pkl"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    vocab = ckpt["vocab"]
    data = load_all()
    assert vocab["pert_id"] == data["vocab"]["pert_id"], "checkpoint vocabulary does not match the data"
    metadata, fp_dict = data["meta"], data["morgan_fps"]

    model = DualEncoder(expr_input_dim=978, fingerprint_dim=2048, n_cells=len(vocab["cell_id"]),
                        n_doses=len(vocab["dose_bin"]), n_times=len(vocab["pert_time"]),
                        embed_dim=256, hidden_dim=512, dropout=0.1).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    drug_vocab = vocab["pert_id"]
    n_drugs = len(drug_vocab)
    fp_t = torch.from_numpy(np.stack([fp_dict.get(p, np.zeros(2048, dtype=np.float32)) for p in drug_vocab],
                                     axis=0).astype(np.float32)).to(device)
    as_long = lambda col: torch.from_numpy(metadata[col].values).long().to(device)
    cell_t, dose_t, time_t, drug_t = (as_long("cell_id_idx"), as_long("dose_bin_idx"),
                                      as_long("pert_time_idx"), as_long("pert_id_idx"))

    # Mean drug-encoder output per compound over all of its signatures
    sum_emb = np.zeros((n_drugs, 256), dtype=np.float64)
    cnt = np.zeros(n_drugs, dtype=np.int64)
    with torch.no_grad():
        for i in tqdm(range(0, len(metadata), 4096), desc="Drug embeddings"):
            sl = slice(i, min(i + 4096, len(metadata)))
            z = model.drug_encoder(fp_t[drug_t[sl]], cell_t[sl], dose_t[sl], time_t[sl]).cpu().numpy()
            np.add.at(sum_emb, drug_t[sl].cpu().numpy(), z)
            np.add.at(cnt, drug_t[sl].cpu().numpy(), 1)
    drug_emb = np.where(cnt[:, None] > 0, sum_emb / np.maximum(cnt[:, None], 1), 0.0)

    # Compounds with a non-zero fingerprint
    valid_fps, valid_embs = [], []
    for i, p in enumerate(drug_vocab):
        fp = fp_dict.get(p)
        if cnt[i] == 0 or fp is None or fp.sum() == 0:
            continue
        valid_fps.append(fp.astype(bool))
        valid_embs.append(drug_emb[i])
    valid_fps, valid_embs = np.stack(valid_fps), np.stack(valid_embs)

    rng = np.random.RandomState(42)
    a = rng.randint(0, len(valid_fps), 50_000)
    b = rng.randint(0, len(valid_fps), 50_000)
    keep = a != b
    a, b = a[keep], b[keep]
    print(f"{len(valid_fps)} compounds, {len(a)} pairs")

    tani = np.array([tanimoto_pair(valid_fps[i], valid_fps[j]) for i, j in tqdm(zip(a, b), total=len(a))],
                    dtype=np.float32)
    eb_a, eb_b = valid_embs[a], valid_embs[b]
    eb_a /= np.linalg.norm(eb_a, axis=1, keepdims=True) + 1e-12
    eb_b /= np.linalg.norm(eb_b, axis=1, keepdims=True) + 1e-12
    cos = (eb_a * eb_b).sum(axis=1)

    rand_emb = np.random.RandomState(0).normal(size=valid_embs.shape).astype(np.float32)
    rand_emb /= np.linalg.norm(rand_emb, axis=1, keepdims=True) + 1e-12
    cos_rand = (rand_emb[a] * rand_emb[b]).sum(axis=1)

    r_model, p_model = spearmanr(tani, cos)
    r_rand, p_rand = spearmanr(tani, cos_rand)

    rng_boot = np.random.RandomState(7)
    boot_model, boot_rand = [], []
    for _ in tqdm(range(1000), desc="Bootstrap"):
        idx = rng_boot.choice(len(tani), size=len(tani), replace=True)
        boot_model.append(spearmanr(tani[idx], cos[idx])[0])
        boot_rand.append(spearmanr(tani[idx], cos_rand[idx])[0])
    ci_model = np.percentile(boot_model, [2.5, 97.5])
    ci_rand = np.percentile(boot_rand, [2.5, 97.5])
    print(f"Model:  Spearman rho = {r_model:.4f} (95% CI {ci_model[0]:.4f} to {ci_model[1]:.4f})")
    print(f"Random: Spearman rho = {r_rand:.4f} (95% CI {ci_rand[0]:.4f} to {ci_rand[1]:.4f})")

    bins = np.linspace(0, 1, 11)
    bin_idx = np.clip(np.digitize(tani, bins) - 1, 0, 9)
    bin_means, bin_stds, bin_counts = np.zeros(10), np.zeros(10), np.zeros(10)
    bin_means_rand, bin_stds_rand = np.zeros(10), np.zeros(10)
    for k in range(10):
        sel = bin_idx == k
        if sel.sum() == 0:
            continue
        bin_counts[k] = sel.sum()
        bin_means[k], bin_stds[k] = cos[sel].mean(), cos[sel].std() / np.sqrt(sel.sum())
        bin_means_rand[k], bin_stds_rand[k] = cos_rand[sel].mean(), cos_rand[sel].std() / np.sqrt(sel.sum())

    out = dict(n_pairs=len(tani), tani=tani, cos=cos, cos_rand=cos_rand,
               spearman_r_model=float(r_model), spearman_p_model=float(p_model),
               spearman_r_random=float(r_rand), spearman_p_random=float(p_rand),
               bootstrap_ci_model=ci_model.tolist(), bootstrap_ci_random=ci_rand.tolist(),
               bin_centers=0.5 * (bins[:-1] + bins[1:]), bin_means_model=bin_means, bin_stds_model=bin_stds,
               bin_means_random=bin_means_rand, bin_stds_random=bin_stds_rand, bin_counts=bin_counts)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
