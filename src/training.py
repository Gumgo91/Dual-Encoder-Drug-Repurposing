"""Training and retrieval evaluation shared by the experiment scripts."""

import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import get_cold_drug_split, get_random_split
from .fingerprints import VARIANTS
from .models import (ChemBERTaDrugEncoder, ChemCPAStyleEncoder, DualEncoder, ExpressionEncoder,
                     GCNDrugEncoder, LearnableDrugEncoder, MorganDrugEncoder,
                     build_pyg_drug_dataset, count_params)
from .paths import CACHE_DIR

ENCODERS = ["learnable", "chemcpa", "gcn_finlayson", "chemberta", "morgan_v1"]


def hit_metrics(scores, k_values=(1, 5, 10, 20, 50)):
    """Hit@k and MRR when query i should retrieve candidate i (the diagonal)."""
    rankings = np.argsort(-scores, axis=1)
    n = scores.shape[0]
    rr = 0.0
    hits = {k: 0 for k in k_values}
    for i in range(n):
        pos = np.where(rankings[i] == i)[0]
        if len(pos) == 0:
            continue
        p = pos[0]
        rr += 1.0 / (p + 1)
        for k in k_values:
            if p < k:
                hits[k] += 1
    return {f"hit@{k}": 100.0 * hits[k] / n for k in k_values} | {"mrr": rr / n, "n": n}


def morgan_table(data):
    """2,048-bit Morgan fingerprints in pert_id vocabulary order."""
    return np.stack([data["morgan_fps"].get(p, np.zeros(2048, dtype=np.float32))
                     for p in data["vocab"]["pert_id"]], axis=0).astype(np.float32)


def variant_table(name, data):
    """Fingerprint variant from canonical SMILES (cached in cache/fp_<name>.npy)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"fp_{name}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    fn, dim = VARIANTS[name]["fn"], VARIANTS[name]["dim"]
    rows = []
    for p in tqdm(data["vocab"]["pert_id"], desc=f"fp {name}"):
        smi = data["pid_to_canon"].get(p)
        v = fn(smi) if smi else None
        rows.append(v if v is not None else np.zeros(dim, dtype=np.float32))
    mat = np.stack(rows, axis=0).astype(np.float32)
    np.save(cache_path, mat)
    return mat


def build_drug_features(name, data, device):
    """Returns (feature table indexed by pert_id_idx or None, extra kwargs for the encoder)."""
    if name in ("morgan_v1", "chemcpa"):
        return torch.from_numpy(morgan_table(data)).to(device), {}
    if name in VARIANTS:
        return torch.from_numpy(variant_table(name, data)).to(device), {}
    if name == "chemberta":
        path = CACHE_DIR / "chemberta_embeddings.npy"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: python scripts/chemberta_embeddings.py")
        emb = np.load(path).astype(np.float32)
        assert emb.shape[0] == len(data["vocab"]["pert_id"])
        return torch.from_numpy(emb).to(device), {}
    if name == "learnable":
        return None, {}
    if name == "gcn_finlayson":
        graphs, _ = build_pyg_drug_dataset(data["vocab"]["pert_id"], data["pid_to_canon"])
        return None, {"drug_data_list": graphs}
    raise ValueError(f"Unknown encoder: {name}")


def build_model(name, data, device, feat_dim=None, **extras):
    n_drugs = len(data["vocab"]["pert_id"])
    ctx = dict(n_cells=len(data["vocab"]["cell_id"]), n_doses=len(data["vocab"]["dose_bin"]),
               n_times=len(data["vocab"]["pert_time"]))
    expr_enc = ExpressionEncoder()
    if name == "morgan_v1":
        drug_enc = MorganDrugEncoder(**ctx)
    elif name in VARIANTS:
        drug_enc = MorganDrugEncoder(fp_dim=VARIANTS[name]["dim"], **ctx)
    elif name == "learnable":
        drug_enc = LearnableDrugEncoder(n_drugs, **ctx)
    elif name == "chemcpa":
        drug_enc = ChemCPAStyleEncoder(**ctx)
    elif name == "gcn_finlayson":
        drug_enc = GCNDrugEncoder(extras["drug_data_list"], **ctx)
    elif name == "chemberta":
        drug_enc = ChemBERTaDrugEncoder(desc_dim=feat_dim, **ctx)
    else:
        raise ValueError(name)
    return DualEncoder(expr_enc, drug_enc).to(device)


def chemcpa_pretrain(model, fp_table, epochs=2, batch_size=512, lr=1e-3):
    """Binary cross-entropy reconstruction pretraining of the chemCPA-style autoencoder."""
    drug_enc = model.drug
    optim = torch.optim.AdamW(list(drug_enc.ae_encoder.parameters()) +
                              list(drug_enc.ae_decoder.parameters()), lr=lr, weight_decay=1e-4)
    n = fp_table.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n)
        loss_sum, n_b = 0.0, 0
        for i in range(0, n, batch_size):
            x = fp_table[perm[i:i + batch_size].to(fp_table.device)]
            loss = F.binary_cross_entropy_with_logits(drug_enc.reconstruct(x), x)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            loss_sum += loss.item()
            n_b += 1
        print(f"    AE epoch {epoch + 1}/{epochs} bce={loss_sum / max(n_b, 1):.4f}", flush=True)
    drug_enc.ae_pretrained = True


def _tensors(data, device):
    meta = data["meta"]
    as_long = lambda col: torch.from_numpy(meta[col].values).long().to(device)
    return (torch.from_numpy(data["expr"]).float().to(device), as_long("cell_id_idx"),
            as_long("dose_bin_idx"), as_long("pert_time_idx"), as_long("pert_id_idx"))


def _fit(model, train_idx, tensors, get_drug_feat, seed, epochs, batch_size, lr):
    """AdamW with a cosine learning-rate schedule over all steps."""
    expr, cell_t, dose_t, time_t, drug_t = tensors
    device = expr.device
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs * (len(train_idx) // batch_size + 1))
    for epoch in range(epochs):
        model.train()
        order = np.random.RandomState(seed + epoch).permutation(train_idx)
        loss_sum, n_b = 0.0, 0
        for i in range(0, len(order), batch_size):
            sub_t = torch.from_numpy(order[i:i + batch_size]).long().to(device)
            z_e, z_d = model(expr[sub_t], get_drug_feat(drug_t[sub_t]),
                             cell_t[sub_t], dose_t[sub_t], time_t[sub_t])
            loss = model.loss_infonce(z_e, z_d)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            sched.step()
            loss_sum += loss.item()
            n_b += 1
        print(f"  epoch {epoch + 1}/{epochs} loss={loss_sum / max(n_b, 1):.3f} "
              f"temp={model.get_temperature():.3f}", flush=True)


def _embed(model, idx, tensors, get_drug_feat, batch=1024):
    expr, cell_t, dose_t, time_t, drug_t = tensors
    model.eval()
    z_es, z_ds = [], []
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            sub_t = torch.from_numpy(idx[i:i + batch]).long().to(expr.device)
            z_e, z_d = model(expr[sub_t], get_drug_feat(drug_t[sub_t]),
                             cell_t[sub_t], dose_t[sub_t], time_t[sub_t])
            z_es.append(z_e.cpu().numpy())
            z_ds.append(z_d.cpu().numpy())
    return np.vstack(z_es), np.vstack(z_ds)


def train_model(name, data, device="cuda", epochs=5, batch_size=512, lr=1e-3, seed=42,
                drug_feat_table=None, extras=None):
    """Train on the 90/10 signature split and evaluate on 10,000 held-out signatures."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    extras = extras or {}
    print(f"\n=== {name} (seed={seed}) ===", flush=True)
    t0 = time.time()
    feat_dim = drug_feat_table.shape[1] if drug_feat_table is not None else None
    model = build_model(name, data, device, feat_dim=feat_dim, **extras)
    n_params = count_params(model)
    print(f"  params: {n_params:,}", flush=True)

    tensors = _tensors(data, device)
    train_idx, val_idx = get_random_split(len(data["meta"]), seed=42)

    if name == "chemcpa":
        chemcpa_pretrain(model, drug_feat_table, epochs=2)

    def get_drug_feat(drug_idx):
        return drug_idx if drug_feat_table is None else drug_feat_table[drug_idx]

    _fit(model, train_idx, tensors, get_drug_feat, seed, epochs, batch_size, lr)

    z_es, z_ds = _embed(model, val_idx, tensors, get_drug_feat)
    if len(z_es) > 10000:  # fixed 10,000-signature retrieval pool
        sub = np.random.RandomState(7).choice(len(z_es), 10000, replace=False)
        z_es, z_ds = z_es[sub], z_ds[sub]
    metrics = hit_metrics(z_es @ z_ds.T)
    metrics.update(dict(name=name, seed=seed, n_params=n_params, time_s=time.time() - t0))
    print(f"  {name}: Hit@1={metrics['hit@1']:.2f}% Hit@10={metrics['hit@10']:.2f}% "
          f"MRR={metrics['mrr']:.4f}", flush=True)
    return metrics, model


def train_on_cold_drug_split(name, data, drug_feat_table, device, split_seed=0, seed=42,
                             epochs=5, batch_size=512, lr=1e-3):
    """Train a Morgan-type encoder on the training compounds of a canonical-SMILES split."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = build_model(name, data, device)
    print(f"  params: {count_params(model):,}", flush=True)
    train_idx = get_cold_drug_split(data["meta"], data["pid_to_canon"], seed=split_seed)[0]
    _fit(model, train_idx, _tensors(data, device), lambda d: drug_feat_table[d], seed, epochs, batch_size, lr)
    return model


TANIMOTO_BINS = [("lo<=0.3", (0.0, 0.3)), ("mid0.3-0.5", (0.3, 0.5)),
                 ("hi0.5-0.7", (0.5, 0.7)), ("vhi>0.7", (0.7, 1.01))]


def evaluate_zero_shot(model, data, device, drug_feat_table, split_seed=0):
    """Retrieval over all held-out signatures, stratified by max Tanimoto to training compounds.

    The rank of a query is the number of candidates with a strictly higher similarity
    than its own drug-condition embedding.
    """
    _, test_idx, train_pids, test_pids, _, _ = get_cold_drug_split(
        data["meta"], data["pid_to_canon"], seed=split_seed)
    z_es, z_ds = _embed(model, test_idx, _tensors(data, device), lambda d: drug_feat_table[d])

    n = len(z_es)
    ranks = np.zeros(n, dtype=np.int32)
    for i in tqdm(range(0, n, 512), desc="  zero-shot ranking"):
        sims = z_es[i:i + 512] @ z_ds.T
        for j in range(sims.shape[0]):
            ranks[i + j] = (sims[j] > sims[j, i + j]).sum()
    rr = 1.0 / (ranks + 1)
    out = dict(zs_n=int(n), zs_hit1=float((ranks < 1).mean() * 100), zs_hit5=float((ranks < 5).mean() * 100),
               zs_hit10=float((ranks < 10).mean() * 100), zs_hit50=float((ranks < 50).mean() * 100),
               zs_mrr=float(rr.mean()), zs_median_rank=int(np.median(ranks)))

    # Maximum Tanimoto similarity (2,048-bit Morgan) of each test compound to the training compounds
    fps = data["morgan_fps"]
    train_fp = np.stack([fps.get(p, np.zeros(2048, dtype=np.float32)) for p in train_pids], axis=0).astype(np.uint8)
    canon_max_tani = {}
    for p in sorted(test_pids):
        fp = fps.get(p)
        canon = data["pid_to_canon"].get(p)
        if fp is None or fp.sum() == 0:
            canon_max_tani[canon] = 0.0
            continue
        fp_u = fp.astype(np.uint8)
        inter = (train_fp & fp_u[None, :]).sum(axis=1)
        union = (train_fp | fp_u[None, :]).sum(axis=1)
        canon_max_tani[canon] = float((inter / np.maximum(union, 1)).max())

    test_canon = [data["pid_to_canon"].get(p, "") for p in data["meta"].iloc[test_idx]["pert_id"].values]
    max_tani = np.array([canon_max_tani.get(c, 0.0) for c in test_canon])
    strata = {}
    for label, (lo, hi) in TANIMOTO_BINS:
        sel = (max_tani >= lo) & (max_tani < hi)
        if sel.sum() < 50:
            strata[label] = None
            continue
        rk = ranks[sel]
        strata[label] = dict(n=int(sel.sum()), hit1=float((rk < 1).mean() * 100),
                             hit10=float((rk < 10).mean() * 100), hit50=float((rk < 50).mean() * 100),
                             mrr=float((1.0 / (rk + 1)).mean()))
    out["strata"] = strata
    return out
