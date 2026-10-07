"""Cached dataset, canonical SMILES and the two data splits used in the paper."""

import pickle

import numpy as np
from rdkit import Chem, RDLogger

from .data_loader import L1000DataLoader
from .paths import CACHE_DIR, DATA_DIR

RDLogger.DisableLog("rdApp.*")


def load_all(force_reload: bool = False) -> dict:
    """Expression matrix, metadata, vocabulary, fingerprints and canonical SMILES.

    The first call parses the GCTX file (several minutes) and writes
    cache/l1000_basics.pkl; later calls read the cache.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "l1000_basics.pkl"
    if cache.exists() and not force_reload:
        with open(cache, "rb") as f:
            return pickle.load(f)

    loader = L1000DataLoader(DATA_DIR)
    expr, meta, vocab = loader.prepare_training_data()
    fps = loader.get_fingerprints()

    with open(DATA_DIR / "drug_smiles.pkl", "rb") as f:
        smi_raw = pickle.load(f)
    pid_to_canon = {}
    for p in vocab["pert_id"]:
        s = smi_raw.get(p)
        mol = Chem.MolFromSmiles(s) if s else None
        pid_to_canon[p] = Chem.MolToSmiles(mol, isomericSmiles=False) if mol else None

    out = dict(expr=expr, meta=meta, vocab=vocab, morgan_fps=fps,
               pid_to_canon=pid_to_canon, smiles_raw=smi_raw)
    with open(cache, "wb") as f:
        pickle.dump(out, f)
    return out


def get_random_split(n_samples: int, seed: int = 42, train_frac: float = 0.9):
    """Signature-level 90/10 split used for the encoder comparison."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_samples)
    n_train = int(train_frac * n_samples)
    return perm[:n_train], perm[n_train:]


def get_cold_drug_split(meta, pid_to_canon, seed: int = 0, test_frac: float = 0.2):
    """Compound-level split: hold out 20% of unique canonical SMILES.

    Returns (train_idx, test_idx, train_pids, test_pids, train_canon, test_canon).
    Signatures of compounds without a parseable SMILES are in neither set.
    """
    canon_to_pids = {}
    for p, c in pid_to_canon.items():
        if c is None:
            continue
        canon_to_pids.setdefault(c, []).append(p)
    canon_list = sorted(canon_to_pids.keys())
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(canon_list))
    n_test = int(test_frac * len(canon_list))
    test_canon = set(canon_list[i] for i in perm[:n_test])
    train_canon = set(canon_list[i] for i in perm[n_test:])
    test_pids = set(p for c in test_canon for p in canon_to_pids[c])
    train_pids = set(p for c in train_canon for p in canon_to_pids[c])

    test_mask = meta["pert_id"].isin(test_pids).values
    train_mask = meta["pert_id"].isin(train_pids).values
    return (np.where(train_mask)[0], np.where(test_mask)[0],
            train_pids, test_pids, train_canon, test_canon)
