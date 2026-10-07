"""Mean-pooled ChemBERTa-77M-MTR embeddings for every compound (input of the ChemBERTa encoder).

Writes cache/chemberta_embeddings.npy in pert_id vocabulary order. The SMILES strings
are taken as given in the LINCS compound metadata; empty strings get a zero vector.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_all  # noqa: E402
from src.paths import CACHE_DIR  # noqa: E402

MODEL_ID = "DeepChem/ChemBERTa-77M-MTR"


def embed(smiles_list, device, batch_size=64):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    try:
        model = AutoModel.from_pretrained(MODEL_ID, use_safetensors=True)
    except OSError:
        model = AutoModel.from_pretrained(MODEL_ID)
    model = model.to(device).eval()
    embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="ChemBERTa"):
            sub = smiles_list[i:i + batch_size]
            valid_idx = [j for j, s in enumerate(sub) if s]
            batch_emb = np.zeros((len(sub), 384), dtype=np.float32)
            if valid_idx:
                enc = tok([sub[j] for j in valid_idx], padding=True, truncation=True,
                          max_length=256, return_tensors="pt").to(device)
                out = model(**enc)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
                batch_emb[valid_idx] = pooled.cpu().numpy().astype(np.float32)
            embs.append(batch_emb)
    return np.concatenate(embs, axis=0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_all()
    smiles = [data["smiles_raw"].get(p, "") or "" for p in data["vocab"]["pert_id"]]
    emb = embed(smiles, device)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_DIR / "chemberta_embeddings.npy", emb)
    print(f"Saved {emb.shape} to {CACHE_DIR / 'chemberta_embeddings.npy'}")


if __name__ == "__main__":
    main()
