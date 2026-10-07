"""Encoders for the encoder comparison (Table 1) and the zero-shot experiments (Table 2).

Every drug encoder maps its chemical input to 256 dimensions, concatenates
cell-line (64), dose (16) and time (16) embeddings, and projects the result
through 352 -> 512 -> 256. The expression encoder is shared by all models.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpressionEncoder(nn.Module):
    def __init__(self, in_dim=978, hidden=512, out_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=-1)


class ContextHead(nn.Module):
    """Cell-line, dose and time embeddings."""

    def __init__(self, n_cells, n_doses, n_times, cell_dim=64, dose_dim=16, time_dim=16):
        super().__init__()
        self.cell = nn.Embedding(n_cells, cell_dim)
        self.dose = nn.Embedding(n_doses, dose_dim)
        self.time = nn.Embedding(n_times, time_dim)
        self.dim = cell_dim + dose_dim + time_dim

    def forward(self, cell_idx, dose_idx, time_idx):
        return torch.cat([self.cell(cell_idx), self.dose(dose_idx), self.time(time_idx)], dim=-1)


def _projection(in_dim, hidden, out_dim, dropout):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class MorganDrugEncoder(nn.Module):
    """Fixed Morgan fingerprint (2,048 bits by default)."""

    def __init__(self, fp_dim=2048, n_cells=30, n_doses=6, n_times=1,
                 hidden=512, out_dim=256, dropout=0.1):
        super().__init__()
        self.fp_proj = nn.Sequential(
            nn.Linear(fp_dim, 256),
            nn.GELU(), nn.LayerNorm(256), nn.Dropout(dropout),
        )
        self.ctx = ContextHead(n_cells, n_doses, n_times)
        self.proj = _projection(256 + self.ctx.dim, hidden, out_dim, dropout)

    def forward(self, drug_feat, cell_idx, dose_idx, time_idx):
        h = torch.cat([self.fp_proj(drug_feat), self.ctx(cell_idx, dose_idx, time_idx)], dim=-1)
        return F.normalize(self.proj(h), p=2, dim=-1)


class LearnableDrugEncoder(nn.Module):
    """Chemistry-free baseline: one trainable 64-d vector per pert_id."""

    def __init__(self, n_drugs, embed_dim=64, n_cells=30, n_doses=6, n_times=1,
                 hidden=512, out_dim=256, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(n_drugs, embed_dim)
        self.proj_drug = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(), nn.LayerNorm(256), nn.Dropout(dropout),
        )
        self.ctx = ContextHead(n_cells, n_doses, n_times)
        self.proj = _projection(256 + self.ctx.dim, hidden, out_dim, dropout)

    def forward(self, drug_idx, cell_idx, dose_idx, time_idx):
        h = torch.cat([self.proj_drug(self.embed(drug_idx)), self.ctx(cell_idx, dose_idx, time_idx)], dim=-1)
        return F.normalize(self.proj(h), p=2, dim=-1)


# --- Finlayson-style GCN -----------------------------------------------------
ATOM_LIST = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B", "Si", "H", "OTHER"]
HYBRID_LIST = ["SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]
ATOM_FEAT_DIM = len(ATOM_LIST) + len(HYBRID_LIST) + 5  # 24


def smiles_to_graph(smi):
    """Atom features (element and hybridization one-hot, degree, formal charge,
    hydrogen count, aromaticity, ring membership) and an undirected edge index."""
    from rdkit import Chem
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    atom_feats = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        sym_oh = [0.0] * len(ATOM_LIST)
        sym_oh[ATOM_LIST.index(sym) if sym in ATOM_LIST else len(ATOM_LIST) - 1] = 1.0
        hyb = atom.GetHybridization().name
        hyb_oh = [0.0] * len(HYBRID_LIST)
        hyb_oh[HYBRID_LIST.index(hyb) if hyb in HYBRID_LIST else len(HYBRID_LIST) - 1] = 1.0
        atom_feats.append(sym_oh + hyb_oh + [
            float(atom.GetDegree()), float(atom.GetFormalCharge()), float(atom.GetTotalNumHs()),
            float(atom.GetIsAromatic()), float(atom.IsInRing()),
        ])
    edge_index = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
    if not edge_index:
        edge_index = [[0, 0]]  # single-atom molecule
    return (np.asarray(atom_feats, dtype=np.float32),
            np.asarray(edge_index, dtype=np.int64).T)


def build_pyg_drug_dataset(pid_list, pid_to_canon):
    """One PyG graph per pert_id (None when the SMILES cannot be parsed)."""
    from torch_geometric.data import Data
    data_list = []
    valid = np.zeros(len(pid_list), dtype=bool)
    for i, pid in enumerate(pid_list):
        out = smiles_to_graph(pid_to_canon.get(pid))
        if out is None:
            data_list.append(None)
            continue
        atom_feats, edge_index = out
        data_list.append(Data(x=torch.from_numpy(atom_feats), edge_index=torch.from_numpy(edge_index)))
        valid[i] = True
    return data_list, valid


class GCNDrugEncoder(nn.Module):
    """Three GCNConv layers (128 hidden, GELU) with global mean pooling."""

    def __init__(self, drug_data_list, atom_feat_dim=ATOM_FEAT_DIM, gcn_hidden=128,
                 n_gcn_layers=3, drug_dim=256, n_cells=30, n_doses=6, n_times=1,
                 hidden=512, out_dim=256, dropout=0.1):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.drug_data_list = drug_data_list
        self.gcn_layers = nn.ModuleList()
        in_dim = atom_feat_dim
        for _ in range(n_gcn_layers):
            self.gcn_layers.append(GCNConv(in_dim, gcn_hidden))
            in_dim = gcn_hidden
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(gcn_hidden)
        self.drug_head = nn.Sequential(
            nn.Linear(gcn_hidden, drug_dim),
            nn.GELU(), nn.LayerNorm(drug_dim), nn.Dropout(dropout),
        )
        self.ctx = ContextHead(n_cells, n_doses, n_times)
        self.proj = _projection(drug_dim + self.ctx.dim, hidden, out_dim, dropout)

    def forward(self, drug_idx, cell_idx, dose_idx, time_idx):
        from torch_geometric.data import Batch, Data
        from torch_geometric.nn import global_mean_pool

        device = drug_idx.device
        unique, inverse = torch.unique(drug_idx, return_inverse=True)
        graphs, zero_mask = [], []
        for u in unique.tolist():
            d = self.drug_data_list[u]
            if d is None:  # compounds without a valid graph get a zero vector
                d = Data(x=torch.zeros(1, ATOM_FEAT_DIM), edge_index=torch.tensor([[0], [0]], dtype=torch.long))
                zero_mask.append(True)
            else:
                zero_mask.append(False)
            graphs.append(d)
        batch = Batch.from_data_list(graphs).to(device)

        h = batch.x
        for gconv in self.gcn_layers:
            h = self.dropout(F.gelu(gconv(h, batch.edge_index)))
        z = self.drug_head(self.norm(global_mean_pool(h, batch.batch)))
        z = z[inverse]
        zero_t = torch.tensor(zero_mask, device=device, dtype=z.dtype)[inverse].unsqueeze(-1)
        z = z * (1 - zero_t)

        h = torch.cat([z, self.ctx(cell_idx, dose_idx, time_idx)], dim=-1)
        return F.normalize(self.proj(h), p=2, dim=-1)


class ChemCPAStyleEncoder(nn.Module):
    """Morgan fingerprint autoencoder (2,048-1,024-512-256), pretrained and then frozen."""

    def __init__(self, fp_dim=2048, latent_dim=256, n_cells=30, n_doses=6, n_times=1,
                 hidden=512, out_dim=256, dropout=0.1, freeze_ae=True):
        super().__init__()
        self.ae_encoder = nn.Sequential(
            nn.Linear(fp_dim, 1024),
            nn.GELU(), nn.LayerNorm(1024), nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.GELU(), nn.LayerNorm(512), nn.Dropout(dropout),
            nn.Linear(512, latent_dim),
        )
        self.ae_decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, 1024),
            nn.GELU(), nn.LayerNorm(1024),
            nn.Linear(1024, fp_dim),
        )
        self.ae_pretrained = False
        self.freeze_ae = freeze_ae
        self.drug_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(), nn.LayerNorm(256), nn.Dropout(dropout),
        )
        self.ctx = ContextHead(n_cells, n_doses, n_times)
        self.proj = _projection(256 + self.ctx.dim, hidden, out_dim, dropout)

    def encode_fp(self, fp):
        if self.ae_pretrained and self.freeze_ae:
            with torch.no_grad():
                return self.ae_encoder(fp)
        return self.ae_encoder(fp)

    def reconstruct(self, fp):
        return self.ae_decoder(self.ae_encoder(fp))

    def forward(self, drug_feat, cell_idx, dose_idx, time_idx):
        h = self.drug_head(self.encode_fp(drug_feat))
        h = torch.cat([h, self.ctx(cell_idx, dose_idx, time_idx)], dim=-1)
        return F.normalize(self.proj(h), p=2, dim=-1)


class ChemBERTaDrugEncoder(nn.Module):
    """Frozen ChemBERTa-77M embedding (384-d) -> 256, same head as the other encoders."""

    def __init__(self, desc_dim=384, n_cells=30, n_doses=6, n_times=1,
                 cell_dim=64, dose_dim=16, time_dim=16, hidden=512, out=256, dropout=0.1):
        super().__init__()
        self.proj_desc = nn.Sequential(
            nn.Linear(desc_dim, 256),
            nn.GELU(), nn.LayerNorm(256), nn.Dropout(dropout),
        )
        self.cell = nn.Embedding(n_cells, cell_dim)
        self.dose = nn.Embedding(n_doses, dose_dim)
        self.time = nn.Embedding(n_times, time_dim)
        self.proj = _projection(256 + cell_dim + dose_dim + time_dim, hidden, out, dropout)

    def forward(self, desc, cell_idx, dose_idx, time_idx):
        h = torch.cat([self.proj_desc(desc), self.cell(cell_idx), self.dose(dose_idx), self.time(time_idx)], dim=-1)
        return F.normalize(self.proj(h), p=2, dim=-1)


class DualEncoder(nn.Module):
    def __init__(self, expr_enc, drug_enc):
        super().__init__()
        self.expr = expr_enc
        self.drug = drug_enc
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, expr, drug_feat, cell_idx, dose_idx, time_idx):
        return self.expr(expr), self.drug(drug_feat, cell_idx, dose_idx, time_idx)

    def loss_infonce(self, z_e, z_d):
        logits = (z_e @ z_d.T) / torch.clamp(self.temperature, min=1e-3)
        labels = torch.arange(z_e.size(0), device=z_e.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

    def get_temperature(self):
        return self.temperature.item()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
