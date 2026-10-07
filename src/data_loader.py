"""Loading and preprocessing of the L1000 Level-5 data (GEO GSE92742)."""

import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from cmapPy.pandasGEXpress.parse_gctx import parse
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from .paths import DATA_DIR
from .utils import (bin_dose, create_categorical_indices, filter_l1000_metadata,
                    get_landmark_genes, parse_dose_um)

GCTX_NAME = "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx"


class L1000DataLoader:
    """Loads signatures, expression profiles and Morgan fingerprints."""

    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = Path(data_dir)
        self.sig_info = None
        self.gene_info = None
        self.pert_info = None
        self.drug_smiles = None
        self.drug_fingerprints = None

    def load_metadata(self):
        read = lambda name: pd.read_csv(self.data_dir / name, sep="\t")
        self.sig_info = read("GSE92742_Broad_LINCS_sig_info.txt.gz")
        self.gene_info = read("GSE92742_Broad_LINCS_gene_info.txt.gz")
        self.pert_info = read("GSE92742_Broad_LINCS_pert_info.txt.gz")
        print(f"Loaded {len(self.sig_info)} signatures, {len(self.gene_info)} genes, "
              f"{len(self.pert_info)} perturbagens")

    def load_expression_data(self, sig_ids: list) -> Tuple[pd.DataFrame, list]:
        """Read the 978 landmark genes for the given signatures from the GCTX file."""
        gene_ids = get_landmark_genes(self.gene_info)
        gctx_path = self.data_dir / GCTX_NAME
        if not gctx_path.exists():
            raise FileNotFoundError(
                f"{gctx_path} not found. Download it from GEO (GSE92742) and decompress it "
                "(see data/README.md).")
        print(f"Reading {len(sig_ids)} signatures x {len(gene_ids)} landmark genes from {gctx_path.name}")
        gct = parse(str(gctx_path), rid=gene_ids, cid=sig_ids)
        return gct.data_df, gene_ids

    def prepare_training_data(self, pert_type: str = "trt_cp",
                              time_hours: int = 24) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, list]]:
        """Return (expression matrix [N, 978], metadata, vocabulary)."""
        if self.sig_info is None:
            self.load_metadata()

        filtered_sig = filter_l1000_metadata(self.sig_info, pert_type=pert_type, time_hours=time_hours)
        print(f"Selected {len(filtered_sig)} signatures (pert_type={pert_type}, {time_hours} h)")

        expr_df, _ = self.load_expression_data(filtered_sig["sig_id"].tolist())

        metadata = filtered_sig[filtered_sig["sig_id"].isin(expr_df.columns)].copy()
        metadata = metadata.set_index("sig_id").loc[expr_df.columns].reset_index()
        metadata["dose_um"] = metadata["pert_idose"].apply(parse_dose_um)
        metadata["dose_bin"] = bin_dose(metadata["dose_um"])
        metadata, vocab = create_categorical_indices(
            metadata, columns=["pert_id", "cell_id", "dose_bin", "pert_time"])

        expr_matrix = expr_df.T.values.astype("float32")
        print(f"Expression matrix: {expr_matrix.shape}; "
              + ", ".join(f"{k}: {len(v)}" for k, v in vocab.items()))

        self.drug_fingerprints = self._generate_fingerprints(metadata["pert_id"].unique().tolist())
        return expr_matrix, metadata, vocab

    def _load_smiles_mapping(self) -> Dict[str, str]:
        smiles_path = self.data_dir / "drug_smiles.pkl"
        if not smiles_path.exists():
            raise FileNotFoundError(f"{smiles_path} not found. Run: python scripts/extract_smiles.py")
        with open(smiles_path, "rb") as f:
            return pickle.load(f)

    def _generate_fingerprints(self, pert_ids: list, radius: int = 2,
                               n_bits: int = 2048) -> Dict[str, np.ndarray]:
        """Morgan fingerprints from the pert_info SMILES; all-zero if missing or unparseable."""
        if self.drug_smiles is None:
            self.drug_smiles = self._load_smiles_mapping()

        fingerprints = {}
        n_missing = 0
        for pert_id in tqdm(pert_ids, desc="Morgan fingerprints"):
            mol = Chem.MolFromSmiles(self.drug_smiles[pert_id]) if pert_id in self.drug_smiles else None
            fp_array = np.zeros(n_bits, dtype=np.float32)
            if mol is None:
                n_missing += 1
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                fp_array[list(fp.GetOnBits())] = 1.0
            fingerprints[pert_id] = fp_array
        print(f"Fingerprints for {len(fingerprints)} compounds ({n_missing} all-zero)")
        return fingerprints

    def get_fingerprints(self) -> Dict[str, np.ndarray]:
        if self.drug_fingerprints is None:
            raise ValueError("Call prepare_training_data() first.")
        return self.drug_fingerprints
