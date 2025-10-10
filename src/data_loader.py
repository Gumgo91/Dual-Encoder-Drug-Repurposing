"""Data loading and preprocessing for L1000 dataset."""

import os
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
from cmapPy.pandasGEXpress.parse_gctx import parse
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

from .utils import (
    filter_l1000_metadata,
    get_landmark_genes,
    parse_dose_um,
    bin_dose,
    create_categorical_indices
)


class L1000DataLoader:
    """Loader for L1000 Level-5 data with filtering and preprocessing."""

    def __init__(self, data_dir: str = "data/l1000"):
        """Initialize data loader.

        Args:
            data_dir: Directory containing L1000 files
        """
        self.data_dir = Path(data_dir)
        self.sig_info = None
        self.gene_info = None
        self.cell_info = None
        self.pert_info = None
        self.expr_matrix = None
        self.metadata_df = None
        self.vocab = None
        self.drug_smiles = None
        self.drug_fingerprints = None

    def load_metadata(self):
        """Load all metadata files."""
        print("Loading metadata files...")

        self.sig_info = pd.read_csv(
            self.data_dir / "GSE92742_Broad_LINCS_sig_info.txt.gz",
            sep="\t"
        )
        self.gene_info = pd.read_csv(
            self.data_dir / "GSE92742_Broad_LINCS_gene_info.txt.gz",
            sep="\t"
        )
        self.cell_info = pd.read_csv(
            self.data_dir / "GSE92742_Broad_LINCS_cell_info.txt.gz",
            sep="\t"
        )
        self.pert_info = pd.read_csv(
            self.data_dir / "GSE92742_Broad_LINCS_pert_info.txt.gz",
            sep="\t"
        )

        print(f"Loaded {len(self.sig_info)} signatures")
        print(f"Loaded {len(self.gene_info)} genes")
        print(f"Loaded {len(self.cell_info)} cell lines")
        print(f"Loaded {len(self.pert_info)} perturbagens")

    def filter_signatures(
        self,
        pert_type: str = "trt_cp",
        time_hours: int = 24,
        min_replicates: int = 2
    ) -> pd.DataFrame:
        """Filter signatures to quality subset.

        Args:
            pert_type: Perturbation type (default: 'trt_cp')
            time_hours: Time point in hours (default: 24)
            min_replicates: Minimum replicates (default: 2)

        Returns:
            Filtered signature info dataframe
        """
        if self.sig_info is None:
            self.load_metadata()

        filtered = filter_l1000_metadata(
            self.sig_info,
            pert_type=pert_type,
            time_hours=time_hours,
            min_replicates=min_replicates
        )

        print(f"Filtered to {len(filtered)} signatures "
              f"(pert_type={pert_type}, time={time_hours}h)")

        return filtered

    def load_expression_data(
        self,
        sig_ids: list,
        use_landmark_only: bool = True
    ) -> Tuple[pd.DataFrame, list]:
        """Load expression matrix for selected signatures.

        Args:
            sig_ids: List of signature IDs to load
            use_landmark_only: Use only 978 landmark genes (default: True)

        Returns:
            Tuple of (expression dataframe, gene IDs)
        """
        print(f"Loading expression data for {len(sig_ids)} signatures...")

        # Get gene IDs
        if use_landmark_only:
            gene_ids = get_landmark_genes(self.gene_info)
            print(f"Using {len(gene_ids)} landmark genes")
        else:
            gene_ids = self.gene_info['pr_gene_id'].astype(str).tolist()
            print(f"Using all {len(gene_ids)} genes")

        # Parse GCTX file
        # Try uncompressed first, then compressed
        gctx_path = self.data_dir / "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx"
        if not gctx_path.exists():
            gctx_path = self.data_dir / "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz"

        if not gctx_path.exists():
            raise FileNotFoundError(
                f"GCTX file not found: {gctx_path}\n"
                "Please download from GEO (GSE92742) and extract if compressed"
            )

        print("Parsing GCTX file (this may take a few minutes)...")
        gct = parse(str(gctx_path), rid=gene_ids, cid=sig_ids)
        expr_df = gct.data_df

        print(f"Loaded expression matrix: {expr_df.shape}")

        return expr_df, gene_ids

    def prepare_training_data(
        self,
        pert_type: str = "trt_cp",
        time_hours: int = 24,
        min_replicates: int = 2,
        use_landmark_only: bool = True,
        cache_path: Optional[str] = None
    ) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, list]]:
        """Prepare complete training dataset.

        Args:
            pert_type: Perturbation type filter
            time_hours: Time point filter
            min_replicates: Minimum replicate filter
            use_landmark_only: Use only landmark genes
            cache_path: Optional path to save/load cached data

        Returns:
            Tuple of (expression matrix, metadata dataframe, vocabulary dict)
        """
        # Check cache
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached data from {cache_path}")
            cached = np.load(cache_path, allow_pickle=True)
            return (
                cached['expr_matrix'],
                pd.DataFrame(cached['metadata'].item()),
                cached['vocab'].item()
            )

        # Load and filter metadata
        print("\n=== Step 1: Loading metadata ===")
        if self.sig_info is None:
            self.load_metadata()

        filtered_sig = self.filter_signatures(
            pert_type=pert_type,
            time_hours=time_hours,
            min_replicates=min_replicates
        )

        # Load expression data
        print("\n=== Step 2: Loading expression data ===")
        expr_df, gene_ids = self.load_expression_data(
            sig_ids=filtered_sig['sig_id'].tolist(),
            use_landmark_only=use_landmark_only
        )

        # Align metadata with expression matrix
        print("\n=== Step 3: Aligning metadata ===")
        metadata = filtered_sig[filtered_sig['sig_id'].isin(expr_df.columns)].copy()
        metadata = metadata.set_index('sig_id').loc[expr_df.columns].reset_index()

        # Process dose information
        print("\n=== Step 4: Processing dose information ===")
        metadata['dose_um'] = metadata['pert_idose'].apply(parse_dose_um)
        metadata['dose_bin'] = bin_dose(metadata['dose_um'])

        # Create categorical indices
        print("\n=== Step 5: Creating categorical indices ===")
        metadata, vocab = create_categorical_indices(
            metadata,
            columns=['pert_id', 'cell_id', 'dose_bin', 'pert_time']
        )

        # Convert expression to numpy array
        expr_matrix = expr_df.T.values.astype('float32')  # Shape: [N_samples, N_genes]

        print(f"\n=== Final dataset ===")
        print(f"Expression matrix: {expr_matrix.shape}")
        print(f"Metadata: {len(metadata)} samples")
        print(f"Vocabularies:")
        for key, val in vocab.items():
            print(f"  {key}: {len(val)} unique values")

        # Cache if requested
        if cache_path:
            print(f"\nSaving cache to {cache_path}")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(
                cache_path,
                expr_matrix=expr_matrix,
                metadata=metadata.to_dict(),
                vocab=vocab,
                gene_ids=gene_ids
            )

        # Generate Morgan fingerprints
        print("\n=== Step 6: Generating Morgan Fingerprints ===")
        drug_fingerprints = self._generate_fingerprints(
            metadata['pert_id'].unique().tolist()
        )

        self.expr_matrix = expr_matrix
        self.metadata_df = metadata
        self.vocab = vocab
        self.drug_fingerprints = drug_fingerprints

        return expr_matrix, metadata, vocab

    def _load_smiles_mapping(self) -> Dict[str, str]:
        """Load SMILES mapping from cached file."""
        smiles_path = Path("data/drug_smiles.pkl")

        if not smiles_path.exists():
            raise FileNotFoundError(
                f"SMILES mapping not found: {smiles_path}\n"
                "Please run: python scripts/extract_smiles.py"
            )

        with open(smiles_path, 'rb') as f:
            drug_smiles = pickle.load(f)

        print(f"Loaded SMILES for {len(drug_smiles)} drugs")
        return drug_smiles

    def _generate_fingerprints(
        self,
        pert_ids: list,
        radius: int = 2,
        n_bits: int = 2048
    ) -> Dict[str, np.ndarray]:
        """Generate Morgan fingerprints for drug compounds.

        Args:
            pert_ids: List of perturbation IDs
            radius: Morgan fingerprint radius (default: 2)
            n_bits: Fingerprint bit length (default: 2048)

        Returns:
            Dictionary mapping pert_id to fingerprint array
        """
        if self.drug_smiles is None:
            self.drug_smiles = self._load_smiles_mapping()

        drug_fingerprints = {}
        missing_smiles = []

        print(f"Generating Morgan fingerprints (radius={radius}, n_bits={n_bits})...")
        for pert_id in tqdm(pert_ids):
            if pert_id not in self.drug_smiles:
                missing_smiles.append(pert_id)
                # Use zero vector for drugs without SMILES
                drug_fingerprints[pert_id] = np.zeros(n_bits, dtype=np.float32)
                continue

            smiles = self.drug_smiles[pert_id]
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                # Invalid SMILES - use zero vector
                drug_fingerprints[pert_id] = np.zeros(n_bits, dtype=np.float32)
                continue

            # Generate Morgan fingerprint
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fp_array = np.zeros(n_bits, dtype=np.float32)
            fp_array[list(fp.GetOnBits())] = 1.0

            drug_fingerprints[pert_id] = fp_array

        if missing_smiles:
            print(f"Warning: {len(missing_smiles)} drugs missing SMILES (using zero vectors)")

        print(f"Generated fingerprints for {len(drug_fingerprints)} drugs")
        return drug_fingerprints

    def get_fingerprints(self) -> Dict[str, np.ndarray]:
        """Get precomputed drug fingerprints.

        Returns:
            Dictionary mapping pert_id to fingerprint array
        """
        if self.drug_fingerprints is None:
            raise ValueError(
                "Fingerprints not generated. Call prepare_training_data() first."
            )
        return self.drug_fingerprints


def load_disease_signature(
    disease_file: str,
    gene_ids: list,
    value_col: str = 'log2fc'
) -> np.ndarray:
    """Load and process disease differential expression signature.

    Args:
        disease_file: Path to disease DE file (CSV)
        gene_ids: List of L1000 gene IDs to match
        value_col: Column name for expression values (default: 'log2fc')

    Returns:
        Disease signature vector aligned to gene_ids
    """
    print(f"Loading disease signature from {disease_file}")

    # Load disease data
    disease_df = pd.read_csv(disease_file)

    # Ensure gene_id column exists
    if 'gene_id' not in disease_df.columns:
        raise ValueError("Disease file must have 'gene_id' column")

    # Convert to series indexed by gene_id
    disease_series = disease_df.set_index('gene_id')[value_col]

    # Align to L1000 gene IDs
    signature = []
    matched = 0
    for gid in gene_ids:
        if gid in disease_series.index:
            signature.append(disease_series[gid])
            matched += 1
        else:
            signature.append(0.0)  # Missing genes set to 0

    signature = np.array(signature, dtype='float32')

    print(f"Matched {matched}/{len(gene_ids)} genes ({matched/len(gene_ids)*100:.1f}%)")

    return signature
