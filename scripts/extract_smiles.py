"""Extract SMILES strings for all drugs from pert_info."""

import pandas as pd
import pickle
from pathlib import Path

def extract_smiles_mapping(output_path: str = "data/drug_smiles.pkl"):
    """Extract SMILES for each drug from pert_info file.

    The pert_info contains chemical structure information including
    canonical SMILES for small molecules.
    """
    # Load pert_info
    pert_info_path = Path("data/GSE92742_Broad_LINCS_pert_info.txt.gz")
    print(f"Loading {pert_info_path}...")
    pert_info = pd.read_csv(pert_info_path, sep="\t", low_memory=False)

    print(f"\nTotal perturbations: {len(pert_info)}")
    print(f"Columns: {pert_info.columns.tolist()}")

    # Filter for small molecules (compounds) only
    compounds = pert_info[pert_info['pert_type'] == 'trt_cp'].copy()
    print(f"\nCompounds (trt_cp): {len(compounds)}")

    # Check SMILES column
    smiles_col = None
    for col in ['canonical_smiles', 'smiles', 'inchi_key', 'pert_iname']:
        if col in compounds.columns:
            print(f"  - Found column: {col}")
            if col in ['canonical_smiles', 'smiles']:
                smiles_col = col

    if smiles_col is None:
        print("\nWarning: No SMILES column found. Checking available structure columns...")
        structure_cols = [c for c in compounds.columns if 'smiles' in c.lower() or 'inchi' in c.lower()]
        print(f"Structure-related columns: {structure_cols}")
        if structure_cols:
            smiles_col = structure_cols[0]

    # Create mapping: pert_id -> SMILES
    drug_smiles = {}
    if smiles_col:
        for _, row in compounds.iterrows():
            pert_id = row['pert_id']
            smiles = row[smiles_col]
            if pd.notna(smiles) and isinstance(smiles, str) and len(smiles) > 0:
                drug_smiles[pert_id] = smiles

    print(f"\nDrugs with valid SMILES: {len(drug_smiles)}")

    # Save mapping
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(drug_smiles, f)

    print(f"\nSaved SMILES mapping to: {output_path}")

    # Show examples
    print("\nExample SMILES:")
    for i, (pert_id, smiles) in enumerate(list(drug_smiles.items())[:5]):
        print(f"  {pert_id}: {smiles[:60]}...")

    return drug_smiles


if __name__ == "__main__":
    import sys
    output = sys.argv[1] if len(sys.argv) > 1 else "data/drug_smiles.pkl"
    extract_smiles_mapping(output)
