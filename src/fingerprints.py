"""Morgan fingerprint variants used in the zero-shot comparison (Supplementary Tables S3-S4)."""

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


def fp_multiradius(smi: str, n_bits: int = 2048):
    """Concatenation of radius 0, 1, 2 and 3 Morgan fingerprints (4 x 2,048 bits)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    parts = []
    for radius in (0, 1, 2, 3):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        parts.append(arr)
    return np.concatenate(parts, axis=0)


def fp_morgan_r2_wide(smi: str, n_bits: int = 8192):
    """Radius-2 Morgan fingerprint hashed to 8,192 bits."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


VARIANTS = {
    "multiradius_concat": dict(fn=fp_multiradius, dim=4 * 2048),
    "morgan_r2_wide_8192": dict(fn=fp_morgan_r2_wide, dim=8192),
}
