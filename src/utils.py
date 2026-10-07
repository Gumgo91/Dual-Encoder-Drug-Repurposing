"""Metadata helpers for the L1000 Level-5 files."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_dose_um(dose_str: str) -> float:
    """Return the leading number of a pert_idose string (e.g. '10 uM' -> 10.0).

    The unit is not converted, so doses reported in nM are binned by their
    numeric value. Unparseable values return NaN.
    """
    try:
        return float(str(dose_str).split()[0])
    except (ValueError, IndexError, AttributeError):
        return np.nan


def bin_dose(doses: pd.Series, bins: Optional[List[float]] = None) -> pd.Series:
    """Bin doses into six categories (missing doses are treated as 1.0)."""
    if bins is None:
        bins = [-1, 0.1, 0.3, 1.0, 3.0, 10.0, 1000.0]
    return pd.cut(doses.fillna(1.0), bins=bins, labels=False)


def filter_l1000_metadata(sig_info: pd.DataFrame, pert_type: str = "trt_cp",
                          time_hours: int = 24) -> pd.DataFrame:
    """Keep signatures of one perturbation type at one time point."""
    filtered = sig_info[(sig_info["pert_type"] == pert_type) &
                        (sig_info["pert_time"] == time_hours)]
    return filtered.reset_index(drop=True)


def get_landmark_genes(gene_info: pd.DataFrame) -> List[str]:
    """IDs of the 978 landmark genes."""
    landmark = gene_info[gene_info["pr_is_lm"] == 1]
    return landmark["pr_gene_id"].astype(str).tolist()


def create_categorical_indices(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, dict]:
    """Add '<col>_idx' integer codes and return the category vocabulary."""
    df = df.copy()
    vocab = {}
    for col in columns:
        df[f"{col}_idx"] = df[col].astype("category").cat.codes.values
        vocab[col] = df[col].astype("category").cat.categories.tolist()
    return df, vocab
