# Data

All inputs come from the LINCS L1000 Level-5 release in NCBI GEO, accession [GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742). Place these files in this directory:

| File | Size |
|---|---|
| `GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx` (decompressed) | 23 GB |
| `GSE92742_Broad_LINCS_sig_info.txt.gz` | 11 MB |
| `GSE92742_Broad_LINCS_gene_info.txt.gz` | 0.2 MB |
| `GSE92742_Broad_LINCS_pert_info.txt.gz` | 1.1 MB |

```bash
BASE=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl
wget $BASE/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz
wget $BASE/GSE92742_Broad_LINCS_sig_info.txt.gz
wget $BASE/GSE92742_Broad_LINCS_gene_info.txt.gz
wget $BASE/GSE92742_Broad_LINCS_pert_info.txt.gz
gunzip GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz
```

Keep the three metadata files compressed. Then run `python scripts/extract_smiles.py` from the repository root to create `drug_smiles.pkl`.
