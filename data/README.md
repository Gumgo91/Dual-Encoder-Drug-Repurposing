# Data Directory

This directory should contain the L1000 dataset files.

## Required Files

Download from [GEO GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742):

1. `GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx` (~14GB)
2. `GSE92742_Broad_LINCS_gene_info.txt`
3. `GSE92742_Broad_LINCS_sig_info.txt`
4. `GSE92742_Broad_LINCS_pert_info.txt`
5. `GSE92742_Broad_LINCS_cell_info.txt`

## Generated Files

After running `scripts/extract_smiles.py`:
- `drug_smiles.pkl` - SMILES strings for 20,413 drugs

## Download Instructions

```bash
# In this directory
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_gene_info.txt.gz
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_sig_info.txt.gz
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_pert_info.txt.gz
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_cell_info.txt.gz

# Extract all
gunzip *.gz
```

## Note

Data files are **not included** in the git repository due to size.
You must download them separately.
