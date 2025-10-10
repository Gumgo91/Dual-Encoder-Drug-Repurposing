# Installation Guide

## System Requirements

- **OS**: Linux, macOS, or Windows
- **Python**: 3.10 or higher
- **GPU**: NVIDIA GPU with CUDA 12.1+ (optional but recommended)
- **RAM**: 16GB minimum, 32GB recommended
- **Disk**: ~50GB for L1000 data

## Step-by-Step Installation

### 1. Install Python

Ensure Python 3.10+ is installed:

```bash
python --version  # Should be 3.10 or higher
```

### 2. Install uv (Package Manager)

```bash
pip install uv
```

### 3. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/l1000-drug-repurposing.git
cd l1000-drug-repurposing
```

### 4. Install Dependencies

```bash
uv sync
```

This will install:
- PyTorch 2.0+ with CUDA 12.1
- RDKit 2023.3.1+
- NumPy, Pandas, SciPy
- Matplotlib, Seaborn
- tqdm, cmapPy

### 5. Verify Installation

```bash
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
uv run python -c "import rdkit; print(f'RDKit: {rdkit.__version__}')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA available: True
RDKit: 2023.x.x
```

## Download L1000 Data

### Option 1: Direct Download (Recommended)

```bash
cd data/

# Download files
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_gene_info.txt.gz
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_sig_info.txt.gz
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_pert_info.txt.gz
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_cell_info.txt.gz

# Extract
gunzip *.gz

cd ..
```

### Option 2: Manual Download

1. Visit [GEO GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742)
2. Download these files to `data/` folder:
   - `GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz`
   - `GSE92742_Broad_LINCS_gene_info.txt.gz`
   - `GSE92742_Broad_LINCS_sig_info.txt.gz`
   - `GSE92742_Broad_LINCS_pert_info.txt.gz`
   - `GSE92742_Broad_LINCS_cell_info.txt.gz`
3. Extract all `.gz` files

### Extract SMILES

```bash
uv run python scripts/extract_smiles.py
```

This creates `data/drug_smiles.pkl` with SMILES strings for 20,413 drugs.

## Verify Setup

```bash
# Quick test
uv run python -c "from src.data_loader import L1000DataLoader; print('Setup complete!')"
```

## Troubleshooting

### CUDA not available

If you see `CUDA available: False`:

1. Install NVIDIA drivers
2. Install CUDA Toolkit 12.1+
3. Reinstall PyTorch with CUDA:
   ```bash
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

### RDKit import error

```bash
uv pip install rdkit
```

### Memory issues

If you encounter memory errors during training:
- Reduce batch size in `train.py` (line 15): `BATCH_SIZE = 256`
- Close other applications
- Use smaller evaluation sample size

### File not found errors

Ensure data files are in correct locations:
```
data/
  ├── GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx
  ├── GSE92742_Broad_LINCS_gene_info.txt
  ├── GSE92742_Broad_LINCS_sig_info.txt
  ├── GSE92742_Broad_LINCS_pert_info.txt
  ├── GSE92742_Broad_LINCS_cell_info.txt
  └── drug_smiles.pkl
```

## Next Steps

After successful installation:

1. **Train model**: See [README.md](README.md#training)
2. **Evaluate**: See [README.md](README.md#evaluation)
3. **Generate figures**: See [README.md](README.md#generate-figures)

## Hardware Recommendations

### Minimum
- CPU: 4 cores
- RAM: 16GB
- GPU: None (CPU training possible but slow)
- Storage: 50GB

### Recommended
- CPU: 8+ cores
- RAM: 32GB
- GPU: NVIDIA RTX 3060+ (8GB VRAM)
- Storage: 100GB SSD

### Performance
- **Training time** (10 epochs):
  - RTX 4060 (8GB): ~4 minutes
  - RTX 3090 (24GB): ~2 minutes
  - CPU only: ~30 minutes
