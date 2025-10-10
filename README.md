# Integrating Chemical Structure and Transcriptional Responses via Dual-Encoder Contrastive Learning for Drug Repurposing

Official implementation of "Integrating Chemical Structure and Transcriptional Responses via Dual-Encoder Contrastive Learning for Drug Repurposing"

**Authors:** Hyunseung Kong¹, Inyoung Kim², and Byoung-Tak Zhang¹,³

¹Interdisciplinary Program in Bioinformatics, Seoul National University
²Department of Defense Science, Korea National Defense University
³Department of Computer Science, Seoul National University

## Abstract

Drug repurposing offers a promising path to accelerate therapeutic discovery by finding new uses for existing compounds. We present a contrastive learning framework that combines chemical structure information via Morgan fingerprints with transcriptomic profiles to enable more effective drug repurposing. Using the L1000 dataset containing 109,721 gene expression experiments across 20,401 unique compounds, our dual-encoder architecture learns to align chemical and biological representations in a shared embedding space.

**Key Results:**
- **39.38% top-1** and **85.23% top-10** retrieval accuracy (2.3× improvement over learnable embeddings)
- **10.87% top-10** zero-shot accuracy on unseen drugs (217× better than random)
- **Spearman correlation r=0.38** between chemical and biological similarity (p<10⁻³⁵)
- **58% fewer parameters** (1.5M vs 3.5M) while achieving superior performance

## Highlights

✨ **State-of-the-art Performance**
- 85.23% top-10 accuracy for practical drug repurposing screens
- 2.3-fold improvement over learnable embeddings without chemical structure

🔬 **True Zero-Shot Generalization**
- Predicts biological responses for completely unseen compounds
- 217-fold better than random chance on novel drugs

🧬 **Validated Structure-Activity Learning**
- Significant correlation between chemical and biological similarity
- Captures activity cliffs and pharmacophore equivalence

⚡ **Efficient Architecture**
- 58% parameter reduction through chemical inductive bias
- Fast training (~25 seconds/epoch on RTX 4060)

## Model Architecture

```
┌─────────────────┐
│ Expression [978]│──→ [512] ──→ [256] ──→ z_expr
└─────────────────┘                         │
                                            ↓
                                    ┌──────────────┐
                                    │   Cosine     │
                                    │  Similarity  │──→ InfoNCE Loss
                                    └──────────────┘
┌─────────────────┐                         ↑
│ Morgan FP [2048]│                         │
│ + Context       │──→ [256] ──────────→ z_drug
└─────────────────┘
```

**Dual-Encoder Design:**
- Expression Encoder: 978 → 512 → 256
- Drug Encoder: 2048 (Morgan FP) + context → 256
- Bidirectional InfoNCE contrastive loss
- L2-normalized embeddings for angular similarity

## Installation

### Requirements
- Python 3.10+
- PyTorch 2.0+ with CUDA 12.1
- RDKit 2023.3.1+
- 16GB RAM (32GB recommended)
- NVIDIA GPU (optional but recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/l1000-drug-repurposing.git
cd l1000-drug-repurposing

# Install uv package manager
pip install uv

# Install dependencies
uv sync
```

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

## Quick Start

### 1. Download L1000 Data

```bash
cd data/

# Download from GEO GSE92742
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_gene_info.txt.gz
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_sig_info.txt.gz
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_pert_info.txt.gz
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_cell_info.txt.gz

# Extract
gunzip *.gz
cd ..
```

### 2. Extract SMILES

```bash
uv run python scripts/extract_smiles.py
```

### 3. Train Model

```bash
uv run python train.py
```

**Training time:** ~4 minutes (10 epochs, RTX 4060)

### 4. Evaluate

```bash
# Standard evaluation
uv run python scripts/comprehensive_evaluation.py --mode metrics

# Zero-shot evaluation
uv run python scripts/zero_shot_evaluation.py --epochs 10
```

### 5. Generate Figures

```bash
uv run python scripts/generate_paper_figures.py
```

## Results

### Standard Evaluation (10K test samples)

| Method | Hit@1 | Hit@5 | Hit@10 | MRR | nDCG@10 | Params |
|--------|-------|-------|--------|-----|---------|--------|
| Random | 0.01% | 0.03% | 0.05% | - | - | - |
| Learnable Emb | 16.94% | 38% | 52% | 0.28 | 0.35 | 3.5M |
| **Ours (Morgan FP)** | **39.38%** | **74.19%** | **85.23%** | **0.5479** | **0.6157** | **1.5M** |
| **Improvement** | **+132%** | **+95%** | **+64%** | **+96%** | **+76%** | **-58%** |

### Zero-Shot Evaluation (Unseen Drugs)

| Metric | Performance | vs Random |
|--------|-------------|-----------|
| Hit@1  | 1.8% | 350× |
| Hit@5  | 6.5% | 262× |
| Hit@10 | 10.9% | **217×** |
| Hit@20 | 17.5% | 175× |
| Hit@50 | 31.5% | 63× |
| MRR    | 0.0520 | - |

**Zero-shot setting:** Model trained on 16,321 drugs, tested on 4,080 completely unseen drugs.

### Chemical-Biological Correlation

- **Spearman r = 0.381** (p < 1×10⁻³⁵)
- Strong correlation between Tanimoto (chemical) and cosine (embedding) similarity
- Captures activity cliffs and structure-activity relationships

## Key Features

### 1. Morgan Fingerprint Integration
- Fixed 2048-bit circular fingerprints (radius=2)
- Encodes established structure-activity principles
- Enables zero-shot prediction for novel compounds

### 2. Contrastive Learning
- Bidirectional InfoNCE loss
- Learnable temperature parameter (τ≈0.07)
- Pulls together matching expression-drug pairs
- Pushes apart non-matching pairs

### 3. Zero-Shot Capability
- Predicts biological responses for unseen compounds
- Only requires SMILES structure (no training data needed)
- Practical for virtual screening of large libraries

### 4. Validated Learning
- Significant chemical-biological correlation (r=0.38)
- Activity cliffs captured in embedding space
- Consistent with medicinal chemistry principles

## Project Structure

```
.
├── src/
│   ├── encoders.py          # Dual-encoder architecture
│   ├── data_loader.py       # L1000 data loading & preprocessing
│   ├── dataset.py           # PyTorch dataset with fingerprints
│   └── evaluation.py        # Retrieval metrics & correlation analysis
│
├── scripts/
│   ├── extract_smiles.py              # Extract SMILES from L1000
│   ├── comprehensive_evaluation.py    # Full evaluation pipeline
│   ├── zero_shot_evaluation.py        # Zero-shot generalization test
│   └── generate_paper_figures.py      # Generate all figures
│
├── train.py                 # Training script
├── pyproject.toml           # Dependencies
├── README.md
├── INSTALL.md              # Detailed installation guide
├── PAPER.md                # Paper information & citation
└── LICENSE
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{kong2024integrating,
  title={Integrating Chemical Structure and Transcriptional Responses via Dual-Encoder Contrastive Learning for Drug Repurposing},
  author={Kong, Hyunseung and Kim, Inyoung and Zhang, Byoung-Tak},
  journal={In preparation},
  year={2024}
}
```

## Dataset

**L1000 (LINCS Program)**
- Source: [GEO GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742)
- Samples: 109,721 gene expression profiles
- Compounds: 20,401 unique drugs
- Genes: 978 landmark genes
- Cell lines: 30 cancer cell lines

**Preprocessing:**
- Filter: compound treatments (trt_cp), 24h timepoint
- Morgan fingerprints: radius=2, 2048-bit
- Split: 80-10-10 (train-val-test) or cold-drug split

## Key Methods

### Morgan Fingerprints
- **Circular fingerprints** capturing local substructures
- **Radius 2, 2048 bits** for optimal coverage
- **Fixed representation** provides chemical inductive bias
- **RDKit implementation** with standardized parameters

### Contrastive Learning
- **Bidirectional InfoNCE loss** for symmetric retrieval
- **L2-normalized embeddings** for angular similarity
- **In-batch negatives** for computational efficiency
- **Temperature scaling** (learnable τ≈0.07)

### Evaluation
- **Hit@k:** Percentage of correct matches in top-k
- **MRR:** Mean reciprocal rank (emphasizes top results)
- **nDCG@10:** Ranking quality with position weighting
- **Chemical-biological correlation:** Tanimoto vs cosine similarity

## Performance Insights

### Why Morgan Fingerprints Work Better

1. **Chemical Inductive Bias**
   - Encodes century of medicinal chemistry knowledge
   - Captures structural similarity principles
   - Regularizes learning toward meaningful patterns

2. **Parameter Efficiency**
   - No need to learn 20K separate embeddings
   - 58% fewer parameters than learnable approach
   - Better generalization with less overfitting

3. **Zero-Shot Capability**
   - Fixed representation works for any molecule
   - No retraining needed for new compounds
   - Practical for virtual screening applications

### Practical Utility

- **85% top-10 accuracy** suitable for experimental validation
- **Typical workflow:** Computational screen → top 10 candidates → wet-lab validation
- **Cost reduction:** 10× more efficient than random screening
- **Speed:** Hours (computational) vs months (traditional)

## Limitations & Future Work

### Current Limitations

1. **Landmark genes only (978/20,000)** - may miss signals
2. **Transcriptomics only** - could integrate proteomics/metabolomics
3. **Computational validation** - requires wet-lab confirmation
4. **Zero-shot gap** - lower performance than standard (expected)

### Future Directions

- **Richer chemical representations:** 3D conformers, quantum properties, GNNs
- **Multi-modal integration:** Morphology, proteomics, metabolomics
- **Interpretability:** Attention mechanisms, feature attribution
- **Meta-learning:** Optimize for few-shot generalization
- **External validation:** CTRP, GDSC, clinical data

## Contact

- **Hyunseung Kong:** hskong@snu.ac.kr
- **GitHub Issues:** [Report bugs/questions](https://github.com/YOUR_USERNAME/l1000-drug-repurposing/issues)

## License

MIT License - see [LICENSE](LICENSE) file for details

## Acknowledgments

- **L1000/LINCS Program:** NIH LINCS Program
- **RDKit:** Open-source cheminformatics toolkit
- **PyTorch:** Deep learning framework

---

**Status:** Research code for paper submission
**Last Updated:** 2024-10-10
