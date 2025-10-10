# Paper Information

**Title:** Contrastive Multimodal Embeddings of Chemical Structure and Transcriptomic Response for Anticancer Drug Repurposing

## Abstract

We present a contrastive learning framework that integrates chemical structure information via Morgan fingerprints with transcriptomic profiles for drug repurposing. On the L1000 dataset (109,721 experiments, 20,401 drugs), our approach achieves **39.38% top-1** and **85.23% top-10** retrieval accuracy, representing a **2.3-fold improvement** over learnable embeddings. In zero-shot evaluation on unseen drugs, the model achieves **10.87% top-10** accuracy, demonstrating generalization capability. Chemical and biological similarity spaces show significant correlation (r=0.38, p<1e-35), validating effective multi-modal integration while reducing parameters by 58%.

## Key Results

### Standard Evaluation
- **Hit@1:** 39.38% (2.3× better than learnable embeddings)
- **Hit@10:** 85.23%
- **MRR:** 0.5479
- **nDCG@10:** 0.6157

### Zero-Shot Evaluation (Unseen Drugs)
- **Hit@1:** 1.75% (350× better than random)
- **Hit@10:** 10.87%
- **Hit@50:** 31.50%

### Chemical-Biological Integration
- **Spearman correlation:** r = 0.3813 (p < 1e-35)
- Significant correlation between chemical and biological similarity spaces

## Reproducibility

All experiments are fully reproducible using the code in this repository.

### Main Results
```bash
# Train model
uv run python train.py

# Evaluate
uv run python scripts/comprehensive_evaluation.py --mode metrics
```

### Zero-Shot Results
```bash
uv run python scripts/zero_shot_evaluation.py --epochs 10
```

### Figures
```bash
uv run python scripts/generate_paper_figures.py
```

## Figures

All figures are available in [`results/paper_figures/`](results/paper_figures/):

- **Figure 1:** Model Architecture
- **Figure 2:** Performance Comparison (Bar charts)
- **Figure 3:** Chemical-Biological Correlation (Scatter plot)
- **Figure 4:** Top-k Retrieval Curves
- **Figure 5:** Zero-Shot Evaluation

## Dataset

- **Source:** [L1000 (LINCS) - GEO GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742)
- **Samples:** 109,721 gene expression profiles
- **Drugs:** 20,401 unique compounds
- **Genes:** 978 landmark genes

## Citation

```bibtex
@article{your2024contrastive,
  title={Contrastive Multimodal Embeddings of Chemical Structure and Transcriptomic Response for Anticancer Drug Repurposing},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## Supplementary Materials

All supplementary materials, including detailed results and additional figures, are available in the paper manuscript.
