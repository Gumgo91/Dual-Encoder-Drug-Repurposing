"""Draw Figures 1-5 of the paper from the files in results/ (written to figures/)."""

import pickle
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import FIGURES_DIR, RESULTS_DIR  # noqa: E402

mpl.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "figure.dpi": 100, "savefig.dpi": 300,
})


def load(name):
    with open(RESULTS_DIR / name, "rb") as f:
        return pickle.load(f)


def save(fig_name, **kwargs):
    for ext in ("png", "pdf"):
        plt.savefig(FIGURES_DIR / f"{fig_name}.{ext}", **kwargs)
    plt.close()
    print("  wrote", FIGURES_DIR / f"{fig_name}.png")


def fig1():
    """Dual-encoder architecture."""
    fig, ax = plt.subplots(figsize=(16.4, 5.0))
    ax.set_xlim(0, 16.4)
    ax.set_ylim(0.55, 4.95)
    ax.axis("off")

    def node(x, y, width, height, label, face, edge, fontsize=18):
        patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.04,rounding_size=0.14",
                               linewidth=1.8, facecolor=face, edgecolor=edge, zorder=2)
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=fontsize,
                weight="semibold", linespacing=1.08, color="#111111", zorder=3)
        return patch, (x + width / 2, y + height / 2)

    def connector(source, target):
        arrow = FancyArrowPatch(source[1], target[1], patchA=source[0], patchB=target[0],
                                arrowstyle="-|>", linewidth=1.7, color="#3F3F3F",
                                mutation_scale=17, shrinkA=2, shrinkB=2, zorder=1)
        ax.add_patch(arrow)

    top_y, bottom_y = 3.55, 0.95
    l1000 = node(0.10, top_y, 3.00, 1.15, "L1000 profile\n978 genes", "#E5F2F8", "#2C6E91", fontsize=20)
    morgan = node(0.10, bottom_y, 3.00, 1.40, "Morgan fingerprint\n2,048-bit + context\n(cell · dose · time)",
                  "#FBE9DE", "#A6492D", fontsize=17.5)
    expr_encoder = node(3.75, top_y, 3.80, 1.15, "Expression encoder\n978 → 512 → 256",
                        "#C9E3EE", "#2C6E91", fontsize=20)
    drug_encoder = node(3.75, bottom_y, 3.80, 1.40, "Drug encoder\nfingerprint + context\n→ 256",
                        "#F2C1B3", "#A6492D", fontsize=19)
    z_expr = node(8.25, top_y, 3.20, 1.15, "z_expr\n256-D · normalized", "#DDF1D7", "#3B7D34", fontsize=19)
    z_drug = node(8.25, bottom_y, 3.20, 1.40, "z_drug\n256-D · normalized", "#DDF1D7", "#3B7D34", fontsize=19)
    objective = node(12.15, 2.00, 4.05, 1.55, "Cosine-similarity\nmatrix\nBidirectional InfoNCE",
                     "#F5E9B9", "#8B741E", fontsize=19)
    for source, target in ((l1000, expr_encoder), (morgan, drug_encoder), (expr_encoder, z_expr),
                           (drug_encoder, z_drug), (z_expr, objective), (z_drug, objective)):
        connector(source, target)

    fig.suptitle("Dual-encoder contrastive learning", fontsize=22, weight="semibold", y=0.96)
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.035, top=0.88)
    save("figure1_architecture")


def fig2():
    """Encoder comparison: mean +/- SE over three training seeds (drawn at print width)."""
    runs = load("encoder_ablation.pkl")
    methods = ["learnable", "chemcpa", "gcn_finlayson", "chemberta", "morgan_v1"]
    labels = ["Learnable-embedding baseline", "chemCPA*-AE", "Finlayson*-style GCN",
              "Frozen ChemBERTa-77M", "Morgan fingerprint (2,048-bit)"]
    colors = ["#4477AA", "#CC6677", "#DDCC77", "#66CCEE", "#EE7733"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))
    x = np.arange(len(methods))
    for ax, key, title in zip(axes, ["hit@1", "hit@10"], ["Hit@1", "Hit@10"]):
        values = np.array([[runs[f"{m}_seed{s}"][key] for s in (42, 7, 11)] for m in methods])
        mean = values.mean(axis=1)
        se = values.std(axis=1, ddof=1) / np.sqrt(values.shape[1])
        ax.bar(x, mean, yerr=se, capsize=2.5, color=colors, width=0.72, error_kw={"linewidth": 0.8})
        top = (mean + se).max()
        for xi, m, s in zip(x, mean, se):
            ax.text(xi, m + s + 0.02 * top, f"{m:.1f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_ylim(0, top * 1.15)
        ax.set_xticks([])
        ax.set_ylabel(f"{title} (%)", fontsize=9)
        ax.set_title(f"{title} (3-seed mean ± SE)", fontsize=9.5)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [Patch(facecolor=c, label=lab) for c, lab in zip(colors, labels)]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False,
               columnspacing=1.2, handlelength=1.2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Encoder ablation under signature-level retrieval", fontsize=10.5, y=0.985)
    fig.text(0.5, 0.905, "10,000 held-out signatures · 3 seeds", ha="center", va="center",
             fontsize=8, color="#555555")
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.22, top=0.80, wspace=0.28)
    save("figure2_encoder_ablation")


def fig3():
    """Canonical-SMILES-disjoint retrieval with the random expectation as an inset."""
    zero_shot = load("zero_shot.pkl")["morgan_v1"]
    ks = [1, 5, 10, 50]
    hit = [zero_shot[f"zs_hit{k}"] for k in ks]
    n_candidates = zero_shot["zs_n"]

    fig, ax = plt.subplots(figsize=(8, 5.4))
    x = np.arange(len(ks))
    bars = ax.bar(x, hit, width=0.62, color="#EE7733", alpha=0.88)
    for bar, value in zip(bars, hit):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.65, f"{value:.2f}%",
                ha="center", va="bottom", fontsize=11, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Retrieval depth (k)")
    ax.set_ylabel("Hit@k (%)")
    fig.suptitle("Canonical-SMILES-disjoint retrieval performance", fontsize=16, y=0.965)
    fig.text(0.5, 0.885, f"{n_candidates:,} test signatures from compounds absent from training",
             ha="center", va="center", fontsize=10, color="#555555")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 36)

    inset = ax.inset_axes([0.10, 0.57, 0.37, 0.27])
    inset.plot(x, [100.0 * k / n_candidates for k in ks], "--o", color="#666666", linewidth=1.5, markersize=4)
    inset.set_xticks(x)
    inset.set_xticklabels([str(k) for k in ks], fontsize=8)
    inset.set_ylim(0, 0.25)
    inset.set_ylabel("Hit@k (%)", fontsize=8)
    inset.set_title("Random expectation", fontsize=9)
    inset.tick_params(axis="y", labelsize=8)
    inset.grid(axis="y", alpha=0.25)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.84)
    save("figure3_zero_shot")


def fig4():
    """Zero-shot Hit@10 by maximum Tanimoto similarity to the training set."""
    strata = load("zero_shot.pkl")["morgan_v1"]["strata"]
    bins = ["lo<=0.3", "mid0.3-0.5", "hi0.5-0.7", "vhi>0.7"]
    bin_labels = ["Low\n≤ 0.3", "Intermediate-low\n> 0.3–0.5", "Intermediate-high\n> 0.5–0.7", "High\n> 0.7"]
    h10 = [strata[b]["hit10"] for b in bins]
    n = [strata[b]["n"] for b in bins]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(bin_labels, h10, color="#EE7733", alpha=0.85)
    for bar, val, ni in zip(bars, h10, n):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val:.2f}%\n(n={ni:,})",
                ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Maximum Tanimoto similarity to training set")
    ax.set_ylabel("Zero-shot Hit@10 (%)")
    ax.set_title("Tanimoto-stratified zero-shot Hit@10 (2,048-bit Morgan fingerprint)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 16)
    plt.tight_layout()
    save("figure4_zero_shot_strata")


def fig5():
    """Tanimoto versus learned-embedding similarity and the random-embedding control (drawn at print width)."""
    d = load("embedding_similarity.pkl")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    ax = axes[0]
    ax.scatter(d["tani"], d["cos"], s=1.2, alpha=0.06, color="#4477AA", linewidths=0, rasterized=True)
    lo, hi = d["bootstrap_ci_model"]
    ax.text(0.04, 0.97, f"Spearman ρ = {d['spearman_r_model']:.3f}\n95% bootstrap CI {lo:.3f}–{hi:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5, linespacing=1.3,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
    ax.set_xlabel("Tanimoto similarity", fontsize=9)
    ax.set_ylabel("Cosine similarity of drug embeddings", fontsize=9)
    ax.set_title("A. Compound pairs", fontsize=9.5)
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    centers, counts = d["bin_centers"], d["bin_counts"]
    ax.errorbar(centers, d["bin_means_model"], yerr=d["bin_stds_model"], marker="o", ms=3.5, lw=1.5,
                capsize=2, elinewidth=0.9, color="#EE7733", label="Morgan fingerprint model")
    ax.errorbar(centers, d["bin_means_random"], yerr=d["bin_stds_random"], marker="o", ms=3.5, lw=1.3,
                capsize=2, elinewidth=0.9, color="#777777", label="Random-embedding control")
    for c, n in zip(centers, counts):
        if n < 200:  # number of pairs in sparse bins, written just above the x-axis
            ax.text(c, -0.055, f"{int(n)}", ha="center", va="top", fontsize=7, color="#7A3A12")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.13, 1.08)
    ax.set_xlabel("Tanimoto similarity (bin center)", fontsize=9)
    ax.set_ylabel("Mean cosine similarity (± SE)", fontsize=9)
    ax.set_title("B. Mean by Tanimoto bin", fontsize=9.5)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    fig.suptitle("Chemical-similarity structure in the learned drug-embedding space", fontsize=10.5, y=0.985)
    fig.subplots_adjust(left=0.08, right=0.975, bottom=0.15, top=0.83, wspace=0.3)
    save("figure5_embedding_similarity")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for fig in (fig1, fig2, fig3, fig4, fig5):
        fig()


if __name__ == "__main__":
    main()
