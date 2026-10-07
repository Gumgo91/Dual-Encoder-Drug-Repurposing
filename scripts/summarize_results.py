"""Print the tables of the paper from the files in results/."""

import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import RESULTS_DIR  # noqa: E402
from src.training import ENCODERS, TANIMOTO_BINS  # noqa: E402

LABELS = {"learnable": "Learnable-embedding baseline", "chemcpa": "chemCPA*-AE",
          "gcn_finlayson": "Finlayson*-style GCN", "chemberta": "ChemBERTa-77M",
          "morgan_v1": "Morgan fingerprint (2,048-bit)"}
BIN_LABELS = {"lo<=0.3": "Low (<= 0.3)", "mid0.3-0.5": "Intermediate-low (> 0.3-0.5)",
              "hi0.5-0.7": "Intermediate-high (> 0.5-0.7)", "vhi>0.7": "High (> 0.7)"}


def load(name):
    with open(RESULTS_DIR / name, "rb") as f:
        return pickle.load(f)


def mean_se(values):
    values = np.asarray(values, dtype=float)
    return values.mean(), values.std(ddof=1) / np.sqrt(len(values))


def encoder_tables():
    runs = load("encoder_ablation.pkl")
    seeds = [42, 7, 11]
    print("Table 1. Encoder comparison (10,000 held-out signatures; mean +/- SE over seeds 42, 7, 11)")
    print(f"{'Method':<32}{'Params':>11}{'Hit@1':>16}{'Hit@5':>16}{'Hit@10':>16}{'MRR':>20}")
    for name in ENCODERS:
        rows = [runs[f"{name}_seed{s}"] for s in seeds]
        cells = []
        for key, fmt in (("hit@1", "{:.2f} +/- {:.2f}"), ("hit@5", "{:.2f} +/- {:.2f}"),
                         ("hit@10", "{:.2f} +/- {:.2f}"), ("mrr", "{:.4f} +/- {:.4f}")):
            cells.append(fmt.format(*mean_se([r[key] for r in rows])))
        print(f"{LABELS[name]:<32}{rows[0]['n_params']:>11,}" + "".join(f"{c:>16}" for c in cells[:3]) + f"{cells[3]:>20}")

    print("\nSupplementary Table S1. Per-seed results")
    for name in ENCODERS:
        for s in seeds:
            r = runs[f"{name}_seed{s}"]
            print(f"{LABELS[name]:<32}{s:>4}{r['n_params']:>11,}{r['hit@1']:>8.2f}{r['hit@5']:>8.2f}"
                  f"{r['hit@10']:>8.2f}{r['mrr']:>9.4f}")

    print("\nTable 3. Learnable-embedding baseline versus Morgan fingerprint")
    for name in ("learnable", "morgan_v1"):
        rows = [runs[f"{name}_seed{s}"] for s in seeds]
        times = [r["time_s"] for r in rows if "time_s" in r]
        time_txt = f"{np.mean(times):.0f} s per 5-epoch run" if times else "n/a"
        print(f"{LABELS[name]:<32} params {rows[0]['n_params']:,}; Hit@1 {mean_se([r['hit@1'] for r in rows])[0]:.2f}%; "
              f"{time_txt}")


def zero_shot_tables():
    zs = load("zero_shot.pkl")
    print("\nTable 2 and Supplementary Table S3. Zero-shot Hit@10 by maximum Tanimoto similarity to training (split 0)")
    names = ["morgan_v1", "multiradius_concat", "morgan_r2_wide_8192"]
    print(f"{'Bin':<32}{'n':>8}{'2,048-bit':>12}{'radius 0-3':>12}{'8,192-bit':>12}")
    for b, _ in TANIMOTO_BINS:
        print(f"{BIN_LABELS[b]:<32}{zs['morgan_v1']['strata'][b]['n']:>8,}"
              + "".join(f"{zs[m]['strata'][b]['hit10']:>11.2f}%" for m in names))
    print(f"{'All':<32}{zs['morgan_v1']['zs_n']:>8,}" + "".join(f"{zs[m]['zs_hit10']:>11.2f}%" for m in names))
    m = zs["morgan_v1"]
    print(f"2,048-bit model, all test signatures: Hit@1 {m['zs_hit1']:.2f}%, Hit@5 {m['zs_hit5']:.2f}%, "
          f"Hit@10 {m['zs_hit10']:.2f}%, Hit@50 {m['zs_hit50']:.2f}% (random Hit@10 {1000 / m['zs_n']:.3f}%)")

    ms = load("zero_shot_multisplit.pkl")
    print("\nSupplementary Table S4. Hit@10 across split seeds 0, 1, 2 (mean +/- SE)")
    per = ms["results_per_split"]
    for b, _ in TANIMOTO_BINS:
        a = ms["aggregate"][b]
        print(f"{BIN_LABELS[b]:<32}{a['morgan_2048']['mean']:>7.2f} +/- {a['morgan_2048']['se']:.2f}"
              f"{a['morgan_8192']['mean']:>9.2f} +/- {a['morgan_8192']['se']:.2f}")
    all_2048 = mean_se([per[s]["morgan_2048"]["hit10_all"] for s in ms["split_seeds"]])
    all_8192 = mean_se([per[s]["morgan_8192"]["hit10_all"] for s in ms["split_seeds"]])
    print(f"{'All':<32}{all_2048[0]:>7.2f} +/- {all_2048[1]:.2f}{all_8192[0]:>9.2f} +/- {all_8192[1]:.2f}")
    low = ms["aggregate"]["lo<=0.3"]
    t = ttest_rel(low["morgan_2048"]["values"], low["morgan_8192"]["values"])
    print(f"Low-similarity bin, paired t-test (2,048 vs 8,192 bits): t = {t.statistic:.2f}, df = 2, p = {t.pvalue:.3f}")


def other_tables():
    sim = load("embedding_similarity.pkl")
    lo, hi = sim["bootstrap_ci_model"]
    rlo, rhi = sim["bootstrap_ci_random"]
    print(f"\nEmbedding similarity (Figure 5): {sim['n_pairs']:,} pairs; Spearman rho = {sim['spearman_r_model']:.3f} "
          f"(95% CI {lo:.3f} to {hi:.3f}); random control rho = {sim['spearman_r_random']:.3f} "
          f"(95% CI {rlo:.3f} to {rhi:.3f})")

    cs = load("chemical_space.pkl")
    t = cs["pairwise_tanimoto"]
    print(f"\nSupplementary Table S2. pert_ids {cs['n_drugs']:,}; valid SMILES {cs['n_valid_fps']:,}; "
          f"mean on-bits {cs['avg_bits_on']:.1f}; pairwise Tanimoto (n = {len(t):,}) mean {t.mean():.3f}, "
          f"median {np.median(t):.3f}, P95 {np.percentile(t, 95):.3f}, P99 {np.percentile(t, 99):.3f}; "
          f"scaffolds {cs['n_unique_scaffolds']:,} ({cs['scaffold_singletons']:,} singletons); top-10/50/100 "
          f"coverage {cs['scaffold_top10_coverage']:.1%} / {cs['scaffold_top50_coverage']:.1%} / "
          f"{cs['scaffold_top100_coverage']:.1%}")
    if "n_unique_canonical" in cs:
        z = cs["zero_shot_test"]
        print(f"Unique canonical SMILES {cs['n_unique_canonical']:,}; pert_ids sharing one "
              f"{cs['n_pids_sharing_canonical']:,}; zero-shot test set {z['n_canonical']:,} compounds, "
              f"{z['n_pids']:,} pert_ids, {z['n_signatures']:,} signatures")
        o = cs["pool_overlap"]
        print(f"Table 1 evaluation pool: compound in training {o['compound']:.1%}; "
              f"same drug-cell-dose-time combination in training {o['condition']:.1%}")


if __name__ == "__main__":
    encoder_tables()
    zero_shot_tables()
    other_tables()
