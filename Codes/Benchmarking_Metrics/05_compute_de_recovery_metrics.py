"""
DE recovery metrics for benchmarking
===========================================================
Fix: AUPRC@50 and CentroidAcc computed only on TEST perturbations
     (GEARS simulation split, model-specific best seed from Excel).

Changes vs compute_de_metrics_all_models.py:
  - Best seed per model loaded from AIVC_model_benchmarking_comprehensive_result.260323.xlsx
  - Test perturbations loaded from GEARS splits/ pkl for that seed
  - Evaluation restricted to test-set perturbations only
  - NDCG@50 also restricted to test-set for consistency

Output:
  260326/de_metrics_test_only_summary.csv
  260326/{ModelName}_de_metrics_per_pert_test.csv
"""

import scanpy as sc
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import average_precision_score

LATEST_DIR = Path("<PROJECT_ROOT>/results_with_scDEED/latest")
OUTPUT_DIR  = Path("<PROJECT_ROOT>/model_performances/deg_ranking_metrics/260326")
OUTPUT_DIR.mkdir(exist_ok=True)

K = 50

# ── Dataset configs ────────────────────────────────────────────────────────────
# data_dir: where GEARS data lives (and where splits/ subdir is)
# split_prefix: filename prefix inside splits/ (some have typo 'oragnoid')
SERIES_CONFIG = {
    "Pan": {
        "gt_path":      Path("<DATA_ROOT>/organoid_allcelltype_pcgenes/perturb_processed.h5ad"),
        "data_dir":     Path("<DATA_ROOT>/organoid_allcelltype_pcgenes"),
        "split_prefix": "oragnoid_allcelltype_pcgenes",   # typo in actual filenames
        "models": ["Pan-NOCAP"],
    },
    "Neuron": {
        "gt_path":      Path("<DATA_ROOT>/organoid_neuron_pcgenes/perturb_processed.h5ad"),
        "data_dir":     Path("<DATA_ROOT>/organoid_neuron_pcgenes"),
        "split_prefix": "organoid_neuron_pcgenes",
        "models": ["Neuro-NOCAP"],
    },
    "Telen": {
        "gt_path":      Path("<DATA_ROOT>/organoid_telencephalicneuron_pcgenes/perturb_processed.h5ad"),
        "data_dir":     Path("<DATA_ROOT>/organoid_telencephalicneuron_pcgenes"),
        "split_prefix": "organoid_telencephalicneuron_pcgenes",
        "models": [
            "Telen-NOCAP", "Telen-BrainTissue", "Telen-WholeHuman",
            "Telen-scLAMBDA", "Telen-GeneCompass", "Telen-Geneformer",
            "Telen-GEARS", "Telen-CellFM", "Telen-EndodermOrganoid",
            "Telen-scFoundation",
        ],
    },
    "BrainCell": {
        "gt_path":      Path("<DATA_ROOT>/corticallineagecell_neuron_pcgenes/perturb_processed.h5ad"),
        "data_dir":     Path("<DATA_ROOT>/corticallineagecell_neuron_pcgenes"),
        "split_prefix": "corticallineagecell_neuron_pcgenes",
        "models": ["BrainCell-NOCAP", "BrainCell-BrainTissue", "BrainCell-WholeHuman"],
    },
    "CellLine": {
        "gt_path":      Path("<DATA_ROOT>/braincell_tian2021_pcgenes/perturb_processed.h5ad"),
        "data_dir":     Path("<DATA_ROOT>/braincell_tian2021_pcgenes"),
        "split_prefix": "braincell_tian2021_pcgenes",
        "models": ["CellLine-NOCAP", "CellLine-BrainTissue"],
    },
}

# ── Load best seed per model from Excel ───────────────────────────────────────
excel_path = Path("<PROJECT_ROOT>/model_performances/AIVC_model_benchmarking_comprehensive_result.260323.xlsx")
seed_df = pd.read_excel(excel_path)[["Model", "seed"]].set_index("Model")
model_seeds = seed_df["seed"].to_dict()
print("Model seeds:")
for m, s in sorted(model_seeds.items()):
    print(f"  {m}: seed={s}")

# ── Helper: load test perturbations for a given dataset + seed ─────────────────
def load_test_perts(data_dir: Path, split_prefix: str, seed: int) -> set:
    pkl_path = data_dir / "splits" / f"{split_prefix}_simulation_{seed}_0.75.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Split file not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        sp = pickle.load(f)
    # keys are like 'ctrl+GENENAME'; strip to just gene name
    test_genes = set()
    for cond in sp["test"]:
        if cond.startswith("ctrl+"):
            gene = cond.split("ctrl+")[1].split("_1+")[0]
            test_genes.add(gene)
    return test_genes

# ── Metric functions ──────────────────────────────────────────────────────────
def gt_key_to_gene(k):
    return k.split("ctrl+")[1].split("_1+")[0]

def ndcg_at_k(pred_ranked_genes, gt_ranked_genes, k):
    gt_set = set(gt_ranked_genes[:k])
    dcg  = sum(1.0 / np.log2(i + 2)
               for i, g in enumerate(pred_ranked_genes[:k])
               if g in gt_set)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(gt_set))))
    return dcg / idcg if idcg > 0 else 0.0

def auprc_at_k(pred_scores, gt_ranked_genes, gene_list, k):
    positive = set(gt_ranked_genes[:k])
    labels   = np.array([1 if g in positive else 0 for g in gene_list])
    if labels.sum() == 0:
        return np.nan
    return float(average_precision_score(labels, pred_scores))

def centroid_accuracy_single(pred_vec, gt_centroid_X, gt_centroids_all, gene_idx):
    """
    For perturbation X: fraction of other perts Y where
    d(pred_X, GT_X) < d(pred_X, GT_Y).
    Uses ALL GT centroids for comparison (full perturbation space),
    but only test perturbations are evaluated as pred_X.
    """
    gt_X_sub  = gt_centroid_X[gene_idx]
    d_to_self = float(np.linalg.norm(pred_vec - gt_X_sub))
    wins = 0; total = 0
    for gene_Y, centroid_Y in gt_centroids_all.items():
        if np.array_equal(centroid_Y, gt_centroid_X):
            continue
        gt_Y_sub = centroid_Y[gene_idx]
        d_to_Y   = float(np.linalg.norm(pred_vec - gt_Y_sub))
        wins  += (d_to_self < d_to_Y)
        total += 1
    return wins / total if total > 0 else np.nan


# ── Main loop ─────────────────────────────────────────────────────────────────
all_rows      = []
all_pert_rows = []

for series_name, cfg in SERIES_CONFIG.items():
    print(f"\n{'#'*60}\n  Series: {series_name}\n{'#'*60}")

    # Load GT
    print("  Loading GT...")
    gt       = sc.read_h5ad(cfg["gt_path"])
    gt_genes = gt.var.index.tolist()
    ctrl_idx  = (gt.obs["control"] == 1).values
    ctrl_mean = np.array(gt.X[ctrl_idx].mean(axis=0)).flatten()
    rank_all  = gt.uns["rank_genes_groups_cov_all"]

    gene_to_gtkey = {gt_key_to_gene(k): k for k in rank_all.keys()}
    all_perts     = sorted(gene_to_gtkey.keys())
    print(f"  GT: {gt.shape}, perturbations={len(all_perts)}")

    # GT centroids for ALL perturbations (used as comparison pool in CentroidAcc)
    print("  Computing GT centroids...")
    gt_centroids = {}
    for gene in all_perts:
        cond = f"ctrl+{gene}"
        mask = (gt.obs["condition"] == cond).values
        if mask.sum() == 0:
            print(f"    [WARN] No cells for condition {cond}")
            continue
        gt_centroids[gene] = np.array(gt.X[mask].mean(axis=0)).flatten()
    valid_perts_all = [g for g in all_perts if g in gt_centroids]
    print(f"  GT centroids: {len(gt_centroids)} / {len(all_perts)}")

    for model_name in cfg["models"]:
        filepath = LATEST_DIR / f"{model_name}.h5ad"
        if not filepath.exists():
            print(f"\n  [SKIP] {model_name} — prediction file not found")
            continue

        if model_name not in model_seeds:
            print(f"\n  [SKIP] {model_name} — seed not found in Excel")
            continue

        seed = int(model_seeds[model_name])

        # Load test perturbations for this model's seed
        try:
            test_perts = load_test_perts(cfg["data_dir"], cfg["split_prefix"], seed)
        except FileNotFoundError as e:
            print(f"\n  [SKIP] {model_name} — {e}")
            continue

        # Filter valid_perts to test only
        test_valid_perts = [g for g in valid_perts_all if g in test_perts]
        print(f"\n  {'='*50}\n    {model_name} (seed={seed})\n  {'='*50}")
        print(f"    Test perts: {len(test_perts)}, valid in GT: {len(test_valid_perts)} / {len(valid_perts_all)} total")

        # Load prediction
        pred       = sc.read_h5ad(filepath)
        pred_genes = pred.var.index.tolist()

        common_genes     = [g for g in gt_genes if g in set(pred_genes)]
        pred_idx         = [pred_genes.index(g) for g in common_genes]
        gt_idx           = [gt_genes.index(g)   for g in common_genes]
        ctrl_mean_common = ctrl_mean[gt_idx]
        common_set       = set(common_genes)
        print(f"    Genes: pred={len(pred_genes)}, GT={len(gt_genes)}, common={len(common_genes)}")

        pred_obs_map = {idx.replace("_perturbed", ""): i
                        for i, idx in enumerate(pred.obs.index)}

        per_pert = []
        skipped  = 0

        for gene in test_valid_perts:
            if gene not in pred_obs_map:
                skipped += 1
                continue

            gt_key           = gene_to_gtkey[gene]
            gt_ranked        = rank_all[gt_key]
            gt_ranked_common = [g for g in gt_ranked if g in common_set]

            row_i      = pred_obs_map[gene]
            pred_expr  = np.array(pred.X[row_i, pred_idx]).flatten()
            pred_delta = pred_expr - ctrl_mean_common
            pred_scores = np.abs(pred_delta)
            pred_ranked = [common_genes[i] for i in np.argsort(-pred_scores)]

            ndcg  = ndcg_at_k(pred_ranked, gt_ranked_common, K)
            ap    = auprc_at_k(pred_scores, gt_ranked_common, common_genes, K)
            ca    = centroid_accuracy_single(
                        pred_vec         = pred_expr,
                        gt_centroid_X    = gt_centroids[gene],
                        gt_centroids_all = gt_centroids,
                        gene_idx         = gt_idx,
                    )

            per_pert.append({
                "Model":        model_name,
                "Series":       series_name,
                "Perturbation": gene,
                "seed":         seed,
                f"NDCG@{K}":    ndcg,
                f"AUPRC@{K}":   ap,
                "CentroidAcc":  ca,
            })

        if skipped:
            print(f"    Skipped {skipped} test perturbations (not in pred obs)")

        df_pert = pd.DataFrame(per_pert)
        all_pert_rows.append(df_pert)

        metric_cols = [f"NDCG@{K}", f"AUPRC@{K}", "CentroidAcc"]
        mean_row = {
            "Model":            model_name,
            "Series":           series_name,
            "seed":             seed,
            "n_test_perts":     len(test_perts),
            "n_perts_evaluated": len(df_pert),
        }
        for col in metric_cols:
            v = round(float(df_pert[col].mean()), 4)
            mean_row[col] = v
            print(f"    {col}: {v:.4f}")
        all_rows.append(mean_row)

        df_pert.to_csv(OUTPUT_DIR / f"{model_name}_de_metrics_per_pert_test.csv", index=False)


# ── Summary ───────────────────────────────────────────────────────────────────
summary = pd.DataFrame(all_rows).set_index("Model")

print("\n\n===== SUMMARY (test-only) =====")
print(summary[[f"NDCG@{K}", f"AUPRC@{K}", "CentroidAcc", "n_test_perts", "n_perts_evaluated"]].to_string())

summary.to_csv(OUTPUT_DIR / "de_metrics_test_only_summary.csv")
print(f"\nSaved → {OUTPUT_DIR}/de_metrics_test_only_summary.csv")

if all_pert_rows:
    all_pert_df = pd.concat(all_pert_rows, ignore_index=True)
    all_pert_df.to_csv(OUTPUT_DIR / "de_metrics_test_only_per_pert.csv", index=False)
    print(f"Saved → {OUTPUT_DIR}/de_metrics_test_only_per_pert.csv")

print("[ALL DONE]")
