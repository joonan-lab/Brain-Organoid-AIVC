import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.sparse as sp
from pathlib import Path

LATEST_DIR = Path("<PROJECT_ROOT>/results_with_scDEED/latest")
OUTPUT_DIR = Path("<PROJECT_ROOT>/model_performances/graph_metrics_all_models")
OUTPUT_DIR.mkdir(exist_ok=True)

# 전체 모델 (Telen-Mouse, Telen-NOCAP-Lite 포함 — 후처리에서 제외 가능)
MODEL_FILES = {name.stem: name for name in sorted(LATEST_DIR.glob("*.h5ad"))}

print(f"Found {len(MODEL_FILES)} models:")
for m in MODEL_FILES: print(f"  {m}")


# ── Metric 1: CV of kNN distance (scale-invariant) ────────────────────────────
def compute_cv_knn_dist(dist_matrix):
    """
    CV = std(per-node mean dist) / mean(per-node mean dist)
    Scale-invariant: comparable across models regardless of PCA magnitude.
    Lower CV → uniform neighborhood density → smooth manifold.
    Higher CV → heterogeneous density → collapse/fragmentation mix.
    """
    D = sp.csr_matrix(dist_matrix)
    per_node_mean = np.array(D.sum(axis=1)).flatten() / np.diff(D.indptr)
    mean_val = float(np.mean(per_node_mean))
    std_val  = float(np.std(per_node_mean))
    return {
        'mean_knn_dist': mean_val,          # for reference only (not comparable across models)
        'cv_knn_dist':   std_val / mean_val if mean_val > 0 else np.nan,
    }


# ── Metric 2: Mean connectivity (UMAP fuzzy weight, 0–1 normalized) ──────────
def compute_mean_connectivity(conn_matrix):
    """
    Per-node mean of UMAP fuzzy set membership weights (0–1).
    Fully comparable across models (normalized by construction).
    Higher → neighbors are strongly connected → dense, smooth structure.
    Lower  → weak connections → isolated fragments.
    Note: collapse (all points at same location) gives high connectivity —
          use alongside CV kNN dist to disambiguate.
    """
    C = sp.csr_matrix(conn_matrix)
    per_node_mean = np.array(C.sum(axis=1)).flatten() / np.diff(C.indptr)
    return {
        'mean_connectivity': float(np.mean(per_node_mean)),
        'std_connectivity':  float(np.std(per_node_mean)),
    }


# ── Metric 3: Near-zero pair fraction (collapse indicator) ────────────────────
def compute_near_zero_fraction(X_pca, n_sample=3000, seed=42):
    """
    Fraction of sampled pairs with pairwise distance < 1% of max distance.
    Captures GLOBAL collapse (single blob). Does NOT capture fragmentation.
    Scale-independent (threshold is relative to max within each model).
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(X_pca.shape[0], size=min(n_sample, X_pca.shape[0]), replace=False)
    X = X_pca[idx]
    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(X, metric='euclidean')
    upper = D[np.triu_indices_from(D, k=1)]
    return {
        'near_zero_fraction': float(np.mean(upper < upper.max() * 0.01)),
        'cv_pairwise_dist':   float(np.std(upper) / np.mean(upper)) if np.mean(upper) > 0 else np.nan,
        '_upper_dists':       upper,
    }


# ── Run ────────────────────────────────────────────────────────────────────────
results = {}

for model_name, filepath in MODEL_FILES.items():
    print(f"\n{'='*55}\n  {model_name}\n{'='*55}")
    adata = sc.read_h5ad(filepath)
    print(f"  Shape: {adata.shape}")

    m1 = compute_cv_knn_dist(adata.obsp['distances'])
    m2 = compute_mean_connectivity(adata.obsp['connectivities'])
    m3 = compute_near_zero_fraction(adata.obsm['X_pca'])

    results[model_name] = {**m1, **m2, **m3}

    print(f"  CV kNN dist       : {m1['cv_knn_dist']:.4f}  (mean kNN: {m1['mean_knn_dist']:.4f} — scale-dep, ref only)")
    print(f"  Mean connectivity : {m2['mean_connectivity']:.4f}")
    print(f"  Near-zero pair %  : {m3['near_zero_fraction']*100:.2f}%")

print("\n[Done] All models processed.")


# ── Summary table ──────────────────────────────────────────────────────────────
rows = []
for m, v in results.items():
    rows.append({
        'Model':              m,
        'cv_knn_dist':        round(v['cv_knn_dist'], 4),
        'mean_connectivity':  round(v['mean_connectivity'], 4),
        'near_zero_fraction': round(v['near_zero_fraction'], 4),
        'mean_knn_dist_ref':  round(v['mean_knn_dist'], 4),   # reference only
        'cv_pairwise_dist':   round(v['cv_pairwise_dist'], 4),
    })

summary_df = pd.DataFrame(rows).set_index('Model').sort_values('cv_knn_dist')
print("\n===== SUMMARY TABLE (sorted by CV kNN dist) =====")
print(summary_df[['cv_knn_dist', 'mean_connectivity', 'near_zero_fraction']].to_string())

summary_df.to_csv(OUTPUT_DIR / "embedding_metrics_all_models.csv")
print(f"\nSaved → {OUTPUT_DIR / 'embedding_metrics_all_models.csv'}")


# ── Figure: 2-panel bar chart (CV kNN dist + Mean connectivity) ───────────────
sort_order = summary_df.index.tolist()

nocap_models = [m for m in sort_order if 'NOCAP' in m]
telen_others = [m for m in sort_order if 'Telen' in m and 'NOCAP' not in m]

def model_color(m):
    if 'NOCAP' in m:    return '#2166AC'
    if 'BrainCell' in m: return '#4DAC26'
    if 'CellLine' in m:  return '#E08214'
    return '#D6604D'

colors = [model_color(m) for m in sort_order]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, col, xlabel, ascending in [
    (axes[0], 'cv_knn_dist',
     'CV of kNN Distance\n(↓ = uniform neighborhood = smooth; scale-invariant)',
     True),
    (axes[1], 'mean_connectivity',
     'Mean kNN Connectivity\n(↑ = strong neighbor links; normalized 0–1)',
     False),
]:
    vals = [summary_df.loc[m, col] for m in sort_order]
    bars = ax.barh(sort_order, vals, color=colors, edgecolor='white', linewidth=0.6)
    for bar, val in zip(bars, vals):
        ax.text(val + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.tick_params(axis='y', labelsize=9)

from matplotlib.patches import Patch
fig.legend(handles=[
    Patch(facecolor='#2166AC', label='NOCAP-based'),
    Patch(facecolor='#4DAC26', label='BrainCell'),
    Patch(facecolor='#E08214', label='CellLine'),
    Patch(facecolor='#D6604D', label='Telen-Foundation'),
], loc='upper center', ncol=4, fontsize=9, bbox_to_anchor=(0.5, 1.02))

plt.suptitle('Embedding Structural Quality Metrics — All Models\n'
             '(Sorted by CV kNN Dist; both metrics are scale-invariant / normalized)',
             fontsize=11, fontweight='bold', y=1.06)
plt.tight_layout()
for ext in ['pdf', 'png']:
    plt.savefig(OUTPUT_DIR / f"embedding_metrics_barplot.{ext}", dpi=300, bbox_inches='tight')
plt.close()
print("Saved → embedding_metrics_barplot.pdf/png")


# ── Figure: pairwise distance histograms ──────────────────────────────────────
n = len(MODEL_FILES)
n_cols = 5
n_rows = (n + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
axes = axes.flatten()
cmap = plt.get_cmap('tab20')

for i, model_name in enumerate(MODEL_FILES.keys()):
    ax = axes[i]
    upper = results[model_name]['_upper_dists']
    cv    = results[model_name]['cv_knn_dist']
    conn  = results[model_name]['mean_connectivity']
    ax.hist(upper, bins=60, color=cmap(i % 20), alpha=0.75, edgecolor='none')
    ax.set_title(f"{model_name}\nCV kNN={cv:.2f}  conn={conn:.3f}", fontsize=8, fontweight='bold')
    ax.set_xlabel('Pairwise dist (PCA)', fontsize=7)
    ax.tick_params(labelsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Pairwise Distance Distributions (3000 sampled perturbations)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
for ext in ['pdf', 'png']:
    plt.savefig(OUTPUT_DIR / f"pairwise_dist_histograms.{ext}", dpi=300, bbox_inches='tight')
plt.close()
print("Saved → pairwise_dist_histograms.pdf/png")

print(f"\n[ALL DONE] Results saved to: {OUTPUT_DIR}")
