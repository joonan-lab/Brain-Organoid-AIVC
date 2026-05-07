"""Evaluate the TrainMean non-parametric baseline across simulation seeds."""
import warnings
warnings.filterwarnings("ignore")

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

DATA_DIR = Path("<DATA_ROOT>/scGPT/data/organoid_telencephalicneuron_pcgenes")
H5AD = DATA_DIR / "perturb_processed.h5ad"
SPLIT_DIR = DATA_DIR / "splits"
OUT = Path("<PROJECT_ROOT>/Section2/BaselineMean/trainmean_TelenNOCAP_8metrics_allseeds.tsv")

print("Reading h5ad...")
adata = sc.read_h5ad(H5AD)
print(adata.shape, "X sparse:", issparse(adata.X))
cond = adata.obs["condition"].astype(str).values

# mean_ctrl is shared across seeds (same h5ad)
ctrl_X = adata.X[cond == "ctrl"]
mean_ctrl = np.asarray(ctrl_X.mean(axis=0)).flatten()
print(f"ctrl cells: {(cond=='ctrl').sum():,}")

geneid2idx = dict(zip(adata.var.index.values, range(adata.n_vars)))
rgg = adata.uns["rank_genes_groups_cov_all"]
key_components = next(iter(rgg.keys())).split("_")


def find_DE_idx(condition_name, top_n=20):
    condition_key = "_".join([key_components[0], condition_name, key_components[2]])
    de_genes = rgg[condition_key][:top_n]
    return [geneid2idx[g] for g in de_genes]


# Cache per-condition mean expression so we don't recompute across seeds
cond_mean_cache = {}


def cond_mean(c):
    if c not in cond_mean_cache:
        m = cond == c
        cond_mean_cache[c] = np.asarray(adata.X[m].mean(axis=0)).flatten()
    return cond_mean_cache[c]


def cond_mean_over_set(cset):
    m = np.isin(cond, list(cset))
    return np.asarray(adata.X[m].mean(axis=0)).flatten()


metric_order = [
    "pearson_delta", "pearson_de_delta",
    "spearmanr_delta", "spearmanr_de_delta",
    "rmse", "rmse_de",
    "mae", "mae_de",
]

rows = []
for seed in range(1, 11):
    sp = SPLIT_DIR / f"organoid_telencephalicneuron_pcgenes_simulation_{seed}_0.75.pkl"
    if not sp.exists():
        print(f"[seed {seed}] split file missing, skip: {sp}")
        continue
    with open(sp, "rb") as f:
        splits = pickle.load(f)
    train_conds = set(splits["train"])
    test_conds = sorted(set(splits["test"]))

    # train_mean across all train cells
    train_mean = cond_mean_over_set(train_conds)

    true_mean_by_condition = np.stack([cond_mean(c) for c in test_conds])
    pred_mean_by_condition = np.broadcast_to(
        train_mean, true_mean_by_condition.shape
    ).copy()
    delta_true = true_mean_by_condition - mean_ctrl
    delta_pred = pred_mean_by_condition - mean_ctrl

    zero_rows = set(np.where(np.all(true_mean_by_condition == 0, axis=1))[0].tolist())
    per_cond = {k: [] for k in metric_order}

    for i, c in enumerate(test_conds):
        if i not in zero_rows:
            per_cond["pearson_delta"].append(pearsonr(delta_true[i], delta_pred[i])[0])
            per_cond["spearmanr_delta"].append(spearmanr(delta_true[i], delta_pred[i])[0])
        per_cond["rmse"].append(np.sqrt(mean_squared_error(true_mean_by_condition[i], pred_mean_by_condition[i])))
        per_cond["mae"].append(mean_absolute_error(true_mean_by_condition[i], pred_mean_by_condition[i]))

        de_idx = find_DE_idx(c, top_n=20)
        t_de = true_mean_by_condition[i][de_idx]
        p_de = pred_mean_by_condition[i][de_idx]
        dt_de = delta_true[i][de_idx]
        dp_de = delta_pred[i][de_idx]
        if not np.all(t_de == 0):
            per_cond["pearson_de_delta"].append(pearsonr(dt_de, dp_de)[0])
            per_cond["spearmanr_de_delta"].append(spearmanr(dt_de, dp_de)[0])
        per_cond["rmse_de"].append(np.sqrt(mean_squared_error(t_de, p_de)))
        per_cond["mae_de"].append(mean_absolute_error(t_de, p_de))

    metrics = {k: float(np.nanmean(v)) for k, v in per_cond.items()}
    metrics["seed"] = seed
    metrics["n_test_conds"] = len(test_conds)
    rows.append(metrics)

    print(f"\n[seed {seed}] n_test_conds={len(test_conds)} test={test_conds}")
    for k in metric_order:
        print(f"  {k:20s}: {metrics[k]:.4f}")

df = pd.DataFrame(rows)[["seed", "n_test_conds"] + metric_order]
df.to_csv(OUT, sep="\t", index=False)

print("\n===== Summary across seeds =====")
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n===== Mean ± std across seeds =====")
mean_row = df[metric_order].mean()
std_row = df[metric_order].std()
for k in metric_order:
    print(f"  {k:20s}: {mean_row[k]:.4f} ± {std_row[k]:.4f}")

print(f"\nSaved -> {OUT}")
