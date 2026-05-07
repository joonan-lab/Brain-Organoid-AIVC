#!/usr/bin/env python3
"""
02_evaluation_GeneCompass.py - final 8 prediction-metric evaluation for GeneCompass + GEARS

Final prediction metrics:
- pearson_delta, pearson_de_delta
- spearmanr_delta, spearmanr_de_delta
- rmse, rmse_de
- mae, mae_de
- Select best model by pearson_de_delta

Usage:
    python 02_evaluation_GeneCompass.py --seeds 1 2 3 4 5 6 7 8 9 10
"""

import os
import sys
import pickle
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

FINAL_PREDICTION_METRICS = ["pearson_delta", "pearson_de_delta", "spearmanr_delta", "spearmanr_de_delta", "rmse", "rmse_de", "mae", "mae_de"]


def _final_prediction_metrics(metrics):
    if "spearman_delta" in metrics and "spearmanr_delta" not in metrics:
        metrics["spearmanr_delta"] = metrics["spearman_delta"]
    if "spearman_de_delta" in metrics and "spearmanr_de_delta" not in metrics:
        metrics["spearmanr_de_delta"] = metrics["spearman_de_delta"]
    return {k: metrics[k] for k in FINAL_PREDICTION_METRICS if k in metrics}


# Add paths - use absolute path like training script
GENECOMPASS_DIR = Path("<DATA_ROOT>/05.Behchmarking_DL/GeneCompass")
GEARS_CODE_DIR = GENECOMPASS_DIR / "downstream_tasks" / "gears" / "gears_code"
sys.path.insert(0, str(GENECOMPASS_DIR))
sys.path.insert(0, str(GEARS_CODE_DIR))

from geares import PertData, GEARS

# Setup logging
log_dir = GENECOMPASS_DIR / 'codes_GeneCompass' / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / f'02_evaluation_GeneCompass_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path("<DATA_ROOT>/05.Behchmarking_DL/data")
# Use 512k normalized data (same as training)
NOCAP_PROCESSED_DIR = DATA_DIR / "NOCAP_Telen_512k_normalized"
EMBEDDING_FILE = GENECOMPASS_DIR / "embeddings" / "nocap_prior_knowledge_64.pickle"
# Use 512k benchmark results directory
RESULTS_DIR = GENECOMPASS_DIR / "results" / "NOCAP_512k_benchmark"

EVAL_BATCH_SIZE = 16
HIDDEN_SIZE = 64
PRIMARY_METRIC = "pearson_de_delta"


# ============================================================================
# Metric Computation Functions (from reference code)
# ============================================================================

def compute_perturbation_metrics_pearson(results, ctrl_adata, non_zero_genes=False):
    """Compute Pearson correlation metrics"""
    metrics = {
        "pearson": [],
        "pearson_de": [],
        "pearson_delta": [],
        "pearson_de_delta": [],
    }

    conditions = np.unique(results["pert_cat"])
    condition2idx = {c: np.where(results["pert_cat"] == c)[0] for c in conditions}

    # Control mean
    ctrl_X = ctrl_adata.X
    if hasattr(ctrl_X, 'toarray'):
        ctrl_X = ctrl_X.toarray()
    mean_ctrl = np.array(ctrl_X.mean(0)).flatten()

    true_perturbed = results["truth"]
    pred_perturbed = results["pred"]

    true_mean_by_condition = np.array([true_perturbed[condition2idx[c]].mean(0) for c in conditions])
    pred_mean_by_condition = np.array([pred_perturbed[condition2idx[c]].mean(0) for c in conditions])

    delta_true = true_mean_by_condition - mean_ctrl
    delta_pred = pred_mean_by_condition - mean_ctrl

    # Get DE genes mapping
    geneid2idx = dict(zip(ctrl_adata.var.index.values, range(len(ctrl_adata.var))))

    def find_DE_genes(condition, top_n=20):
        try:
            # Try direct key lookup first (key format: "CONDITION" like "SOX5+ctrl")
            if condition in ctrl_adata.uns["rank_genes_groups_cov_all"]:
                de_genes = ctrl_adata.uns["rank_genes_groups_cov_all"][condition][:top_n]
                return [geneid2idx[g] for g in de_genes if g in geneid2idx]

            # Fall back to prefixed key format (key format: "prefix_CONDITION_suffix")
            key_components = next(iter(ctrl_adata.uns["rank_genes_groups_cov_all"].keys())).split("_")
            if len(key_components) >= 3:
                condition_key = "_".join([key_components[0], condition, key_components[2]])
                de_genes = ctrl_adata.uns["rank_genes_groups_cov_all"][condition_key][:top_n]
                return [geneid2idx[g] for g in de_genes if g in geneid2idx]
            return []
        except:
            return []

    for i, cond in enumerate(conditions):
        # All genes
        r_all, _ = pearsonr(true_mean_by_condition[i], pred_mean_by_condition[i])
        metrics["pearson"].append(r_all if not np.isnan(r_all) else 0)

        r_delta, _ = pearsonr(delta_true[i], delta_pred[i])
        metrics["pearson_delta"].append(r_delta if not np.isnan(r_delta) else 0)

        # DE genes
        de_idx = find_DE_genes(cond)
        if len(de_idx) > 0:
            r_de, _ = pearsonr(true_mean_by_condition[i][de_idx], pred_mean_by_condition[i][de_idx])
            metrics["pearson_de"].append(r_de if not np.isnan(r_de) else 0)

            r_de_delta, _ = pearsonr(delta_true[i][de_idx], delta_pred[i][de_idx])
            metrics["pearson_de_delta"].append(r_de_delta if not np.isnan(r_de_delta) else 0)
        else:
            metrics["pearson_de"].append(metrics["pearson"][-1])
            metrics["pearson_de_delta"].append(metrics["pearson_delta"][-1])

    return {k: float(np.mean(v)) for k, v in metrics.items()}


def compute_perturbation_metrics_spearman(results, ctrl_adata, non_zero_genes=False):
    """Compute Spearman correlation metrics"""
    metrics = {
        "spearman": [],
        "spearman_de": [],
        "spearman_delta": [],
        "spearman_de_delta": [],
    }

    conditions = np.unique(results["pert_cat"])
    condition2idx = {c: np.where(results["pert_cat"] == c)[0] for c in conditions}

    ctrl_X = ctrl_adata.X
    if hasattr(ctrl_X, 'toarray'):
        ctrl_X = ctrl_X.toarray()
    mean_ctrl = np.array(ctrl_X.mean(0)).flatten()

    true_perturbed = results["truth"]
    pred_perturbed = results["pred"]

    true_mean_by_condition = np.array([true_perturbed[condition2idx[c]].mean(0) for c in conditions])
    pred_mean_by_condition = np.array([pred_perturbed[condition2idx[c]].mean(0) for c in conditions])

    delta_true = true_mean_by_condition - mean_ctrl
    delta_pred = pred_mean_by_condition - mean_ctrl

    geneid2idx = dict(zip(ctrl_adata.var.index.values, range(len(ctrl_adata.var))))

    def find_DE_genes(condition, top_n=20):
        try:
            # Try direct key lookup first (key format: "CONDITION" like "SOX5+ctrl")
            if condition in ctrl_adata.uns["rank_genes_groups_cov_all"]:
                de_genes = ctrl_adata.uns["rank_genes_groups_cov_all"][condition][:top_n]
                return [geneid2idx[g] for g in de_genes if g in geneid2idx]

            # Fall back to prefixed key format (key format: "prefix_CONDITION_suffix")
            key_components = next(iter(ctrl_adata.uns["rank_genes_groups_cov_all"].keys())).split("_")
            if len(key_components) >= 3:
                condition_key = "_".join([key_components[0], condition, key_components[2]])
                de_genes = ctrl_adata.uns["rank_genes_groups_cov_all"][condition_key][:top_n]
                return [geneid2idx[g] for g in de_genes if g in geneid2idx]
            return []
        except:
            return []

    for i, cond in enumerate(conditions):
        r_all, _ = spearmanr(true_mean_by_condition[i], pred_mean_by_condition[i])
        metrics["spearman"].append(r_all if not np.isnan(r_all) else 0)

        r_delta, _ = spearmanr(delta_true[i], delta_pred[i])
        metrics["spearman_delta"].append(r_delta if not np.isnan(r_delta) else 0)

        de_idx = find_DE_genes(cond)
        if len(de_idx) > 0:
            r_de, _ = spearmanr(true_mean_by_condition[i][de_idx], pred_mean_by_condition[i][de_idx])
            metrics["spearman_de"].append(r_de if not np.isnan(r_de) else 0)

            r_de_delta, _ = spearmanr(delta_true[i][de_idx], delta_pred[i][de_idx])
            metrics["spearman_de_delta"].append(r_de_delta if not np.isnan(r_de_delta) else 0)
        else:
            metrics["spearman_de"].append(metrics["spearman"][-1])
            metrics["spearman_de_delta"].append(metrics["spearman_delta"][-1])

    return {k: float(np.mean(v)) for k, v in metrics.items()}


def compute_perturbation_metrics_mse(results, ctrl_adata, non_zero_genes=False):
    """Compute MSE metrics"""
    metrics = {
        "mse": [],
        "mse_de": [],
        "mse_delta": [],
        "mse_de_delta": [],
    }

    conditions = np.unique(results["pert_cat"])
    condition2idx = {c: np.where(results["pert_cat"] == c)[0] for c in conditions}

    ctrl_X = ctrl_adata.X
    if hasattr(ctrl_X, 'toarray'):
        ctrl_X = ctrl_X.toarray()
    mean_ctrl = np.array(ctrl_X.mean(0)).flatten()

    true_perturbed = results["truth"]
    pred_perturbed = results["pred"]

    true_mean_by_condition = np.array([true_perturbed[condition2idx[c]].mean(0) for c in conditions])
    pred_mean_by_condition = np.array([pred_perturbed[condition2idx[c]].mean(0) for c in conditions])

    delta_true = true_mean_by_condition - mean_ctrl
    delta_pred = pred_mean_by_condition - mean_ctrl

    geneid2idx = dict(zip(ctrl_adata.var.index.values, range(len(ctrl_adata.var))))

    def find_DE_genes(condition, top_n=20):
        try:
            # Try direct key lookup first (key format: "CONDITION" like "SOX5+ctrl")
            if condition in ctrl_adata.uns["rank_genes_groups_cov_all"]:
                de_genes = ctrl_adata.uns["rank_genes_groups_cov_all"][condition][:top_n]
                return [geneid2idx[g] for g in de_genes if g in geneid2idx]

            # Fall back to prefixed key format (key format: "prefix_CONDITION_suffix")
            key_components = next(iter(ctrl_adata.uns["rank_genes_groups_cov_all"].keys())).split("_")
            if len(key_components) >= 3:
                condition_key = "_".join([key_components[0], condition, key_components[2]])
                de_genes = ctrl_adata.uns["rank_genes_groups_cov_all"][condition_key][:top_n]
                return [geneid2idx[g] for g in de_genes if g in geneid2idx]
            return []
        except:
            return []

    for i, cond in enumerate(conditions):
        metrics["mse"].append(mean_squared_error(true_mean_by_condition[i], pred_mean_by_condition[i]))
        metrics["mse_delta"].append(mean_squared_error(delta_true[i], delta_pred[i]))

        de_idx = find_DE_genes(cond)
        if len(de_idx) > 0:
            metrics["mse_de"].append(mean_squared_error(true_mean_by_condition[i][de_idx], pred_mean_by_condition[i][de_idx]))
            metrics["mse_de_delta"].append(mean_squared_error(delta_true[i][de_idx], delta_pred[i][de_idx]))
        else:
            metrics["mse_de"].append(metrics["mse"][-1])
            metrics["mse_de_delta"].append(metrics["mse_delta"][-1])

    return {k: float(np.mean(v)) for k, v in metrics.items()}


def compute_perturbation_metrics_rmse(results, ctrl_adata, non_zero_genes=False):
    """Compute RMSE metrics"""
    mse_metrics = compute_perturbation_metrics_mse(results, ctrl_adata, non_zero_genes)
    return {
        "rmse": np.sqrt(mse_metrics["mse"]),
        "rmse_de": np.sqrt(mse_metrics["mse_de"]),
    }


def compute_perturbation_metrics_mae(results, ctrl_adata, non_zero_genes=False):
    """Compute final MAE metrics."""
    metrics = {"mae": [], "mae_de": []}
    conditions = np.unique(results["pert_cat"])
    condition2idx = {c: np.where(results["pert_cat"] == c)[0] for c in conditions}
    true_perturbed = results["truth"]
    pred_perturbed = results["pred"]
    true_mean_by_condition = np.array([true_perturbed[condition2idx[c]].mean(0) for c in conditions])
    pred_mean_by_condition = np.array([pred_perturbed[condition2idx[c]].mean(0) for c in conditions])
    geneid2idx = dict(zip(ctrl_adata.var.index.values, range(len(ctrl_adata.var))))

    def find_DE_genes(condition, top_n=20):
        try:
            if condition in ctrl_adata.uns["rank_genes_groups_cov_all"]:
                de_genes = ctrl_adata.uns["rank_genes_groups_cov_all"][condition][:top_n]
                return [geneid2idx[g] for g in de_genes if g in geneid2idx]
            key_components = next(iter(ctrl_adata.uns["rank_genes_groups_cov_all"].keys())).split("_")
            if len(key_components) >= 3:
                condition_key = "_".join([key_components[0], condition, key_components[2]])
                de_genes = ctrl_adata.uns["rank_genes_groups_cov_all"][condition_key][:top_n]
                return [geneid2idx[g] for g in de_genes if g in geneid2idx]
            return []
        except Exception:
            return []

    for i, cond in enumerate(conditions):
        metrics["mae"].append(mean_absolute_error(true_mean_by_condition[i], pred_mean_by_condition[i]))
        de_idx = find_DE_genes(cond)
        if len(de_idx) > 0:
            metrics["mae_de"].append(mean_absolute_error(true_mean_by_condition[i][de_idx], pred_mean_by_condition[i][de_idx]))
        else:
            metrics["mae_de"].append(metrics["mae"][-1])
    return {k: float(np.mean(v)) for k, v in metrics.items()}


# ============================================================================
# Evaluation
# ============================================================================

def eval_perturb(loader, model, device):
    """Run model in inference mode using test loader"""
    model.eval()

    pert_cat = []
    pred = []
    truth = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Get condition names
            if hasattr(batch, 'pert'):
                pert_cat.extend(batch.pert)
            elif hasattr(batch, 'condition'):
                pert_cat.extend(batch.condition)

            # Forward pass
            output = model(batch)
            p = output['pred'] if isinstance(output, dict) else output
            t = batch.y

            pred.append(p.cpu().numpy())
            truth.append(t.cpu().numpy())

    results = {
        "pert_cat": np.array(pert_cat),
        "pred": np.vstack(pred),
        "truth": np.vstack(truth),
    }

    return results


def evaluate_one_seed(seed, device='cuda'):
    """Evaluate model for one seed"""
    logger.info(f"Evaluating seed {seed}")

    result_dir = RESULTS_DIR / f"seed_{seed}"

    # Check for model files
    model_path = None
    for name in ["model.pt", "ckpt.pth", "best_model.pt"]:
        if (result_dir / name).exists():
            model_path = result_dir / name
            break

    if model_path is None:
        logger.error(f"No model found at {result_dir}")
        return None

    logger.info(f"Found model at {model_path}")

    # Load data
    pert_data = PertData(str(NOCAP_PROCESSED_DIR))
    pert_data.load(data_path=str(NOCAP_PROCESSED_DIR))
    pert_data.prepare_split(split='simulation', seed=seed)
    pert_data.get_dataloader(batch_size=EVAL_BATCH_SIZE, test_batch_size=EVAL_BATCH_SIZE)

    # Initialize model
    gears_model = GEARS(pert_data, device=device)
    gears_model.model_initialize(
        damoxing_type='prior_knowledge',
        hidden_size=HIDDEN_SIZE,
        embedding_file=str(EMBEDDING_FILE)
    )

    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        gears_model.model.load_state_dict(checkpoint['model_state_dict'])
    else:
        gears_model.model.load_state_dict(checkpoint)
    gears_model.model.eval()

    # Run evaluation on test loader
    logger.info("Running inference on test set...")
    test_loader = pert_data.dataloader['test_loader']
    test_res = eval_perturb(test_loader, gears_model.model, device)

    logger.info(f"Test set: {len(test_res['pert_cat'])} samples, {len(np.unique(test_res['pert_cat']))} conditions")

    # Get control adata for metric computation
    ctrl_adata = pert_data.adata[pert_data.adata.obs['condition'] == 'ctrl']

    # Compute final 8 prediction metrics
    metrics = {}

    pearson_metrics = compute_perturbation_metrics_pearson(test_res, ctrl_adata)
    logger.info(f"Pearson metrics: {pearson_metrics}")
    metrics.update(pearson_metrics)

    spearman_metrics = compute_perturbation_metrics_spearman(test_res, ctrl_adata)
    logger.info(f"Spearman metrics: {spearman_metrics}")
    metrics.update(spearman_metrics)

    rmse_metrics = compute_perturbation_metrics_rmse(test_res, ctrl_adata)
    logger.info(f"RMSE metrics: {rmse_metrics}")
    metrics.update(rmse_metrics)

    mae_metrics = compute_perturbation_metrics_mae(test_res, ctrl_adata)
    logger.info(f"MAE metrics: {mae_metrics}")
    metrics.update(mae_metrics)

    metrics = _final_prediction_metrics(metrics)

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(result_dir / "metrics_final8.csv", index=False)

    logger.info(f"Seed {seed} complete: {PRIMARY_METRIC} = {metrics[PRIMARY_METRIC]:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Final 8 prediction-metric evaluation for GeneCompass + GEARS")
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 11)),
                        help='Seeds to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Final 8 prediction-metric evaluation for GeneCompass + GEARS")
    logger.info("=" * 60)
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Primary metric: {PRIMARY_METRIC}")

    all_metrics = []

    for seed in args.seeds:
        try:
            torch.cuda.empty_cache()
            metrics = evaluate_one_seed(seed, args.device)
            if metrics:
                metrics['seed'] = seed
                all_metrics.append(metrics)
        except Exception as e:
            logger.error(f"Error evaluating seed {seed}: {e}")
            import traceback
            traceback.print_exc()

    if all_metrics:
        # Save combined results
        df = pd.DataFrame(all_metrics)
        df.to_csv(RESULTS_DIR / "Seed1to10_model_performance.csv", index=False)

        # Find best seed
        best_idx = df[PRIMARY_METRIC].idxmax()
        best_seed = int(df.loc[best_idx, 'seed'])
        best_score = df.loc[best_idx, PRIMARY_METRIC]

        logger.info("=" * 60)
        logger.info("Evaluation Summary")
        logger.info("=" * 60)
        for _, row in df.iterrows():
            logger.info(f"Seed {int(row['seed'])}: {PRIMARY_METRIC} = {row[PRIMARY_METRIC]:.4f}")

        logger.info(f"\nBest seed: {best_seed} ({PRIMARY_METRIC} = {best_score:.4f})")

        # Save best model selection
        selection = {
            'model': 'Telen-GeneCompass',
            'primary_metric': PRIMARY_METRIC,
            'all_results': {str(int(row['seed'])): row[PRIMARY_METRIC] for _, row in df.iterrows()},
            'best_seed': best_seed,
            'best_score': best_score
        }
        with open(RESULTS_DIR / "best_model_selection.json", 'w') as f:
            json.dump(selection, f, indent=2)

    logger.info("=" * 60)
    logger.info("Evaluation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
