#!/usr/bin/env python
"""
02_evaluation_scLAMBDA.py - Evaluation script for scLAMBDA benchmarking

This script:
1. Loads trained models for each seed
2. Computes the final 8 prediction metrics on test set
3. Selects best model by pearson_de_delta
4. Saves metrics in standardized format

Final prediction metrics:
- pearson_delta, pearson_de_delta
- spearmanr_delta, spearmanr_de_delta
- rmse, rmse_de
- mae, mae_de

Usage:
    python 02_evaluation_scLAMBDA.py --data_dir ./data --model_dir ./results/NOCAP_benchmark
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch

# Add scLAMBDA to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from sclambda.model import Model

FINAL_PREDICTION_METRICS = ["pearson_delta", "pearson_de_delta", "spearmanr_delta", "spearmanr_de_delta", "rmse", "rmse_de", "mae", "mae_de"]


def _final_prediction_metrics(metrics):
    if "spearman_delta" in metrics and "spearmanr_delta" not in metrics:
        metrics["spearmanr_delta"] = metrics["spearman_delta"]
    if "spearman_de_delta" in metrics and "spearmanr_de_delta" not in metrics:
        metrics["spearmanr_de_delta"] = metrics["spearman_de_delta"]
    return {k: metrics[k] for k in FINAL_PREDICTION_METRICS if k in metrics}

from sclambda.networks import Net
from sclambda.utils import data_split


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate scLAMBDA models')
    parser.add_argument('--data_dir', type=str,
                        default='<DATA_ROOT>/05.Behchmarking_DL/scLAMBDA/data',
                        help='Directory with preprocessed data')
    parser.add_argument('--model_dir', type=str,
                        default='<DATA_ROOT>/05.Behchmarking_DL/scLAMBDA/results/NOCAP_benchmark',
                        help='Directory with trained models')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save evaluation results (default: model_dir)')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Seeds to evaluate (default: all available)')
    parser.add_argument('--ctrl_size', type=int, default=5000,
                        help='Number of control cells for prediction')
    parser.add_argument('--de_top_n', type=int, default=20,
                        help='Number of top DE genes for DE metrics')
    return parser.parse_args()


def compute_metrics(pred_mean, truth_mean, ctrl_mean, de_idx=None):
    """
    Compute final prediction metrics for one perturbation condition.

    Args:
        pred_mean: Predicted mean expression (n_genes,)
        truth_mean: Ground truth mean expression (n_genes,)
        ctrl_mean: Control mean expression (n_genes,)
        de_idx: Indices of DE genes (optional)

    Returns:
        Dictionary with all metrics
    """
    # Compute deltas
    pred_delta = pred_mean - ctrl_mean
    truth_delta = truth_mean - ctrl_mean

    metrics = {}

    # All genes metrics
    # Pearson
    r, _ = pearsonr(pred_mean, truth_mean)
    metrics['pearson'] = r if not np.isnan(r) else 0

    r, _ = pearsonr(pred_delta, truth_delta)
    metrics['pearson_delta'] = r if not np.isnan(r) else 0

    # Spearman
    r, _ = spearmanr(pred_mean, truth_mean)
    metrics['spearman'] = r if not np.isnan(r) else 0

    r, _ = spearmanr(pred_delta, truth_delta)
    metrics['spearman_delta'] = r if not np.isnan(r) else 0

    # RMSE
    mse = mean_squared_error(truth_mean, pred_mean)
    mse_delta = mean_squared_error(truth_delta, pred_delta)
    metrics['rmse'] = np.sqrt(mse)
    metrics['rmse_delta'] = np.sqrt(mse_delta)

    # MAE
    metrics['mae'] = mean_absolute_error(truth_mean, pred_mean)
    metrics['mae_delta'] = mean_absolute_error(truth_delta, pred_delta)

    # DE genes metrics (if DE indices provided)
    if de_idx is not None and len(de_idx) > 0:
        pred_mean_de = pred_mean[de_idx]
        truth_mean_de = truth_mean[de_idx]
        pred_delta_de = pred_delta[de_idx]
        truth_delta_de = truth_delta[de_idx]

        # Pearson DE
        r, _ = pearsonr(pred_mean_de, truth_mean_de)
        metrics['pearson_de'] = r if not np.isnan(r) else 0

        r, _ = pearsonr(pred_delta_de, truth_delta_de)
        metrics['pearson_de_delta'] = r if not np.isnan(r) else 0

        # Spearman DE
        r, _ = spearmanr(pred_mean_de, truth_mean_de)
        metrics['spearman_de'] = r if not np.isnan(r) else 0

        r, _ = spearmanr(pred_delta_de, truth_delta_de)
        metrics['spearman_de_delta'] = r if not np.isnan(r) else 0

        # RMSE DE
        mse_de = mean_squared_error(truth_mean_de, pred_mean_de)
        mse_de_delta = mean_squared_error(truth_delta_de, pred_delta_de)
        metrics['rmse_de'] = np.sqrt(mse_de)
        metrics['rmse_de_delta'] = np.sqrt(mse_de_delta)

        # MAE DE
        metrics['mae_de'] = mean_absolute_error(truth_mean_de, pred_mean_de)
        metrics['mae_de_delta'] = mean_absolute_error(truth_delta_de, pred_delta_de)
    else:
        # No DE genes available
        metrics['pearson_de'] = np.nan
        metrics['pearson_de_delta'] = np.nan
        metrics['spearman_de'] = np.nan
        metrics['spearman_de_delta'] = np.nan
        metrics['rmse_de'] = np.nan
        metrics['rmse_de_delta'] = np.nan
        metrics['mae_de'] = np.nan
        metrics['mae_de_delta'] = np.nan

    return _final_prediction_metrics(metrics)


def get_de_genes(adata, condition, n_top=20):
    """
    Get DE gene indices for a condition.

    Args:
        adata: AnnData object with DE gene info in uns
        condition: Perturbation condition name
        n_top: Number of top DE genes

    Returns:
        List of DE gene indices
    """
    # Try to find DE genes in adata.uns
    if 'rank_genes_groups_cov_all' in adata.uns:
        # Find matching key
        for key in adata.uns['rank_genes_groups_cov_all'].keys():
            if condition in key:
                de_genes = adata.uns['rank_genes_groups_cov_all'][key][:n_top]
                # Convert gene names to indices
                gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
                de_idx = [gene_to_idx[g] for g in de_genes if g in gene_to_idx]
                return de_idx

    # Fallback: compute DE genes on the fly
    try:
        ctrl_adata = adata[adata.obs['condition'] == 'ctrl'].copy()
        pert_adata = adata[adata.obs['condition'] == condition].copy()

        if pert_adata.n_obs == 0:
            return []

        import anndata as ad
        combined = ad.concat([ctrl_adata, pert_adata])
        combined.obs['group'] = ['ctrl'] * ctrl_adata.n_obs + ['pert'] * pert_adata.n_obs

        sc.tl.rank_genes_groups(combined, 'group', method='wilcoxon',
                                reference='ctrl', n_genes=n_top)

        de_genes = combined.uns['rank_genes_groups']['names']['pert'][:n_top]
        gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
        de_idx = [gene_to_idx[g] for g in de_genes if g in gene_to_idx]
        return de_idx
    except Exception as e:
        print(f"Warning: Could not compute DE genes for {condition}: {e}")
        return []


def evaluate_model(model, adata, gene_embeddings, test_conditions, ctrl_mean, args):
    """
    Evaluate a trained model on test conditions.

    Args:
        model: scLAMBDA Model object
        adata: AnnData object
        gene_embeddings: Gene embedding dictionary
        test_conditions: List of test perturbation conditions
        ctrl_mean: Control mean expression
        args: Command line arguments

    Returns:
        DataFrame with metrics for each condition
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.Net.eval()

    all_metrics = []

    with torch.no_grad():
        for cond in tqdm(test_conditions, desc="Evaluating"):
            # Skip if condition not in gene embeddings
            if cond not in gene_embeddings:
                print(f"Warning: {cond} not in gene embeddings, skipping")
                continue

            # Get ground truth
            pert_adata = adata[adata.obs['condition'] == cond]
            if pert_adata.n_obs == 0:
                print(f"Warning: No cells for {cond}, skipping")
                continue

            truth_mean = np.array(pert_adata.X.mean(axis=0)).flatten()

            # Get prediction
            pert_emb = gene_embeddings[cond]
            ctrl_x = model.ctrl_x

            val_p = torch.from_numpy(
                np.tile(pert_emb, (ctrl_x.shape[0], 1))
            ).float().to(device)

            x_hat, p_hat, mean_z, log_var_z, s = model.Net(ctrl_x, val_p)
            # Flatten ctrl_mean in case it's a 2D matrix
            ctrl_mean_flat = np.array(model.ctrl_mean).flatten()
            pred_mean = x_hat.mean(dim=0).cpu().numpy() + ctrl_mean_flat

            # Get DE genes
            de_idx = get_de_genes(adata, cond, n_top=args.de_top_n)

            # Compute metrics
            metrics = compute_metrics(pred_mean, truth_mean, ctrl_mean, de_idx)
            metrics['condition'] = cond
            metrics['n_de_genes'] = len(de_idx) if de_idx else 0

            all_metrics.append(metrics)

    return pd.DataFrame(all_metrics)


def main():
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = args.model_dir

    print("=" * 60)
    print("scLAMBDA Evaluation Script")
    print("=" * 60)
    print(f"Start time: {datetime.now().isoformat()}")

    # Load preprocessed data
    data_dir = Path(args.data_dir)
    print(f"\nLoading data from {data_dir}...")

    adata = sc.read_h5ad(data_dir / 'adata_preprocessed.h5ad')
    print(f"Loaded adata: {adata.shape}")

    with open(data_dir / 'gene_embeddings.pkl', 'rb') as f:
        gene_embeddings = pickle.load(f)
    print(f"Loaded {len(gene_embeddings)} gene embeddings")

    # Find available seeds
    model_dir = Path(args.model_dir)
    if args.seeds is None:
        seed_dirs = sorted([d for d in model_dir.iterdir()
                           if d.is_dir() and d.name.startswith('seed_')])
        args.seeds = [int(d.name.split('_')[1]) for d in seed_dirs]

    print(f"Evaluating seeds: {args.seeds}")

    # Control mean expression
    ctrl_adata = adata[adata.obs['condition'] == 'ctrl']
    ctrl_mean = np.array(ctrl_adata.X.mean(axis=0)).flatten()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate each seed
    all_seed_metrics = []

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Evaluating seed {seed}")
        print("=" * 60)

        seed_dir = model_dir / f'seed_{seed}'
        model_path = seed_dir / 'ckpt.pth'

        if not model_path.exists():
            print(f"Model not found at {model_path}, skipping")
            continue

        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Recreate data split with same seed
        adata_split, split = data_split(
            adata.copy(),
            split_type='single',
            train_gene_set_size=0.75,
            seed=seed
        )

        # Get test conditions
        test_conditions = split['test_subgroup']['unseen_single']
        print(f"Test conditions: {len(test_conditions)}")

        # Build model (need to initialize network structure)
        model = Model(
            adata=adata_split,
            gene_emb=gene_embeddings,
            multi_gene=False,
            training_epochs=1,  # Not training, just need structure
            ctrl_size=args.ctrl_size,
            model_path=str(seed_dir),
            seed=seed
        )

        # Create the network structure (normally done in train())
        from sclambda.networks import Net as NetClass
        model.Net = NetClass(
            x_dim=model.x_dim,
            p_dim=model.p_dim,
            latent_dim=model.latent_dim,
            hidden_dim=model.hidden_dim,
            use_tg_coord=model.use_tg_coord
        )

        # Load trained weights
        model.Net.load_state_dict(checkpoint['Net'])
        model.Net.to(device)

        # Evaluate
        metrics_df = evaluate_model(
            model, adata_split, gene_embeddings, test_conditions, ctrl_mean, args
        )

        # Add seed column
        metrics_df['seed'] = seed

        # Save per-seed metrics
        metrics_df.to_csv(seed_dir / 'test_metrics.csv', index=False)

        # Aggregate metrics for this seed
        seed_summary = {
            'seed': seed,
            'n_conditions': len(metrics_df),
        }

        # Mean of each metric
        metric_cols = [c for c in metrics_df.columns
                       if c not in ['condition', 'seed', 'n_de_genes']]
        for col in metric_cols:
            seed_summary[col] = metrics_df[col].mean()

        all_seed_metrics.append(seed_summary)
        print(f"\nSeed {seed} summary:")
        for k, v in seed_summary.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        # Clear GPU memory
        del model
        torch.cuda.empty_cache()

    # Create summary DataFrame
    summary_df = pd.DataFrame(all_seed_metrics)

    # Save summary
    output_dir = Path(args.output_dir)
    summary_df.to_csv(output_dir / 'all_seeds_metrics.csv', index=False)

    # Find best seed by pearson_de_delta
    if 'pearson_de_delta' in summary_df.columns:
        best_idx = summary_df['pearson_de_delta'].idxmax()
        best_seed = summary_df.loc[best_idx, 'seed']
        best_metric = summary_df.loc[best_idx, 'pearson_de_delta']
    else:
        # Fallback to pearson_delta
        best_idx = summary_df['pearson_delta'].idxmax()
        best_seed = summary_df.loc[best_idx, 'seed']
        best_metric = summary_df.loc[best_idx, 'pearson_delta']

    # Save best seed info
    best_seed_info = {
        'best_seed': int(best_seed),
        'best_pearson_de_delta': float(best_metric),
        'all_seeds_evaluated': list(args.seeds),
        'evaluated_at': datetime.now().isoformat()
    }

    with open(output_dir / 'best_seed_summary.json', 'w') as f:
        json.dump(best_seed_info, f, indent=2)

    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print(f"\nBest seed: {best_seed} (pearson_de_delta = {best_metric:.4f})")
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
