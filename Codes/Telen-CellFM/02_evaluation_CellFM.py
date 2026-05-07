"""
02_evaluation_CellFM.py - Evaluate trained GEARS model with final 8 prediction metrics
Metrics: Pearson/Spearman delta and DE-delta correlations; RMSE/MAE and DE-restricted RMSE/MAE.

Usage:
    python 02_evaluation_CellFM.py \
        --data_dir /path/to/gears_data \
        --data_name nocap_perturbation \
        --emb_path /path/to/embeddings.npy \
        --model_path /path/to/best_model.pt \
        --output_dir /path/to/results \
        --seed 1 \
        --max_cells 600 \
        --device cuda:0
"""

import os
import sys
import argparse
import numpy as np
import torch
import pickle
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict
from anndata import AnnData

CODES_DIR = '<ENV_ROOT>/nvme/model3/CellFM'
ROOT_DIR = '<ENV_ROOT>/nvme/model3/CellFM'
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, CODES_DIR)

from gears import PertData

FINAL_PREDICTION_METRICS = ["pearson_delta", "pearson_de_delta", "spearmanr_delta", "spearmanr_de_delta", "rmse", "rmse_de", "mae", "mae_de"]


def _final_prediction_metrics(metrics):
    if "spearman_delta" in metrics and "spearmanr_delta" not in metrics:
        metrics["spearmanr_delta"] = metrics["spearman_delta"]
    if "spearman_de_delta" in metrics and "spearmanr_de_delta" not in metrics:
        metrics["spearmanr_de_delta"] = metrics["spearman_de_delta"]
    return {k: metrics[k] for k in FINAL_PREDICTION_METRICS if k in metrics}

from cellfm_gears_model import GEARS_Model_CellEmb, load_cell_embeddings

# ==================== Metric Functions ====================

def compute_pearson_metrics(results: Dict, ctrl_adata: AnnData, return_raw: bool = False) -> Dict:
    """Compute Pearson correlation metrics."""
    metrics = {"pearson": [], "pearson_de": [], "pearson_delta": [], "pearson_de_delta": []}
    
    conditions = np.unique(results["pert_cat"])
    condition2idx = {c: np.where(results["pert_cat"] == c)[0] for c in conditions}
    
    mean_ctrl = np.array(ctrl_adata.X.mean(0)).flatten()
    true_perturbed = results["truth"]
    pred_perturbed = results["pred"]
    
    true_mean_by_condition = np.array([true_perturbed[condition2idx[c]].mean(0) for c in conditions])
    pred_mean_by_condition = np.array([pred_perturbed[condition2idx[c]].mean(0) for c in conditions])
    
    delta_true = true_mean_by_condition - mean_ctrl
    delta_pred = pred_mean_by_condition - mean_ctrl
    
    geneid2idx = dict(zip(ctrl_adata.var.index.values, range(len(ctrl_adata.var))))
    
    for i, cond in enumerate(conditions):
        # All genes
        r, _ = pearsonr(true_mean_by_condition[i], pred_mean_by_condition[i])
        metrics["pearson"].append(r)
        r_delta, _ = pearsonr(delta_true[i], delta_pred[i])
        metrics["pearson_delta"].append(r_delta)
        
        # DE genes
        de_idx = find_DE_genes(ctrl_adata, cond, geneid2idx)
        if de_idx:
            r_de, _ = pearsonr(true_mean_by_condition[i][de_idx], pred_mean_by_condition[i][de_idx])
            metrics["pearson_de"].append(r_de)
            r_de_delta, _ = pearsonr(delta_true[i][de_idx], delta_pred[i][de_idx])
            metrics["pearson_de_delta"].append(r_de_delta)
    
    if not return_raw:
        for k in metrics:
            if metrics[k]:
                metrics[k] = float(np.nanmean(metrics[k]))
            else:
                metrics[k] = float('nan')
    return metrics


def compute_spearman_metrics(results: Dict, ctrl_adata: AnnData, return_raw: bool = False) -> Dict:
    """Compute Spearman correlation metrics."""
    metrics = {"spearman": [], "spearman_de": [], "spearman_delta": [], "spearman_de_delta": []}
    
    conditions = np.unique(results["pert_cat"])
    condition2idx = {c: np.where(results["pert_cat"] == c)[0] for c in conditions}
    
    mean_ctrl = np.array(ctrl_adata.X.mean(0)).flatten()
    true_perturbed = results["truth"]
    pred_perturbed = results["pred"]
    
    true_mean_by_condition = np.array([true_perturbed[condition2idx[c]].mean(0) for c in conditions])
    pred_mean_by_condition = np.array([pred_perturbed[condition2idx[c]].mean(0) for c in conditions])
    
    delta_true = true_mean_by_condition - mean_ctrl
    delta_pred = pred_mean_by_condition - mean_ctrl
    
    geneid2idx = dict(zip(ctrl_adata.var.index.values, range(len(ctrl_adata.var))))
    
    for i, cond in enumerate(conditions):
        # All genes
        r, _ = spearmanr(true_mean_by_condition[i], pred_mean_by_condition[i])
        metrics["spearman"].append(r)
        r_delta, _ = spearmanr(delta_true[i], delta_pred[i])
        metrics["spearman_delta"].append(r_delta)
        
        # DE genes
        de_idx = find_DE_genes(ctrl_adata, cond, geneid2idx)
        if de_idx:
            r_de, _ = spearmanr(true_mean_by_condition[i][de_idx], pred_mean_by_condition[i][de_idx])
            metrics["spearman_de"].append(r_de)
            r_de_delta, _ = spearmanr(delta_true[i][de_idx], delta_pred[i][de_idx])
            metrics["spearman_de_delta"].append(r_de_delta)
    
    if not return_raw:
        for k in metrics:
            if metrics[k]:
                metrics[k] = float(np.nanmean(metrics[k]))
            else:
                metrics[k] = float('nan')
    return metrics


def compute_rmse_metrics(results: Dict, ctrl_adata: AnnData, return_raw: bool = False) -> Dict:
    """Compute RMSE metrics."""
    metrics = {"rmse": [], "rmse_de": [], "rmse_delta": [], "rmse_de_delta": []}
    
    conditions = np.unique(results["pert_cat"])
    condition2idx = {c: np.where(results["pert_cat"] == c)[0] for c in conditions}
    
    mean_ctrl = np.array(ctrl_adata.X.mean(0)).flatten()
    true_perturbed = results["truth"]
    pred_perturbed = results["pred"]
    
    true_mean_by_condition = np.array([true_perturbed[condition2idx[c]].mean(0) for c in conditions])
    pred_mean_by_condition = np.array([pred_perturbed[condition2idx[c]].mean(0) for c in conditions])
    
    delta_true = true_mean_by_condition - mean_ctrl
    delta_pred = pred_mean_by_condition - mean_ctrl
    
    geneid2idx = dict(zip(ctrl_adata.var.index.values, range(len(ctrl_adata.var))))
    
    for i, cond in enumerate(conditions):
        # All genes
        metrics["rmse"].append(np.sqrt(mean_squared_error(true_mean_by_condition[i], pred_mean_by_condition[i])))
        metrics["rmse_delta"].append(np.sqrt(mean_squared_error(delta_true[i], delta_pred[i])))
        
        # DE genes
        de_idx = find_DE_genes(ctrl_adata, cond, geneid2idx)
        if de_idx:
            metrics["rmse_de"].append(np.sqrt(mean_squared_error(true_mean_by_condition[i][de_idx], pred_mean_by_condition[i][de_idx])))
            metrics["rmse_de_delta"].append(np.sqrt(mean_squared_error(delta_true[i][de_idx], delta_pred[i][de_idx])))
    
    if not return_raw:
        for k in metrics:
            if metrics[k]:
                metrics[k] = float(np.nanmean(metrics[k]))
            else:
                metrics[k] = float('nan')
    return metrics


def compute_mae_metrics(results: Dict, ctrl_adata: AnnData, return_raw: bool = False) -> Dict:
    """Compute MAE metrics."""
    metrics = {"mae": [], "mae_de": [], "mae_delta": [], "mae_de_delta": []}
    
    conditions = np.unique(results["pert_cat"])
    condition2idx = {c: np.where(results["pert_cat"] == c)[0] for c in conditions}
    
    mean_ctrl = np.array(ctrl_adata.X.mean(0)).flatten()
    true_perturbed = results["truth"]
    pred_perturbed = results["pred"]
    
    true_mean_by_condition = np.array([true_perturbed[condition2idx[c]].mean(0) for c in conditions])
    pred_mean_by_condition = np.array([pred_perturbed[condition2idx[c]].mean(0) for c in conditions])
    
    delta_true = true_mean_by_condition - mean_ctrl
    delta_pred = pred_mean_by_condition - mean_ctrl
    
    geneid2idx = dict(zip(ctrl_adata.var.index.values, range(len(ctrl_adata.var))))
    
    for i, cond in enumerate(conditions):
        # All genes
        metrics["mae"].append(mean_absolute_error(true_mean_by_condition[i], pred_mean_by_condition[i]))
        metrics["mae_delta"].append(mean_absolute_error(delta_true[i], delta_pred[i]))
        
        # DE genes
        de_idx = find_DE_genes(ctrl_adata, cond, geneid2idx)
        if de_idx:
            metrics["mae_de"].append(mean_absolute_error(true_mean_by_condition[i][de_idx], pred_mean_by_condition[i][de_idx]))
            metrics["mae_de_delta"].append(mean_absolute_error(delta_true[i][de_idx], delta_pred[i][de_idx]))
    
    if not return_raw:
        for k in metrics:
            if metrics[k]:
                metrics[k] = float(np.nanmean(metrics[k]))
            else:
                metrics[k] = float('nan')
    return metrics


def find_DE_genes(ctrl_adata: AnnData, condition: str, geneid2idx: dict, top_n: int = 20):
    """Find DE genes for a condition."""
    if "rank_genes_groups_cov_all" not in ctrl_adata.uns:
        return []
    
    # Find matching key
    for key in ctrl_adata.uns["rank_genes_groups_cov_all"].keys():
        if condition in key:
            de_genes = ctrl_adata.uns["rank_genes_groups_cov_all"][key][:top_n]
            return [geneid2idx[g] for g in de_genes if g in geneid2idx]
    return []


def run_inference(model, data_loader, device):
    """Run inference and collect predictions."""
    model.eval()
    pert_cats = []
    preds = []
    truths = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            pred = model(batch)
            
            # Get perturbation category
            if hasattr(batch, 'pert'):
                pert_cats.extend(batch.pert)
            else:
                # Fallback: use condition_name if available
                pert_cats.extend(['unknown'] * pred.shape[0])
            
            preds.append(pred.cpu().numpy())
            truths.append(batch.y.cpu().numpy())
    
    results = {
        "pert_cat": np.array(pert_cats),
        "pred": np.vstack(preds),
        "truth": np.vstack(truths)
    }
    return results


# ==================== Main ====================

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate GEARS model with final 8 prediction metrics')
    parser.add_argument('--data_dir', type=str, default='<ENV_ROOT>/nvme/model3/CellFM/gears_data',
                        help='Path to GEARS data directory')
    parser.add_argument('--data_name', type=str, default='nocap_perturbation',
                        help='Name of the dataset')
    parser.add_argument('--emb_path', type=str,
                        default='<ENV_ROOT>/nvme/model3/CellFM/embeddings/cellfm_ctrl_emb_nocap_perturbation_adapted_700cells.npy',
                        help='Path to cell embeddings')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint (default: output_dir/best_model.pt)')
    parser.add_argument('--output_dir', type=str,
                        default='<ENV_ROOT>/nvme/model3/CellFM/results/cellfm/seed_1',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size')
    parser.add_argument('--max_cells', type=int, default=600, help='Max cells to use')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Parameters
    data_dir = args.data_dir
    data_name = args.data_name
    emb_path = args.emb_path
    output_dir = args.output_dir
    model_path = args.model_path if args.model_path else os.path.join(output_dir, 'best_model.pt')
    seed = args.seed
    hidden_size = args.hidden_size
    max_cells = args.max_cells
    device = torch.device(args.device)

    os.makedirs(output_dir, exist_ok=True)

    print('=' * 70)
    print('GEARS Model Evaluation - final 8 prediction metrics')
    print('=' * 70)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print(f'Device: {device}')

    # Load perturbation data
    print('\n[1/5] Loading perturbation data...')
    pert_data = PertData(data_dir)
    pert_data.load(data_path=os.path.join(data_dir, data_name))
    pert_data.prepare_split(split='simulation', seed=seed, train_gene_set_size=0.75)
    pert_data.get_dataloader(batch_size=10, test_batch_size=32)
    print(f'  Total genes: {pert_data.adata.n_vars}')
    print(f'  Total cells: {pert_data.adata.n_obs}')

    # Get ctrl adata for metrics
    ctrl_adata = pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"].copy()
    print(f'  Control cells: {ctrl_adata.n_obs}')

    # Load cell-level embeddings
    print('\n[2/5] Loading cell-level embeddings...')
    ctrl_emb = load_cell_embeddings(emb_path, device=device, max_cells=max_cells)

    # Setup model
    print('\n[3/5] Initializing GEARS model...')
    gene2go_path = os.path.join(data_dir, 'gene2go.pkl')
    with open(gene2go_path, 'rb') as f:
        gene2go = pickle.load(f)
    num_perts = len(gene2go)
    num_genes = pert_data.adata.n_vars

    # Minimal GO graph
    G_go = torch.LongTensor([[i, i] for i in range(num_perts)]).T
    G_go_weight = torch.ones(num_perts)
    G_coexpress = torch.LongTensor([[i, i] for i in range(num_genes)]).T
    G_coexpress_weight = torch.ones(num_genes)

    model_args = {
        'num_genes': num_genes,
        'num_perts': num_perts,
        'hidden_size': hidden_size,
        'uncertainty': False,
        'num_go_gnn_layers': 2,
        'decoder_hidden_size': hidden_size,
        'num_gene_gnn_layers': 2,
        'no_perturb': False,
        'cell_fitness_pred': False,
        'G_go': G_go,
        'G_go_weight': G_go_weight,
        'G_coexpress': G_coexpress,
        'G_coexpress_weight': G_coexpress_weight,
        'device': device,
        'model_type': 'emb',
    }

    model = GEARS_Model_CellEmb(model_args, ctrl_emb=ctrl_emb)

    # Load best model
    print(f'  Loading model from {model_path}...')
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    del checkpoint
    model = model.to(device)
    print('  Model loaded successfully!')

    # Run inference on test set
    print('\n[4/5] Running inference on test set...')
    test_loader = pert_data.dataloader['test_loader']
    test_results = run_inference(model, test_loader, device)
    print(f'  Test samples: {len(test_results["pred"])}')
    print(f'  Unique conditions: {len(np.unique(test_results["pert_cat"]))}')

    # Compute final 8 prediction metrics
    print('\n[5/5] Computing metrics...')
    all_metrics = {}

    # Pearson
    print('  Computing Pearson metrics...')
    pearson_metrics = compute_pearson_metrics(test_results, ctrl_adata)
    all_metrics.update(pearson_metrics)

    # Spearman
    print('  Computing Spearman metrics...')
    spearman_metrics = compute_spearman_metrics(test_results, ctrl_adata)
    all_metrics.update(spearman_metrics)

    # RMSE
    print('  Computing RMSE metrics...')
    rmse_metrics = compute_rmse_metrics(test_results, ctrl_adata)
    all_metrics.update(rmse_metrics)

    # MAE
    print('  Computing MAE metrics...')
    mae_metrics = compute_mae_metrics(test_results, ctrl_adata)
    all_metrics.update(mae_metrics)

    # Print results
    print('\n' + '=' * 70)
    all_metrics = _final_prediction_metrics(all_metrics)

    print('Evaluation Results - final 8 prediction metrics')
    print('=' * 70)
    print('\nCorrelation Metrics:')
    print(f'  Pearson (delta):   {all_metrics.get("pearson_delta", "N/A"):.4f}')
    print(f'  Pearson (DE delta):{all_metrics.get("pearson_de_delta", "N/A"):.4f}')
    print(f'  Spearman (delta):  {all_metrics.get("spearmanr_delta", "N/A"):.4f}')
    print(f'  Spearman (DE delta):{all_metrics.get("spearmanr_de_delta", "N/A"):.4f}')

    print('\nError Metrics:')
    print(f'  RMSE:              {all_metrics.get("rmse", "N/A"):.4f}')
    print(f'  RMSE (DE):         {all_metrics.get("rmse_de", "N/A"):.4f}')
    print(f'  MAE:               {all_metrics.get("mae", "N/A"):.4f}')
    print(f'  MAE (DE):          {all_metrics.get("mae_de", "N/A"):.4f}')

    # Save metrics
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.pkl')
    with open(metrics_path, 'wb') as f:
        pickle.dump(all_metrics, f)
    print(f'\nMetrics saved to {metrics_path}')

    # Save as CSV for easy viewing
    metrics_df = pd.DataFrame([all_metrics])
    csv_path = os.path.join(output_dir, 'evaluation_metrics.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f'Metrics saved to {csv_path}')

    print('\n' + '=' * 70)
    print('Evaluation Complete!')
    print('=' * 70)
