"""
Evaluate trained Telen-GEARS models with the final 8 prediction-metric panel.

Originally extracted from:
  model_training_organoid_telencephalicneuron.ipynb (cells 7, 8)

Metrics per seed (8 total):
  pearson_delta, pearson_de_delta
  spearmanr_delta, spearmanr_de_delta
  rmse, rmse_de
  mae, mae_de

Best-seed selection uses pearson_de_delta.
"""
import argparse
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scipy.stats import pearsonr, spearmanr

from gears import PertData, GEARS
from gears.inference import evaluate


def _set_all_seeds(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _base_metric(results: Dict, ctrl_adata: AnnData, stat_fn):
    """Compute {metric, metric_de, metric_delta, metric_de_delta} using stat_fn(a, b) → scalar."""
    conditions = np.unique(results['pert_cat'])
    assert 'ctrl' not in conditions
    cond2idx = {c: np.where(results['pert_cat'] == c)[0] for c in conditions}
    mean_ctrl = np.asarray(ctrl_adata.X.mean(0)).flatten()

    true_p = results['truth']
    pred_p = results['pred']
    mean_de_idxs = results.get('mean_de_idx') or results.get('de_idx')

    metric, metric_de, metric_delta, metric_de_delta = [], [], [], []
    for c in conditions:
        idx = cond2idx[c]
        true_c = np.asarray(true_p[idx]).mean(0)
        pred_c = np.asarray(pred_p[idx]).mean(0)
        de_idx = mean_de_idxs[c] if isinstance(mean_de_idxs, dict) else None
        metric.append(stat_fn(true_c, pred_c))
        if de_idx is not None and len(de_idx):
            metric_de.append(stat_fn(true_c[de_idx], pred_c[de_idx]))
            metric_de_delta.append(stat_fn(true_c[de_idx] - mean_ctrl[de_idx],
                                           pred_c[de_idx] - mean_ctrl[de_idx]))
        metric_delta.append(stat_fn(true_c - mean_ctrl, pred_c - mean_ctrl))

    def _mean(x):
        return float(np.mean(x)) if len(x) else float('nan')
    return _mean(metric), _mean(metric_de), _mean(metric_delta), _mean(metric_de_delta)


def _pearson_stat(a, b):
    r, _ = pearsonr(a, b)
    return r


def _spearman_stat(a, b):
    r, _ = spearmanr(a, b)
    return r


def _rmse_stat(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _mae_stat(a, b):
    return float(np.mean(np.abs(a - b)))


def compute_all_metrics(results, ctrl_adata):
    p  = _base_metric(results, ctrl_adata, _pearson_stat)
    s  = _base_metric(results, ctrl_adata, _spearman_stat)
    r  = _base_metric(results, ctrl_adata, _rmse_stat)
    m  = _base_metric(results, ctrl_adata, _mae_stat)
    return {
        'pearson_delta': p[2], 'pearson_de_delta': p[3],
        'spearmanr_delta': s[2], 'spearmanr_de_delta': s[3],
        'rmse': r[0], 'rmse_de': r[1],
        'mae': m[0], 'mae_de': m[1],
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Telen-GEARS across seeds (final 8 prediction metrics)')
    parser.add_argument('--pert_data_dir', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, default='organoid_telencephalicneuron_pcgenes')
    parser.add_argument('--gene_set_path', type=str, required=True)
    parser.add_argument('--models_dir', type=str, required=True,
                        help='Directory containing Seed_{k}_model subfolders')
    parser.add_argument('--out_csv', type=str, required=True)
    parser.add_argument('--seed_start', type=int, default=1)
    parser.add_argument('--seed_end', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    _set_all_seeds(42)

    rows = []
    for seed in range(args.seed_start, args.seed_end + 1):
        pert_data = PertData(args.pert_data_dir, gene_set_path=args.gene_set_path)
        pert_data.load(data_name=args.dataset_name,
                       data_path=str(Path(args.pert_data_dir) / args.dataset_name))
        pert_data.prepare_split(split='simulation', seed=seed)
        pert_data.get_dataloader(batch_size=args.batch_size,
                                 test_batch_size=args.test_batch_size)

        gears_model = GEARS(pert_data, device=args.device,
                            weight_bias_track=False,
                            proj_name='gears_telen',
                            exp_name=f'gears_telen_seed_{seed}')
        gears_model.load_pretrained(str(Path(args.models_dir) / f'Seed_{seed}_model'))

        test_loader = pert_data.dataloader['test_loader']
        test_res = evaluate(test_loader, gears_model.model,
                            gears_model.config['uncertainty'],
                            device=args.device)
        ctrl_adata = pert_data.adata[pert_data.adata.obs['condition'] == 'ctrl']

        metrics = compute_all_metrics(test_res, ctrl_adata)
        metrics['seed'] = seed
        rows.append(metrics)
        print(f'[Seed {seed}] pearson_de_delta = {metrics["pearson_de_delta"]:.4f}', flush=True)

    df = pd.DataFrame(rows).set_index('seed')
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv)
    print(f'Wrote {args.out_csv}')
    print('Mean ± std across seeds:')
    print(df.agg(['mean', 'std']).T)
    print(f'Best seed by pearson_de_delta: {df["pearson_de_delta"].idxmax()}')


if __name__ == '__main__':
    main()
