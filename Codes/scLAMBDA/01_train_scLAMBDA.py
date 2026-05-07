#!/usr/bin/env python
"""
01_train_scLAMBDA.py - Preprocess data and train scLAMBDA for NOCAP benchmarking

This script:
1. Converts the NOCAP 512k condition format into scLAMBDA-compatible labels
2. Verifies control/perturbation counts and gene-embedding coverage
3. Saves reusable preprocessed inputs
4. Trains scLAMBDA models across multiple seeds

Usage:
    python 01_train_scLAMBDA.py --adata_path <input.h5ad> --data_dir ./data --output_dir ./results --seeds 1 2 3 4 5
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

import torch

# Add scLAMBDA to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from sclambda.model import Model
from sclambda.utils import data_split


def convert_condition_format(adata):
    """
    Convert condition column from 'ctrl+{gene}' format to standardized format.

    New dataset format:
    - 'ctrl' -> control cells
    - 'ctrl+CHD8' -> perturbed cells (need to extract 'CHD8')

    Output format (what scLAMBDA expects):
    - 'ctrl' -> control cells
    - 'CHD8' -> perturbed cells (gene name only)
    """
    print("Converting condition format...")

    # Get original conditions
    original_conditions = adata.obs['condition'].unique()
    print(f"Original conditions: {len(original_conditions)}")
    print(f"Sample: {list(original_conditions[:5])}")

    # Create new condition column
    def extract_gene(cond):
        if cond == 'ctrl':
            return 'ctrl'
        elif cond.startswith('ctrl+'):
            return cond.replace('ctrl+', '')
        else:
            return cond

    adata.obs['condition'] = adata.obs['condition'].apply(extract_gene)

    # Verify conversion
    new_conditions = adata.obs['condition'].unique()
    print(f"Converted conditions: {len(new_conditions)}")
    print(f"Sample: {list(new_conditions[:5])}")

    # Print distribution
    ctrl_count = (adata.obs['condition'] == 'ctrl').sum()
    pert_count = (adata.obs['condition'] != 'ctrl').sum()
    print(f"Control cells: {ctrl_count}")
    print(f"Perturbed cells: {pert_count}")

    return adata


def compute_de_genes_fast(adata, condition_col='condition', n_top=20, n_ctrl_subsample=5000):
    """
    Compute differentially expressed genes using fast method.
    """
    print("Computing DE genes (optimized)...")

    from scipy import sparse

    # Get unique conditions (excluding control)
    conditions = [c for c in adata.obs[condition_col].unique() if c != 'ctrl']

    # Get control cells and subsample for speed
    ctrl_mask = adata.obs[condition_col] == 'ctrl'
    ctrl_idx = np.where(ctrl_mask)[0]

    if len(ctrl_idx) > n_ctrl_subsample:
        np.random.seed(42)
        ctrl_idx = np.random.choice(ctrl_idx, n_ctrl_subsample, replace=False)

    # Pre-compute control mean
    ctrl_X = adata.X[ctrl_idx]
    if sparse.issparse(ctrl_X):
        ctrl_X = ctrl_X.toarray()
    ctrl_mean = ctrl_X.mean(axis=0)

    print(f"Using {len(ctrl_idx)} control cells for DE computation")

    # Store DE genes per condition
    rank_genes_groups_cov_all = {}
    non_zeros_gene_idx = {}
    top_non_dropout_de_20 = {}

    gene_names = adata.var_names.tolist()

    for cond in tqdm(conditions, desc="Computing DE genes"):
        cond_mask = adata.obs[condition_col] == cond
        cond_idx = np.where(cond_mask)[0]

        if len(cond_idx) == 0:
            key = f"ctrl_{cond}_1"
            rank_genes_groups_cov_all[key] = []
            non_zeros_gene_idx[key] = np.array([])
            top_non_dropout_de_20[key] = []
            continue

        # Subsample if too many cells
        if len(cond_idx) > 5000:
            cond_idx = np.random.choice(cond_idx, 5000, replace=False)

        cond_X = adata.X[cond_idx]
        if sparse.issparse(cond_X):
            cond_X = cond_X.toarray()
        cond_mean = cond_X.mean(axis=0)

        # Compute log fold change
        log_fc = cond_mean - ctrl_mean

        # Get top genes by absolute fold change
        top_idx = np.argsort(np.abs(log_fc))[::-1][:n_top]
        de_genes = [gene_names[i] for i in top_idx]

        key = f"ctrl_{cond}_1"
        rank_genes_groups_cov_all[key] = de_genes

        # Non-zero gene indices
        non_zero_idx = np.where(cond_mean > 0)[0]
        non_zeros_gene_idx[key] = non_zero_idx

        # Top non-dropout DE genes
        de_genes_expressed = [g for g in de_genes if cond_mean[gene_names.index(g)] > 0]
        top_non_dropout_de_20[key] = de_genes_expressed[:n_top]

    adata.uns['rank_genes_groups_cov_all'] = rank_genes_groups_cov_all
    adata.uns['non_zeros_gene_idx'] = non_zeros_gene_idx
    adata.uns['top_non_dropout_de_20'] = top_non_dropout_de_20

    print(f"Computed DE genes for {len(conditions)} conditions")
    return adata


def prepare_sclambda_inputs(args):
    """Prepare scLAMBDA input files when cached preprocessing outputs are unavailable or refresh is requested."""
    data_dir = Path(args.data_dir)
    adata_out = data_dir / 'adata_preprocessed.h5ad'
    embeddings_out = data_dir / 'gene_embeddings.pkl'
    info_out = data_dir / 'preprocessing_info.pkl'

    if not args.force_preprocess and adata_out.exists() and embeddings_out.exists():
        print(f"Using existing preprocessed inputs in {data_dir}")
        return data_dir

    if args.adata_path is None:
        raise ValueError("--adata_path is required when preprocessed files are missing or --force_preprocess is set")
    if args.gene_embeddings_path is None:
        raise ValueError("--gene_embeddings_path is required when preprocessed files are missing or --force_preprocess is set")

    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("scLAMBDA Data Preparation")
    print("=" * 60)

    print(f"\nLoading data from {args.adata_path}...")
    adata = sc.read_h5ad(args.adata_path)
    print(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")

    print("\n" + "-" * 40)
    adata = convert_condition_format(adata)

    conditions = adata.obs['condition'].unique()
    print(f"\nTotal unique conditions: {len(conditions)} (including ctrl)")
    print(f"\nUsing full dataset: {adata.n_obs} cells")

    if adata.X.max() > 100:
        print("\nData appears to be counts, applying log normalization...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        print("\nData appears to be already normalized (max value < 100)")

    if 'gene_name' not in adata.var.columns:
        adata.var['gene_name'] = adata.var_names

    if args.compute_de:
        print("\n" + "-" * 40)
        adata = compute_de_genes_fast(adata)

    gene_names = adata.var_names.tolist()

    print(f"\nLoading gene embeddings from {args.gene_embeddings_path}...")
    with open(args.gene_embeddings_path, 'rb') as f:
        gene_embeddings = pickle.load(f)

    covered = [g for g in gene_names if g in gene_embeddings]
    missing = [g for g in gene_names if g not in gene_embeddings]
    print(f"Gene embedding coverage: {len(covered)}/{len(gene_names)} ({100*len(covered)/len(gene_names):.1f}%)")

    if missing:
        print(f"Missing embeddings for {len(missing)} genes - creating deterministic placeholder embeddings")
        embedding_dim = len(next(iter(gene_embeddings.values())))
        np.random.seed(42)
        for gene in missing:
            gene_embeddings[gene] = np.random.randn(embedding_dim).astype(np.float32)

    print("\n" + "-" * 40)
    print(f"Saving prepared inputs to {data_dir}...")

    adata.write(adata_out)
    print(f"Saved adata to {adata_out}")

    with open(embeddings_out, 'wb') as f:
        pickle.dump(gene_embeddings, f)
    print(f"Saved gene embeddings to {embeddings_out}")

    condition_counts = adata.obs['condition'].value_counts()
    info = {
        'n_cells': adata.n_obs,
        'n_genes': adata.n_vars,
        'n_conditions': len(conditions),
        'conditions': list(conditions),
        'n_ctrl_cells': int(condition_counts.get('ctrl', 0)),
        'n_pert_cells': int(adata.n_obs - condition_counts.get('ctrl', 0)),
        'embedding_dim': len(next(iter(gene_embeddings.values()))),
        'embedding_coverage': len(covered) / len(gene_names),
        'source_file': args.adata_path,
        'preprocessing_format': 'NOCAP_512k_condition_labels'
    }
    with open(info_out, 'wb') as f:
        pickle.dump(info, f)
    print(f"Saved preprocessing info to {info_out}")

    print("\nDataset summary:")
    print(f"  - Total cells: {info['n_cells']:,}")
    print(f"  - Control cells: {info['n_ctrl_cells']:,}")
    print(f"  - Perturbed cells: {info['n_pert_cells']:,}")
    print(f"  - Genes: {info['n_genes']:,}")
    print(f"  - Perturbation conditions: {info['n_conditions'] - 1}")
    print(f"  - Embedding dim: {info['embedding_dim']}")
    print(f"  - Embedding coverage: {info['embedding_coverage']*100:.1f}%")

    return data_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare data and train scLAMBDA for perturbation prediction')
    parser.add_argument('--adata_path', type=str,
                        default='<DATA_ROOT>/05.Behchmarking_DL/data/Output_telenNeuron_pcgenes_normalized_250507.h5ad',
                        help='Path to NOCAP h5ad input file')
    parser.add_argument('--data_dir', type=str,
                        default='<DATA_ROOT>/05.Behchmarking_DL/scLAMBDA/data',
                        help='Directory for prepared scLAMBDA input files')
    parser.add_argument('--gene_embeddings_path', type=str,
                        default='<DATA_ROOT>/05.Behchmarking_DL/scLAMBDA/data/gene_embeddings.pkl',
                        help='Path to gene embeddings')
    parser.add_argument('--compute_de', action='store_true',
                        help='Compute DE genes during data preparation')
    parser.add_argument('--force_preprocess', action='store_true',
                        help='Regenerate prepared input files even if they already exist')
    parser.add_argument('--output_dir', type=str,
                        default='<DATA_ROOT>/05.Behchmarking_DL/scLAMBDA/results/NOCAP_benchmark',
                        help='Directory to save trained models and summaries')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                        help='Random seeds for training')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Training batch size')
    parser.add_argument('--latent_dim', type=int, default=30,
                        help='Latent dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension')
    parser.add_argument('--lambda_mi', type=float, default=200,
                        help='Mutual information regularization weight')
    parser.add_argument('--ctrl_size', type=int, default=None,
                        help='Number of control cells to use for validation (None=use all)')
    parser.add_argument('--train_gene_set_size', type=float, default=0.75,
                        help='Fraction of perturbations for training')
    parser.add_argument('--max_train_cells', type=int, default=None,
                        help='Maximum training cells (subsample if exceeded, for memory control)')
    return parser.parse_args()


def train_seed(adata, gene_embeddings, seed, args, output_dir):
    """Train scLAMBDA for one seed using the original Model.train() method."""

    print(f"\n{'='*60}")
    print(f"Training seed {seed}")
    print("=" * 60)

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Data split
    print(f"Splitting data with seed {seed}...")
    adata_split, split = data_split(
        adata.copy(),
        split_type='single',
        train_gene_set_size=args.train_gene_set_size,
        seed=seed
    )

    # Subsample training data if max_train_cells is specified
    if args.max_train_cells is not None:
        # Get training cells (non-test perturbations)
        train_perts = set(split['train'])
        train_mask = adata_split.obs['condition'].isin(train_perts) | (adata_split.obs['condition'] == 'ctrl')
        n_train = train_mask.sum()

        if n_train > args.max_train_cells:
            print(f"Subsampling training data: {n_train} -> {args.max_train_cells} cells")
            # Get indices of training cells
            train_idx = np.where(train_mask)[0]
            # Randomly select max_train_cells
            np.random.seed(seed)
            keep_idx = np.random.choice(train_idx, args.max_train_cells, replace=False)
            # Also keep all test cells
            test_idx = np.where(~train_mask)[0]
            all_keep_idx = np.concatenate([keep_idx, test_idx])
            all_keep_idx = np.sort(all_keep_idx)
            # Subsample adata
            adata_split = adata_split[all_keep_idx].copy()
            print(f"After subsampling: {adata_split.n_obs} total cells")

    # Create model directory
    seed_dir = output_dir / f'seed_{seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Build and train model using original scLAMBDA API
    print("Building scLAMBDA model...")
    model = Model(
        adata=adata_split,
        gene_emb=gene_embeddings,
        multi_gene=False,  # Single-gene perturbation
        training_epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        lambda_MI=args.lambda_mi,
        ctrl_size=args.ctrl_size,
        model_path=str(seed_dir),
        seed=seed
    )

    print(f"Model built with {model.x_dim} genes, {model.p_dim}-dim perturbation embeddings")
    print(f"Training cells: {model.adata_train.shape[0]}")
    print(f"Validation perturbations: {len(model.pert_val)}")

    # Train using original method
    print("\nStarting training...")
    start_time = datetime.now()
    model.train()
    elapsed = datetime.now() - start_time

    print(f"Training complete! Time: {elapsed}")

    # Save split info
    split_info = {
        'seed': seed,
        'train_perts': list(split['train']),
        'val_perts': list(split.get('val', [])),
        'test_perts': list(split['test_subgroup']['unseen_single']),
        'n_train_cells': model.adata_train.shape[0],
        'n_val_perts': len(model.pert_val),
        'training_time_minutes': elapsed.total_seconds() / 60,
        'completed_at': datetime.now().isoformat()
    }

    with open(seed_dir / 'training_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)

    # Clear GPU memory
    del model
    torch.cuda.empty_cache()

    return split_info


def main():
    args = parse_args()

    print("=" * 60)
    print("scLAMBDA Data Preparation and Training")
    print("=" * 60)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")

    data_dir = prepare_sclambda_inputs(args)

    print(f"\nLoading prepared data from {data_dir}...")
    adata = sc.read_h5ad(data_dir / 'adata_preprocessed.h5ad')
    print(f"Loaded adata: {adata.shape}")

    with open(data_dir / 'gene_embeddings.pkl', 'rb') as f:
        gene_embeddings = pickle.load(f)
    print(f"Loaded {len(gene_embeddings)} gene embeddings")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in args.seeds:
        seed_dir = output_dir / f'seed_{seed}'

        if (seed_dir / 'ckpt.pth').exists():
            print(f"\nSeed {seed} already trained (ckpt.pth exists), skipping...")
            if (seed_dir / 'training_info.json').exists():
                with open(seed_dir / 'training_info.json', 'r') as f:
                    info = json.load(f)
                results.append(info)
            continue

        try:
            info = train_seed(adata, gene_embeddings, seed, args, output_dir)
            results.append(info)
        except Exception as e:
            print(f"Error training seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if results:
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(output_dir / 'training_summary.csv', index=False)

        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        print(f"Successfully trained {len(results)} seeds")
        for r in results:
            print(f"  Seed {r['seed']}: {r['training_time_minutes']:.1f} min")


if __name__ == '__main__':
    main()
