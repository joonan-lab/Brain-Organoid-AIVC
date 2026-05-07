#!/usr/bin/env python
"""
03_prediction_scLAMBDA.py - Full prediction matrix generation and output formatting for scLAMBDA

This script:
1. Loads the best trained model (by pearson_de_delta)
2. Generates predictions for all genes as perturbation targets
3. Creates the full prediction matrix
4. Saves the benchmarking-standardized Telen-scLAMBDA.pickle output

Per BENCHMARKING_GUIDE.txt:
- Output: Telen-scLAMBDA.pickle (GxG matrix as pandas DataFrame)
- Values: RAW predicted expression (NOT delta)
- Index: gene names (response genes)
- Columns: '{gene}_perturbed' format

Usage:
    python 03_prediction_scLAMBDA.py --data_dir ./data --model_dir ./results/NOCAP_benchmark --output_dir ./predictions
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

import torch

# Add scLAMBDA to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from sclambda.model import Model
from sclambda.networks import Net
from sclambda.utils import data_split


def parse_args():
    parser = argparse.ArgumentParser(description='Generate full prediction matrix with scLAMBDA')
    parser.add_argument('--data_dir', type=str,
                        default='<DATA_ROOT>/05.Behchmarking_DL/scLAMBDA/data',
                        help='Directory with preprocessed data')
    parser.add_argument('--model_dir', type=str,
                        default='<DATA_ROOT>/05.Behchmarking_DL/scLAMBDA/results/NOCAP_benchmark',
                        help='Directory with trained models')
    parser.add_argument('--output_dir', type=str,
                        default='<DATA_ROOT>/05.Behchmarking_DL/scLAMBDA/results/NOCAP_benchmark/predictions',
                        help='Directory to save predictions')
    parser.add_argument('--best_seed', type=int, default=None,
                        help='Seed to use (default: auto-detect from best_seed_summary.json)')
    parser.add_argument('--ctrl_size', type=int, default=500,
                        help='Number of control cells for prediction (smaller for speed)')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of genes to predict at once (for memory efficiency)')
    parser.add_argument('--gene_start', type=int, default=0,
                        help='Start index for gene predictions (for parallel runs)')
    parser.add_argument('--gene_end', type=int, default=None,
                        help='End index for gene predictions (for parallel runs)')
    parser.add_argument('--save_format', type=str, default='both',
                        choices=['pkl', 'npy', 'both'],
                        help='Intermediate prediction matrix format')
    parser.add_argument('--standardized_output', type=str, default=None,
                        help='Path for benchmarking-standardized output (default: output_dir/Telen-scLAMBDA.pickle)')
    parser.add_argument('--verify_standardized', action='store_true',
                        help='Reload and verify the standardized DataFrame after saving')
    return parser.parse_args()


def predict_perturbation(model, gene_name, gene_embeddings, device):
    """
    Predict expression response for a single gene perturbation.

    Args:
        model: scLAMBDA Model object
        gene_name: Name of gene to perturb
        gene_embeddings: Gene embedding dictionary
        device: Torch device

    Returns:
        Predicted RAW expression (n_genes,) or None if gene not in embeddings
        Note: Returns raw expression (not delta) per BENCHMARKING_GUIDE.txt
    """
    if gene_name not in gene_embeddings:
        return None

    model.Net.eval()

    with torch.no_grad():
        # Get perturbation embedding
        pert_emb = gene_embeddings[gene_name]
        ctrl_x = model.ctrl_x

        # Create perturbation tensor
        val_p = torch.from_numpy(
            np.tile(pert_emb, (ctrl_x.shape[0], 1))
        ).float().to(device)

        # Forward pass
        # Note: model operates in delta space (ctrl_mean subtracted during init)
        x_hat, p_hat, mean_z, log_var_z, s = model.Net(ctrl_x, val_p)

        # Get mean prediction and convert back to raw expression
        # x_hat is in delta space, add ctrl_mean to get raw expression
        pred_delta = x_hat.mean(dim=0).cpu().numpy()
        ctrl_mean = np.array(model.ctrl_mean).flatten()
        pred_raw = pred_delta + ctrl_mean

    return pred_raw


def save_standardized_prediction(prediction_matrix, genes, output_path, verify=False):
    """Save prediction matrix as a benchmarking-standardized pandas DataFrame."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = prediction_matrix.shape
    if n_rows != len(genes) or n_cols != len(genes):
        raise ValueError(f"Full standardized output requires a square matrix matching {len(genes)} genes; got {prediction_matrix.shape}")

    columns = [f"{g}_perturbed" for g in genes]
    df_standardized = pd.DataFrame(
        prediction_matrix.T,
        index=genes,
        columns=columns,
        dtype=np.float32
    )

    with open(output_path, 'wb') as f:
        pickle.dump(df_standardized, f)

    if verify:
        with open(output_path, 'rb') as f:
            df_check = pickle.load(f)
        assert isinstance(df_check, pd.DataFrame), f"Expected DataFrame, got {type(df_check)}"
        assert df_check.shape == (len(genes), len(genes)), f"Shape mismatch: {df_check.shape}"
        assert list(df_check.index) == list(genes), "Index mismatch"
        assert list(df_check.columns) == columns, "Column mismatch"
        print("Standardized output verification PASSED")

    return df_standardized



def main():
    args = parse_args()

    print("=" * 60)
    print("scLAMBDA Full Prediction Matrix Generation")
    print("=" * 60)
    print(f"Start time: {datetime.now().isoformat()}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load preprocessed data
    data_dir = Path(args.data_dir)
    print(f"\nLoading data from {data_dir}...")

    adata = sc.read_h5ad(data_dir / 'adata_preprocessed.h5ad')
    print(f"Loaded adata: {adata.shape}")

    with open(data_dir / 'gene_embeddings.pkl', 'rb') as f:
        gene_embeddings = pickle.load(f)
    print(f"Loaded {len(gene_embeddings)} gene embeddings")

    # Get all gene names
    all_genes = adata.var_names.tolist()
    n_genes = len(all_genes)
    print(f"Total genes: {n_genes}")

    # Determine which seed to use
    model_dir = Path(args.model_dir)

    if args.best_seed is not None:
        best_seed = args.best_seed
    else:
        # Try to load from best_seed_summary.json
        best_seed_path = model_dir / 'best_seed_summary.json'
        if best_seed_path.exists():
            with open(best_seed_path, 'r') as f:
                best_info = json.load(f)
            best_seed = best_info['best_seed']
            print(f"Using best seed from evaluation: {best_seed}")
        else:
            # Default to seed 1
            best_seed = 1
            print(f"No best seed info found, using seed {best_seed}")

    # Load model
    seed_dir = model_dir / f'seed_{best_seed}'
    model_path = seed_dir / 'ckpt.pth'

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    print(f"\nLoading model from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Recreate data split with same seed
    adata_split, split = data_split(
        adata.copy(),
        split_type='single',
        train_gene_set_size=0.75,
        seed=best_seed
    )

    # Build model
    model = Model(
        adata=adata_split,
        gene_emb=gene_embeddings,
        multi_gene=False,
        training_epochs=1,
        ctrl_size=args.ctrl_size,
        model_path=str(seed_dir),
        seed=best_seed
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
    model.Net.eval()

    print(f"Model loaded successfully")
    print(f"Control cells for prediction: {model.ctrl_x.shape[0]}")

    # Determine gene range for prediction
    gene_start = args.gene_start
    gene_end = args.gene_end if args.gene_end is not None else n_genes

    genes_to_predict = all_genes[gene_start:gene_end]
    print(f"\nPredicting perturbations for genes {gene_start} to {gene_end}")
    print(f"Total predictions to make: {len(genes_to_predict)}")

    # Check embedding coverage
    genes_with_emb = [g for g in genes_to_predict if g in gene_embeddings]
    genes_without_emb = [g for g in genes_to_predict if g not in gene_embeddings]

    print(f"Genes with embeddings: {len(genes_with_emb)}/{len(genes_to_predict)}")
    if genes_without_emb:
        print(f"Warning: {len(genes_without_emb)} genes missing embeddings")

    # Initialize prediction matrix
    # Rows = perturbed genes (genes_to_predict)
    # Columns = response genes (all_genes)
    prediction_matrix = np.zeros((len(genes_to_predict), n_genes), dtype=np.float32)

    # Track which genes were successfully predicted
    predicted_genes = []
    failed_genes = []

    # Generate predictions
    print("\nGenerating predictions...")
    start_time = datetime.now()

    for i, gene in enumerate(tqdm(genes_to_predict, desc="Predicting")):
        pred_delta = predict_perturbation(model, gene, gene_embeddings, device)

        if pred_delta is not None:
            prediction_matrix[i, :] = pred_delta
            predicted_genes.append(gene)
        else:
            failed_genes.append(gene)

        # Periodic GPU memory cleanup
        if (i + 1) % 1000 == 0:
            torch.cuda.empty_cache()

    elapsed = datetime.now() - start_time
    print(f"\nPrediction complete! Time: {elapsed}")
    print(f"Successfully predicted: {len(predicted_genes)}/{len(genes_to_predict)}")
    if failed_genes:
        print(f"Failed predictions: {len(failed_genes)}")

    # Save prediction matrix
    print(f"\nSaving prediction matrix to {output_dir}...")

    # Create gene index mapping
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    perturbed_gene_to_row = {g: i for i, g in enumerate(genes_to_predict)}

    # Save metadata
    metadata = {
        'n_perturbed_genes': len(genes_to_predict),
        'n_response_genes': n_genes,
        'gene_start': gene_start,
        'gene_end': gene_end,
        'perturbed_genes': genes_to_predict,
        'response_genes': all_genes,
        'predicted_genes': predicted_genes,
        'failed_genes': failed_genes,
        'best_seed': best_seed,
        'ctrl_size': args.ctrl_size,
        'prediction_time': str(elapsed),
        'created_at': datetime.now().isoformat()
    }

    # Save based on format
    if args.save_format in ['pkl', 'both']:
        # Pickle format with metadata
        output_pkl = output_dir / f'prediction_matrix_{gene_start}_{gene_end}.pkl'
        with open(output_pkl, 'wb') as f:
            pickle.dump({
                'matrix': prediction_matrix,
                'metadata': metadata
            }, f)
        print(f"Saved pickle: {output_pkl}")

    if args.save_format in ['npy', 'both']:
        # NumPy format
        output_npy = output_dir / f'prediction_matrix_{gene_start}_{gene_end}.npy'
        np.save(output_npy, prediction_matrix)
        print(f"Saved numpy: {output_npy}")

        # Also save metadata as JSON
        metadata_json = output_dir / f'prediction_metadata_{gene_start}_{gene_end}.json'
        # Convert lists for JSON
        metadata_for_json = metadata.copy()
        with open(metadata_json, 'w') as f:
            json.dump(metadata_for_json, f, indent=2, default=str)
        print(f"Saved metadata: {metadata_json}")

    # If this is a full prediction (all genes), also save in standard format
    if gene_start == 0 and gene_end >= n_genes:
        print("\nSaving full prediction matrix in standard format...")

        # Standard pickle format (internal format with metadata)
        standard_pkl = output_dir / 'prediction_matrix.pkl'
        with open(standard_pkl, 'wb') as f:
            pickle.dump({
                'matrix': prediction_matrix,
                'genes': all_genes,
                'metadata': metadata
            }, f)
        print(f"Saved standard format: {standard_pkl}")

        # Also save as numpy with gene list
        np.savez(
            output_dir / 'prediction_matrix.npz',
            matrix=prediction_matrix,
            genes=np.array(all_genes)
        )
        print(f"Saved npz format: {output_dir / 'prediction_matrix.npz'}")

        print("\nSaving benchmarking-standardized format...")
        standardized_path = Path(args.standardized_output) if args.standardized_output else output_dir / 'Telen-scLAMBDA.pickle'
        df_standardized = save_standardized_prediction(
            prediction_matrix,
            all_genes,
            standardized_path,
            verify=args.verify_standardized
        )
        print(f"Saved benchmarking format: {standardized_path}")
        print(f"  Shape: {df_standardized.shape}")
        print(f"  Index (genes): {df_standardized.index[:3].tolist()}...")
        print(f"  Columns: {df_standardized.columns[:3].tolist()}...")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Matrix shape: {prediction_matrix.shape}")
    print(f"Matrix size: {prediction_matrix.nbytes / 1e9:.2f} GB")
    print(f"Non-zero entries: {np.count_nonzero(prediction_matrix)}")
    print(f"Mean absolute value: {np.abs(prediction_matrix).mean():.4f}")
    print(f"Max absolute value: {np.abs(prediction_matrix).max():.4f}")


if __name__ == '__main__':
    main()
