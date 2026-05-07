#!/usr/bin/env python3
"""
03_prediction_GeneCompass.py - gene-by-gene perturbation prediction for GeneCompass + GEARS

Prediction utility for GeneCompass + GEARS outputs.

Matrix structure:
- Size: 19424 x 19424
- Rows = genes being knocked out (perturbed genes)
- Columns = response genes (all genes in vocabulary)
- Values = predicted expression changes after perturbation

Following BENCHMARKING_GUIDE.txt requirements.

Usage:
    python 03_prediction_GeneCompass.py --device cuda:0
"""

import os
import sys
import pickle
import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import anndata as ad
import torch
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm

# Add paths - use absolute paths
GENECOMPASS_DIR = Path("<DATA_ROOT>/05.Behchmarking_DL/GeneCompass")
GEARS_CODE_DIR = GENECOMPASS_DIR / "downstream_tasks" / "gears" / "gears_code"
sys.path.insert(0, str(GENECOMPASS_DIR))
sys.path.insert(0, str(GEARS_CODE_DIR))

# Setup logging
log_dir = GENECOMPASS_DIR / 'codes_GeneCompass' / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / f'03_prediction_GeneCompass_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class PredictionConfig:
    """Prediction configuration"""
    # Paths
    DATA_DIR = Path("<DATA_ROOT>/05.Behchmarking_DL/data")
    # Use 512k normalized data (same as training)
    NOCAP_PROCESSED_DIR = DATA_DIR / "NOCAP_Telen_512k_normalized"
    EMBEDDING_FILE = GENECOMPASS_DIR / "embeddings" / "nocap_prior_knowledge_64.pickle"
    # Use 512k benchmark results directory
    RESULTS_DIR = GENECOMPASS_DIR / "results" / "NOCAP_512k_benchmark"

    # Output path for standardized format
    OUTPUT_DIR = Path("<DATA_ROOT>/07.Benchmarking_Downstream_Analysis/data")

    # Prediction parameters
    BATCH_SIZE = 64
    POOL_SIZE = 300
    HIDDEN_SIZE = 64


# ============================================================================
# Custom Cell Graph Creation (without 'flag' column)
# ============================================================================

def create_cell_graph_for_prediction(X, pert_idx, pert_gene):
    """
    Create cell graph for prediction without requiring 'flag' column.

    Based on GEARS create_cell_graph_for_prediction:
    return Data(x=torch.Tensor(X).T, pert_idx=pert_idx, pert=pert_gene, emb=emb)

    Args:
        X: Gene expression vector (1D array, shape: n_genes)
        pert_idx: List of perturbation gene indices
        pert_gene: List of perturbation gene names

    Returns:
        PyG Data object
    """
    # Flatten X if needed
    if len(X.shape) > 1:
        X = X.flatten()

    # Create Data object matching GEARS format
    # x should be (n_genes, 1) - expression values for each gene
    data = Data(
        x=torch.Tensor(X).unsqueeze(1),  # Shape: (n_genes, 1)
        pert_idx=pert_idx,  # Keep as list, not tensor
        pert='+'.join(pert_gene) + '+ctrl' if pert_gene else 'ctrl',
        emb=0  # Dummy value - not used when use_simple_embedding=True
    )

    return data


def create_base_cell_graphs(ctrl_adata, device, num_samples=300, seed=42):
    """
    Pre-create base cell graphs from control cells (without perturbation info).
    These can be reused for all gene perturbations by just changing pert_idx.

    Args:
        ctrl_adata: Control adata
        device: Torch device
        num_samples: Number of control cells to sample
        seed: Random seed for reproducibility

    Returns:
        List of base PyG Data objects on device, sampled indices
    """
    np.random.seed(seed)

    # Sample control cells
    n_cells = len(ctrl_adata)
    if n_cells > num_samples:
        rand_idx = np.random.choice(n_cells, num_samples, replace=False)
    else:
        rand_idx = np.arange(n_cells)

    # Get expression matrix
    if hasattr(ctrl_adata.X, 'toarray'):
        X_matrix = ctrl_adata.X[rand_idx].toarray()
    else:
        X_matrix = ctrl_adata.X[rand_idx]

    # Create base cell graphs (with placeholder pert_idx)
    base_graphs = []
    for i in range(len(rand_idx)):
        X = X_matrix[i]
        if len(X.shape) > 1:
            X = X.flatten()

        data = Data(
            x=torch.Tensor(X).unsqueeze(1),  # Shape: (n_genes, 1)
            pert_idx=[0],  # Placeholder - will be updated per gene
            pert='placeholder+ctrl',  # Placeholder
            emb=0
        )
        data = data.to(device)
        base_graphs.append(data)

    logger.info(f"Created {len(base_graphs)} base cell graphs")
    return base_graphs, rand_idx


def update_graphs_for_perturbation(base_graphs, pert_idx, pert_gene):
    """
    Update base graphs with new perturbation info (in-place modification).

    Args:
        base_graphs: List of base PyG Data objects
        pert_idx: Perturbation gene index (int)
        pert_gene: Perturbation gene name (str)
    """
    pert_name = f'{pert_gene}+ctrl'
    for graph in base_graphs:
        graph.pert_idx = [pert_idx]
        graph.pert = pert_name


def create_cell_graphs_for_prediction(pert_gene, ctrl_adata, gene_names, device, num_samples=300):
    """
    Create cell graphs for prediction without requiring 'flag' column.

    Args:
        pert_gene: Gene to perturb (string or list)
        ctrl_adata: Control adata
        gene_names: List of gene names
        device: Torch device
        num_samples: Number of control cells to sample

    Returns:
        List of PyG Data objects
    """
    # Ensure pert_gene is a list
    if isinstance(pert_gene, str):
        pert_gene = [pert_gene]

    # Get perturbation indices
    gene_names_array = np.array(gene_names)
    pert_idx = []
    for p in pert_gene:
        matches = np.where(gene_names_array == p)[0]
        if len(matches) > 0:
            pert_idx.append(matches[0])
        else:
            logger.warning(f"Gene {p} not found in gene list")
            return []

    # Sample control cells
    n_cells = len(ctrl_adata)
    if n_cells > num_samples:
        rand_idx = np.random.choice(n_cells, num_samples, replace=False)
    else:
        rand_idx = np.arange(n_cells)

    # Get expression matrix
    if hasattr(ctrl_adata.X, 'toarray'):
        X_matrix = ctrl_adata.X[rand_idx].toarray()
    else:
        X_matrix = ctrl_adata.X[rand_idx]

    # Create cell graphs
    cell_graphs = []
    for i in range(len(rand_idx)):
        data = create_cell_graph_for_prediction(X_matrix[i], pert_idx, pert_gene)
        data = data.to(device)
        cell_graphs.append(data)

    return cell_graphs


# ============================================================================
# Model Loading and Prediction
# ============================================================================

def get_best_seed():
    """Get the best seed from evaluation results"""
    selection_file = PredictionConfig.RESULTS_DIR / "best_model_selection.json"
    if selection_file.exists():
        with open(selection_file, 'r') as f:
            selection = json.load(f)
        return selection['best_seed']

    logger.warning("No evaluation results found, using seed 1")
    return 1


def load_model_and_data(device='cuda'):
    """Load the best trained model and data"""
    best_seed = get_best_seed()
    logger.info(f"Using best seed: {best_seed}")

    result_dir = PredictionConfig.RESULTS_DIR / f"seed_{best_seed}"

    # Find model file
    model_path = None
    for name in ["model.pt", "ckpt.pth", "best_model.pt"]:
        if (result_dir / name).exists():
            model_path = result_dir / name
            break

    if model_path is None:
        raise FileNotFoundError(f"No model found at {result_dir}")

    # Load data
    data_path = PredictionConfig.NOCAP_PROCESSED_DIR

    try:
        from geares import PertData, GEARS
    except ImportError:
        from gears import PertData, GEARS

    pert_data = PertData(str(data_path))
    pert_data.load(data_path=str(data_path))
    pert_data.prepare_split(split='simulation', seed=best_seed)
    pert_data.get_dataloader(batch_size=PredictionConfig.BATCH_SIZE,
                             test_batch_size=PredictionConfig.BATCH_SIZE)

    # Initialize model
    gears_model = GEARS(pert_data, device=device)
    gears_model.model_initialize(
        damoxing_type='prior_knowledge',
        hidden_size=PredictionConfig.HIDDEN_SIZE,
        embedding_file=str(PredictionConfig.EMBEDDING_FILE)
    )

    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        gears_model.model.load_state_dict(checkpoint['model_state_dict'])
    else:
        gears_model.model.load_state_dict(checkpoint)
    gears_model.model.eval()

    logger.info(f"Loaded model from {model_path}")

    return gears_model, pert_data, best_seed


def predict_single_gene(model, pert_data, gene_name, device='cuda', pool_size=300):
    """
    Predict expression changes for a single gene perturbation.
    Uses the model directly with properly constructed input.

    Args:
        model: GEARS model (not gears_model wrapper)
        pert_data: PertData object
        gene_name: Gene to perturb
        device: Device
        pool_size: Number of control cells

    Returns:
        Mean predicted expression (n_genes,)
    """
    # Get control cells
    ctrl_mask = pert_data.adata.obs['condition'] == 'ctrl'
    ctrl_adata = pert_data.adata[ctrl_mask]
    gene_list = list(pert_data.adata.var_names)

    # Check gene exists
    if gene_name not in gene_list:
        return None

    # Create cell graphs
    cell_graphs = create_cell_graphs_for_prediction(
        gene_name, ctrl_adata, gene_list, device, num_samples=pool_size
    )

    if not cell_graphs:
        return None

    # Run inference
    loader = DataLoader(cell_graphs, batch_size=PredictionConfig.BATCH_SIZE, shuffle=False)

    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            try:
                output = model(batch)
                if isinstance(output, dict):
                    pred = output['pred']
                else:
                    pred = output
                predictions.append(pred.cpu().numpy())
            except Exception as e:
                logger.warning(f"Error during forward pass for {gene_name}: {e}")
                return None

    if predictions:
        predictions = np.vstack(predictions)
        return np.mean(predictions, axis=0)

    return None


def predict_with_base_graphs(model, base_graphs, pert_idx, pert_gene, device='cuda'):
    """
    Predict expression using pre-created base graphs (OPTIMIZED).
    Only updates pert_idx and pert name, avoiding graph recreation.

    Args:
        model: GEARS model
        base_graphs: Pre-created base cell graphs
        pert_idx: Perturbation gene index
        pert_gene: Perturbation gene name
        device: Device

    Returns:
        Mean predicted expression (n_genes,)
    """
    # Update graphs with new perturbation info (in-place)
    update_graphs_for_perturbation(base_graphs, pert_idx, pert_gene)

    # Run inference
    loader = DataLoader(base_graphs, batch_size=PredictionConfig.BATCH_SIZE, shuffle=False)

    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in loader:
            try:
                output = model(batch)
                if isinstance(output, dict):
                    pred = output['pred']
                else:
                    pred = output
                predictions.append(pred.cpu().numpy())
            except Exception as e:
                logger.warning(f"Error during forward pass for {pert_gene}: {e}")
                return None

    if predictions:
        predictions = np.vstack(predictions)
        return np.mean(predictions, axis=0)

    return None


def generate_gxg_matrix_v2(gears_model, pert_data, device='cuda'):
    """
    Generate the complete GxG prediction matrix.
    """
    gene_list = list(pert_data.adata.var_names)
    n_genes = len(gene_list)

    logger.info(f"Generating GxG matrix for {n_genes} genes...")

    # Initialize matrix
    gxg_matrix = np.zeros((n_genes, n_genes), dtype=np.float32)

    # Get model
    model = gears_model.model

    # Get control expression for computing delta
    ctrl_mask = pert_data.adata.obs['condition'] == 'ctrl'
    ctrl_adata = pert_data.adata[ctrl_mask]
    if hasattr(ctrl_adata.X, 'toarray'):
        ctrl_mean = ctrl_adata.X.toarray().mean(axis=0)
    else:
        ctrl_mean = ctrl_adata.X.mean(axis=0)

    # Predict each gene
    successful = 0
    failed = 0

    for i, gene_name in enumerate(tqdm(gene_list, desc="Predicting genes")):
        try:
            pred = predict_single_gene(
                model, pert_data, gene_name, device,
                pool_size=PredictionConfig.POOL_SIZE
            )

            if pred is not None:
                # Store raw predicted expression (NOT delta)
                gxg_matrix[i, :] = pred
                successful += 1
            else:
                # For failed predictions, use control mean as fallback
                gxg_matrix[i, :] = ctrl_mean
                failed += 1

        except Exception as e:
            logger.error(f"Error for gene {gene_name}: {e}")
            gxg_matrix[i, :] = ctrl_mean  # Fallback to control mean
            failed += 1

        # Periodic cleanup
        if (i + 1) % 100 == 0:
            torch.cuda.empty_cache()
            logger.info(f"Progress: {i+1}/{n_genes} ({successful} successful, {failed} failed)")

    logger.info(f"Prediction complete: {successful} successful, {failed} failed")

    return gxg_matrix, gene_list


def generate_gxg_matrix_partial(gears_model, pert_data, gene_start, gene_end, device='cuda'):
    """
    Generate a partial GxG prediction matrix for a range of genes.
    OPTIMIZED: Pre-creates base cell graphs once and reuses them.
    """
    gene_list = list(pert_data.adata.var_names)
    n_genes = len(gene_list)

    # Get range
    genes_to_process = gene_list[gene_start:gene_end]
    n_to_process = len(genes_to_process)

    logger.info(f"Generating partial GxG matrix: genes {gene_start} to {gene_end} ({n_to_process} genes)")

    # Initialize partial matrix (rows = genes in range, cols = all genes)
    partial_matrix = np.zeros((n_to_process, n_genes), dtype=np.float32)

    # Get model
    model = gears_model.model

    # Get control expression for fallback
    ctrl_mask = pert_data.adata.obs['condition'] == 'ctrl'
    ctrl_adata = pert_data.adata[ctrl_mask]
    if hasattr(ctrl_adata.X, 'toarray'):
        ctrl_mean = ctrl_adata.X.toarray().mean(axis=0)
    else:
        ctrl_mean = ctrl_adata.X.mean(axis=0)

    # OPTIMIZATION: Pre-create base cell graphs ONCE
    logger.info("Pre-creating base cell graphs (one-time operation)...")
    base_graphs, _ = create_base_cell_graphs(
        ctrl_adata, device,
        num_samples=PredictionConfig.POOL_SIZE,
        seed=42  # Fixed seed for reproducibility
    )

    # Build gene name to index mapping
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    # Predict each gene in range using optimized approach
    successful = 0
    failed = 0

    for i, gene_name in enumerate(tqdm(genes_to_process, desc=f"Predicting genes {gene_start}-{gene_end}")):
        try:
            # Get gene index
            if gene_name not in gene_to_idx:
                logger.warning(f"Gene {gene_name} not found")
                partial_matrix[i, :] = ctrl_mean
                failed += 1
                continue

            pert_idx = gene_to_idx[gene_name]

            # Use optimized prediction with pre-created graphs
            pred = predict_with_base_graphs(
                model, base_graphs, pert_idx, gene_name, device
            )

            if pred is not None:
                # Store raw predicted expression (NOT delta)
                partial_matrix[i, :] = pred
                successful += 1
            else:
                # Fallback to control mean
                partial_matrix[i, :] = ctrl_mean
                failed += 1

        except Exception as e:
            logger.error(f"Error for gene {gene_name}: {e}")
            partial_matrix[i, :] = ctrl_mean  # Fallback to control mean
            failed += 1

        # Periodic cleanup
        if (i + 1) % 1000 == 0:
            torch.cuda.empty_cache()
            logger.info(f"Progress: {i+1}/{n_to_process} ({successful} successful, {failed} failed)")

    logger.info(f"Partial prediction complete: {successful} successful, {failed} failed")

    return partial_matrix, genes_to_process


def combine_partial_results(gene_list):
    """
    Combine partial GxG matrices from multiple GPU runs into a single matrix.
    """
    n_genes = len(gene_list)
    gxg_matrix = np.zeros((n_genes, n_genes), dtype=np.float32)

    # Find all partial files
    partial_files = list(PredictionConfig.RESULTS_DIR.glob("gxg_partial_*.pkl"))
    logger.info(f"Found {len(partial_files)} partial result files")

    for partial_file in partial_files:
        with open(partial_file, 'rb') as f:
            data = pickle.load(f)

        gene_start = data['gene_start']
        gene_end = data['gene_end']
        partial_matrix = data['matrix']

        logger.info(f"Loading partial result: genes {gene_start} to {gene_end}")
        gxg_matrix[gene_start:gene_end, :] = partial_matrix

    return gxg_matrix


def save_standardized_output(gxg_matrix, gene_list, best_seed):
    """
    Save output in standardized format per BENCHMARKING_GUIDE.txt
    """
    logger.info("Saving in standardized BENCHMARKING_GUIDE format...")

    # Create DataFrame with proper format
    # Rows = Response genes, Columns = Perturbed genes
    df = pd.DataFrame(
        gxg_matrix.T,  # Transpose
        index=gene_list,
        columns=[f'{g}_perturbed' for g in gene_list]
    )

    # Save to standardized location
    output_file = PredictionConfig.OUTPUT_DIR / "Telen-GeneCompass.pickle"
    PredictionConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(df, f)

    logger.info(f"Standardized output saved to: {output_file}")
    logger.info(f"  Shape: {df.shape}")

    # Save metadata
    metadata = {
        'model': 'Telen-GeneCompass',
        'best_seed': best_seed,
        'n_genes': len(gene_list),
        'shape': list(df.shape)
    }
    with open(PredictionConfig.OUTPUT_DIR / "Telen-GeneCompass_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return output_file


def main():
    parser = argparse.ArgumentParser(description="GxG Matrix Generation for GeneCompass")
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--gene_start', type=int, default=0, help='Start gene index')
    parser.add_argument('--gene_end', type=int, default=None, help='End gene index')
    parser.add_argument('--combine_only', action='store_true', help='Only combine existing partial results')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("GxG Matrix Generation for GeneCompass + GEARS")
    logger.info("=" * 60)
    logger.info(f"Device: {args.device}")
    logger.info(f"Gene range: {args.gene_start} to {args.gene_end}")

    # Load model and data
    gears_model, pert_data, best_seed = load_model_and_data(args.device)
    gene_list = list(pert_data.adata.var_names)
    n_genes = len(gene_list)

    if args.combine_only:
        # Combine partial results
        logger.info("Combining partial results...")
        gxg_matrix = combine_partial_results(gene_list)
        save_standardized_output(gxg_matrix, gene_list, best_seed)
        return

    # Determine gene range
    gene_start = args.gene_start
    gene_end = args.gene_end if args.gene_end is not None else n_genes

    # Generate partial matrix
    partial_matrix, processed_genes = generate_gxg_matrix_partial(
        gears_model, pert_data, gene_start, gene_end, args.device
    )

    # Save partial result
    partial_path = PredictionConfig.RESULTS_DIR / f"gxg_partial_{gene_start}_{gene_end}.pkl"
    with open(partial_path, 'wb') as f:
        pickle.dump({
            'matrix': partial_matrix,
            'gene_start': gene_start,
            'gene_end': gene_end,
            'gene_list': gene_list,
            'best_seed': best_seed
        }, f)
    logger.info(f"Partial matrix saved to: {partial_path}")

    # If this was a full run, save standardized output
    if gene_start == 0 and gene_end == n_genes:
        save_standardized_output(partial_matrix, gene_list, best_seed)

        # Also save raw matrix
        raw_path = PredictionConfig.RESULTS_DIR / "gxg_prediction_matrix.pkl"
        with open(raw_path, 'wb') as f:
            pickle.dump({
                'matrix': partial_matrix,
                'gene_list': gene_list,
                'best_seed': best_seed
            }, f)
        logger.info(f"Raw matrix saved to: {raw_path}")

    logger.info("=" * 60)
    logger.info("Prediction complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
