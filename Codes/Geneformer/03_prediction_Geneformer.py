### Last update: 2025. 12. 20 by IGK

# ==========================================
# Geneformer Prediction (unseen generation)
# Generate in silico perturbation predictions
# ==========================================

import argparse
import os
import sys
import shutil
import pickle
import json
import logging
import time
import tempfile
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import scanpy as sc
import gc
import random

# GEARS
from gears import PertData

# Geneformer
from geneformer import TranscriptomeTokenizer, InSilicoPerturber
from geneformer import perturber_utils as pu
from geneformer.emb_extractor import get_embs

warnings.filterwarnings("ignore")


# =======================
# Utils & Logger
# =======================
def set_seed(seed):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_dir: Path, prefix: str, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("Geneformer_pred")
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
    
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{prefix}_prediction.log"
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# =======================
# MLP Regression Head (Decoder)
# =======================
class GeneformerRegressionHead(nn.Module):
    """
    MLP regression head for mapping embeddings to gene expression.
    Strategy: Linear -> LayerNorm -> ReLU -> Dropout -> Linear
    """
    def __init__(self, in_dim, out_dim, hidden_dim=2048, dropout_rate=0.3):
        super().__init__()
        
        dense_dim = hidden_dim if hidden_dim else in_dim // 2
        
        self.net = nn.Sequential(
            # 1. Dimension reduction
            nn.Linear(in_dim, dense_dim),
            # 2. LayerNorm
            nn.LayerNorm(dense_dim),
            # 3. Activation
            nn.ReLU(),
            # 4. Regularization
            nn.Dropout(dropout_rate),
            # 5. Output Projection
            nn.Linear(dense_dim, out_dim),
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, in_dim) - condition embeddings
        Returns:
            (batch_size, out_dim) - predicted gene expression
        """
        return self.net(x)


# =======================
# Helper Functions
# =======================
def tokenize_adata(adata, output_dir=None, n_jobs=1):
    """Tokenize AnnData for Geneformer"""
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as adata_dir:
        adata.write_h5ad(os.path.join(adata_dir, 'adata.h5ad'))
        tk = TranscriptomeTokenizer({"condition": "condition"}, nproc=n_jobs)
        tk.tokenize_data(
            data_directory=adata_dir,
            output_directory=str(output_dir),
            output_prefix="tokenized",
            file_format="h5ad"
        )
    return f"{output_dir}/tokenized.dataset"

def get_perturbed_embedding(isp, model, input_data):
    """Apply in silico perturbation and extract CLS embeddings."""
    isp.max_len = pu.get_model_input_size(model)
    layer_to_quant = pu.quant_layers(model) + isp.emb_layer
    
    def make_group_perturbation_batch(example):
        example_input_ids = example["input_ids"]
        example["tokens_to_perturb"] = isp.tokens_to_perturb
        indices_to_perturb = [
            example_input_ids.index(token) if token in example_input_ids else None
            for token in isp.tokens_to_perturb
        ]
        indices_to_perturb = [i for i in indices_to_perturb if i is not None]
        
        if len(indices_to_perturb) > 0:
            example["perturb_index"] = indices_to_perturb
        else:
            example["perturb_index"] = [-100]
            
        if isp.perturb_type == "delete":
            example = pu.delete_indices(example)
        elif isp.perturb_type == "overexpress":
            example = pu.overexpress_tokens(example, isp.max_len, isp.special_token)
            example["n_overflow"] = pu.calc_n_overflow(
                isp.max_len,
                example["length"], 
                isp.tokens_to_perturb, 
                indices_to_perturb,
            )
        return example
    
    perturbed_data = input_data.map(make_group_perturbation_batch, num_proc=isp.nproc)
    
    if isp.perturb_type == "overexpress":
        input_data = input_data.add_column("n_overflow", perturbed_data["n_overflow"])
        input_data = input_data.map(pu.truncate_by_n_overflow_special, num_proc=isp.nproc)
    
    perturbation_cls_emb = get_embs(
        model,
        perturbed_data,
        "cls",
        layer_to_quant,
        isp.pad_token_id,
        isp.forward_batch_size,
        token_gene_dict=isp.token_gene_dict,
        summary_stat=None,
        silent=True,
    )
    return perturbation_cls_emb.cpu().detach().numpy()


# =======================
# Main Prediction Logic
# =======================
def main(args):
    total_start = time.time()
    
    # Output Directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Logger
    if args.gene_start is not None and args.gene_end is not None:
        prefix = f"gene_{args.gene_start}_{args.gene_end}"
    else:
        prefix = "Telen-Geneformer_all"
    
    log_dir = output_dir / 'logs'
    logger = setup_logger(log_dir, prefix, verbose=args.verbose)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Geneformer In Silico Prediction")
    logger.info(f"{'='*80}")
    logger.info(f"Arguments: {vars(args)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # -----------------------
    # 1. Load Data
    # -----------------------
    logger.info("[1/5] Loading PertData...")
    pert_data = PertData(args.data_dir, default_pert_graph=False)
    pert_data.load(data_path=os.path.join(args.data_dir, args.data_name))
    adata = pert_data.adata
    
    # Ensure n_counts is numeric
    if "n_counts" in adata.obs.columns:
        adata.obs["n_counts"] = pd.to_numeric(adata.obs["n_counts"], errors="coerce")
        
    gene_ids_dataset = adata.var_names.tolist() # Gene symbols
    
    # Load Gene Mapping (Symbol -> Ensembl)
    with open("/data1/Geneformer/geneformer/ensembl_mapping_dict_gc104M.pkl", "rb") as f:
        gene2ensembl = pickle.load(f)
    logger.info(f"  Loaded {len(gene2ensembl)} gene mappings")

    # -----------------------
    # 2. Prepare Control Pool
    # -----------------------
    logger.info(f"\n[2/5] Preparing control cell pool ({args.pool_size} cells)...")
    ctrl_adata_full = adata[adata.obs['condition'] == 'ctrl'].copy()
    n_ctrl_total = ctrl_adata_full.n_obs
    
    pool_size = min(args.pool_size, n_ctrl_total)
    set_seed(42)
    
    if n_ctrl_total > pool_size:
        idx = np.random.choice(n_ctrl_total, size=pool_size, replace=False)
        ctrl_adata = ctrl_adata_full[idx].copy()
    else:
        ctrl_adata = ctrl_adata_full
    
    logger.info(f"  Tokenizing {pool_size} control cells...")
    ctrl_tk_dir = output_dir / f'temp_{pool_size}_ctrl_tokens'
    
    if ctrl_tk_dir.exists():
        shutil.rmtree(ctrl_tk_dir)
        
    ctrl_tok_path = tokenize_adata(ctrl_adata, output_dir=ctrl_tk_dir, n_jobs=args.num_workers)
    ctrl_dataset = pu.load_and_filter(None, 1, ctrl_tok_path)
    logger.info(f"  Control tokens ready: {len(ctrl_dataset)} cells")

    # -----------------------
    # 3. Load Models
    # -----------------------
    logger.info(f"\n[3/5] Loading models (Seed {args.seed})...")
    seed_model_dir = Path(args.model_dir) / f"seed_{args.seed}"
    classifier_path = seed_model_dir / "classifier"
    regression_path = seed_model_dir / f"Seed_{args.seed}_best_model.pt"
    
    # A. Classifier
    try:
        # Load num_classes from pickle
        id_class_dict_path = seed_model_dir / "id_class_dict.pkl"
        with open(id_class_dict_path, "rb") as f:
            id_class_dict = pickle.load(f)
        num_classes = len(id_class_dict)
        
        geneformer_model = pu.load_model(
            "CellClassifier", 
            num_classes=num_classes, 
            model_directory=str(classifier_path),
            mode="eval"
        )
        geneformer_model.eval()
        geneformer_model.to(device)
        logger.info("  Classifier loaded")
    except Exception as e:
        logger.error(f"Failed to load Classifier: {e}")
        return

    # B. Regression Head
    try:
        reg_checkpoint = torch.load(regression_path, map_location=device, weights_only=False)
        reg_head = GeneformerRegressionHead(
            in_dim=reg_checkpoint['in_dim'],
            out_dim=reg_checkpoint['out_dim'],
            hidden_dim=2048, # Force 2048
            dropout_rate=0.3
        )
        reg_head.load_state_dict(reg_checkpoint['model_state_dict'])
        reg_head.to(device)
        reg_head.eval()
        logger.info("  Regression Head loaded")
    except Exception as e:
        logger.error(f"Failed to load Regression Head: {e}")
        return

    # -----------------------
    # 4. Prepare Gene List
    # -----------------------
    logger.info("\n[4/5] Preparing prediction targets...")
    
    # Determine which genes to predict
    if args.pred_gene_list:
        with open(args.pred_gene_list, "r") as f:
            target_genes = f.read().splitlines()
        target_genes = [g for g in target_genes if g in gene_ids_dataset]
    else:
        target_genes = gene_ids_dataset # Use all genes in dataset
        
    # Range Slicing
    if args.gene_start is not None or args.gene_end is not None:
        target_genes = target_genes[args.gene_start:args.gene_end]
        
    # Filter out already done
    saved_files = [f for f in os.listdir(output_dir) if f.endswith("_predicted.h5ad")]
    saved_genes = [f.replace("_predicted.h5ad", "") for f in saved_files]
    target_genes = sorted(list(set(target_genes) - set(saved_genes)))
    
    logger.info(f"  Target genes count: {len(target_genes)}")
    if len(target_genes) == 0:
        logger.info("  All genes predicted. Exiting.")
        return

    # -----------------------
    # 5. Prediction Loop
    # -----------------------
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting Prediction Loop")
    logger.info(f"{'='*80}\n")
    
    # Pre-calculate base isp object for efficiency
    isp_base = InSilicoPerturber(
        perturb_type=args.perturbation_type,
        model_type="CellClassifier",
        emb_mode="cls",
        max_ncells=args.pool_size,
        emb_layer=0,
        forward_batch_size=args.forward_batch_size,
        nproc=args.num_workers
    )
    # Apply additional filters once (e.g. length check)
    ctrl_dataset_filt = isp_base.apply_additional_filters(ctrl_dataset)
    
    for i, gene_name in enumerate(target_genes):
        start_t = time.time()
        
        # 1. Map to Ensembl
        #ensembl_id = gene2ensembl.get(gene_name)
        ensembl_id = adata.var.loc[adata.var['gene_name']==gene_name, 'ensembl_id'].values[0]
        
        # 2. Check validity
        if not ensembl_id:
            logger.warning(f"[{i+1}/{len(target_genes)}] Skipping {gene_name}: Ensembl ID not found")
            continue
            
        # 3. Create ISP for specific gene
        # Note: We reconstruct ISP because genes_to_perturb changes
        try:
            isp = InSilicoPerturber(
                perturb_type=args.perturbation_type,
                genes_to_perturb=[ensembl_id],
                model_type="CellClassifier",
                emb_mode="cls",
                max_ncells=args.max_ncells_ctrl,
                emb_layer=0,
                forward_batch_size=args.forward_batch_size,
                nproc=args.num_workers
            )
        except RuntimeError:
            # Geneformer 라이브러리 버그로 인해 RuntimeError가 발생할 수 있음
            # (토큰 사전에 없는 유전자일 경우)
            logger.warning(f"Skipping {gene_name}: Not found in model token dictionary.")
            continue
        except Exception as e:
            # 그 외 다른 에러가 나도 죽지 않고 넘어가도록 처리
            logger.warning(f"Skipping {gene_name}: Error during ISP initialization - {e}")
            continue
        
        # 4. Get Embedding (In silico perturbation)
        try:
            # ctrl_dataset_filt is already tokenized, just need to perturb
            pert_emb = get_perturbed_embedding(isp, 
                                               geneformer_model, 
                                               ctrl_dataset_filt
                                               )
            # 5. Regress to Gene Expression
            with torch.no_grad():
                inp = torch.tensor(pert_emb, dtype=torch.float32).to(device) # (100, 512)
                pred_expr = reg_head(inp).cpu().numpy() # (100, 19417)
                
            # 6. Save as AnnData
            adata_pred = sc.AnnData(pred_expr)
            adata_pred.obs_names = [f"Cell_{k}" for k in range(pred_expr.shape[0])]
            adata_pred.var_names = gene_ids_dataset # All 19417 genes
            adata_pred.obs["perturbation"] = gene_name
            adata_pred.obs["condition"] = gene_name # For consistency
            
            out_path = output_dir / f"{gene_name}_predicted.h5ad"
            adata_pred.write(out_path, compression="gzip")
            
            elapsed = time.time() - start_t
            logger.info(f"[{i+1}/{len(target_genes)}] {gene_name} predicted. Shape: {pred_expr.shape}. Time: {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"[{i+1}/{len(target_genes)}] Error predicting {gene_name}: {e}")
            continue
            
        # Cleanup per iteration to prevent OOM
        del isp
        gc.collect()

    logger.info(f"\nPrediction Complete. Total time: {(time.time() - total_start)/60:.2f} mins")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Geneformer Prediction Script")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to GEARS perturbation data directory')
    parser.add_argument('--data_name', type=str, required=True, help='Dataset name')
    parser.add_argument('--model_dir', type=str, required=True, help='Path containing "models" folder (where seed_* folders are)')
    parser.add_argument('--geneformer_model_dir', type=str, required=True, help='Pretrained Geneformer path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for h5ad files')
    parser.add_argument('--seed', type=int, default=1, help='Seed of the trained model to use')
    parser.add_argument('--pool_size', type=int, default=100, help='Number of control cells to use (default: 100)')
    parser.add_argument('--max_ncells_ctrl', type=int, default=100, help='Max ctrl cells for ISP')
    parser.add_argument('--perturbation_type', type=str, default='delete', choices=['delete', 'overexpress'])
    parser.add_argument('--forward_batch_size', type=int, default=96, help='Inference batch size')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pred_gene_list', type=str, default=None, help='Path to txt file with specific genes to predict')
    parser.add_argument('--gene_start', type=int, default=None, help='Start index for gene slicing')
    parser.add_argument('--gene_end', type=int, default=None, help='End index for gene slicing')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    main(args)