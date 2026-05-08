# Last update: 2025. 12. 17 by IGK

"""
Geneformer Fine-tuning Train Script
Train multiple models with different seeds (1-10)

Environment: t2 (Quadro RTX 8000 (48GB) x 2)

Pipeline:
1. Fine-tune Geneformer CellClassifier on perturbation data
2. Generate condition embeddings via in silico perturbation on control cells (Single-GPU for safety)
3. Train MLP regression head to map embeddings to mean expression
"""

# ========================
# Imports
# ========================
from pathlib import Path
import shutil
import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import json
import pickle
import tempfile
import argparse
from datetime import datetime
from collections import defaultdict
import copy
import logging
import time
import scanpy as sc
import os, gc
import random

# GEARS
import gears.version
assert gears.version.__version__ == '0.1.2'
from gears import PertData

# Geneformer
from geneformer import TranscriptomeTokenizer, Classifier, InSilicoPerturber
from geneformer import perturber_utils as pu
from geneformer.emb_extractor import get_embs


"""set random seed."""
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEFAULT_ENSEMBL_MAPPING_DICT = Path(__file__).resolve().parent / "resources" / "ensembl_mapping_dict_gc104M.pkl"

def resolve_ensembl_mapping_dict(args):
    mapping_path = getattr(args, "ensembl_mapping_dict", None) or os.environ.get("GENEFORMER_ENSEMBL_MAPPING_DICT") or DEFAULT_ENSEMBL_MAPPING_DICT
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Geneformer Ensembl mapping dictionary not found: {mapping_path}. "
            "Provide --ensembl_mapping_dict or set GENEFORMER_ENSEMBL_MAPPING_DICT."
        )
    return mapping_path


# =======================
# Logger
# =======================
def setup_logger(log_file_path: Path):
    logger = logging.getLogger("Geneformer_train_logger")

    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
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


def train_regression_head(
    X_train, Y_train, X_val, Y_val,
    logger,
    seed,
    n_epochs=15,
    lr=5e-5,
    batch_size=32,
    early_stop_patience=5,
    accum_steps=1,
    log_interval = 100,
    use_amp=True,
    device='cuda',
    num_workers=1,
    dropout_rate=0.3,
):
    """
    Train MLP regression head with validation and best checkpoint selection.
    
    Args:
        X_train: (n_train, hidden_dim) - training embeddings
        Y_train: (n_train, n_genes) - training expression
        X_val: (n_val, hidden_dim) - validation embeddings
        Y_val: (n_val, n_genes) - validation expression
        n_epochs: number of training epochs
        lr: learning rate
        batch_size: batch size
        early_stop_patience: early stopping patience
        device: 'cuda' or 'cpu'
    
    Returns:
        best_model: best model state dict
        best_val_loss: best validation loss
        training_history: dict with train/val losses
    """
    in_dim = X_train.shape[1]
    out_dim = Y_train.shape[1]
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32)
    
    # Create dataloaders
    logger.info("Creating DataLoaders for training and validation...")
    train_dataset = TensorDataset(X_train_t, Y_train_t)
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              num_workers=num_workers,
                              shuffle=True,
                              pin_memory=True,
                              prefetch_factor=2,
                              persistent_workers=True if num_workers > 0 else False
                              )
    
    val_dataset = TensorDataset(X_val_t, Y_val_t)
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            num_workers=num_workers,
                            shuffle=True,
                            pin_memory=True,
                            prefetch_factor=2,
                            persistent_workers=True if num_workers > 0 else False
                            )
    logger.info(f"Using default DataLoaders: train={len(train_loader)} batches, val={len(val_loader)} batches")
    
    # Initialize model
    logger.info("Initializing model...")
    model = GeneformerRegressionHead(in_dim=in_dim, 
                                     out_dim=out_dim, 
                                     hidden_dim=2048,
                                     dropout_rate=dropout_rate)
    model = model.to(device)
    
    # Optimizer and loss
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        verbose=True
    )
    
    amp = use_amp and (device == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    
    # Training
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    num_batches = len(train_loader)
    
    for epoch in range(n_epochs):
        ### Train ###
        model.train()
        
        train_loss_epoch = 0
        train_total_loss, train_total_mse = 0.0, 0.0
        epoch_start = time.time()        
        start_time = time.time()
        
        for step, batch_data in enumerate(train_loader):
            xb, yb = batch_data
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=amp, dtype=torch.bfloat16):
                pred = model(xb)
                loss = criterion(pred, yb)
                loss = loss / accum_steps
                
            # Backward with Scaler
            scaler.scale(loss).backward()
            
            # Gradient Clipping (Scaler 사용 시 unscale 먼저 해야 함)
            # Gradient Accumulation Logic
            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                    
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss_epoch += loss.item() * xb.size(0)
            
            train_total_loss += loss.item()
            train_total_mse += loss.item()
            
            if step % log_interval == 0 and step > 0:
                lr_curr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = train_total_loss / log_interval
                cur_mse = train_total_mse / log_interval
                logger.info(
                    f"| epoch {epoch:3d} | {step:4d}/{num_batches:4d} batches | "
                    f"lr {lr_curr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.4f} | mse {cur_mse:5.4f} | "
                    f"eff_bs {batch_size}"
                )
                train_total_loss = 0.0
                train_total_mse = 0.0
                start_time = time.time()
        
        train_loss_epoch /= len(train_dataset)
        train_losses.append(train_loss_epoch)
        
        ### Validate ###
        model.eval()
        val_loss_epoch = 0        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                
                with torch.cuda.amp.autocast(enabled=amp):
                    pred = model(xb)
                    loss = criterion(pred, yb)
                
                val_loss_epoch += loss.item() * xb.size(0)
        
        val_loss_epoch /= len(val_dataset)
        val_losses.append(val_loss_epoch)
        
        scheduler.step(val_loss_epoch)
        
        epoch_time = time.time() - epoch_start
        logger.info(
            f"| epoch {epoch:3d} (seed {seed}) | "
            f"time: {epoch_time:5.2f}s | val_loss: {val_loss_epoch:5.4f} |"
        )
        
        # Check if best model
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            logger.info(f"Best model updated!")
            logger.info(
                f"  Epoch {epoch+1}/{n_epochs} | "
                f"Train loss: {train_loss_epoch:.6f} | "
                f"Val loss: {val_loss_epoch:.6f} | "
                f"Collapsed time={epoch_time:.2f}"
            )
        else:
            patience_counter += 1        
            # Early stopping
            if early_stop_patience is not None:
                if patience_counter >= early_stop_patience:
                    logger.info(f"  Early stopping at epoch {epoch+1}")
                    break
            
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': np.argmin(val_losses) + 1,
    }
    
    return best_model_state, best_val_loss, training_history


# =======================
# Helper Functions
# =======================
def tokenize_adata(adata, output_dir = None, n_jobs=1):
    """Tokenize AnnData for Geneformer"""
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as adata_dir:
        adata.write_h5ad(os.path.join(adata_dir, 'adata.h5ad'))
        tk = TranscriptomeTokenizer({"condition": "condition"},
                                    nproc=n_jobs)
        tk.tokenize_data(
            data_directory=adata_dir,
            output_directory=str(output_dir),
            output_prefix="tokenized",
            file_format="h5ad"
        )
    return f"{output_dir}/tokenized.dataset"


def parse_condition_to_genes(condition_str, gene2ensembl):
    """
    Parse condition string into list of Ensembl IDs.
    
    Args:
        condition_str: e.g., "GENE1+GENE2+ctrl" or "ENSG00000123456+ctrl"
        gene2ensembl: dict mapping gene symbols to Ensembl IDs
    
    Returns:
        list of Ensembl IDs
    """
    gene_list = list(set(condition_str.split("+")) - set(["ctrl"]))
    
    # Convert to Ensembl IDs
    ensembl_genes = []
    for gene in gene_list:
        if gene.startswith("ENSG"):
            # Already Ensembl ID
            ensembl_genes.append(gene)
        elif gene in gene2ensembl:
            # Gene symbol -> Ensembl ID
            ensembl_genes.append(gene2ensembl[gene])
    
    return ensembl_genes


def validate_gene_mapping(conditions, gene2ensembl, logger):
    """
    Validate gene mapping before training.
    
    Args:
        conditions: list of condition strings
        gene2ensembl: dict mapping gene symbols to Ensembl IDs
        logger: Logger
    
    Returns:
        valid_conditions: list of valid conditions
        invalid_conditions: dict of {condition: reason}
    """
    valid_conditions = []
    invalid_conditions = {}
    
    logger.info("Validating gene mapping for all conditions...")
    
    for cond in conditions:
        if cond == "ctrl":
            valid_conditions.append(cond)
            continue
        
        genes = parse_condition_to_genes(cond, gene2ensembl)
        
        if len(genes) == 0:
            gene_list = list(set(cond.split("+")) - set(["ctrl"]))
            invalid_conditions[cond] = f"No mappable genes in {gene_list}"
        else:
            valid_conditions.append(cond)
    
    if invalid_conditions:
        logger.warning(f"Found {len(invalid_conditions)} unmappable conditions:")
        for cond, reason in list(invalid_conditions.items())[:5]:
            logger.warning(f"  - {cond}: {reason}")
        if len(invalid_conditions) > 5:
            logger.warning(f"  ... and {len(invalid_conditions) - 5} more")
    
    logger.info(f"Valid conditions: {len(valid_conditions)}/{len(conditions)}")
    
    return valid_conditions, invalid_conditions


def get_control_embedding(isp, model, ctrl_dataset, logger):
    """
    Extract control cell embeddings (to be cached and reused).
    
    Returns:
        np.ndarray: Shape (n_ctrl_cells, hidden_dim)
    """
    logger.info("  Extracting control cell embeddings...")
    
    isp.max_len = pu.get_model_input_size(model)
    layer_to_quant = pu.quant_layers(model) + isp.emb_layer
    
    # Get control embeddings without perturbation
    ctrl_cls_emb = get_embs(
        model,
        ctrl_dataset,
        "cls",
        layer_to_quant,
        isp.pad_token_id,
        isp.forward_batch_size,
        token_gene_dict=isp.token_gene_dict,
        summary_stat=None,
        silent=True,
    )
    
    ctrl_emb_array = ctrl_cls_emb.cpu().detach().numpy()
    logger.info(f"  Control embeddings shape: {ctrl_emb_array.shape}")
    
    return ctrl_emb_array


def get_perturbed_embedding(isp, model, input_data):
    """
    Apply in silico perturbation and extract CLS embeddings.
    
    Returns:
        np.ndarray: Shape (n_cells_used, hidden_dim)
    """
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


def mean_expr_for_condition(adata_subset, condition):
    """Mean normalized expression vector for a given condition"""
    sub = adata_subset[adata_subset.obs['condition'] == condition, :]
    return np.array(sub.X.mean(axis=0)).ravel()


def build_training_pairs(
    train_adata, 
    train_conditions,
    ctrl_tok_path,  
    geneformer_model, 
    perturbation_type, 
    max_ncells_ctrl,
    forward_batch_size,
    n_jobs,
    gene2ensembl,
    logger
):
    """
    Generate (embedding, mean expression) pairs for train conditions.
    
    Args:
        adata_subset: AnnData object
        train_conditions: list of condition strings
        ctrl_tok_path: path to tokenized control data
        geneformer_model: fine-tuned Geneformer model
        perturbation_type: 'delete' or 'overexpress'
        max_ncells_ctrl: max number of control cells
        gene2ensembl: dict mapping gene symbols to Ensembl IDs
    
    Returns:
        X: np.ndarray of shape (n_conditions, hidden_dim)
        Y: np.ndarray of shape (n_conditions, n_genes)
        used_conditions: list of conditions actually used
    """
    # Load and filter control dataset
    logger.info("Loading control dataset for ISP...")
    ctrl_dataset = pu.load_and_filter(None, 1, ctrl_tok_path)
    
    isp_base = InSilicoPerturber(perturb_type=perturbation_type,
                                 model_type="CellClassifier",
                                 emb_mode="cls",
                                 max_ncells=max_ncells_ctrl,
                                 emb_layer=0,
                                 forward_batch_size=forward_batch_size,
                                 nproc=n_jobs,
                                )
    ctrl_dataset = isp_base.apply_additional_filters(ctrl_dataset)
    logger.info(f"Filtered control dataset: {len(ctrl_dataset)} cells")
    
    # Get hidden_dim from model
    if hasattr(geneformer_model, 'config'):
        hidden_dim = geneformer_model.config.hidden_size
    else:
        # Default for Geneformer
        hidden_dim = 512
    
    # Caching control embeddings
    logger.info("Computing control embeddings...")
    ctrl_embeddings = get_control_embedding(isp_base, geneformer_model, ctrl_dataset, logger)
    ctrl_emb_mean = ctrl_embeddings.mean(axis=0)  # (hidden_dim,)
    logger.info(f"Control embedding mean shape: {ctrl_emb_mean.shape}")
    
    # Generate embeddings for each condition
    X_list = []
    Y_list = []
    used_conditions = []
    
    logger.info(f"Processing {len(train_conditions)} conditions...")
    
    for i, cond in enumerate(train_conditions):
        logger.info(f"  Progress: {i+1}/{len(train_conditions)} conditions")
            
        if cond == "ctrl":
            # Use actual control embedding mean (not zero)
            X_list.append(ctrl_emb_mean)
            Y_list.append(mean_expr_for_condition(train_adata, cond))
            used_conditions.append(cond)
            continue
        
        # Parse genes
        genes = parse_condition_to_genes(cond, gene2ensembl)
        if len(genes) == 0:
            print(f"   Skipping {cond}: no mappable genes")
            continue
        
        # Create in silico perturber for this condition
        isp = InSilicoPerturber(
            perturb_type=perturbation_type,
            genes_to_perturb=genes,
            model_type="CellClassifier",
            emb_mode="cls",
            max_ncells=max_ncells_ctrl,
            emb_layer=0,
            forward_batch_size=forward_batch_size,
            nproc=n_jobs,
        )
        
        # Get embeddings
        try:
            pert_emb = get_perturbed_embedding(isp,
                                               geneformer_model,
                                               ctrl_dataset)
            emb_mean = pert_emb.mean(axis=0)  # (hidden_dim,)
            # Get mean expression for this condition
            expr_mean = mean_expr_for_condition(train_adata, cond)  # (n_genes,)
            
            X_list.append(emb_mean)
            Y_list.append(expr_mean)
            used_conditions.append(cond)
        except Exception as e:
            print(f"   Skipping condition {cond}: {str(e)}")
            continue
        
    if len(X_list) == 0:
        raise RuntimeError("No conditions produced valid (X, Y) pairs. "
                           "Check gene mapping and perturbation settings.")
    
    X = np.vstack(X_list)  # (n_conditions, hidden_dim)
    Y = np.vstack(Y_list)  # (n_conditions, n_genes)
    
    logger.info(f"Generated {len(used_conditions)} training pairs")
    logger.info(f"  X shape: {X.shape} (conditions x embedding_dim)")
    logger.info(f"  Y shape: {Y.shape} (conditions x genes)")
    
    return X, Y, used_conditions


# =======================
# Main
# =======================
def main(parser):
    args = parser.parse_args()

    GENEFORMER_MODEL_LOCATION = args.geneformer_model_dir
    today_date = datetime.now().strftime('%y%m%d')
        
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    models_dir = save_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = save_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(log_dir / f"Geneformer_Seed{args.seed_start}to{args.seed_end}_training_log.txt")
    logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f'Saving to {save_dir}')
    logger.info(f'Model saving directory: {models_dir}')
    
    # GPU 정보
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    logger.info(f"Batch size of regression: {args.regression_batch_size}")
    logger.info(f'Batch size of Geneformer Classifier() and InSilicoPerturber(): {args.forward_batch_size}')
    logger.info(f"Seed range: {args.seed_start} to {args.seed_end}")
    
    # =======================
    # Load Data
    # =======================
    logger.info("\n" + "="*60)
    logger.info("Loading Data")
    logger.info("="*60)

    # Data load
    pert_data = PertData(args.data_dir, default_pert_graph=False)

    if args.preprocess:
        logger.info("Running preprocessing...")
        load_adata = sc.read_h5ad(os.path.join(args.data_dir, args.data_name + '.h5ad'))
        load_adata.uns['log1p'] = {}
        load_adata.uns['log1p']['base'] = None
        pert_data.new_data_process(dataset_name=args.data_name, adata=load_adata)
        logger.info("Preprocessing finished.")
    else:
        logger.info(f"Loading data from {args.data_dir}")
        pert_data.load(data_path=os.path.join(args.data_dir, args.data_name))

    adata = pert_data.adata
    
    if "n_counts" in adata.obs.columns:
        adata.obs["n_counts"] = pd.to_numeric(adata.obs["n_counts"], errors="coerce")
    
    if 'ensembl_id' not in adata.var.columns:
        logger.warning("'ensembl_id' must be in var.")
             
    if not adata.var.index[0].startswith('ENSG'):
        if 'ensembl_id' in adata.var.columns:
             adata.var_names = adata.var['ensembl_id'].tolist()
        else:
             logger.warning("Could not find Ensembl IDs in var. Using index directly.")
    
    logger.info(f"Loaded data: {adata.shape[0]} cells x {adata.shape[1]} genes")
    logger.info(f"Total conditions: {adata.obs['condition'].nunique()}")
    logger.info(f"Index of var features is Ensembl IDs: {adata.var_names[0].startswith('ENSG')}")
    
    # =======================
    # Data Validation
    # =======================
    # Load gene to ENSEMBL mapping (for parsing condition strings)
    logger.info("\n" + "="*60)
    logger.info("Loading Gene Mapping")
    logger.info("="*60)

    mapping_path = resolve_ensembl_mapping_dict(args)
    logger.info(f"Using Geneformer Ensembl mapping dictionary: {mapping_path}")
    with open(mapping_path, "rb") as f:
        gene2ensembl = pickle.load(f)

    logger.info(f"Loaded gene2ensembl mapping: {len(gene2ensembl)} genes")
    
    # Tokeniztion for ctrl cell data
    logger.info("\n" + "="*60)
    logger.info("Tokenizing Control Data")
    logger.info("="*60)
    
    ctrl_tk_dir = os.path.join(save_dir, 'ctrl_tokenized_data')
    ctrl_adata = adata[adata.obs['condition'] == 'ctrl', :].copy()
    if not os.path.exists(ctrl_tk_dir) or not os.listdir(ctrl_tk_dir):
        logger.info("Tokenizing control data...")
        ctrl_adata_tok = tokenize_adata(ctrl_adata, output_dir=ctrl_tk_dir, n_jobs=args.num_workers)
    else:
        logger.info("Using existing control tokenized data.")
        ctrl_adata_tok = f"{ctrl_tk_dir}/tokenized.dataset"
    logger.info(f"Control data: {ctrl_adata.shape[0]} cells x {ctrl_adata.shape[1]} genes")
    logger.info(f"Tokenized to: {ctrl_adata_tok}")

    # =======================
    # Training Loop
    # =======================
    training_summary = []
    total_seeds = args.seed_end - args.seed_start + 1

    for seed_idx, seed in enumerate(range(args.seed_start, args.seed_end + 1), 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Model {seed_idx}/{total_seeds}: Seed {seed}")
        logger.info(f"{'='*60}")
        
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # =======================
        # Prepare data split
        # =======================
        logger.info(f"[Seed {seed}] Preparing data split...")
        pert_data.prepare_split(
            split="simulation",
            seed=seed,
            train_gene_set_size=args.train_gene_set_size
        )
        
        tmp_train_conditions = pert_data.set2conditions.get('train', [])
        test_conditions = pert_data.set2conditions.get('test', [])
        
        # Validate gene mapping early
        valid_train_conditions, invalid_train = validate_gene_mapping(
            tmp_train_conditions, gene2ensembl, logger
        )
                
        if len(invalid_train) > 0:
            logger.warning(f"  Removed {len(invalid_train)} invalid conditions")
            train_conditions = valid_train_conditions
        else:
            train_conditions = tmp_train_conditions
        
        
        logger.info(f"  - Train conditions: {len(train_conditions)}")
        logger.info(f"  - Test conditions: {len(test_conditions)}")
        
        # Filter data
        train_adata = adata[adata.obs['condition'].isin(train_conditions), :].copy()
        logger.info(f"[Seed {seed}] Training data: {train_adata.shape[0]} cells x {train_adata.shape[1]} genes")
        
        # Create seed-specific directory
        seed_model_dir = models_dir / f"seed_{seed}"
        seed_model_dir.mkdir(exist_ok=True)
        
        # =======================
        # Tokenization
        # =======================
        logger.info(f"[Seed {seed}] Tokenizing data...")
        
        tokenization_dir = seed_model_dir / "tokenized_data"
        
        
        if tokenization_dir.exists() and (tokenization_dir / 'dataset_info.json'):
            logger.info(f"[Seed {seed}] Found existing tokenized train dataset")
            train_adata_tok = f"{tokenization_dir}/train/tokenized.dataset"
        else:
            tokenization_dir.mkdir(exist_ok=True)
            train_adata_tok = tokenize_adata(train_adata, output_dir=tokenization_dir / "train", n_jobs=args.num_workers)
            logger.info(f"[Seed {seed}] Tokenization complete")
        
        # =======================
        # 1. Fine-tune Classifier
        # =======================
        logger.info(f"[Seed {seed}] Fine-tuning Geneformer Classifier...")
        
        classifier_save_path = seed_model_dir / "classifier"
        
        if classifier_save_path.exists() and (classifier_save_path / "config.json").exists():
            logger.info(f"[Seed {seed}] Found existing fine-tuned classifier at {classifier_save_path}. Skipping fine-tuning classifier.")
            classifier_location = classifier_save_path
        else:
            temp_training_dir = seed_model_dir / "temp_training"
            if temp_training_dir.exists():
                shutil.rmtree(temp_training_dir)
            temp_training_dir.mkdir(parents=True, exist_ok=True)
            
            #with tempfile.TemporaryDirectory() as temp_training_dir:
            try:
                # Initialize classifier with custom training args
                cc = Classifier(classifier="cell",
                                cell_state_dict={"state_key": "condition", "states": "all"},
                                freeze_layers=12,
                                num_crossval_splits=1,
                                max_ncells_per_class=args.ncells_training,
                                forward_batch_size=args.forward_batch_size * 2, # 48 * 2
                                nproc=args.num_workers,
                                training_args={
                                    "num_train_epochs": args.num_classifier_epochs,
                                    "evaluation_strategy": "epoch",
                                    "save_strategy": "epoch",
                                    "load_best_model_at_end": True,
                                    "metric_for_best_model": "eval_macro_f1",
                                    "save_total_limit": 1,  # Only keep best checkpoint
                                    "per_device_train_batch_size": args.forward_batch_size, # 48
                                    "per_device_eval_batch_size": args.forward_batch_size * 2, # 48 * 2
                                    "gradient_accumulation_steps": args.accum_steps,
                                    "gradient_checkpointing": True, 
                                    "fp16": True,
                                    "dataloader_num_workers": 8,
                                    "dataloader_pin_memory": True,
                                    "seed": seed,
                                })
                
                # Prepare data
                logger.info(f"[Seed {seed}] Preparing classifier data...")
                cc.prepare_data(input_data_file=train_adata_tok,
                                output_directory=str(temp_training_dir),
                                output_prefix="classification",
                                split_id_dict=None)
                
                # Train classifier
                logger.info(f"[Seed {seed}] Starting classifier training...")
                train_start = datetime.now()
                
                try:
                    cc.validate(model_directory=GENEFORMER_MODEL_LOCATION,
                                prepared_input_data_file=f"{temp_training_dir}/classification_labeled_train.dataset",
                                id_class_dict_file=f"{temp_training_dir}/classification_id_class_dict.pkl",
                                output_directory=str(temp_training_dir),
                                output_prefix="classification_result")
                except Exception as e:
                    logger.error(f"[Seed {seed}] Warning: Validation/Plotting failed ({e})")
                    logger.error(f"[Seed {seed}] However, training might be finished. Attempting to retrieve model...")              
                
                classifier_train_time = (datetime.now() - train_start).total_seconds() / 60
                logger.info(f"[Seed {seed}] Classifier training completed in {classifier_train_time:.1f} minutes")
                
                # Find trained model directory
                candidate_dirs = list(Path(temp_training_dir).glob("*geneformer_cellClassifier*"))
                trained_model_source = None
                
                if candidate_dirs:
                    # 보통 ksplit1 폴더 안에 최종 모델이 있습니다.
                    possible_path = candidate_dirs[0] / "ksplit1"
                    if possible_path.exists():
                        trained_model_source = possible_path
                    else:
                        # ksplit 구조가 아닐 경우 상위 폴더를 지정
                        trained_model_source = candidate_dirs[0]
                
                if trained_model_source and trained_model_source.exists():
                    logger.info(f"[Seed {seed}] Found trained model at: {trained_model_source}")
                    
                    # Copy classifier to permanent location
                    classifier_location = seed_model_dir / "classifier"
                    if classifier_location.exists():
                        shutil.rmtree(classifier_location)
                    shutil.copytree(trained_model_source, classifier_location)
                    
                    # Save id_class_dict
                    shutil.copy(
                        f"{temp_training_dir}/classification_id_class_dict.pkl",
                        seed_model_dir / "id_class_dict.pkl"
                    )
                    logger.info(f"[Seed {seed}] Classifier saved to: {classifier_location.relative_to(save_dir)}")
                    
                else:
                    # 모델이 정말로 없는 경우 (학습 자체가 초반에 터짐)
                    raise FileNotFoundError(f"Critical: Could not find any trained model in {temp_training_dir}. Training failed.")
            
            finally:
                # [선택] 디버깅을 위해 에러 시 폴더를 남기고 싶으면 아래를 주석 처리하세요.
                # 성공했으면 임시 폴더 삭제
                if 'temp_training_dir' in locals() and temp_training_dir.exists():
                    # 안전을 위해 모델 복사가 확실히 끝났을 때만 지우는 로직을 추가할 수도 있지만,
                    # 여기서는 그냥 깔끔하게 지웁니다. (위에서 copytree로 이미 복사함)
                    shutil.rmtree(temp_training_dir)
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # =======================
        # 2. Generate Embeddings via In Silico Perturbation
        # =======================
        logger.info(f"[Seed {seed}] Generating condition embeddings...")
        
        # Load fine-tuned classifier
        num_classes = len(train_conditions)
        geneformer_model = pu.load_model("CellClassifier",
                                         num_classes=num_classes,
                                         model_directory=str(classifier_location),
                                         mode="eval")
        geneformer_model.eval()
        for p in geneformer_model.parameters():
            p.requires_grad = False
        
        # Build training pairs
        X, Y, used_conditions = build_training_pairs(train_adata=train_adata,
                                                     train_conditions=train_conditions,
                                                     ctrl_tok_path=ctrl_adata_tok,
                                                     geneformer_model=geneformer_model,
                                                     perturbation_type=args.perturbation_type,
                                                     max_ncells_ctrl=args.max_ncells_ctrl,
                                                     forward_batch_size=args.forward_batch_size,
                                                     n_jobs=args.num_workers,
                                                     gene2ensembl=gene2ensembl,
                                                     logger=logger)
        
        logger.info(f"[Seed {seed}] Generated embeddings:")
        logger.info(f"  - X shape: {X.shape} (conditions x embedding_dim)")
        logger.info(f"  - Y shape: {Y.shape} (conditions x genes)")
        logger.info(f"  - Used conditions: {len(used_conditions)}/{len(train_conditions)}")
        
        # Clean up model from GPU
        del geneformer_model
        torch.cuda.empty_cache()
         
        # =======================
        # 3. Train Regression Head (MLP)
        # =======================
        logger.info(f"[Seed {seed}] Training MLP regression head...")
        
        # Split into train/val (0.75/0.25)
        n_conditions = X.shape[0]
        indices = np.arange(n_conditions)
        np.random.shuffle(indices)
        
        split_idx = int(args.train_gene_set_size * n_conditions)
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]
        
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]
        
        logger.info(f"[Seed {seed}] Train: {len(train_idx)} conditions, Val: {len(val_idx)} conditions")
        
        # Train MLP
        regression_start = datetime.now()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        best_model_state, best_val_loss, training_history = train_regression_head(
            X_train, Y_train, X_val, Y_val,
            logger=logger,
            seed=seed,
            n_epochs=args.num_regression_epochs,
            lr=args.regression_lr,
            batch_size=args.regression_batch_size,
            accum_steps=args.regression_accum_steps,
            early_stop_patience=args.regression_early_stop,
            num_workers=args.num_workers,
            device=device
        )
        
        regression_train_time = (datetime.now() - regression_start).total_seconds()
        
        logger.info(f"[Seed {seed}] Regression head trained in {regression_train_time:.1f} seconds")
        logger.info(f"  - Best val loss: {best_val_loss:.6f}")
        logger.info(f"  - Best epoch: {training_history['best_epoch']}")
        
        # Save regression head (state dict + architecture info)
        regression_data = {
            'model_state_dict': best_model_state,
            'in_dim': X.shape[1],
            'out_dim': Y.shape[1],
            'hidden_dim': 2048,
            'best_val_loss': best_val_loss,
            'training_history': training_history,
        }
        regression_path = seed_model_dir / f"Seed_{seed}_best_model.pt"
        torch.save(regression_data, regression_path)
        
        logger.info(f"[Seed {seed}] Best regression model saved")
        
        # =======================
        # Save Training Metadata
        # =======================
        training_info = {
            "seed": seed,
            "classifier_location": str(classifier_location.relative_to(save_dir)),
            "regression_location": str(regression_path.relative_to(save_dir)),
            "dataset_name": args.data_name,
            "perturbation_type": args.perturbation_type,
            "train_gene_set_size": args.train_gene_set_size,
            "train_conditions": train_conditions,
            "test_conditions": test_conditions,
            "used_train_conditions": used_conditions,
            "num_training_conditions": len(train_conditions),
            "num_used_conditions": len(used_conditions),
            "num_test_conditions": len(test_conditions),
            "num_training_cells": train_adata.shape[0],
            "num_ctrl_cells": ctrl_adata.shape[0],
            "num_genes": train_adata.shape[1],
            "ncells_per_condition": args.ncells_training,
            "max_ncells_ctrl": args.max_ncells_ctrl,
            "num_classifier_epochs": args.num_classifier_epochs,
            "num_regression_epochs": args.num_regression_epochs,
            "regression_lr": args.regression_lr,
            "regression_batch_size": args.regression_batch_size,
            "regression_early_stop": args.regression_early_stop,
            "regression_best_val_loss": float(best_val_loss),
            "regression_best_epoch": int(training_history['best_epoch']),
            "regression_num_train_conditions": len(train_idx),
            "regression_num_val_conditions": len(val_idx),
            "training_date": datetime.now().isoformat(),
            "pretrained_model": GENEFORMER_MODEL_LOCATION
        }
        
        with open(seed_model_dir / "training_info.json", 'w') as f:
            json.dump(training_info, f, indent=4)
        
        training_summary.append(training_info)
        
        logger.info(f"[Seed {seed}] Training info saved")
        logger.info(f"[Seed {seed}] Seed {seed} complete!")

    # =======================
    # Save Training Summary
    # =======================
    summary_data = {
        "dataset_name": args.data_name,
        "data_dir": str(args.data_dir),
        "perturbation_type": args.perturbation_type,
        "train_gene_set_size": args.train_gene_set_size,
        "pretrained_model": GENEFORMER_MODEL_LOCATION,
        "num_models": len(training_summary),
        "seed_range": [args.seed_start, args.seed_end],
        "classifier_config": {
            "ncells_per_condition": args.ncells_training,
            "num_epochs": args.num_classifier_epochs,
        },
        "regression_config": {
            "type": "MLP",
            "num_epochs": args.num_regression_epochs,
            "learning_rate": args.regression_lr,
            "batch_size": args.regression_batch_size,
            "early_stop_patience": args.regression_early_stop,
            "hidden_dim": 1024,
        },
        "isp_config": {
            "max_ncells_ctrl": args.max_ncells_ctrl,
        },
        "num_genes": adata.shape[1],
        "models": training_summary
    }

    summary_path = models_dir / f"Seed{args.seed_start}to{args.seed_end}_training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=4)

    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"Models saved: {models_dir}")
    logger.info(f"Summary saved: {summary_path.name}")
    logger.info(f"Total models trained: {len(training_summary)}")
    logger.info(f"Seeds: {args.seed_start}-{args.seed_end}")
    logger.info("="*60)


# =======================
# Argument Parser
# =======================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Geneformer models for perturbation prediction with MLP regression head.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, required=True, help='Path to GEARS perturbation data directory')
    parser.add_argument('--data_name', type=str, required=True, help='Name of the dataset saved in data directory')
    parser.add_argument('--preprocess', action='store_true', help='Whether to preprocess the data')
    parser.add_argument('--perturbation_type', type=str, required=True, choices=['delete', 'overexpress'], default='delete', help='Geneformer perturbation type')
    parser.add_argument('--ncells_training', type=int, default=300, help='Max number of cells per condition for classifier training')
    parser.add_argument('--max_ncells_ctrl', type=int, default=200, help='Max number of ctrl cells for in silico perturbation')
    parser.add_argument('--train_gene_set_size', type=float, default=0.75, help='Proportion of genes for training (0.0-1.0)')
    parser.add_argument('--seed_start', type=int, default=1, help='Starting seed number')
    parser.add_argument('--seed_end', type=int, default=10, help='Ending seed number (inclusive)')
    parser.add_argument('--num_classifier_epochs', type=int, default=15, help='Number of epochs for classifier fine-tuning')
    parser.add_argument('--forward_batch_size', type=int, default=32, help='Batch size for forward passes during embedding extraction')
    parser.add_argument('--num_regression_epochs', type=int, default=15, help='Number of epochs for regression head training')
    parser.add_argument('--regression_lr', type=float, default=1e-4, help='Learning rate for regression head')
    parser.add_argument('--regression_batch_size', type=int, default=1024, help='Batch size for regression head training')
    parser.add_argument('--accum_steps', type=int, default=4, help='Step size of accumulation for Geneformer training')
    parser.add_argument('--regression_accum_steps', type=int, default=1, help='Step size of accumulation for regression head training')
    parser.add_argument('--regression_early_stop', type=int, default=5, help='Early stopping patience for regression head')
    parser.add_argument('--save_dir', type=str, required=True, help='Working directory containing results, configs, etc.')
    parser.add_argument('--geneformer_model_dir', type=str, required=True, help='Path to pretrained Geneformer model')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of CPU cores')
    
    main(parser)