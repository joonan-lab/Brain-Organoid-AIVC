# Last update: 2025. 12. 19 by IGK

"""
Geneformer fine-tuned model evaluation
Find best model among multiple seeds

Environment: t2 (Quadro RTX 8000 (48GB) x 2)
"""

#======================
# Imports
#======================

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
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

FINAL_PREDICTION_METRICS = ["pearson_delta", "pearson_de_delta", "spearmanr_delta", "spearmanr_de_delta", "rmse", "rmse_de", "mae", "mae_de"]


def _final_prediction_metrics(metrics):
    if "spearman_delta" in metrics and "spearmanr_delta" not in metrics:
        metrics["spearmanr_delta"] = metrics["spearman_delta"]
    if "spearman_de_delta" in metrics and "spearmanr_de_delta" not in metrics:
        metrics["spearmanr_de_delta"] = metrics["spearman_de_delta"]
    return {k: metrics[k] for k in FINAL_PREDICTION_METRICS if k in metrics}

from sklearn.metrics import mean_squared_error, mean_absolute_error
import scanpy as sc
import gc

# GEARS
from gears import PertData

# Geneformer
from geneformer import TranscriptomeTokenizer, InSilicoPerturber
from geneformer import perturber_utils as pu
from geneformer.emb_extractor import get_embs

warnings.filterwarnings("ignore")

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
    logger = logging.getLogger("Geneformer_eval_logger")
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
# Model Definitions (Must match Training Script)
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
    
    # Geneformer tokenizer works on directory of .h5ad
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
    
    # Convert gene symbol to Ensembl IDs
    ensembl_genes = []
    for gene in gene_list:
        if gene.startswith("ENSG"):
            ensembl_genes.append(gene)
        elif gene in gene2ensembl:
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


# =========================================================
# Metrics Calculation
# =========================================================
def compute_all_metrics_efficient(
    results: dict,
    ctrl_adata: sc.AnnData,
    non_zero_genes: bool = False,
    return_raw: bool = False,
) -> dict:
    """   
    Args:
        results: Dict with 'pred', 'truth', 'pert_cat'
        ctrl_adata: Control condition adata
        non_zero_genes: Whether to only consider non-zero genes
        return_raw: Whether to return raw metrics
    
    Returns:
        Dict with final 8 prediction metrics
    """
    
    conditions = np.unique(results["pert_cat"])
    assert "ctrl" not in conditions, "ctrl should not be in test conditions"
    condition2idx = {c: np.where(results["pert_cat"] == c)[0] for c in conditions}
    
    true_perturbed = results["truth"]
    pred_perturbed = results["pred"]
    
    mean_ctrl = np.array(ctrl_adata.X.mean(0)).flatten()
    
    # ---------- condition별 mean ----------
    # Mean by condition
    true_mean_by_condition = np.array(
        [true_perturbed[condition2idx[c]].mean(0) for c in conditions]
    )
    pred_mean_by_condition = np.array(
        [pred_perturbed[condition2idx[c]].mean(0) for c in conditions]
    )

    # Delta
    true_delta_by_condition = true_mean_by_condition - mean_ctrl
    pred_delta_by_condition = pred_mean_by_condition - mean_ctrl
    
    zero_rows = np.where(np.all(true_mean_by_condition == 0, axis=1))[0].tolist()    
    
    # ---------- DE gene 인덱스 찾기 ----------
    if "gene_name" in ctrl_adata.var.columns:
        gene_names = ctrl_adata.var["gene_name"].tolist()
    else:
        gene_names = ctrl_adata.var.index.tolist()
    name_to_idx = {g: i for i, g in enumerate(gene_names)}

    def find_DE_indices(adata, condition, name_to_idx, non_zero_genes=False, top_n=20):
        try:
            key_components = next(iter(adata.uns["rank_genes_groups_cov_all"].keys())).split("_")
            condition_key = "_".join([key_components[0], condition, key_components[2]])
            de_genes = adata.uns["rank_genes_groups_cov_all"][condition_key]
            if non_zero_genes:
                 # scFoundation 코드에 있는 부분 (Geneformer에선 uns에 이 키가 없을 수 있으므로 주의)
                 if "top_non_dropout_de_20" in adata.uns:
                     de_genes = adata.uns["top_non_dropout_de_20"][condition_key]
            de_genes = de_genes[:top_n]
            return [name_to_idx[g] for g in de_genes if g in name_to_idx]
        except KeyError:
            return []
    
    de_idx = {
        c: find_DE_indices(ctrl_adata, c, name_to_idx, non_zero_genes)
        for c in conditions
    }
    
    # Control Means for DE genes
    mean_ctrl_de = [
        mean_ctrl[de_idx[c]] if len(de_idx[c]) > 0 else np.zeros(0, dtype=float) 
        for c in conditions
    ]
    
    # DE genes
    true_mean_de = []
    pred_mean_de = []
    for i, c in enumerate(conditions):
        if len(de_idx[c]) > 0:
            true_mean_de.append(true_mean_by_condition[i, de_idx[c]])
            pred_mean_de.append(pred_mean_by_condition[i, de_idx[c]])
        else:
            true_mean_de.append(np.zeros(0, dtype=float))
            pred_mean_de.append(np.zeros(0, dtype=float))
    true_mean_de = np.array(true_mean_de, dtype=object)
    pred_mean_de = np.array(pred_mean_de, dtype=object)
    
    # DE delta
    true_delta_de = []
    pred_delta_de = []
    for i, c in enumerate(conditions):
        if len(de_idx[c]) > 0:
            true_delta_de.append(true_mean_de[i] - mean_ctrl_de[i])
            pred_delta_de.append(pred_mean_de[i] - mean_ctrl_de[i])
        else:
            true_delta_de.append(np.zeros(0, dtype=float))
            pred_delta_de.append(np.zeros(0, dtype=float))
    true_delta_de = np.array(true_delta_de, dtype=object)
    pred_delta_de = np.array(pred_delta_de, dtype=object)
    
    zero_rows_de = [
        i for i in range(len(conditions)) 
        if len(true_mean_de[i]) == 0 or np.all(true_mean_de[i] == 0)
    ]
    
    # ========== 메트릭 계산 헬퍼 함수 ==========
    def compute_corr_over_genes(x, y, conditions, skip_rows, corr_func, non_zero_mask=None):
        """Correlation over genes"""
        res_list = []
        for i, c in enumerate(conditions):
            if i in skip_rows:
                continue
            x_, y_ = x[i], y[i]
            if isinstance(x_, np.ndarray) and x_.dtype == object:
                x_ = np.array(x_, dtype=float)
                y_ = np.array(y_, dtype=float)
            if non_zero_mask is not None:
                mask = non_zero_mask[i]
                x_ = x_[mask]
                y_ = y_[mask]
            if len(x_) > 1:  # Need at least 2 points
                res_list.append(corr_func(x_, y_)[0])
        return res_list
    
    def compute_error_over_conditions(y_true, y_pred, conditions, error_func, non_zero_mask=None):
        """condition별 error (RMSE/MAE) 계산"""
        res_list = []
        for i, c in enumerate(conditions):
            x_ = y_true[i]
            y_ = y_pred[i]
            if isinstance(x_, np.ndarray) and x_.dtype == object:
                x_ = np.array(x_, dtype=float)
                y_ = np.array(y_, dtype=float)
            if non_zero_mask is not None:
                mask = non_zero_mask[i]
                x_ = x_[mask]
                y_ = y_[mask]
            if len(x_) > 0:
                res_list.append(error_func(x_, y_))
        return res_list
    
    # ========== 모든 메트릭 한 번에 계산! ==========
    metrics = {}
    
    non_zero_mask_all = (true_mean_by_condition != 0) if non_zero_genes else None
    
    # Pearson (4개)
    metrics["pearson"] = compute_corr_over_genes(
        true_mean_by_condition, pred_mean_by_condition, 
        conditions, zero_rows, pearsonr, non_zero_mask_all
    )
    metrics["pearson_delta"] = compute_corr_over_genes(
        true_delta_by_condition, pred_delta_by_condition,
        conditions, zero_rows, pearsonr, non_zero_mask_all
    )
    metrics["pearson_de"] = compute_corr_over_genes(
        true_mean_de, pred_mean_de,
        conditions, zero_rows_de, pearsonr
    )
    metrics["pearson_de_delta"] = compute_corr_over_genes(
        true_delta_de, pred_delta_de,
        conditions, zero_rows_de, pearsonr
    )
    
    # Spearman (4개)
    metrics["spearmanr"] = compute_corr_over_genes(
        true_mean_by_condition, pred_mean_by_condition,
        conditions, zero_rows, spearmanr, non_zero_mask_all
    )
    metrics["spearmanr_delta"] = compute_corr_over_genes(
        true_delta_by_condition, pred_delta_by_condition,
        conditions, zero_rows, spearmanr, non_zero_mask_all
    )
    metrics["spearmanr_de"] = compute_corr_over_genes(
        true_mean_de, pred_mean_de,
        conditions, zero_rows_de, spearmanr
    )
    metrics["spearmanr_de_delta"] = compute_corr_over_genes(
        true_delta_de, pred_delta_de,
        conditions, zero_rows_de, spearmanr
    )
    
    # RMSE (4개)
    def rmse_func(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    metrics["rmse"] = compute_error_over_conditions(
        true_mean_by_condition, pred_mean_by_condition,
        conditions, rmse_func, non_zero_mask_all
    )
    metrics["rmse_delta"] = compute_error_over_conditions(
        true_delta_by_condition, pred_delta_by_condition,
        conditions, rmse_func, non_zero_mask_all
    )
    metrics["rmse_de"] = compute_error_over_conditions(
        true_mean_de, pred_mean_de,
        conditions, rmse_func
    )
    metrics["rmse_de_delta"] = compute_error_over_conditions(
        true_delta_de, pred_delta_de,
        conditions, rmse_func
    )
    
    # MAE (4개)
    metrics["mae"] = compute_error_over_conditions(
        true_mean_by_condition, pred_mean_by_condition,
        conditions, mean_absolute_error, non_zero_mask_all
    )
    metrics["mae_delta"] = compute_error_over_conditions(
        true_delta_by_condition, pred_delta_by_condition,
        conditions, mean_absolute_error, non_zero_mask_all
    )
    metrics["mae_de"] = compute_error_over_conditions(
        true_mean_de, pred_mean_de,
        conditions, mean_absolute_error
    )
    metrics["mae_de_delta"] = compute_error_over_conditions(
        true_delta_de, pred_delta_de,
        conditions, mean_absolute_error
    )
    
    # ========== 평균 계산 ==========
    if not return_raw:
        for k in metrics:
            if len(metrics[k]) > 0:
                metrics[k] = float(np.mean(metrics[k]))
            else:
                metrics[k] = np.nan
    
    return metrics



# ==========================================
# Evaluate Single Seed Function (New)
# ==========================================
def evaluate_seed(seed, args, pert_data, ctrl_dataset, gene2ensembl, device, logger):
    """
    Evaluates a single seed.
    Returns: metrics dictionary or None if failed.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating Seed {seed}")
    logger.info(f"{'='*60}")
    
    save_dir = Path(args.save_dir)
    
    # 1. Determine Test Conditions
    pert_data.prepare_split(split="simulation", seed=seed, train_gene_set_size=0.75)
    test_conditions = pert_data.set2conditions['test']
    adata = pert_data.adata
    logger.info(f"Found {len(test_conditions)} test conditions")
    
    # 2. Load Models
    seed_model_dir = Path(args.model_dir) / f"seed_{seed}"
    classifier_path = seed_model_dir / "classifier"
    regression_path = seed_model_dir / f"Seed_{seed}_best_model.pt"
    
    if not classifier_path.exists() or not regression_path.exists():
        logger.error(f"Model files not found for Seed {seed}. Skipping.")
        return None
        
    # A. Load Classifier
    logger.info("Loading Classifier...")
    try:
        # [수정] 학습할 때 저장한 클래스 정보(id_class_dict.pkl)를 불러옵니다.
        id_class_dict_path = seed_model_dir / "id_class_dict.pkl"
        
        if id_class_dict_path.exists():
            with open(id_class_dict_path, "rb") as f:
                id_class_dict = pickle.load(f)
            num_classes = len(id_class_dict) # 예: 41
            logger.info(f"Detected num_classes from pickle: {num_classes}")
        else:
            # 혹시 파일이 없다면 config.json에서 찾기 시도
            config_path = classifier_path / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                num_classes = config.get("num_labels", 0) # 여기서 가져옴
                logger.info(f"Detected num_classes from config: {num_classes}")
            else:
                logger.warning("Could not find class count info. Defaulting to 0 (May fail).")
                num_classes = 0

        # [수정] 찾아낸 정확한 num_classes를 넣어줍니다.
        geneformer_model = pu.load_model(
            "CellClassifier", 
            num_classes=num_classes, # 0 대신 실제 클래스 개수 입력
            model_directory=str(classifier_path),
            mode="eval"
        )
        geneformer_model.eval()
        geneformer_model.to(device)
    except Exception as e:
        logger.error(f"Failed to load Classifier: {e}")
        return None
        
    # B. Load Regression Head
    logger.info("Loading Regression Head...")
    try:
        reg_checkpoint = torch.load(regression_path, map_location=device, weights_only=False)
        reg_head = GeneformerRegressionHead(
            in_dim=reg_checkpoint['in_dim'],
            out_dim=reg_checkpoint['out_dim'],
            hidden_dim=2048, #reg_checkpoint['hidden_dim']
        )
        reg_head.load_state_dict(reg_checkpoint['model_state_dict'])
        reg_head.to(device)
        reg_head.eval()
    except Exception as e:
        logger.error(f"Failed to load Regression Head: {e}")
        # Clean up classifier if regression fails
        del geneformer_model
        torch.cuda.empty_cache()
        return None

    # 3. Prediction Loop
    logger.info("Generating predictions...")
    
    # Pre-calculate control embedding mean
    isp_base = InSilicoPerturber(
        perturb_type=args.perturbation_type,
        model_type="CellClassifier",
        emb_mode="cls",
        max_ncells=args.max_ncells_ctrl,
        emb_layer=0,
        forward_batch_size=args.forward_batch_size,
        nproc=args.num_workers
    )
    ctrl_dataset_filt = isp_base.apply_additional_filters(ctrl_dataset)
    ctrl_emb = get_control_embedding(isp_base, geneformer_model, ctrl_dataset_filt, logger)
    ctrl_emb_mean = ctrl_emb.mean(axis=0)
    
    preds_list = []
    truths_list = []
    conds_list = []
    
    for i, cond in enumerate(test_conditions):
        if (i+1) % 10 == 0:
            logger.info(f"  Processing condition {i+1}/{len(test_conditions)}")
        
        try:
            truth_expr = mean_expr_for_condition(adata, cond)
        except Exception as e:
            logger.warning(f"Could not get truth for {cond}: {e}")
            continue
            
        current_emb = None
        if cond == "ctrl":
            current_emb = ctrl_emb_mean
        else:
            genes = parse_condition_to_genes(cond, gene2ensembl)
            if not genes:
                continue
            
            isp = InSilicoPerturber(
                perturb_type=args.perturbation_type,
                genes_to_perturb=genes,
                model_type="CellClassifier",
                emb_mode="cls",
                max_ncells=args.max_ncells_ctrl,
                emb_layer=0,
                forward_batch_size=args.forward_batch_size,
                nproc=args.num_workers
            )
            try:
                pert_emb = get_perturbed_embedding(isp, geneformer_model, ctrl_dataset_filt)
                current_emb = pert_emb.mean(axis=0)
            except Exception as e:
                logger.warning(f"Perturbation failed for {cond}: {e}")
                continue
        
        with torch.no_grad():
            inp = torch.tensor(current_emb, dtype=torch.float32).unsqueeze(0).to(device)
            pred_expr = reg_head(inp).cpu().numpy().flatten()
            
        preds_list.append(pred_expr)
        truths_list.append(truth_expr)
        conds_list.append(cond)
    
    # Clean up models immediately after inference
    del geneformer_model, reg_head
    torch.cuda.empty_cache()
    gc.collect()

    if len(preds_list) == 0:
        logger.error("No valid predictions generated.")
        return None
        
    # Format Results
    preds_arr = np.array(preds_list)
    truths_arr = np.array(truths_list)
    conds_arr = np.array(conds_list)
    
    results = {
        "pred": preds_arr,
        "truth": truths_arr,
        "pert_cat": conds_arr
    }
    
    # 4. Compute Metrics
    logger.info("Computing metrics...")
    ctrl_adata = adata[adata.obs['condition'] == 'ctrl', :].copy()
    
    # Ensure DE info exists in ctrl_adata
    if 'rank_genes_groups_cov_all' not in ctrl_adata.uns:
        if 'rank_genes_groups_cov_all' in adata.uns:
            ctrl_adata.uns['rank_genes_groups_cov_all'] = adata.uns['rank_genes_groups_cov_all']
    
    metrics = compute_all_metrics_efficient(
        results, ctrl_adata, non_zero_genes=False, return_raw=False
    )
    metrics['seed'] = seed
    
    logger.info("Seed Results:")
    for k, v in metrics.items():
        if k != 'seed':
            logger.info(f"  {k}: {v:.4f}")
        
    # Save Predictions
    if args.save_predictions:
        pred_out = save_dir / "predictions" / f"seed_{seed}"
        pred_out.mkdir(parents=True, exist_ok=True)
        np.save(pred_out / "predictions.npy", preds_arr)
        np.save(pred_out / "targets.npy", truths_arr)
        np.save(pred_out / "conditions.npy", conds_arr)
        
    return metrics



# =======================
# Main Evaluation Loop
# =======================
def main(args):
    total_start = time.time()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = save_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(log_dir / f"Geneformer_evaluation_Seed{args.seed_start}to{args.seed_end}_log.txt")
    logger.info(f"Starting Geneformer Evaluation for seed {args.seed_start} to {args.seed_end}...")
    logger.info(f"Arguments: {vars(args)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # 1. Load Data
    logger.info(f"Loading data from {args.data_dir}...")
    pert_data = PertData(args.data_dir, default_pert_graph=False)
    pert_data.load(data_path=os.path.join(args.data_dir, args.data_name))
    adata = pert_data.adata
    
    if "n_counts" in adata.obs.columns:
        logger.info("Converting 'n_counts' to numeric...")
        adata.obs["n_counts"] = pd.to_numeric(adata.obs["n_counts"], errors="coerce")
    
    # Gene Mapping
    mapping_path = resolve_ensembl_mapping_dict(args)
    logger.info(f"Using Geneformer Ensembl mapping dictionary: {mapping_path}")
    with open(mapping_path, "rb") as f:
        gene2ensembl = pickle.load(f)
        
    # 2. Tokenize Control Data
    ctrl_tk_dir = save_dir / 'ctrl_tokenized_data'
    ctrl_adata = adata[adata.obs['condition'] == 'ctrl', :].copy()
    
    # DE Gene check (Ensure 'rank_genes_groups_cov_all' exists)
    if 'rank_genes_groups_cov_all' not in ctrl_adata.uns:
        if 'rank_genes_groups_cov_all' in adata.uns:
            ctrl_adata.uns['rank_genes_groups_cov_all'] = adata.uns['rank_genes_groups_cov_all']
        else:
            logger.warning("DE gene information not found! DE metrics will be empty.")
    
    if not ctrl_tk_dir.exists():
        logger.info("Tokenizing control data for inference...")
        ctrl_tok_path = tokenize_adata(ctrl_adata, output_dir=ctrl_tk_dir, n_jobs=args.num_workers)
    else:
        ctrl_tok_path = str(ctrl_tk_dir / "tokenized.dataset")
        logger.info(f"Using existing control tokens at {ctrl_tok_path}")

    ctrl_dataset = pu.load_and_filter(None, 1, ctrl_tok_path)
    

    # 3. Loop over Seeds
    all_metrics = []
    
    for seed in range(args.seed_start, args.seed_end + 1):
        metrics = evaluate_seed(
            seed=seed,
            args=args,
            pert_data=pert_data,
            ctrl_dataset=ctrl_dataset,
            gene2ensembl=gene2ensembl,
            device=device,
            logger=logger
        )
        
        if metrics is not None:
            all_metrics.append(metrics)

    # =======================
    # Summary
    # =======================
    if all_metrics:
        logger.info(f"\n{'='*80}")
        logger.info("SUMMARY ACROSS ALL SEEDS")
        logger.info(f"{'='*80}")
        
        df_metrics = pd.DataFrame(all_metrics)
        summary = df_metrics.drop('seed', axis=1).agg(['mean', 'std'])
        
        logger.info("\nMean ± Std:")
        for col in summary.columns:
            m = summary.loc['mean', col]
            s = summary.loc['std', col]
            logger.info(f"  {col}: {m:.4f} ± {s:.4f}")
            
        df_metrics.to_csv(Path(args.model_dir) / f"Seed{args.seed_start}to{args.seed_end}_metrics.csv", index=False)
        summary.to_csv(Path(args.model_dir) / "metrics_summary.csv")
        
        if 'pearson_de_delta' in df_metrics.columns:
            best_idx = df_metrics['pearson_de_delta'].idxmax()
            best_seed = df_metrics.loc[best_idx, 'seed']
            best_val = df_metrics.loc[best_idx, 'pearson_de_delta']
            logger.info(f"\nBest Seed: {int(best_seed)} (Pearson DE Delta: {best_val:.4f})")
            
    logger.info(f"Total time: {(time.time() - total_start)/60:.2f} minutes")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Geneformer Evaluation Script")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to GEARS perturbation data directory')
    parser.add_argument('--data_name', type=str, required=True, help='Dataset name')
    parser.add_argument('--model_dir', type=str, required=True, help='Path containing "models" folder from training')
    parser.add_argument('--geneformer_model_dir', type=str, required=True, help='Pretrained Geneformer path (for config)')
    parser.add_argument('--save_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--seed_start', type=int, default=1)
    parser.add_argument('--seed_end', type=int, default=10)
    parser.add_argument('--perturbation_type', type=str, default='delete', choices=['delete', 'overexpress'])
    parser.add_argument('--max_ncells_ctrl', type=int, default=200, help='Max ctrl cells for ISP')
    parser.add_argument('--forward_batch_size', type=int, default=100, help='Inference batch size')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--save_predictions', action='store_true')
    
    args = parser.parse_args()
    main(args)