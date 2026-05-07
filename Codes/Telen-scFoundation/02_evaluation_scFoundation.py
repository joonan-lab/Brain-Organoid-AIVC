# =======================
# Imports
# =======================

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict
import time
import numpy as np
import pandas as pd
import torch
import logging
from torch import nn
from torch_geometric.loader import DataLoader as GeoDataLoader

FINAL_PREDICTION_METRICS = ["pearson_delta", "pearson_de_delta", "spearmanr_delta", "spearmanr_de_delta", "rmse", "rmse_de", "mae", "mae_de"]


def _final_prediction_metrics(metrics):
    if "spearman_delta" in metrics and "spearmanr_delta" not in metrics:
        metrics["spearmanr_delta"] = metrics["spearman_delta"]
    if "spearman_de_delta" in metrics and "spearmanr_de_delta" not in metrics:
        metrics["spearmanr_de_delta"] = metrics["spearman_de_delta"]
    return {k: metrics[k] for k in FINAL_PREDICTION_METRICS if k in metrics}

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from anndata import AnnData

# scFoundation 유틸들
sys.path.append("<ENV_ROOT>/scFoundation/model")
sys.path.insert(0, "<ENV_ROOT>/scFoundation/GEARS")

from gears import PertData
from modules.encoders import MAEAutobinencoder

warnings.filterwarnings("ignore")

# =======================
# 설정
# =======================
test_batch_size = 128

# OS gene 리스트
OS_GENE_TSV = "<DATA_ROOT>/scFoundation/preprocessing/OS_scRNA_gene_index.19264.tsv"
os_gene_df = pd.read_csv(OS_GENE_TSV, sep="\t")
ref_gene_ids = os_gene_df["gene_name"].tolist()
N_GENE = len(ref_gene_ids)


# =======================
# Logger
# =======================
def setup_logger(log_file_path: Path):
    logger = logging.getLogger("eval_logger")
    
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
    
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# =======================
# Utils
# =======================
def create_reorder_indices(gene_ids_dataset: list, ref_gene_ids: list):
    """Gene reordering indices 생성"""
    """Dataset gene order -> creating index for reordering by ref_gene_ids (OS)"""
    ds_gene_to_idx = {g: i for i, g in enumerate(gene_ids_dataset)}
    
    indices = []
    for ref_gene in ref_gene_ids:
        idx = ds_gene_to_idx.get(ref_gene, -1)
        indices.append(idx)
    
    return torch.tensor(indices, dtype=torch.long)

def reorder_matrix_fast(mat: torch.Tensor, 
                        reorder_indices: torch.Tensor,
                        device: torch.device) -> torch.Tensor:
    """
    mat: (B, G_dataset)
    reorder_indices: (G_ref,), 값은 dataset index 또는 -1
    -> out: (B, G_ref)
    """
    B = mat.shape[0]
    G = len(reorder_indices)
    out = torch.zeros((B, G), dtype=mat.dtype, device=device)
    
    valid_mask = reorder_indices >= 0
    valid_indices = reorder_indices[valid_mask]
    
    if len(valid_indices) > 0:
        out[:, valid_mask] = mat[:, valid_indices]
    
    return out

def create_gene_to_ref_mapping(gene_ids_dataset: list, ref_gene_ids: list):
    """Dataset gene index -> ref gene index mapping"""
    ref_gene_to_idx = {g: i for i, g in enumerate(ref_gene_ids)}
    
    gene_to_ref_idx = {}
    for i, gene in enumerate(gene_ids_dataset):
        if gene in ref_gene_to_idx:
            gene_to_ref_idx[i] = ref_gene_to_idx[gene]
    
    return gene_to_ref_idx

def create_pert_flags_fast(batch_data, batch_size, gene_to_ref_idx: dict, device):
    """
    Perturbation flags
    Result shape: (batch_size, N_GENE)  (ref_gene_ids)
    """
    pert_flags = torch.zeros((batch_size, N_GENE), dtype=torch.long, device=device)
    
    for idx, pert_list in enumerate(batch_data.pert_idx):
        if len(pert_list) > 0 and pert_list[0] != -1:
            valid_ref_indices = [gene_to_ref_idx[gene_idx] 
                                for gene_idx in pert_list 
                                if gene_idx in gene_to_ref_idx]
            
            if valid_ref_indices:
                pert_flags[idx, valid_ref_indices] = 1
    
    return pert_flags


# =======================
# Model
# =======================
class ScFoundationPerturbModel(nn.Module):
    """scFoundation-based perturbation prediction model"""
    
    def __init__(self, 
                 ckpt_path: str, 
                 bin_set: str = "autobin_resolution_append", 
                 highres: int = 0):
        super().__init__()
        config = {
            "load_path": ckpt_path,
            "bin_set": bin_set,
            "highres": highres,
        }
        self.encoder = MAEAutobinencoder(config)
        hidden_dim = self.encoder.model_config["decoder"]["hidden_dim"]
        
        self.pert_emb = nn.Embedding(2, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.hidden_dim = hidden_dim
    
    def forward(self, x_values: torch.Tensor, total_counts: torch.Tensor, 
                pert_flags: torch.Tensor) -> torch.Tensor:
        """
        x_values: (B, N_GENE)  (ref_gene orders)
        total_counts: (B,)
        pert_flags: (B, N_GENE)
        """
        x_query = torch.cat(
            (x_values, total_counts.unsqueeze(1)),
            dim=1,
        ).to(dtype=torch.float32)
        
        enc = self.encoder(x_query)
        
        # Sequence length 맞추기
        B, L_enc, _ = enc.shape
        L_flags = pert_flags.size(1)
        
        if L_flags < L_enc:
            pad = torch.zeros(
                (B, L_enc - L_flags),
                dtype=pert_flags.dtype,
                device=pert_flags.device,
            )
            pert_flags_aligned = torch.cat([pert_flags, pad], dim=1)
        elif L_flags > L_enc:
            pert_flags_aligned = pert_flags[:, :L_enc]
        else:
            pert_flags_aligned = pert_flags
        
        pert_emb = self.pert_emb(pert_flags_aligned.long())
        enc = enc + pert_emb
        pred = self.head(enc).squeeze(-1)
        
        return pred

# =======================
# Compute model performance metrics
# =======================
def compute_all_metrics_efficient(
    results: Dict,
    ctrl_adata: AnnData,
    non_zero_genes: bool = False,
    return_raw: bool = False,
) -> Dict:
    """   
    Args:
        results: Dict with 'pred', 'truth', 'pert_cat'
        ctrl_adata: Control condition adata
        non_zero_genes: Whether to only consider non-zero genes
        return_raw: Whether to return raw metrics
    
    Returns:
        Dict with final 8 prediction metrics
    """
    
    # ========== 공통 계산 (한 번만!) ==========
    conditions = np.unique(results["pert_cat"])
    assert "ctrl" not in conditions, "ctrl should not be in test conditions"
    condition2idx = {c: np.where(results["pert_cat"] == c)[0] for c in conditions}
    
    #mean_ctrl = np.array(ctrl_adata.X.mean(0)).flatten()
    
    true_perturbed = results["truth"]
    pred_perturbed = results["pred"]
    
    # ref_gene_ids 기준 name→index 매핑
    ref_name_to_idx = {g: i for i, g in enumerate(ref_gene_ids)}
    
    # ---------- ctrl mean을 ref_gene_ids 순서로 맞추기 ----------
    if "gene_name" in ctrl_adata.var.columns:
        ctrl_gene_names = ctrl_adata.var["gene_name"].tolist()
    else:
        ctrl_gene_names = ctrl_adata.var.index.tolist()
    
    ctrl_mean_dataset = np.array(ctrl_adata.X.mean(0)).flatten()  # (G_dataset,)
    mean_ctrl = np.zeros(len(ref_gene_ids), dtype=float)          # (G_ref,)
    
    for ds_idx, g in enumerate(ctrl_gene_names):
        ref_idx = ref_name_to_idx.get(g, None)
        if ref_idx is not None:
            mean_ctrl[ref_idx] = ctrl_mean_dataset[ds_idx]
    
    # ---------- condition별 mean (ref 순서) ----------
    # Mean by condition
    true_mean_by_condition = np.array(
        [true_perturbed[condition2idx[c]].mean(0) for c in conditions]
    ) # (C, G_ref)
    pred_mean_by_condition = np.array(
        [pred_perturbed[condition2idx[c]].mean(0) for c in conditions]
    ) # (C, G_ref)
    

    L_eval = true_perturbed.shape[1]   # 지금 평가한 gene 수

    # mean_ctrl도 그 길이에 맞춰 잘라 쓰기
    mean_ctrl = mean_ctrl[:L_eval]
    true_mean_by_condition = true_mean_by_condition[:, :L_eval]
    pred_mean_by_condition = pred_mean_by_condition[:, :L_eval]

    # Delta
    true_delta_by_condition = true_mean_by_condition - mean_ctrl
    pred_delta_by_condition = pred_mean_by_condition - mean_ctrl
    
    zero_rows = np.where(np.all(true_mean_by_condition == 0, axis=1))[0].tolist()    
    
    # DE genes
    def find_DE_genes(adata, condition, geneid2idx, non_zero_genes=False, top_n=20):
        key_components = next(iter(adata.uns["rank_genes_groups_cov_all"].keys())).split("_")
        condition_key = "_".join([key_components[0], condition, key_components[2]])
        de_genes = adata.uns["rank_genes_groups_cov_all"][condition_key]
        if non_zero_genes:
            de_genes = adata.uns["top_non_dropout_de_20"][condition_key]
        de_genes = de_genes[:top_n]
        de_idx = [geneid2idx[i] for i in de_genes]
        return de_idx
    
    # ---------- DE gene 인덱스를 ref 기준으로 ----------
    def find_DE_ref_indices(adata, condition, ref_name_to_idx, non_zero_genes=False, top_n=20, L_eval=None):
        key_components = next(iter(adata.uns["rank_genes_groups_cov_all"].keys())).split("_")
        condition_key = "_".join([key_components[0], condition, key_components[2]])
        de_genes = adata.uns["rank_genes_groups_cov_all"][condition_key]
        if non_zero_genes:
            de_genes = adata.uns["top_non_dropout_de_20"][condition_key]
        de_genes = de_genes[:top_n]
        de_ref_idx = [ref_name_to_idx[g] for g in de_genes if g in ref_name_to_idx]
        if L_eval is not None:
            de_ref_idx = [i for i in de_ref_idx if i < L_eval]
        return de_ref_idx
    
    de_idx = {
        c: find_DE_ref_indices(ctrl_adata, c, ref_name_to_idx, non_zero_genes, L_eval=L_eval)
        for c in conditions
    }
    
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
                # object array (DE) 방어용
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
    
    non_zero_mask_all = true_mean_by_condition != 0 if non_zero_genes else None
    
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
    
    return _final_prediction_metrics(metrics)


# =======================
# Evaluation 함수
# =======================
def evaluate_model(
    model: nn.Module,
    test_loader,
    device,
    reorder_indices: torch.Tensor,
    gene_to_ref_idx: dict,
    max_seq_len: int,
    logger=None,
    seed=None,
):
    """
    모델 evaluation - GEARS 형식으로 결과 반환
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_pert_cats = []
    
    total_batches = len(test_loader)
    start_time = time.time()

    with torch.no_grad():
        for b_idx, batch_data in enumerate(test_loader):
            batch_size_actual = len(batch_data.y)
            batch_data.to(device)
            
            x: torch.Tensor = batch_data.x
            n_genes = int(x.shape[0] / batch_size_actual)
            
            ori_gene_values = x[:, 0].view(batch_size_actual, n_genes)
            target_gene_values = batch_data.y
            
            # Gene reordering -> ref_gene_ids
            x_query = reorder_matrix_fast(ori_gene_values, reorder_indices, device)
            target_values = reorder_matrix_fast(target_gene_values, reorder_indices, device)
            
            # Perturbation flags
            pert_flags = create_pert_flags_fast(batch_data, batch_size_actual, gene_to_ref_idx, device)
            
            # Apply max_seq_len
            input_gene_size = x_query.size(1)      # 보통 19264
            gene_idx = torch.arange(input_gene_size, dtype=torch.long, device=device)

            if input_gene_size > max_seq_len:
                perm = torch.randperm(input_gene_size, device=device)
                gene_idx = perm[:max_seq_len]
                #gene_idx = gene_idx[:max_seq_len]

            x_query = x_query[:, gene_idx]         # (B, L_eval)
            target_values = target_values[:, gene_idx]
            pert_flags = pert_flags[:, gene_idx]

            total_counts = torch.log10(ori_gene_values.sum(dim=1) + 1).to(device)
            
            # Prediction
            output = model(x_query, total_counts, pert_flags)
            
            '''
            # Output length 맞추기
            if output.shape[1] > N_GENE:
                output = output[:, :N_GENE]
            elif output.shape[1] < N_GENE:
                pad = torch.zeros(
                    (batch_size_actual, N_GENE - output.shape[1]),
                    device=device
                )
                output = torch.cat([output, pad], dim=1)
            '''
            # 저장
            all_predictions.append(output.cpu().numpy())
            all_targets.append(target_values.cpu().numpy())
            all_pert_cats.extend(batch_data.pert)

            # ========================
            # Logging processe
            # ========================
            if logger is not None and (b_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_per_batch = elapsed / (b_idx + 1)
                remaining_batches = total_batches - (b_idx + 1)
                eta_sec = remaining_batches * avg_per_batch
                eta_min = eta_sec / 60.0

                percent = (b_idx + 1) / total_batches
                bar_len = 30
                filled = int(bar_len * percent)
                bar = "█" * filled + "░" * (bar_len - filled)

                logger.info(
                    f"[Seed {seed}] [{bar}] {percent*100:6.2f}% "
                    f"({b_idx + 1}/{total_batches} batches, "
                    f"ETA ~ {eta_min:.1f} min)"
                )
    
    # Concatenate
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    pert_cats = np.array(all_pert_cats)
    
    results = {
        "pred": predictions.astype(float),
        "truth": targets.astype(float),
        "pert_cat": pert_cats,
    }
    
    return results


# =======================
# Visualization
# =======================
def plot_results(predictions, targets, save_dir, seed):
    """결과 시각화"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    n_samples = min(10000, len(predictions.flatten()))
    idx = np.random.choice(len(predictions.flatten()), n_samples, replace=False)
    
    pred_flat = predictions.flatten()[idx]
    target_flat = targets.flatten()[idx]
    
    # Scatter
    axes[0].scatter(target_flat, pred_flat, alpha=0.3, s=1)
    axes[0].plot([target_flat.min(), target_flat.max()], 
                 [target_flat.min(), target_flat.max()], 
                 'r--', linewidth=2)
    axes[0].set_xlabel('True Expression', fontsize=12)
    axes[0].set_ylabel('Predicted Expression', fontsize=12)
    axes[0].set_title(f'Seed {seed}: Prediction vs True', fontsize=14)
    
    # Hexbin
    axes[1].hexbin(target_flat, pred_flat, gridsize=50, cmap='Blues', mincnt=1)
    axes[1].plot([target_flat.min(), target_flat.max()], 
                 [target_flat.min(), target_flat.max()], 
                 'r--', linewidth=2)
    axes[1].set_xlabel('True Expression', fontsize=12)
    axes[1].set_ylabel('Predicted Expression', fontsize=12)
    axes[1].set_title(f'Seed {seed}: Density Plot', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'seed_{seed}_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()


# =======================
# 메인 함수
# =======================
def main(args):
    import time
    total_start = time.time()
    
    # Setup
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(save_dir / "scFoundation_model_evaluation.log")
    logger.info(f"Starting evaluation with performance metrics...")
    logger.info(f"Arguments: {vars(args)}")
    
    # 데이터 로드
    logger.info(f"Loading data from {args.data_dir}...")
    pert_data = PertData(args.data_dir)
    pert_data.load(data_path=os.path.join(args.data_dir, args.data_name))
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Gene names
    gene_ids_dataset = pert_data.adata.var["gene_name"].tolist()
    
    # Preprocessing (한 번만)
    logger.info("Creating reorder indices and gene mappings...")
    reorder_indices = create_reorder_indices(gene_ids_dataset, ref_gene_ids).to(device)
    gene_to_ref_idx = create_gene_to_ref_mapping(gene_ids_dataset, ref_gene_ids)
    logger.info("Preprocessing complete")
    
    # Control adata 준비
    ctrl_adata = pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"].copy()
    logger.info(f"Control cells: {ctrl_adata.n_obs}")
    
    # Encoder 한 번만 로드! (중요한 최적화)
    logger.info("Loading shared encoder...")
    base_model = ScFoundationPerturbModel(args.pretrain_model)
    shared_encoder = base_model.encoder
    shared_pert_emb = base_model.pert_emb
    hidden_dim = base_model.hidden_dim
    shared_encoder.to(device)
    shared_pert_emb.to(device)
    shared_encoder.eval()  # 항상 eval mode
    logger.info("Shared encoder loaded")
    
    # Seed별 evaluation
    all_metrics = []
    
    for seed in range(args.seed_start, args.seed_end + 1):
        seed_start = time.time()
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating Seed {seed}")
        logger.info(f"{'='*80}")
        
        # Data split
        pert_data.prepare_split(
            split="simulation",
            seed=seed,
            train_gene_set_size=0.75
        )
        pert_data.get_dataloader(
            batch_size=test_batch_size,
            test_batch_size=test_batch_size
        )
        
        # DataLoader 최적화!
        test_dataset = pert_data.dataloader['test_loader'].dataset
        test_loader = GeoDataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=4,  # 최적화!
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        logger.info(f"Test set: {len(test_loader)} batches (optimized DataLoader)")
        
        # Head만 로드! (Encoder 재사용)
        model_path = Path(args.model_dir) / f"Seed_{seed}_best_model.pt"
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            continue
        
        logger.info(f"Loading head weights from {model_path}...")
        
        # 새 모델 생성 (shared encoder 재사용)
        class OptimizedModel(nn.Module):
            def __init__(self, encoder, pert_emb, hidden_dim):
                super().__init__()
                self.encoder = encoder
                self.pert_emb = pert_emb
                self.head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
                self.hidden_dim = hidden_dim
            
            def forward(self, x_values, total_counts, pert_flags):
                x_query = torch.cat((x_values, total_counts.unsqueeze(1)), dim=1).to(dtype=torch.float32)
                enc = self.encoder(x_query)
                
                B, L_enc, _ = enc.shape
                L_flags = pert_flags.size(1)
                
                if L_flags < L_enc:
                    pad = torch.zeros((B, L_enc - L_flags), dtype=pert_flags.dtype, device=pert_flags.device)
                    pert_flags_aligned = torch.cat([pert_flags, pad], dim=1)
                elif L_flags > L_enc:
                    pert_flags_aligned = pert_flags[:, :L_enc]
                else:
                    pert_flags_aligned = pert_flags
                
                pert_emb_out = self.pert_emb(pert_flags_aligned.long())
                enc = enc + pert_emb_out
                pred = self.head(enc).squeeze(-1)
                return pred
        
        model = OptimizedModel(shared_encoder, shared_pert_emb, hidden_dim)
        
        # Head & pert_emb만 로드
        state_dict = torch.load(model_path, map_location=device)
        model.head.load_state_dict({
            k.replace('head.', ''): v 
            for k, v in state_dict.items() 
            if k.startswith('head.')
        })
        model.pert_emb.load_state_dict({
            k.replace('pert_emb.', ''): v 
            for k, v in state_dict.items() 
            if k.startswith('pert_emb.')
        })
        
        model.to(device)
        logger.info("Head loaded (encoder reused)")
        
        # Evaluation
        logger.info("Running evaluation...")
        eval_start = time.time()
        results = evaluate_model(
            model, test_loader, device, reorder_indices, gene_to_ref_idx, max_seq_len=args.max_seq_len, logger=logger, seed=seed,
        )
        eval_time = time.time() - eval_start
        logger.info(f"Evaluation took {eval_time:.2f}s")
        logger.info(f"Predictions shape: {results['pred'].shape}")
        logger.info(f"Unique perturbations: {len(np.unique(results['pert_cat']))}")
        
        # 모든 메트릭 한 번에 계산!
        logger.info("Computing ALL metrics (efficient)...")
        metrics_start = time.time()
        metrics = compute_all_metrics_efficient(
            results, ctrl_adata, non_zero_genes=False, return_raw=False
        )
        metrics_time = time.time() - metrics_start
        logger.info(f"Metrics computation took {metrics_time:.2f}s")
        
        metrics['seed'] = seed
        all_metrics.append(metrics)
        
        # Log metrics
        logger.info(f"\n{'='*60}")
        logger.info(f"Seed {seed} Results:")
        logger.info(f"{'='*60}")
        for key, value in metrics.items():
            if key != 'seed':
                logger.info(f"  {key}: {value:.4f}")
        
        seed_time = time.time() - seed_start
        logger.info(f"Seed {seed} total time: {seed_time:.2f}s")
        logger.info(f"{'='*60}\n")
        
        # Visualization
        if args.plot:
            logger.info("Creating plots...")
            plot_results(results['pred'], results['truth'], save_dir / "plots", seed)
            logger.info("Plots saved")
        
        # Save predictions
        if args.save_predictions:
            pred_dir = save_dir / "predictions" / f"seed_{seed}"
            pred_dir.mkdir(parents=True, exist_ok=True)
            
            np.save(pred_dir / "predictions.npy", results['pred'])
            np.save(pred_dir / "targets.npy", results['truth'])
            np.save(pred_dir / "pert_cats.npy", results['pert_cat'])
            logger.info(f"Predictions saved to {pred_dir}")
    
    # Summary statistics
    if all_metrics:
        logger.info(f"\n{'='*80}")
        logger.info(f"SUMMARY ACROSS ALL SEEDS")
        logger.info(f"{'='*80}")
        
        df_metrics = pd.DataFrame(all_metrics)
        
        # 평균 및 표준편차
        summary = df_metrics.drop('seed', axis=1).agg(['mean', 'std'])
        
        logger.info("\nMean ± Std:")
        for col in summary.columns:
            mean_val = summary.loc['mean', col]
            std_val = summary.loc['std', col]
            logger.info(f"  {col}: {mean_val:.4f} ± {std_val:.4f}")
        
        # CSV 저장
        df_metrics.to_csv(save_dir / f"Seed{args.seed_start}to{args.seed_end}_model_preformance.txt", index=False, sep='\t')
        summary.to_csv(save_dir / "metrics_summary_efficient.csv")
        logger.info(f"\nMetrics saved to {save_dir}")
        
        # Best seed
        best_seed = df_metrics.loc[df_metrics['pearson_de_delta'].idxmax(), 'seed']
        best_pearson_de_delta = df_metrics.loc[df_metrics['pearson_de_delta'].idxmax(), 'pearson_de_delta']
        logger.info(f"\nBest seed: {int(best_seed)} (Pearson DE delta: {best_pearson_de_delta:.4f})")
        
        total_time = time.time() - total_start
        logger.info(f"\n Total evaluation time: {total_time/60:.2f} minutes")
        logger.info(f"{'='*80}")
        logger.info(f" Evaluation complete!")
        logger.info(f"{'='*80}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="EFFICIENT evaluation with GEARS metrics"
    )
    
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--data_name', type=str, required=True)
    parser.add_argument('--pretrain_model', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--seed_start', type=int, default=1)
    parser.add_argument('--seed_end', type=int, default=10)
    parser.add_argument('--max_seq_len', type=int, default=2000)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--save_predictions', action='store_true')
    
    args = parser.parse_args()
    main(args)