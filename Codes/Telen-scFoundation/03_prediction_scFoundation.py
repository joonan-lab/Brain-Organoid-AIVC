"""
scFoundation Perturbation Prediction
- Chunk 타일링 방식 (max_seq_len 기준)
- Perturbation gene 항상 포함
- Training과 입력 형태 일관성 유지
- GPU memory 효율적
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from torch import nn
from anndata import AnnData
import logging

# =======================
# scFoundation / GEARS path
# =======================
sys.path.append("<ENV_ROOT>/scFoundation/model")
sys.path.insert(0, "<ENV_ROOT>/scFoundation/GEARS")

from gears import PertData
from modules.encoders import MAEAutobinencoder

warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)


# =======================
# OS gene list (19,264 genes)
# =======================
OS_GENE_TSV = "<DATA_ROOT>/scFoundation/preprocessing/OS_scRNA_gene_index.19264.tsv"
os_gene_df = pd.read_csv(OS_GENE_TSV, sep="\t")
ref_gene_ids: List[str] = os_gene_df["gene_name"].tolist()
N_GENE = len(ref_gene_ids)
print(f"[INFO] Loaded OS gene list: {N_GENE} genes")

# =======================
# Utils
# =======================
def setup_logger(log_dir: Path, prefix: str, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("scfoundation_pred")

    # 중복 핸들러 방지
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{prefix}_prediction.log"

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)  # 파일에는 최대한 많이 남기기

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logging to {log_file}")
    return logger


def create_reorder_indices(
    gene_ids_dataset: List[str],
    ref_gene_ids: List[str],
) -> torch.Tensor:
    """Dataset gene order → ref_gene_ids order mapping"""
    ds_gene_to_idx: Dict[str, int] = {g: i for i, g in enumerate(gene_ids_dataset)}
    indices = [ds_gene_to_idx.get(ref_g, -1) for ref_g in ref_gene_ids]
    return torch.tensor(indices, dtype=torch.long)


def reorder_matrix_fast(
    mat: torch.Tensor,
    reorder_indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Reorder matrix columns to ref_gene_ids order"""
    B = mat.shape[0]
    G = len(reorder_indices)
    out = torch.zeros((B, G), dtype=mat.dtype, device=device)
    
    valid_mask = reorder_indices >= 0
    valid_indices = reorder_indices[valid_mask]
    
    if len(valid_indices) > 0:
        out[:, valid_mask] = mat[:, valid_indices]
    
    return out


# =======================
# Model Classes
# =======================
class ScFoundationPerturbModel(nn.Module):
    """Base model for loading pretrained encoder"""
    
    def __init__(
        self,
        ckpt_path: str,
        bin_set: str = "autobin_resolution_append",
        highres: int = 0,
    ):
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


class OptimizedModel(nn.Module):
    """Fine-tuned model with seed-specific head & pert_emb"""
    
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
        """
        Args:
            x_values: (B, L) gene expression
            total_counts: (B,) total counts
            pert_flags: (B, L) perturbation flags
        Returns:
            (B, L) predicted expression
        """
        x_query = torch.cat(
            (x_values, total_counts.unsqueeze(1)), dim=1
        ).to(dtype=torch.float32)
        
        enc = self.encoder(x_query)  # (B, L_enc, H)
        B, L_enc, _ = enc.shape
        L_flags = pert_flags.size(1)
        
        # Align pert_flags to encoder length
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
        
        pert_emb_out = self.pert_emb(pert_flags_aligned.long())
        enc = enc + pert_emb_out
        pred = self.head(enc).squeeze(-1)  # (B, L_enc)
        
        if pred.size(1) > L_flags:
            pred = pred[:, :L_flags]
        
        return pred  # (B, L_flags)


# =======================
# Load Trained Model
# =======================
def load_trained_model(
    pretrain_model_path: str,
    model_dir: Path,
    device: torch.device,
    seed: int = None,
) -> OptimizedModel:
    """
    Load best fine-tuned model
    
    Args:
        pretrain_model_path: scFoundation pretrained checkpoint
        model_dir: Directory with Seed_*_best_model.pt
        device: cuda or cpu
        seed: Specific seed to use (None = auto-select best)
    
    Returns:
        OptimizedModel with loaded weights
    """
    model_dir = Path(model_dir)
    logger = logging.getLogger("scfoundation_pred")
    
    # Auto-select best seed by pearson_de_delta
    if seed is None:
        metrics_path = model_dir / "Seed1to10_model_performance.txt"
        if metrics_path.exists():
            df = pd.read_csv(metrics_path, sep="\t")
            if "pearson_de_delta" in df.columns:
                best_row = df["pearson_de_delta"].idxmax()
                seed = int(df.loc[best_row, "seed"])
                logger.info(f"[INFO] Auto-selected best seed: {seed} (pearson_de_delta={df.loc[best_row, 'pearson_de_delta']:.4f})")
            else:
                seed = 1
                logger.warning("[WARN] pearson_de_delta not found, using seed=1")
        else:
            seed = 1
            logger.warning("[WARN] Seed1to10_model_performance.txt not found, using seed=1")
    else:
        logger.info(f"[INFO] Using specified seed: {seed}")
    
    ckpt_path = model_dir / f"Seed_{seed}_best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    # Load base model (pretrained encoder)
    logger.info(f"[INFO] Loading pretrained encoder from {pretrain_model_path}...")
    base_model = ScFoundationPerturbModel(pretrain_model_path)
    encoder = base_model.encoder.to(device)
    pert_emb = base_model.pert_emb.to(device)
    hidden_dim = base_model.hidden_dim
    
    model = OptimizedModel(encoder, pert_emb, hidden_dim).to(device)
    
    # Load fine-tuned head & pert_emb
    logger.info(f"[INFO] Loading fine-tuned weights from {ckpt_path}...")
    state_dict = torch.load(ckpt_path, map_location=device)
    
    head_state = {
        k.replace("head.", ""): v
        for k, v in state_dict.items()
        if k.startswith("head.")
    }
    model.head.load_state_dict(head_state)
    
    pert_state = {
        k.replace("pert_emb.", ""): v
        for k, v in state_dict.items()
        if k.startswith("pert_emb.")
    }
    model.pert_emb.load_state_dict(pert_state)
    
    model.eval()
    logger.info(f"[INFO] Model loaded successfully (seed={seed})")
    
    return model


# =======================
# Core Prediction Function
# =======================
def predict(
    model: OptimizedModel,
    gene_idx_ref: int,
    ctrl_expr_ref: torch.Tensor,      # (pool_size, N_GENE)
    total_counts_full: torch.Tensor,  # (pool_size,)
    max_seq_len: int,
    device: torch.device,
    pred_batch_size: int,
    shuffle_genes: bool = True,
) -> np.ndarray:
    """
    단일 perturbation gene에 대한 full (pool_size x N_GENE) 예측.

    - 항상 perturbation gene을 포함한 상태로
    - 나머지 gene들을 chunk로 쪼개서
    - 모든 gene을 정확히 한 번씩 예측.
    """
    model.eval()
    pool_size, num_genes = ctrl_expr_ref.shape

    assert num_genes == N_GENE

    if max_seq_len is None or max_seq_len <= 0:
        max_seq_len = num_genes
    if max_seq_len < 2:
        raise ValueError("max_seq_len must be >= 2")

    # perturbation gene 제외한 나머지 gene index
    all_idx = torch.arange(num_genes, device=device)
    other_idx = all_idx[all_idx != gene_idx_ref]

    # 한 번 퍼뮤테이션 (optional, reproducible하게 하려면 seed 고정)
    if shuffle_genes:
        perm = torch.randperm(len(other_idx), device=device)
        other_idx = other_idx[perm]

    # chunk 하나에서 쓸 타겟 gene 수
    chunk_target_size = max_seq_len - 1

    # 결과 저장용 full matrix
    preds_full = torch.zeros((pool_size, num_genes), dtype=torch.float32, device=device)

    # ---- gene chunk loop ----
    for chunk_start in range(0, len(other_idx), chunk_target_size):
        chunk_targets = other_idx[chunk_start:chunk_start + chunk_target_size]  # (<= 1999,)

        gene_idx_eval = torch.cat([
            torch.tensor([gene_idx_ref], device=device),
            chunk_targets
        ])  # (L_eval,)  =  [pert gene] + [이 chunk의 gene들]

        L_eval = gene_idx_eval.size(0)

        # ---- cell batch loop ----
        for start in range(0, pool_size, pred_batch_size):
            end = min(pool_size, start + pred_batch_size)
            bsz = end - start

            batch_expr_eval = ctrl_expr_ref[start:end][:, gene_idx_eval]    # (bsz, L_eval)
            batch_counts    = total_counts_full[start:end]                  # (bsz,)

            pert_flags = torch.zeros((bsz, L_eval), dtype=torch.long, device=device)
            pert_flags[:, 0] = 1   # 첫 번째 위치 = perturbation gene

            with torch.no_grad():
                batch_preds = model(
                    batch_expr_eval,      # (bsz, L_eval)
                    batch_counts,         # (bsz,)
                    pert_flags,           # (bsz, L_eval)
                )  # -> (bsz, L_eval)

            # full space에 쓰기
            preds_full[start:end][:, gene_idx_eval] = batch_preds

    # 필요하면 여기서 perturbation gene column을 원래 ctrl 값으로 덮어쓸 수도 있음
    # preds_full[:, gene_idx_ref] = ctrl_expr_ref[:, gene_idx_ref]  # optional

    return preds_full.cpu().numpy()


# =======================
# Main Function
# =======================
def main(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # output_dir가 이미 존재하니까 여기서 logger 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.gene_start is not None and args.gene_end is not None:
        prefix = f"gpu{args.gpu_id}_gene{args.gene_start}_{args.gene_end}_pred.log"
    else:
        prefix = "Telen-scFoundation_gene"

    logger = setup_logger(output_dir, prefix, verbose=args.verbose)

    logger.info(f"\n{'='*80}")
    logger.info(f"scFoundation Perturbation Prediction")
    logger.info(f"{'='*80}")
    logger.info(f"Device: {device}")
    logger.info(f"Parameters:")
    logger.info(f"  - max_seq_len: {args.max_seq_len}")
    logger.info(f"  - pred_batch_size: {args.pred_batch_size}")
    logger.info(f"  - pool_size: {args.pool_size}")
    logger.info(f"{'='*80}\n")
    
    # -----------------------
    # Load Data
    # -----------------------
    logger.info("[1/5] Loading PertData...")
    pert_data = PertData(args.data_dir)
    pert_data.load(data_path=os.path.join(args.data_dir, args.data_name))
    
    adata = pert_data.adata
    gene_ids_dataset = adata.var["gene_name"].tolist()
    logger.info(f"  Loaded {adata.n_obs} cells × {adata.n_vars} genes")
    
    # -----------------------
    # Control Cell Pool
    # -----------------------
    logger.info("\n[2/5] Preparing control cell pool...")
    ctrl_adata_full = adata[adata.obs["condition"] == "ctrl"].copy()
    n_ctrl_total = ctrl_adata_full.n_obs
    logger.info(f"  Total control cells: {n_ctrl_total}")
    
    pool_size = min(args.pool_size, n_ctrl_total)
    np.random.seed(42)
    if n_ctrl_total > pool_size:
        idx = np.random.choice(n_ctrl_total, size=pool_size, replace=False)
        ctrl_adata = ctrl_adata_full[idx].copy()
    else:
        ctrl_adata = ctrl_adata_full
    
    pool_size = ctrl_adata.n_obs
    logger.info(f"  Using {pool_size} control cells")
    
    # Extract expression matrix
    X_ctrl = ctrl_adata.X
    if not isinstance(X_ctrl, np.ndarray):
        X_ctrl = X_ctrl.toarray()
    X_ctrl = X_ctrl.astype(np.float32)
    
    # -----------------------
    # Reorder to ref_gene_ids
    # -----------------------
    logger.info("\n[3/5] Reordering genes to OS reference order...")
    reorder_indices = create_reorder_indices(gene_ids_dataset, ref_gene_ids).to(device)
    
    ctrl_expr_torch = torch.tensor(X_ctrl, dtype=torch.float32, device=device)
    ctrl_expr_ref = reorder_matrix_fast(
        ctrl_expr_torch, reorder_indices, device
    )  # (pool_size, 19264)
    
    total_counts_full = torch.log10(ctrl_expr_ref.sum(dim=1) + 1.0)  # (pool_size,)
    logger.info(f"  Control pool ready: {ctrl_expr_ref.shape}")
    
    # -----------------------
    # Load Model
    # -----------------------
    logger.info("\n[4/5] Loading fine-tuned model...")
    model = load_trained_model(
        pretrain_model_path=args.pretrain_model,
        model_dir=Path(args.model_dir),
        device=device,
        seed=args.seed,
    )
    
    # -----------------------
    # Prepare Prediction Genes
    # -----------------------
    logger.info("\n[5/5] Preparing prediction gene list...")
    
    # Check already saved files
    saved_files = [f for f in os.listdir(output_dir) if f.endswith("_predicted.h5ad")]
    saved_names = [f.rsplit("_predicted.h5ad", 1)[0] for f in saved_files]
    logger.info(f"  Found {len(saved_names)} already predicted genes")
    
    # Build prediction gene list
    if args.pred_gene_list is not None:
        with open(args.pred_gene_list, "r") as f:
            pred_genes = f.read().splitlines()
        pred_genes = [g for g in pred_genes if g in ref_gene_ids]
        logger.info(f"  Loaded {len(pred_genes)} genes from {args.pred_gene_list}")
    else:
        pred_genes = ref_gene_ids.copy()
        logger.info(f"  Using all {len(pred_genes)} genes")
    
    # Apply gene range slicing
    if args.gene_start is not None or args.gene_end is not None:
        pred_genes = pred_genes[args.gene_start:args.gene_end]
        logger.info(f"  Sliced to genes [{args.gene_start}:{args.gene_end}] = {len(pred_genes)} genes")
    
    # Exclude already saved genes
    pred_genes = sorted(list(set(pred_genes) - set(saved_names)))
    logger.info(f"  Remaining genes to predict: {len(pred_genes)}")
    
    if len(pred_genes) == 0:
        logger.info("\n[DONE] All genes already predicted! Exiting.")
        return
    
    # Gene name → index mapping
    ref_name_to_idx = {g: i for i, g in enumerate(ref_gene_ids)}
    
    # -----------------------
    # Prediction Loop
    # -----------------------
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting Prediction")
    logger.info(f"{'='*80}\n")
    
    total_start = time.time()
    
    for idx_g, genename in enumerate(pred_genes, start=1):
        if genename not in ref_name_to_idx:
            logger.warning(f"[WARN] {genename} not in ref_gene_ids. Skip.")
            continue
        
        gene_idx_ref = ref_name_to_idx[genename]
        logger.info(
            f"[{idx_g}/{len(pred_genes)}] Predicting {genename} "
            f"(idx={gene_idx_ref})...",
        )
        
        start_time = time.time()
        
        # Chunk-based prediction
        preds = predict(
            model=model,
            gene_idx_ref=gene_idx_ref,
            ctrl_expr_ref=ctrl_expr_ref,
            total_counts_full=total_counts_full,
            max_seq_len=args.max_seq_len,
            device=device,
            pred_batch_size=args.pred_batch_size,
            shuffle_genes=False,  # deterministic 원하면 False, 살짝 random 섞고 싶으면 True
        )  # → (pool_size, 19264)
        
        # Save as AnnData
        adata_pred = AnnData(preds)
        adata_pred.obs_names = [f"Cell_{i}" for i in range(preds.shape[0])]
        adata_pred.var_names = ref_gene_ids
        adata_pred.obs["perturbation"] = genename
        
        out_path = output_dir / f"{genename}_predicted.h5ad"
        adata_pred.write(out_path, compression="gzip")
        
        elapsed = time.time() - start_time
        logger.info(
            f"  ✓ Done in {elapsed:.2f}s → {out_path.name}\n"
        )
    
    # -----------------------
    # Summary
    # -----------------------
    total_time = time.time() - total_start
    logger.info(f"\n{'='*80}")
    logger.info(f"Prediction Complete!")
    logger.info(f"{'='*80}")
    logger.info(f"Total genes predicted: {len(pred_genes)}")
    logger.info(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    logger.info(f"Average time per gene: {total_time/len(pred_genes):.2f} seconds")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'='*80}\n")


# =======================
# CLI
# =======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="scFoundation Perturbation Prediction (Chunk Method)")
    
    # Data
    parser.add_argument( "--data_dir", type=str, required=True, help="Path to data directory",)
    parser.add_argument( "--data_name", type=str, required=True, help="Name of dataset",)
    
    # Model
    parser.add_argument( "--pretrain_model", type=str, required=True, help="Path to scFoundation pretrained checkpoint",)
    parser.add_argument( "--model_dir", type=str, required=True, help="Directory with Seed_*_best_model.pt files",)
    parser.add_argument( "--seed", type=int, default=None, help="Seed to use (None = auto-select best)",)
    
    # Output
    parser.add_argument( "--output_dir", type=str, required=True, help="Directory to save predicted h5ad files",)
    
    # Prediction settings
    parser.add_argument( "--pool_size", type=int, default=1000, help="Number of control cells (default: 1000)",)
    parser.add_argument( "--max_seq_len", type=int, default=2000, help="Max genes per forward pass (default: 5000)",)
    parser.add_argument( "--pred_batch_size", type=int, default=256, help="Batch size for cells (default: 256)",)
    
    # Gene selection
    parser.add_argument( "--pred_gene_list", type=str, default=None, help="Path to txt file with gene names (one per line)",)
    parser.add_argument( "--gene_start", type=int, default=None, help="Start index for gene slicing",)
    parser.add_argument( "--gene_end", type=int, default=None, help="End index for gene slicing",)
    
    # System
    parser.add_argument( "--gpu_id", type=int, default=0, help="CUDA GPU id (default: 0)",)
    parser.add_argument( "--verbose", action="store_true", help="Print detailed progress",)
    
    args = parser.parse_args()
    main(args)
