# =======================
# Imports
# =======================

import argparse
import os
import sys
import time
import copy
import warnings
from pathlib import Path
import scanpy as sc

import numpy as np
import pandas as pd
import torch
import logging
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader as GeoDataLoader

# scFoundation 유틸들
sys.path.append("<ENV_ROOT>/scFoundation/model")
sys.path.insert(0, "<ENV_ROOT>/scFoundation/GEARS")

from gears import PertData
from modules.encoders import MAEAutobinencoder

warnings.filterwarnings("ignore")

# =======================
# Set hyperparameters
# =======================
lr = 1e-4
batch_size = 128
test_batch_size = 128
epochs = 15
schedule_interval = 1
early_stop = 5
log_interval = 100
amp = True
accumulation_steps = 1 # Gradient accumulation
# DataLoader 최적화
num_workers = 2  # Train용
num_workers_val = 1  # Validation용


# =======================
# OS gene list - scFoundation used
# =======================
OS_GENE_TSV = "<DATA_ROOT>/scFoundation/preprocessing/OS_scRNA_gene_index.19264.tsv"
os_gene_df = pd.read_csv(OS_GENE_TSV, sep="\t")
ref_gene_ids = os_gene_df["gene_name"].tolist()


# =======================
# Logger
# =======================
def setup_logger(log_file_path: Path):
    logger = logging.getLogger("scfoundation_train_logger")

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
def masked_mse_loss(pred: torch.Tensor,
                    target: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """
    pred, target: (B, G)
    mask: (B, G)  - True/1 인 위치만 loss 계산
    """
    diff2 = (pred - target) ** 2
    diff2 = diff2 * mask.float()
    denom = mask.float().sum()
    if denom.item() == 0:
        return torch.tensor(0.0, device=pred.device)
    return diff2.sum() / denom

def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN gradient in {name}")
                return False
            if torch.isinf(param.grad).any():
                print(f"Inf gradient in {name}")
                return False
    return True

# OPTIMIZED: 미리 계산된 indices로 빠른 reorder
def create_reorder_indices(gene_ids_dataset: list, ref_gene_ids: list):
    """
    gene_ids_dataset -> ref_gene_ids 매핑
    Returns: tensor of indices
    """
    ds_gene_to_idx = {g: i for i, g in enumerate(gene_ids_dataset)}
    
    indices = []
    for ref_gene in ref_gene_ids:
        idx = ds_gene_to_idx.get(ref_gene, -1)
        indices.append(idx)
    
    return torch.tensor(indices, dtype=torch.long)

def reorder_matrix_fast(mat: torch.Tensor, 
                        reorder_indices: torch.Tensor,
                        device: torch.device) -> torch.Tensor:

    B = mat.shape[0]
    G = len(reorder_indices)
    out = torch.zeros((B, G), dtype=mat.dtype, device=device)
    
    # Valid indices만 처리
    valid_mask = reorder_indices >= 0
    valid_indices = reorder_indices[valid_mask]
    
    if len(valid_indices) > 0:
        out[:, valid_mask] = mat[:, valid_indices]
    
    return out

# OPTIMIZED: pert_flags 생성 최적화
def create_gene_to_ref_mapping(gene_ids_dataset: list, ref_gene_ids: list):
    """
    dataset gene index -> ref gene index 매핑
    """
    ref_gene_to_idx = {g: i for i, g in enumerate(ref_gene_ids)}
    
    gene_to_ref_idx = {}
    for i, gene in enumerate(gene_ids_dataset):
        if gene in ref_gene_to_idx:
            gene_to_ref_idx[i] = ref_gene_to_idx[gene]
    
    return gene_to_ref_idx


def create_pert_flags_fast(batch_data, batch_size, gene_to_ref_idx: dict, device):

    pert_flags = torch.zeros((batch_size, 19264), dtype=torch.long, device=device)
    
    for idx, pert_list in enumerate(batch_data.pert_idx):
        if len(pert_list) > 0 and pert_list[0] != -1:
            # Vectorized로 처리
            valid_ref_indices = [gene_to_ref_idx[gene_idx] 
                                for gene_idx in pert_list 
                                if gene_idx in gene_to_ref_idx]
            
            if valid_ref_indices:
                pert_flags[idx, valid_ref_indices] = 1
    
    return pert_flags


# =======================
# scFoundation Model
# =======================
class ScFoundationPerturbModel(nn.Module):
    """
    scFoundation-based perturbation prediction model -> scFoundation MAEAutobinencoder + perturbation embedding + regression head
    """
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

        # perturbation binary embedding (0/1)
        self.pert_emb = nn.Embedding(2, hidden_dim)
        print(f"Perturbation embedding initialized: 2 tokens x {hidden_dim} dim")

        # 작은 회귀 head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.hidden_dim = hidden_dim

    def forward(self, x_values: torch.Tensor, total_counts: torch.Tensor, 
                pert_flags: torch.Tensor) -> torch.Tensor:
        """
        x_values: (B, L)  - OS gene 순서 또는 subset
        total_counts: (B,) - log10(count + 1)
        pert_flags: (B, L) - 0/1
        """
        # scFoundation encoder input: gene expression + total count 한 칼럼
        x_query = torch.cat(
            (x_values, total_counts.unsqueeze(1)),      # (B, L+1)
            dim=1,
        ).to(dtype=torch.float32)

        # scFoundation encoder
        enc = self.encoder(x_query)   # (B, L_enc, H)  보통 L_enc = L + 2 정도

        # ---- seq length 맞추기 ----
        # enc_len 기준으로 pert_flags 길이 조정
        B, L_enc, _ = enc.shape
        L_flags = pert_flags.size(1)

        if L_flags < L_enc:
            # 부족한 토큰 위치는 perturbation 없음(0)으로 패딩
            pad = torch.zeros(
                (B, L_enc - L_flags),
                dtype=pert_flags.dtype,
                device=pert_flags.device,
            )
            pert_flags_aligned = torch.cat([pert_flags, pad], dim=1)
        elif L_flags > L_enc:
            # 혹시라도 길이가 더 길다면 앞부분만 사용
            pert_flags_aligned = pert_flags[:, :L_enc]
        else:
            pert_flags_aligned = pert_flags

        pert_emb = self.pert_emb(pert_flags_aligned.long())  # (B, L_enc, H)

        enc = enc + pert_emb
        pred = self.head(enc).squeeze(-1)  # (B, L_enc)

        return pred


# =======================
# Train / Eval funtions
# =======================
def train(
    model: nn.Module,
    train_loader,
    device,
    reorder_indices: torch.Tensor,
    gene_to_ref_idx: dict,
    scaler,
    optimizer,
    scheduler,
    logger,
    epoch: int,
    args
):
    model.train()

    total_loss, total_mse = 0.0, 0.0
    start_time = time.time()
    num_batches = len(train_loader)

    # Gradient accumulation
    optimizer.zero_grad()

    for step, batch_data in enumerate(train_loader):
        batch_size_actual = len(batch_data.y)
        batch_data.to(device)

        x: torch.Tensor = batch_data.x
        n_genes = int(x.shape[0] / batch_size_actual)

        ori_gene_values = x[:, 0].view(batch_size_actual, n_genes)
        target_gene_values = batch_data.y

        # OS gene 순서로 reorder
        x_query = reorder_matrix_fast(ori_gene_values, reorder_indices, device)
        target_values = reorder_matrix_fast(target_gene_values, reorder_indices, device)

        # perturbation flags (OS gene 기준)
        pert_flags = create_pert_flags_fast(batch_data, batch_size_actual, 
                                            gene_to_ref_idx, device)

        # max_seq_len 적용 (gene subset sampling)
        input_gene_size = x_query.size(1)
        gene_idx = torch.arange(input_gene_size, dtype=torch.long, device=device)
        if input_gene_size > args.max_seq_len:
            # Random subset for each batch (like scGPT)
            perm = torch.randperm(input_gene_size, device=device)
            gene_idx = perm[:args.max_seq_len]

        x_query = x_query[:, gene_idx]          # (B, max_seq_len)
        target_values = target_values[:, gene_idx]
        pert_flags = pert_flags[:, gene_idx]

        # total_counts는 원래 input 전체에서 계산
        total_counts = torch.log10(ori_gene_values.sum(dim=1) + 1).to(device)
        
        with torch.cuda.amp.autocast(enabled=amp):
            output = model(x_query, total_counts, pert_flags)

            if output.size(1) > target_values.size(1):
                output_use = output[:, : target_values.size(1)]
            else:
                output_use = output     

            if torch.isnan(output).any() or torch.isinf(output).any():
                print("Model output contains NaN or Inf! Skip batch.")
                continue

            masked_positions = torch.ones_like(target_values, dtype=torch.bool)
            loss = masked_mse_loss(output_use, target_values, masked_positions)
            
            # Gradient accumulation
            loss = loss / accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                print("Loss is NaN or Inf! Skip batch.")
                continue

        scaler.scale(loss).backward()

        # Gradient accumulation: accumulation_steps마다 update
        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if check_gradients(model):
                scaler.step(optimizer)
            else:
                print("Bad gradients, skipping optimizer step")
            
            optimizer.zero_grad()
            scaler.update()

        total_loss += (loss.item() * accumulation_steps)
        total_mse += (loss.item() * accumulation_steps)

        if step % log_interval == 0 and step > 0:
            lr_curr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            logger.info(
                f"| epoch {epoch:3d} | {step:4d}/{num_batches:4d} batches | "
                f"lr {lr_curr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.4f} | mse {cur_mse:5.4f} | "
                f"eff_bs {batch_size * accumulation_steps}"
            )
            total_loss = 0.0
            total_mse = 0.0
            start_time = time.time()

def eval(
    model: nn.Module,
    val_loader,
    device,
    reorder_indices: torch.Tensor,
    gene_to_ref_idx: dict,
    args
):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_data in val_loader:
            batch_size_actual = len(batch_data.y)
            batch_data.to(device)
            x: torch.Tensor = batch_data.x
            n_genes = int(x.shape[0] / batch_size_actual)

            ori_gene_values = x[:, 0].view(batch_size_actual, n_genes)
            target_gene_values = batch_data.y

            x_query = reorder_matrix_fast(ori_gene_values, reorder_indices, device)
            target_values = reorder_matrix_fast(target_gene_values, reorder_indices, device)

            pert_flags = create_pert_flags_fast(batch_data, 
                                                batch_size_actual,
                                                gene_to_ref_idx, 
                                                device)

            # max_seq_len 적용 (validation도 train과 동일한 gene subset 전략)
            input_gene_size = x_query.size(1)
            gene_idx = torch.arange(input_gene_size, dtype=torch.long, device=device)
            if input_gene_size > args.max_seq_len:
                perm = torch.randperm(input_gene_size, device=device)
                gene_idx = perm[:args.max_seq_len]

            x_query = x_query[:, gene_idx]
            target_values = target_values[:, gene_idx]
            pert_flags = pert_flags[:, gene_idx]

            total_counts = torch.log10(ori_gene_values.sum(dim=1) + 1).to(device)

            with torch.cuda.amp.autocast(enabled=amp):
                output = model(x_query, total_counts, pert_flags)

                if output.size(1) > target_values.size(1):
                    output_use = output[:, : target_values.size(1)]
                else:
                    output_use = output

                masked_positions = torch.ones_like(target_values, dtype=torch.bool)
                loss = masked_mse_loss(output_use, target_values, masked_positions)

            total_loss += loss.item()

    return total_loss / len(val_loader)


# =======================
# Main
# =======================
def main(parser):
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving to {save_dir}")

    logger = setup_logger(save_dir / "test_training.log")
    logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Arguments: {vars(args)}")

    # GPU 정보
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Accumulation steps: {accumulation_steps}")
    logger.info(f"Effective batch size: {batch_size * accumulation_steps}")
    logger.info(f"Num workers: {num_workers}")
    logger.info(f"Seed range: {args.seed_start} to {args.seed_end}")

    # Data load
    pert_data = PertData(args.data_dir)

    if args.preprocess:
        logger.info("Running preprocessing...")
        adata = sc.read_h5ad(os.path.join(args.data_dir, args.data_name + '.h5ad'))
        adata.uns['log1p'] = {}
        adata.uns['log1p']['base'] = None
        pert_data.new_data_process(dataset_name=args.data_name, adata=adata)
        logger.info("Preprocessing finished.")
    else:
        logger.info(f"Loading data from {args.data_dir}")
        pert_data.load(data_path=os.path.join(args.data_dir, args.data_name))

    # Seed loop
    for seed in range(args.seed_start, args.seed_end + 1):
        start_time = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"SEED {seed} - START")
        logger.info(f"{'='*60}")

        # Data split
        pert_data.prepare_split(
            split="simulation", 
            seed=seed, 
            train_gene_set_size=0.75
        )
        pert_data.get_dataloader(
            batch_size=batch_size, 
            test_batch_size=test_batch_size
        )

        genes = pert_data.adata.var["gene_name"].tolist()
        gene_ids_dataset = genes

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        logger.info("Creating reorder indices and gene mappings...")
        reorder_indices = create_reorder_indices(gene_ids_dataset, ref_gene_ids).to(device)
        gene_to_ref_idx = create_gene_to_ref_mapping(gene_ids_dataset, ref_gene_ids)
        logger.info("Preprocessing complete")

        # Load train/validation data    
        train_loader = pert_data.dataloader['train_loader']
        val_loader = pert_data.dataloader['val_loader']
        
        try:
            train_dataset = train_loader.dataset
            val_dataset = val_loader.dataset
            
            logger.info("Creating optimized DataLoaders with num_workers...")
            train_loader = GeoDataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True if num_workers > 0 else False
            )
            val_loader = GeoDataLoader(
                val_dataset,
                batch_size=test_batch_size,
                shuffle=False,
                num_workers=num_workers_val,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True if num_workers_val > 0 else False
            )
            logger.info(f"Optimized DataLoaders created: train={len(train_loader)} batches, val={len(val_loader)} batches")
        except Exception as e:
            # Failed -> using GEARS dataloader
            logger.warning(f"Failed to create optimized DataLoaders: {e}")
            logger.warning("Using GEARS default DataLoaders (without num_workers optimization)")
            train_loader = pert_data.dataloader['train_loader']
            val_loader = pert_data.dataloader['val_loader']
            logger.info(f"Using default DataLoaders: train={len(train_loader)} batches, val={len(val_loader)} batches")

        # Model initialization
        logger.info("Initializing model...")
        model = ScFoundationPerturbModel(args.pretrain_model)
        model.to(device)

        # Freeze setting
        if args.finetune_method == "frozen":
            logger.info("Freezing encoder parameters...")
            for p in model.encoder.parameters():
                p.requires_grad = False
            logger.info("Encoder frozen")

        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"MODEL PARAMETERS")
        logger.info(f"{'='*60}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        if args.finetune_method == 'frozen':
            encoder_params = sum(p.numel() for p in model.encoder.parameters())
            head_params = sum(p.numel() for p in model.head.parameters())
            pert_emb_params = sum(p.numel() for p in model.pert_emb.parameters())
            
            logger.info(f"\nParameter breakdown:")
            logger.info(f"  - Encoder: {encoder_params:,} (frozen)")
            logger.info(f"  - Head: {head_params:,} (trainable)")
            logger.info(f"  - Pert_emb: {pert_emb_params:,} (trainable)")
            
            expected_trainable = head_params + pert_emb_params
            if abs(trainable_params - expected_trainable) > 10:
                logger.warning(f"Trainable params mismatch!")
            else:
                logger.info(f"Freeze verified: {trainable_params:,} trainable params")
        
        logger.info(f"{'='*60}\n")

        # Optimizer for model training
        if args.finetune_method == 'frozen':
            trainable_params_list = [p for p in model.parameters() if p.requires_grad]
            optimizer = Adam(trainable_params_list, lr=lr)
            logger.info("Optimizer: Adam (trainable params only)")
        elif args.finetune_method == 'finetune_lr_1':
            encoder_params = []
            other_params = []
            for n, p in model.named_parameters():
                if ".encoder." in n:
                    encoder_params.append(p)
                else:
                    other_params.append(p)
            optimizer = Adam([
                {"params": other_params, "lr": lr},
                {"params": encoder_params, "lr": lr * 0.1},
            ])
            logger.info(f"Optimizer: Adam (encoder lr={lr*0.1}, others lr={lr})")
        else:
            optimizer = Adam(model.parameters(), lr=lr)
            logger.info(f"Optimizer: Adam (all params, lr={lr})")

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=schedule_interval, gamma=0.9
        )
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

        best_val_loss = float("inf")
        best_model = None
        patience = 0

        # Training loop
        logger.info(f"\nStarting training for {epochs} epochs...")
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            train(
                model, 
                train_loader, 
                device, 
                reorder_indices, 
                gene_to_ref_idx,
                scaler, 
                optimizer, 
                scheduler, 
                logger, 
                epoch, 
                args
            )

            val_loss = eval(model, 
                            val_loader, 
                            device, 
                            reorder_indices, 
                            gene_to_ref_idx, 
                            args
                       )

            elapsed = time.time() - epoch_start
            logger.info(
                f"| epoch {epoch:3d} (seed {seed}) | "
                f"time: {elapsed:5.2f}s | val_loss: {val_loss:5.4f} |"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)
                logger.info(f"Best model updated! val_loss={best_val_loss:.4f}")
                patience = 0
            else:
                patience += 1
                if patience >= early_stop:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            scheduler.step()

        # Save best model
        if best_model is not None:
            model_path = save_dir / f"Seed_{seed}_best_model.pt"
            torch.save(best_model.state_dict(), model_path)
            logger.info(f"💾 Best model saved: {model_path}")

        total_time = time.time() - start_time
        logger.info(f"\nSeed {seed} finished in {total_time/60:.1f} minutes")
        logger.info(f"Best val_loss: {best_val_loss:.4f}")
        logger.info(f"{'='*60}\n")

    logger.info("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fully optimized scFoundation training for perturbation prediction with binary condition token")

    parser.add_argument('--data_dir', type=str, required=True, help='Path for perturbation dataset (GEARS PertData name)')
    parser.add_argument('--data_name', type=str, required=True, help='Name for perturbation dataset (GEARS PertData name)')
    parser.add_argument('--pretrain_model', type=str, required=True, help='Path to scFoundation checkpoint (e.g., /path/to/models.ckpt)')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save trained models (Seed_*_best_model.pt)')
    parser.add_argument('--preprocess', action='store_true', help="If set, run PertData preprocessing with new_data_process")
    parser.add_argument('--finetune_method', type=str, default='frozen', choices=['frozen', 'finetune', 'finetune_lr_1'], 
                        help='Finetune method: frozen (default), finetune, or finetune_lr_1')
    parser.add_argument('--seed_start', type=int, default=1)
    parser.add_argument('--seed_end', type=int, default=10)
    parser.add_argument('--max_seq_len', type=int, default=2000)

    main(parser)