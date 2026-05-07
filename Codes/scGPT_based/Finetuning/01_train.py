import argparse

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--adata_path', type=str, required=False, help='Path to .h5ad file')
parser.add_argument('--pert_name', type=str, required=True, help='Name for perturbation dataset')
parser.add_argument('--pretrain_model', type=str, required=True, help='Directory of pretrained model')
parser.add_argument('--save_dir', type=str, required=True, help='Directory to save trained models')
parser.add_argument('--preprocess', action='store_true', help="If set, run PertData preprocessing")
args = parser.parse_args()


import scanpy as sc
from gears import PertData, GEARS
if args.preprocess:
    print("Running PertData preprocessing...")
    new_adata2 = sc.read_h5ad(args.adata_path)
    pert_data = PertData('./data')  # or any other save path
    pert_data.new_data_process(dataset_name=args.pert_name, adata=new_adata2)
else:
    print("Skipping preprocessing step (--preprocess not set)")


import os
import json
import sys
import time
import copy
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Union, Optional
import warnings

import torch
import numpy as np
import matplotlib
from torch import nn
from torch.nn import functional as F
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from torch_geometric.loader import DataLoader
from gears import PertData, GEARS
from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis
from gears.utils import create_cell_graph_dataset_for_prediction

sys.path.insert(0, "../")

import scgpt as scg
from scgpt.model import TransformerGenerator
from scgpt.loss import (
    masked_mse_loss,
    criterion_neg_log_bernoulli,
    masked_relative_error,
)
from scgpt.tokenizer import tokenize_batch, pad_batch, tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import compute_perturbation_metrics, set_seed, map_raw_id_to_vocab_id

matplotlib.rcParams["savefig.transparent"] = False
warnings.filterwarnings("ignore")

set_seed(42)


# settings for data prcocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
pad_value = 0  # for padding values
pert_pad_id = 0
include_zero_gene = "all"
max_seq_len = 1536

# settings for training
MLM = True  # whether to use masked language modeling, currently it is always on.
CLS = False  # celltype classification objective
CCE = False  # Contrastive cell embedding objective
MVC = False  # Masked value prediction for cell embedding
ECS = False  # Elastic cell similarity objective
amp = True
load_model = args.pretrain_model
load_param_prefixs = [
    "encoder",
    "value_encoder",
    "transformer_encoder",
]

# settings for optimizer
lr = 1e-4  # or 1e-4
batch_size = 64
eval_batch_size = 64
epochs = 15
schedule_interval = 1
early_stop = 5

# settings for the model
embsize = 512  # embedding dimension
d_hid = 512  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 12  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8  # number of heads in nn.MultiheadAttention
n_layers_cls = 3
dropout = 0  # dropout probability
use_fast_transformer = True  # whether to use fast transformer

# logging
log_interval = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_dir = Path(args.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)
print(f"saving to {save_dir}")

logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")
# log running date and current git commit
logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")


#### Iteration ####
for seed in range(1, 11):

    start_time = time.time()
    print(f'For seed {seed}, start model training')
    
    pert_data_path = f"data/{args.pert_name}"
    pert_data = PertData(data_path=pert_data_path)
    pert_data.load(data_name=args.pert_name, data_path=pert_data_path)
    pert_data.prepare_split(split='simulation', seed=seed)  # get data split with seed
    pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)  # prepare data loader

    if load_model is not None:
        model_dir = Path(load_model)
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"

        vocab = GeneVocab.from_file(vocab_file)
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)

        pert_data.adata.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
        logger.info(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )
        genes = pert_data.adata.var["gene_name"].tolist()

        # model
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        logger.info(
            f"Resume model from {model_file}, the model args will override the "
            f"config {model_config_file}."
        )
        embsize = model_configs["embsize"]
        nhead = model_configs["nheads"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
        n_layers_cls = model_configs["n_layers_cls"]
    else:
        genes = pert_data.adata.var["gene_name"].tolist()
        vocab = Vocab(
            VocabPybind(genes + special_tokens, None)
        )  # bidirectional lookup [gene <-> int]
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(
        [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
    )
    n_genes = len(genes)

    ntokens = len(vocab)  # size of vocabulary
    model = TransformerGenerator(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=n_layers_cls,
        n_cls=1,
        vocab=vocab,
        dropout=dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        pert_pad_id=pert_pad_id,
        use_fast_transformer=use_fast_transformer,
    )
    if load_param_prefixs is not None and load_model is not None:
        # only load params that start with the prefix
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if any([k.startswith(prefix) for prefix in load_param_prefixs])
        }
        for k, v in pretrained_dict.items():
            logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    elif load_model is not None:
        try:
            model.load_state_dict(torch.load(model_file))
            logger.info(f"Loading all model params from {model_file}")
        except:
            # only load params that are in the model and match the size
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_file)
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                logger.info(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
    model.to(device)


    criterion = masked_mse_loss
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)


    def train(model: nn.Module, train_loader: torch.utils.data.DataLoader) -> None:
        """
        Train the model for one epoch.
        """
        model.train()
        total_loss, total_mse = 0.0, 0.0
        start_time = time.time()

        num_batches = len(train_loader)
        for batch, batch_data in enumerate(train_loader):
            batch_size = len(batch_data.y)
            batch_data.to(device)
            x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
            ori_gene_values = x[:, 0].view(batch_size, n_genes)
            pert_flags = x[:, 1].long().view(batch_size, n_genes)
            target_gene_values = batch_data.y  # (batch_size, n_genes)

            if include_zero_gene in ["all", "batch-wise"]:
                if include_zero_gene == "all":
                    input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
                else:
                    input_gene_ids = (
                        ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                    )
                # sample input_gene_id
                if len(input_gene_ids) > max_seq_len:
                    input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                        :max_seq_len
                    ]
                input_values = ori_gene_values[:, input_gene_ids]
                input_pert_flags = pert_flags[:, input_gene_ids]
                target_values = target_gene_values[:, input_gene_ids]

                mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
                mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

                # src_key_padding_mask = mapped_input_gene_ids.eq(vocab[pad_token])
                src_key_padding_mask = torch.zeros_like(
                    input_values, dtype=torch.bool, device=device
                )

            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = model(
                    mapped_input_gene_ids,
                    input_values,
                    input_pert_flags,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=CLS,
                    CCE=CCE,
                    MVC=MVC,
                    ECS=ECS,
                )
                output_values = output_dict["mlm_output"]

                masked_positions = torch.ones_like(
                    input_values, dtype=torch.bool
                )  # Use all
                loss = loss_mse = criterion(output_values, target_values, masked_positions)

            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                    error_if_nonfinite=False if scaler.is_enabled() else True,
                )
                if len(w) > 0:
                    logger.warning(
                        f"Found infinite gradient. This may be caused by the gradient "
                        f"scaler. The current scale is {scaler.get_scale()}. This warning "
                        "can be ignored if no longer occurs after autoscaling of the scaler."
                    )
            scaler.step(optimizer)
            scaler.update()

            # torch.cuda.empty_cache()

            total_loss += loss.item()
            total_mse += loss_mse.item()
            if batch % log_interval == 0 and batch > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                cur_mse = total_mse / log_interval
                # ppl = math.exp(cur_loss)
                logger.info(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} |"
                )
                total_loss = 0
                total_mse = 0
                start_time = time.time()


    def eval_perturb(
        loader: DataLoader, model: TransformerGenerator, device: torch.device
    ) -> Dict:
        """
        Run model in inference mode using a given data loader
        """

        model.eval()
        model.to(device)
        pert_cat = []
        pred = []
        truth = []
        pred_de = []
        truth_de = []
        results = {}
        logvar = []

        for itr, batch in enumerate(loader):
            batch.to(device)
            pert_cat.extend(batch.pert)

            with torch.no_grad():
                p = model.pred_perturb(
                    batch,
                    include_zero_gene=include_zero_gene,
                    gene_ids=gene_ids,
                )
                t = batch.y
                pred.extend(p.cpu())
                truth.extend(t.cpu())

                # Differentially expressed genes
                for itr, de_idx in enumerate(batch.de_idx):
                    pred_de.append(p[itr, de_idx])
                    truth_de.append(t[itr, de_idx])

        # all genes
        results["pert_cat"] = np.array(pert_cat)
        pred = torch.stack(pred)
        truth = torch.stack(truth)
        results["pred"] = pred.detach().cpu().numpy().astype(float)
        results["truth"] = truth.detach().cpu().numpy().astype(float)

        pred_de = torch.stack(pred_de)
        truth_de = torch.stack(truth_de)
        results["pred_de"] = pred_de.detach().cpu().numpy().astype(float)
        results["truth_de"] = truth_de.detach().cpu().numpy().astype(float)

        return results
    
    best_val_loss = float("inf")
    best_val_corr = 0
    best_model = None
    patience = 0

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_loader = pert_data.dataloader["train_loader"]
        valid_loader = pert_data.dataloader["val_loader"]

        train(
            model,
            train_loader,
        )

        val_res = eval_perturb(valid_loader, model, device)
        val_metrics = compute_perturbation_metrics(
            val_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
        )
        logger.info(f"val_metrics at epoch {epoch}: ")
        logger.info(val_metrics)

        elapsed = time.time() - epoch_start_time
        logger.info(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | ")

        val_score = val_metrics["pearson"]
        if val_score > best_val_corr:
            best_val_corr = val_score
            best_model = copy.deepcopy(model)
            logger.info(f"Best model with score {val_score:5.4f}")
            patience = 0
        else:
            patience += 1
            if patience >= early_stop:
                logger.info(f"Early stop at epoch {epoch}")
                break

        scheduler.step()

    # Save model
    torch.save(best_model.state_dict(), save_dir / f"Seed_{seed}_best_model.pt")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"For seed {seed}, done model training / ({elapsed_time:.2f} seconds)")


