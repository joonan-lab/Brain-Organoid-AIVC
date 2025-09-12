import argparse

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--pert_name', type=str, required=True, help='Name for perturbation dataset')
parser.add_argument('--pretrain_model', type=str, required=True, help='Directory of pretrained model')
parser.add_argument('--save_dir', type=str, required=True, help='Directory to save trained models')
args = parser.parse_args()

import os
import gc
import json
import sys
import time
import copy
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Union, Optional
import warnings

import torch
import numpy as np
import pandas as pd
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

torch.cuda.empty_cache()

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
early_stop = 10

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
scg.utils.add_file_handler(logger, save_dir / "run_eval.log")
# log running date and current git commit
logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")

from scipy.stats import spearmanr
from anndata import AnnData
# === Spearman correlation metrics ===
def compute_perturbation_metrics_spearman(
    results: Dict,
    ctrl_adata: AnnData,
    non_zero_genes: bool = False,
    return_raw: bool = False,
) -> Dict:
    """
    Given results from a model run and the ground truth, compute metrics

    Args:
        results (:obj:`Dict`): The results from a model run
        ctrl_adata (:obj:`AnnData`): The adata of the control condtion
        non_zero_genes (:obj:`bool`, optional): Whether to only consider non-zero
            genes in the ground truth when computing metrics
        return_raw (:obj:`bool`, optional): Whether to return the raw metrics or
            the mean of the metrics. Default is False.

    Returns:
        :obj:`Dict`: The metrics computed
    """
    from scipy.stats import spearmanr


    # metrics:
    #   spearmanr correlation of expression on all genes, on DE genes,
    #   spearmanr correlation of expression change on all genes, on DE genes,

    metrics_across_genes = {
        "spearmanr": [],
        "spearmanr_de": [],
        "spearmanr_delta": [],
        "spearmanr_de_delta": [],
    }

    metrics_across_conditions = {
        "spearmanr": [],
        "spearmanr_delta": [],
    }

    conditions = np.unique(results["pert_cat"])
    assert not "ctrl" in conditions, "ctrl should not be in test conditions"
    condition2idx = {c: np.where(results["pert_cat"] == c)[0] for c in conditions}

    mean_ctrl = np.array(ctrl_adata.X.mean(0)).flatten()  # (n_genes,)
    assert ctrl_adata.X.max() <= 1000, "gene expression should be log transformed"

    true_perturbed = results["truth"]  # (n_cells, n_genes)
    assert true_perturbed.max() <= 1000, "gene expression should be log transformed"
    true_mean_perturbed_by_condition = np.array(
        [true_perturbed[condition2idx[c]].mean(0) for c in conditions]
    )  # (n_conditions, n_genes)
    true_mean_delta_by_condition = true_mean_perturbed_by_condition - mean_ctrl
    zero_rows = np.where(np.all(true_mean_perturbed_by_condition == 0, axis=1))[
        0
    ].tolist()
    zero_cols = np.where(np.all(true_mean_perturbed_by_condition == 0, axis=0))[
        0
    ].tolist()

    pred_perturbed = results["pred"]  # (n_cells, n_genes)
    pred_mean_perturbed_by_condition = np.array(
        [pred_perturbed[condition2idx[c]].mean(0) for c in conditions]
    )  # (n_conditions, n_genes)
    pred_mean_delta_by_condition = pred_mean_perturbed_by_condition - mean_ctrl

    def corr_over_genes(x, y, conditions, res_list, skip_rows=[], non_zero_mask=None):
        """compute spearmanr correlation over genes for each condition"""
        for i, c in enumerate(conditions):
            if i in skip_rows:
                continue
            x_, y_ = x[i], y[i]
            if non_zero_mask is not None:
                x_ = x_[non_zero_mask[i]]
                y_ = y_[non_zero_mask[i]]
            res_list.append(spearmanr(x_, y_)[0])

    corr_over_genes(
        true_mean_perturbed_by_condition,
        pred_mean_perturbed_by_condition,
        conditions,
        metrics_across_genes["spearmanr"],
        zero_rows,
        non_zero_mask=true_mean_perturbed_by_condition != 0 if non_zero_genes else None,
    )
    corr_over_genes(
        true_mean_delta_by_condition,
        pred_mean_delta_by_condition,
        conditions,
        metrics_across_genes["spearmanr_delta"],
        zero_rows,
        non_zero_mask=true_mean_perturbed_by_condition != 0 if non_zero_genes else None,
    )

    def find_DE_genes(adata, condition, geneid2idx, non_zero_genes=False, top_n=20):
        """
        Find the DE genes for a condition
        """
        key_components = next(
            iter(adata.uns["rank_genes_groups_cov_all"].keys())
        ).split("_")
        assert len(key_components) == 3, "rank_genes_groups_cov_all key is not valid"

        condition_key = "_".join([key_components[0], condition, key_components[2]])

        de_genes = adata.uns["rank_genes_groups_cov_all"][condition_key]
        if non_zero_genes:
            de_genes = adata.uns["top_non_dropout_de_20"][condition_key]
            # de_genes = adata.uns["rank_genes_groups_cov_all"][condition_key]
            # de_genes = de_genes[adata.uns["non_zeros_gene_idx"][condition_key]]
            # assert len(de_genes) > top_n

        de_genes = de_genes[:top_n]

        de_idx = [geneid2idx[i] for i in de_genes]

        return de_idx, de_genes

    geneid2idx = dict(zip(ctrl_adata.var.index.values, range(len(ctrl_adata.var))))
    de_idx = {
        c: find_DE_genes(ctrl_adata, c, geneid2idx, non_zero_genes)[0]
        for c in conditions
    }
    mean_ctrl_de = np.array(
        [mean_ctrl[de_idx[c]] for c in conditions]
    )  # (n_conditions, n_diff_genes)

    true_mean_perturbed_by_condition_de = np.array(
        [
            true_mean_perturbed_by_condition[i, de_idx[c]]
            for i, c in enumerate(conditions)
        ]
    )  # (n_conditions, n_diff_genes)
    zero_rows_de = np.where(np.all(true_mean_perturbed_by_condition_de == 0, axis=1))[
        0
    ].tolist()
    true_mean_delta_by_condition_de = true_mean_perturbed_by_condition_de - mean_ctrl_de

    pred_mean_perturbed_by_condition_de = np.array(
        [
            pred_mean_perturbed_by_condition[i, de_idx[c]]
            for i, c in enumerate(conditions)
        ]
    )  # (n_conditions, n_diff_genes)
    pred_mean_delta_by_condition_de = pred_mean_perturbed_by_condition_de - mean_ctrl_de

    corr_over_genes(
        true_mean_perturbed_by_condition_de,
        pred_mean_perturbed_by_condition_de,
        conditions,
        metrics_across_genes["spearmanr_de"],
        zero_rows_de,
    )
    corr_over_genes(
        true_mean_delta_by_condition_de,
        pred_mean_delta_by_condition_de,
        conditions,
        metrics_across_genes["spearmanr_de_delta"],
        zero_rows_de,
    )

    if not return_raw:
        for k, v in metrics_across_genes.items():
            metrics_across_genes[k] = np.mean(v)
        for k, v in metrics_across_conditions.items():
            metrics_across_conditions[k] = np.mean(v)
    metrics = metrics_across_genes

    return metrics

from sklearn.metrics import mean_squared_error

def compute_perturbation_metrics_rmse(
    results: Dict,
    ctrl_adata: AnnData,
    non_zero_genes: bool = False,
    return_raw: bool = False,
) -> Dict:
    """
    Compute RMSE metrics from perturbation results.
    """

    metrics = {
        "rmse": [],
        "rmse_de": [],
        "rmse_delta": [],
        "rmse_de_delta": [],
    }

    conditions = np.unique(results["pert_cat"])
    condition2idx = {c: np.where(results["pert_cat"] == c)[0] for c in conditions}

    mean_ctrl = np.array(ctrl_adata.X.mean(0)).flatten()

    true_perturbed = results["truth"]
    pred_perturbed = results["pred"]

    true_mean_by_condition = np.array(
        [true_perturbed[condition2idx[c]].mean(0) for c in conditions]
    )
    pred_mean_by_condition = np.array(
        [pred_perturbed[condition2idx[c]].mean(0) for c in conditions]
    )

    delta_true = true_mean_by_condition - mean_ctrl
    delta_pred = pred_mean_by_condition - mean_ctrl

    for i, cond in enumerate(conditions):
        if non_zero_genes:
            non_zero_mask = true_mean_by_condition[i] != 0
            true_vec = true_mean_by_condition[i][non_zero_mask]
            pred_vec = pred_mean_by_condition[i][non_zero_mask]
            delta_true_vec = delta_true[i][non_zero_mask]
            delta_pred_vec = delta_pred[i][non_zero_mask]
        else:
            true_vec = true_mean_by_condition[i]
            pred_vec = pred_mean_by_condition[i]
            delta_true_vec = delta_true[i]
            delta_pred_vec = delta_pred[i]

        metrics["rmse"].append(np.sqrt(mean_squared_error(true_vec, pred_vec)))
        metrics["rmse_delta"].append(np.sqrt(mean_squared_error(delta_true_vec, delta_pred_vec)))

    # ============ DE Genes RMSE ============
    geneid2idx = dict(zip(ctrl_adata.var.index.values, range(len(ctrl_adata.var))))
    def find_DE_genes(condition, top_n=20):
        key_components = next(iter(ctrl_adata.uns["rank_genes_groups_cov_all"].keys())).split("_")
        condition_key = "_".join([key_components[0], condition, key_components[2]])
        de_genes = ctrl_adata.uns["rank_genes_groups_cov_all"][condition_key][:top_n]
        return [geneid2idx[g] for g in de_genes]

    for i, cond in enumerate(conditions):
        de_idx = find_DE_genes(cond)
        true_vec_de = true_mean_by_condition[i][de_idx]
        pred_vec_de = pred_mean_by_condition[i][de_idx]
        delta_true_de = delta_true[i][de_idx]
        delta_pred_de = delta_pred[i][de_idx]

        metrics["rmse_de"].append(np.sqrt(mean_squared_error(true_vec_de, pred_vec_de)))
        metrics["rmse_de_delta"].append(np.sqrt(mean_squared_error(delta_true_de, delta_pred_de)))

    if not return_raw:
        for k in metrics:
            metrics[k] = float(np.mean(metrics[k]))

    return metrics


from sklearn.metrics import mean_absolute_error

def compute_perturbation_metrics_mae(
    results: Dict,
    ctrl_adata: AnnData,
    non_zero_genes: bool = False,
    return_raw: bool = False,
) -> Dict:
    """
    Compute MAE metrics from perturbation results.
    """

    metrics = {
        "mae": [],
        "mae_de": [],
        "mae_delta": [],
        "mae_de_delta": [],
    }

    conditions = np.unique(results["pert_cat"])
    condition2idx = {c: np.where(results["pert_cat"] == c)[0] for c in conditions}

    mean_ctrl = np.array(ctrl_adata.X.mean(0)).flatten()

    true_perturbed = results["truth"]
    pred_perturbed = results["pred"]

    true_mean_by_condition = np.array(
        [true_perturbed[condition2idx[c]].mean(0) for c in conditions]
    )
    pred_mean_by_condition = np.array(
        [pred_perturbed[condition2idx[c]].mean(0) for c in conditions]
    )

    delta_true = true_mean_by_condition - mean_ctrl
    delta_pred = pred_mean_by_condition - mean_ctrl

    for i, cond in enumerate(conditions):
        if non_zero_genes:
            non_zero_mask = true_mean_by_condition[i] != 0
            true_vec = true_mean_by_condition[i][non_zero_mask]
            pred_vec = pred_mean_by_condition[i][non_zero_mask]
            delta_true_vec = delta_true[i][non_zero_mask]
            delta_pred_vec = delta_pred[i][non_zero_mask]
        else:
            true_vec = true_mean_by_condition[i]
            pred_vec = pred_mean_by_condition[i]
            delta_true_vec = delta_true[i]
            delta_pred_vec = delta_pred[i]

        metrics["mae"].append(mean_absolute_error(true_vec, pred_vec))
        metrics["mae_delta"].append(mean_absolute_error(delta_true_vec, delta_pred_vec))

    # ============ DE Genes MAE ============
    geneid2idx = dict(zip(ctrl_adata.var.index.values, range(len(ctrl_adata.var))))
    def find_DE_genes(condition, top_n=20):
        key_components = next(iter(ctrl_adata.uns["rank_genes_groups_cov_all"].keys())).split("_")
        condition_key = "_".join([key_components[0], condition, key_components[2]])
        de_genes = ctrl_adata.uns["rank_genes_groups_cov_all"][condition_key][:top_n]
        return [geneid2idx[g] for g in de_genes]

    for i, cond in enumerate(conditions):
        de_idx = find_DE_genes(cond)
        true_vec_de = true_mean_by_condition[i][de_idx]
        pred_vec_de = pred_mean_by_condition[i][de_idx]
        delta_true_de = delta_true[i][de_idx]
        delta_pred_de = delta_pred[i][de_idx]

        metrics["mae_de"].append(mean_absolute_error(true_vec_de, pred_vec_de))
        metrics["mae_de_delta"].append(mean_absolute_error(delta_true_de, delta_pred_de))

    if not return_raw:
        for k in metrics:
            metrics[k] = float(np.mean(metrics[k]))

    return metrics

all_test_metrics = []

for seed in range(1, 11):
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
    
    model.load_state_dict(torch.load(f'{Path(args.save_dir)}/Seed_{seed}_best_model.pt'))
    
    # Get model performance for test dataset
    test_loader = pert_data.dataloader["test_loader"]
    test_res = eval_perturb(test_loader, model, device)
    metrics = {}
    # test_metrics, test_pert_res = compute_metrics(test_res)
    test_metrics = compute_perturbation_metrics(
        test_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
    )
    print(test_metrics)
    metrics.update(test_metrics)
    
    # test_metrics, test_pert_res = compute_metrics(test_res)
    test_metrics = compute_perturbation_metrics_spearman(
        test_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
    )
    print(test_metrics)
    metrics.update(test_metrics)
    
    rmse_metrics = compute_perturbation_metrics_rmse(
        test_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
    )
    print(rmse_metrics)
    metrics.update(rmse_metrics)
    
    mae_metrics = compute_perturbation_metrics_mae(
        test_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
    )
    print(mae_metrics)
    metrics.update(mae_metrics)
    
    all_test_metrics.append(metrics)

# Convert all_test_metrics to a DataFrame for better viewing and further analysis
test_metrics_df = pd.DataFrame(all_test_metrics)
test_metrics_df.to_csv(f"{Path(args.save_dir)}/Seed1to10_model_performance.txt", sep="\t", index=False)

