# NOCAP-AIVC Benchmarking Code

This directory contains the GitHub code package for benchmarking NOCAP-AIVC perturbation-response models and external single-cell foundation/modeling baselines.

The package is scripts-only. Large single-cell matrices, trained checkpoints, third-party pretrained weights, generated prediction matrices and protected cohort data are not included in `Codes/`; model checkpoint entries are maintained at the repository level under `Models/` where applicable.

## Overview

The benchmarking workflow evaluates perturbation-response prediction in neural organoid contexts. It includes:

- scGPT-backbone NOCAP workflows under `scGPT_backbone/`
- External architecture baselines: Telen-scLAMBDA, Telen-scFoundation, Telen-Geneformer, Telen-GeneCompass, Telen-CellFM and Telen-GEARS
- A non-parametric TrainMean baseline under `BaselineMean/`
- Downstream post-analysis for prediction accuracy, embedding quality, differential-expression recovery, Moran's I and integrated rank aggregation

## Directory layout

```text
Codes/
├── README.md
├── requirements.txt
├── environment.yml
├── paths.example.yaml
├── CODE_FILE_MANIFEST.txt
├── CODE_FILE_MANIFEST.json
├── BaselineMean/             # TrainMean non-parametric baseline
├── Benchmarking_Metrics/     # metric computation and final rank aggregation
├── Telen-CellFM/             # CellFM encoder with GEARS-style perturbation decoder
├── Telen-GEARS/              # GEARS workflow
├── Telen-GeneCompass/        # GeneCompass workflow
├── Telen-Geneformer/         # Geneformer workflow
├── Telen-scFoundation/       # scFoundation workflow
├── scGPT_backbone/     # scGPT pretraining/fine-tuning/evaluation/prediction scripts
└── Telen-scLAMBDA/           # scLAMBDA workflow
```

Model-specific workflow directories use reviewer-facing `Telen-` prefixes except for `BaselineMean/`, `Benchmarking_Metrics/`, and the shared `scGPT_backbone/` workflow. Script filenames are intentionally kept architecture-specific to avoid unnecessary execution changes.

The `scGPT_backbone/` directory contains both unpacked scripts and the corresponding small zip archives:

```text
scGPT_backbone/
├── Finetuning/
│   ├── 01_train.py
│   ├── 02_evaluation.py
│   └── 03_prediction.py
├── Finetuning.zip
├── Pretraining/
│   └── pretrain.py
└── Pretraining.zip
```

## System requirements

### Operating systems tested

The scripts were prepared and checked on Linux x86_64. Manuscript-scale runs were performed on a Linux GPU server; standard CPU-only Linux workstations are sufficient for code inspection, environment construction and lightweight metric utilities.

### Software versions tested

The bundled environment files specify the versions used during code-package preparation, including Python 3.10, R 4.4.3, Scanpy 1.11.5, AnnData 0.11.4, scikit-learn 1.5.2, PyTorch 2.6.0, Transformers 4.56.1 and the R packages listed in `environment.yml`. External model workflows additionally require their upstream packages or local checkouts, including `gears`, `scgpt`, `geneformer`, `sclambda`, scFoundation `modules`, and GeneCompass/GEARS-compatible `geares` where applicable.

### Hardware requirements

Full model training and prediction require GPU resources appropriate for the selected model and dataset size. The original large-scale benchmarking used NVIDIA GPU hardware. CPU-only execution is suitable for static inspection and some post-analysis utilities, but not for practical full-scale foundation-model fine-tuning or transcriptome-wide prediction.

## Installation guide

```bash
conda env create -f environment.yml
conda activate nocap-aivc-benchmark
```

or:

```bash
pip install -r requirements.txt
```

Typical installation time is approximately 20-45 minutes for the conda environment on a Linux workstation with broadband internet, excluding CUDA-specific PyTorch wheel selection, external model repository setup and large checkpoint downloads. A pip-only setup is typically faster but may require manual installation of R packages and external model packages.

The Python requirements are pinned to versions used during code-package preparation where available. `torch==2.6.0` is listed without a CUDA build suffix; install the CUDA-compatible wheel for your system if GPU execution is required. `torchtext` is required by the scGPT-backbone scripts, and `openpyxl` is required for Excel input/output through pandas.

The R post-analysis scripts additionally require packages such as `Seurat`, `MuDataSeurat`, `scDEED`, `reticulate`, `dplyr`, `tidyr`, `ggplot2`, `readxl`, `writexl`, `foreach`, `doParallel`, `patchwork`, `RColorBrewer` and `extrafont`. Code-package-QA observed versions included R 4.4.3, Seurat 5.3.0, MuDataSeurat 0.0.0.9000, reticulate 1.43.0, dplyr 1.1.4, tidyr 1.3.1, ggplot2 4.0.0, readxl 1.4.5, writexl 1.5.4, foreach 1.5.2, doParallel 1.0.17, patchwork 1.3.2, RColorBrewer 1.1.3 and extrafont 0.20. `MuDataSeurat`, `scDEED`, `VGAM`, `resample` and `distances` may need installation from upstream R sources if unavailable from conda-forge.

## Data and external resource requirements

This code package does not include large or restricted inputs. To reproduce the benchmark, users need to provide local copies of the following resources where permitted by licenses and data-use agreements.

### Core inputs

- Preprocessed NOCAP telencephalic-neuron AnnData object used for model fine-tuning/evaluation
- NOCAP Extended Atlas or model-specific processed derivatives
- Train/test perturbation splits
- Gene annotation and protein-coding gene universe tables
- ASD/NDD risk gene-set tables used for Moran's I, enrichment and post-analysis
- Differential-expression gene lists stored in expected AnnData `.uns` fields or generated by preprocessing scripts

### External model resources

- scGPT pretrained checkpoints or scripts to pretrain from the specified corpus
- scFoundation public checkpoint and gene index
- Geneformer checkpoint/tokenizer resources. The release includes the small Ensembl mapping dictionary used by the Geneformer workflow at `Telen-Geneformer/resources/ensembl_mapping_dict_gc104M.pkl`; this path can be overridden with `--ensembl_mapping_dict` or the `GENEFORMER_ENSEMBL_MAPPING_DICT` environment variable.
- GeneCompass checkpoint and prior-knowledge resources
- CellFM checkpoint or PyTorch-compatible port resources
- GEARS/scLAMBDA dependencies and graph/gene-embedding resources

## Instructions for use with local data

This repository is not a self-contained data archive. Download the permitted h5ad matrices, prediction matrices and model resources from the repositories listed in the manuscript data-availability statement, then copy and edit the path template:

```bash
cd Codes
cp paths.example.yaml paths.yaml
```

Use `paths.yaml`, command-line arguments and the placeholders documented below to point each workflow to local data, checkpoints and output directories. Protected ASD cohort data used for genomic validation cannot be redistributed through this repository and should be accessed only under the relevant data-use agreements.

### Path placeholders

Most scripts use placeholders or local path constants from the original analysis environment. Replace these with local resource paths before execution. Common placeholders include:

- `<PROJECT_ROOT>`
- `<DATA_ROOT>`
- `<SHARED_ROOT>`
- `<ENV_ROOT>`
- `/path/to/...`

## Run order

This is a high-level execution outline. Exact command-line arguments depend on local paths, external checkpoints and upstream package availability.

### Telen-scGPT-backbone models

```bash
python scGPT_backbone/Pretraining/pretrain.py
python scGPT_backbone/Finetuning/01_train.py
python scGPT_backbone/Finetuning/02_evaluation.py
python scGPT_backbone/Finetuning/03_prediction.py  # uses 1,000 sampled control cells per perturbation
```

### Telen-scLAMBDA

```bash
python Telen-scLAMBDA/01_train_scLAMBDA.py
python Telen-scLAMBDA/02_evaluation_scLAMBDA.py
python Telen-scLAMBDA/03_prediction_scLAMBDA.py
```

### Telen-scFoundation

```bash
python Telen-scFoundation/01_train_scFoundation.py
python Telen-scFoundation/02_evaluation_scFoundation.py
python Telen-scFoundation/03_prediction_scFoundation.py
```

### Telen-Geneformer

```bash
python Telen-Geneformer/01_train_Geneformer.py
python Telen-Geneformer/02_evaluation_Geneformer.py
python Telen-Geneformer/03_prediction_Geneformer.py
```

### Telen-GeneCompass

```bash
python Telen-GeneCompass/01_train_GeneCompass.py
python Telen-GeneCompass/02_evaluation_GeneCompass.py
python Telen-GeneCompass/03_prediction_GeneCompass.py
```

### Telen-CellFM

```bash
python Telen-CellFM/01_train_CellFM.py
python Telen-CellFM/02_evaluation_CellFM.py
python Telen-CellFM/03_prediction_CellFM.py
```

### Telen-GEARS baseline

```bash
python Telen-GEARS/00_preprocess_GEARS.py
python Telen-GEARS/01_train_GEARS.py
python Telen-GEARS/02_evaluation_GEARS.py
python Telen-GEARS/03_prediction_GEARS.py
```

### TrainMean baseline

```bash
python BaselineMean/01_evaluation_TrainMean.py
```

### Benchmarking post-analysis

```bash
python Benchmarking_Metrics/00_convert_predictions_to_anndata.py
Rscript Benchmarking_Metrics/01_scdeed_grid_search.R
python Benchmarking_Metrics/02_apply_optimal_umap.py
python Benchmarking_Metrics/03_compute_pca_moran_i.py
python Benchmarking_Metrics/04_gather_model_performance_metrics.py
python Benchmarking_Metrics/05_compute_de_recovery_metrics.py
python Benchmarking_Metrics/06_compute_embedding_quality_metrics.py
python Benchmarking_Metrics/07_identify_perturbation_gene_clusters.py
Rscript Benchmarking_Metrics/08_aggregate_benchmarking_metrics.R
```

## Evaluation metrics

This code package follows the revised manuscript/SI benchmarking framework: 13 model-comparison metrics plus an integrated `Total` rank.

### Prediction-accuracy metrics

The final model comparison uses the following 8 prediction metrics. Some evaluator scripts may retain additional diagnostic columns for provenance, but these are not used in the final integrated rank:

| Metric | Direction | Meaning |
|---|---:|---|
| `pearson_delta` | higher is better | Pearson correlation between predicted and observed expression-change vectors |
| `pearson_de_delta` | higher is better | Pearson correlation on top differentially expressed genes, using expression-change vectors |
| `spearmanr_delta` | higher is better | Spearman correlation between predicted and observed expression-change vectors |
| `spearmanr_de_delta` | higher is better | Spearman correlation on top differentially expressed genes, using expression-change vectors |
| `rmse` | lower is better | Root mean squared error on expression profiles |
| `rmse_de` | lower is better | Root mean squared error restricted to top differentially expressed genes |
| `mae` | lower is better | Mean absolute error on expression profiles |
| `mae_de` | lower is better | Mean absolute error restricted to top differentially expressed genes |

The DE-restricted metrics use the top-20 differentially expressed genes for each held-out perturbation where available.

### Post-analysis metrics

The downstream benchmarking summary adds 5 post-analysis metrics:

| Metric | Direction | Meaning |
|---|---:|---|
| `mean_knn_dist_ref` | higher is better | Mean k-nearest-neighbor distance relative to the reference embedding space |
| `cv_knn_dist` | lower is better | Coefficient of variation of k-nearest-neighbor distances |
| `AUPRC@50` | higher is better | Differential-expression recovery among top-ranked genes |
| `CentroidAcc` | higher is better | Centroid-level recovery of perturbation-specific DE signal |
| `Moran_I_ASD185_EAGLE` | higher is better | PCA-space Moran's I for ASD-risk-gene coherence |

### Integrated rank

`Benchmarking_Metrics/08_aggregate_benchmarking_metrics.R` aggregates within-metric ranks with fixed group weights:

- Prediction accuracy: 30%
- Embedding quality: 25%
- Differential-expression recovery: 25%
- Perturbation-space coherence: 20%

Non-significant Moran's I values are treated as unavailable for the coherence component, and the remaining group weights are redistributed within the affected model.

### Best-seed selection

For methods trained across seeds, the best-seed prediction matrix is selected by `pearson_de_delta`. This metric is retained because downstream prediction-space analyses require one representative prediction matrix per model.

## Model-specific notes

### Telen-scGPT-backbone workflow note

`scGPT_backbone/Finetuning/03_prediction.py` samples up to 1,000 control cells per perturbation as the prediction background, matching the manuscript-facing description. The `Finetuning.zip` and `Pretraining.zip` archives duplicate the unpacked scGPT-backbone scripts for compatibility with the staged Dropbox package.

### Telen-scLAMBDA workflow note

`Telen-scLAMBDA/01_train_scLAMBDA.py` includes data preparation before model training, and `Telen-scLAMBDA/03_prediction_scLAMBDA.py` writes the benchmarking-standardized prediction output directly.

### Telen-GeneCompass and Telen-CellFM workflow notes

`Telen-GeneCompass/01_train_GeneCompass.py` integrates preprocessing, coexpression precomputation and multi-seed training. In this code package, `Telen-CellFM` denotes the CellFM encoder coupled to a GEARS-style perturbation decoder; `Telen-CellFM/01_train_CellFM.py` integrates checkpoint adaptation, control-cell embedding extraction, GO graph preparation and downstream perturbation-decoder training. Their prediction scripts write benchmarking-ready outputs after prediction generation.

## File manifest

`CODE_FILE_MANIFEST.txt` and `CODE_FILE_MANIFEST.json` describe the Dropbox-staged file layout and checksums for the `Codes/` package.

## Notes

This directory is intended for code-package and reproducibility guidance. It is not a self-contained data archive.
