# AI-Driven Framework Modeling Perturbation Responses in Brain Organoids

This repository contains code and model resources for the NOCAP-AIVC perturbation-response modeling study. The project uses the Neural Organoid Cell Atlas with Perturbation (NOCAP) to train and benchmark models for predicting transcriptional responses to genetic perturbation in human brain organoid-derived neural contexts.

## Repository contents


```text
.
├── README.md                 # repository-level overview
├── Codes/                    # scripts-only benchmarking code package
├── Models/                   # Git LFS pointers for pre-trained and fine-tuned model checkpoints
├── .gitattributes            # Git LFS tracking for model checkpoint files
└── .gitignore                # excludes local data, outputs, caches and environment files

```

The complete revised benchmarking code package is under `Codes/`. See `Codes/README.md` for setup, run order, metric definitions, data/checkpoint requirements, and the exact script layout used by this Dropbox-staged GitHub package.

## Data resources

Large single-cell matrices, generated prediction matrices, and protected cohort data are not stored directly in this repository. The manuscript-associated public resources are deposited separately:

- Integrated and batch-corrected NOCAP atlas: https://doi.org/10.5281/zenodo.17104715
- Predicted gene-expression matrices from fine-tuned models: https://doi.org/10.5281/zenodo.17106100
- Integrated and batch-corrected mouse brain perturbation atlas: https://doi.org/10.5281/zenodo.17112344

Users should download permitted datasets and generated matrices from the relevant archive, then edit `Codes/paths.example.yaml` into a local `paths.yaml` or adapt script-level path variables for their environment.

## Model resources

Pre-trained and fine-tuned model checkpoint entries are organized under `Models/`:

```text
Models/
├── Pretrained/
│   ├── BrainTissue_pretrained/
│   ├── EndodermOrganoid_pretrained/
│   ├── Mouse_pretrained/
│   └── NOCAP_pretrained/
└── Finetuned/
    ├── CellLine-BrainTissue.pt
    ├── CellLine-NOCAP.pt
    ├── Neuro-NOCAP.pt
    ├── Pan-NOCAP.pt
    ├── Telen-BrainTissue.pt
    ├── Telen-EndodermOrganoid.pt
    ├── Telen-GEARS/
    ├── Telen-Mouse.pt
    ├── Telen-NOCAP-Lite.pt
    └── Telen-NOCAP.pt
```

Checkpoint files are tracked through Git LFS where applicable. If a cloned repository contains pointer files instead of full model files, install Git LFS and run:

```bash
git lfs install
git lfs pull
```

## Code package

The `Codes/` directory is a scripts-only benchmarking package. It includes:

- scGPT-backbone NOCAP model pretraining, fine-tuning, evaluation and prediction scripts under `Codes/scGPT_based/`
- External baseline workflows for scLAMBDA, scFoundation, Geneformer, GeneCompass, CellFM and GEARS
- A non-parametric TrainMean baseline
- Downstream metric computation and integrated rank aggregation under `Codes/Benchmarking_Metrics/`
- Environment and path templates: `Codes/environment.yml`, `Codes/requirements.txt`, and `Codes/paths.example.yaml`
- File manifests: `Codes/CODE_FILE_MANIFEST.txt` and `Codes/CODE_FILE_MANIFEST.json`

The code package is intended for reproducibility guidance and code inspection. It is not a self-contained data archive; users must provide large inputs, external pretrained checkpoints, upstream model packages and generated prediction matrices where permitted.

## Quick start

```bash
cd Codes
conda env create -f environment.yml
conda activate nocap-aivc-benchmark
cp paths.example.yaml paths.yaml
```

Then edit `paths.yaml` and any script-level path placeholders to point to local data, checkpoint and output locations. See `Codes/README.md` for the model-specific run order.

## Repository URL

Canonical GitHub URL:

https://github.com/joonan-lab/Brain-Organoid-AIVC
