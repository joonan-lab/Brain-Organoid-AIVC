# NOCAP-AIVC Model Checkpoints

This directory contains model checkpoint entries associated with the NOCAP-AIVC perturbation-response modeling study. Large binary checkpoints should be handled through Git LFS when this Dropbox-staged repository is pushed to GitHub.

## Directory layout

```text
Models/
├── Pretrained/   # scGPT-backbone pretrained model directories
└── Finetuned/    # fine-tuned perturbation-response model checkpoints
```

## Pretrained models

Current pretrained model directories:

```text
Models/Pretrained/
├── BrainTissue_pretrained/
├── EndodermOrganoid_pretrained/
├── Mouse_pretrained/
└── NOCAP_pretrained/
```

Each scGPT-style pretrained directory contains:

- `args.json`
- `best_model.pt`
- `vocab.json`

## Fine-tuned models

Current fine-tuned checkpoint entries include the original scGPT-backbone model variants and added benchmarking baselines:

```text
Models/Finetuned/
├── BrainCell-BrainTissue.pt
├── BrainCell-NOCAP.pt
├── BrainCell-WholeHuman.pt
├── CellLine-BrainTissue.pt
├── CellLine-NOCAP.pt
├── Neuro-NOCAP.pt
├── Pan-NOCAP.pt
├── Telen-BrainTissue.pt
├── Telen-EndodermOrganoid.pt
├── Telen-GEARS/
├── Telen-Mouse.pt
├── Telen-NOCAP-Lite.pt
├── Telen-NOCAP.pt
├── Telen-WholeHuman.pt
├── Telen-scLAMBDA/
├── Telen-GeneCompass/
├── Telen-Geneformer/
└── Telen-CellFM/
```

### BrainCell scGPT-backbone fine-tuned checkpoints

```text
Models/Finetuned/
├── BrainCell-BrainTissue.pt
├── BrainCell-WholeHuman.pt
└── BrainCell-NOCAP.pt
```

These are the selected BrainCell fine-tuned scGPT-backbone checkpoints renamed as flat model entries for consistency with the other single-file fine-tuned checkpoints. The source seeds are seed 6 for BrainCell-BrainTissue, seed 9 for BrainCell-WholeHuman, and seed 4 for BrainCell-NOCAP.

### Telen-WholeHuman

```text
Models/Finetuned/Telen-WholeHuman.pt
```

This is the selected seed 8 Telen-WholeHuman fine-tuned checkpoint, renamed from `Telen-WholeHuman_Seed_8_best_model.pt` as a flat model entry for consistency with the other single-file fine-tuned checkpoints.

### Telen-scLAMBDA

```text
Models/Finetuned/Telen-scLAMBDA/
├── ckpt.pth
├── best_seed_summary.json
└── Telen-scLAMBDA_all_seeds_metrics.csv
```

The selected checkpoint is seed 8, based on `pearson_de_delta` in `best_seed_summary.json`.

### Telen-GeneCompass

```text
Models/Finetuned/Telen-GeneCompass/
├── best_model_selection.json
├── Seed1to10_model_performance.csv
└── seed_10/
    ├── model.pt
    ├── ckpt.pth
    └── config.pkl
```

Only seed 10 is included. It is the selected seed in `best_model_selection.json` under the `pearson_de_delta` criterion.

### Telen-Geneformer

```text
Models/Finetuned/Telen-Geneformer/
├── Seed1to10_metrics.csv
├── metrics_summary.csv
├── selection_note.json
└── seed_2/
    ├── Seed_2_best_model.pt
    ├── training_info.json
    └── classifier/
        └── config.json
```

Only seed 2 is included. In the available Geneformer seed evaluation table, seed 2 has the highest `pearson_de_delta` and is the included checkpoint for the Telen-Geneformer benchmark entry.

### Telen-CellFM

```text
Models/Finetuned/Telen-CellFM/
├── all_seeds_metrics_16.csv
├── evaluation_results_final.csv
└── adapted_cellfm/
    ├── best_checkpoint.pth
    └── training_config.json
```

The adapted CellFM checkpoint is included here for model-resource provenance. The full final Telen-CellFM perturbation-decoder checkpoint is too large for GitHub/LFS distribution and is archived separately as a two-part Zenodo deposit because it exceeds Zenodo's 50 GB per-record bucket quota.

Download both parts and reconstruct the original checkpoint with:

```
cat Telen-CellFM.pt.part01 Telen-CellFM.pt.part02 > Telen-CellFM.pt
sha256sum Telen-CellFM.pt
```

Zenodo records:

- Part 1: `Telen-CellFM.pt.part01` (DOI: `10.5281/zenodo.20078742`; SHA256: `1866138adcd9e2398c463d18eb8315b5a6b908e3c31f5ece7c43bb27884bd085`)
- Part 2: `Telen-CellFM.pt.part02` (DOI: `10.5281/zenodo.20079060`; SHA256: `3ec7e86d40477943e58c6a688dacbf987339cf56614c9baaf6afe9ab4863922e`)
- Reconstructed `Telen-CellFM.pt` SHA256: `f30f4d0df6554492d94be4e9f94455c8b3abc2d3e7169bc02f80b4f0d1ef5c57`

This file corresponds to the selected seed 3 local checkpoint from `results_v2/seed_3/best_model.pt`.

## Git LFS

Checkpoint-like files should be tracked with Git LFS:

```text
*.pt
*.pth
*.pkl
*.ckpt
*.safetensors
```

Before pushing to GitHub, verify LFS tracking with:

```bash
git lfs track
git status --short
```

If a clone contains pointer files instead of full checkpoint binaries, install Git LFS and run:

```bash
git lfs install
git lfs pull
```

## Notes

- This directory is not a complete data archive.
- Large single-cell matrices and prediction matrices are deposited separately as described in the repository README.
- Third-party upstream pretrained resources may require separate downloads according to their original licenses.
