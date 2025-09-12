# **AI-Driven Framework Modeling Perturbation Responses in Brain Organoids**

This study presents an AI-driven framework to prioritize autism spectrum disorder (ASD) risk genes. Training our model on a 3.6 million-cell atlas of genetically perturbed human brain organoids, we performed *in silico* screens to identify genes strongly associated with ASD. The paper serves as a powerful, validated AI framework to accelerate gene discovery for complex genetic disorders.

We achieve this by building a foundation model using Neural Organoid Cell Atlas with Perturbation (NOCAP), fine-tuning with cell-type-specific perturbation data, predicting perturbation effects of all protein-coding genes, and evaluating the model’s predictive performance.

## [1] Data resource

### Neural Organoid Cell Atlas with Perturbation (NOCAP)

NOCAP is a fully integrated single-cell transcriptomic atlas comprising approximately 3.6 million cells derived from human brain organoids.

The integrated and batch-corrected NOCAP and mouse brain organoid atlas are deposited at Zenodo: (link). 

### Pre-trained and fine-tuned models

You can find all pre-trained and fine-tuned models used in our paper at the **“Models”** directory.

### Predicted gene expression matrix

The resulting gene expression matrices from each model are deposited at Zenodo: (link).

## [2] Pre-training a foundation model

A total of four pre-trained models were used in this paper, all of which were based on the foundation model architecture of scGPT ([https://github.com/bowang-lab/scGPT](https://github.com/bowang-lab/scGPT)). The procedures for data preprocessing before pre-training are detailed in `Codes/Pretraining/Preprocessing`. Model pre-training can be conducted using the code `pretrain.py` located in `Codes/Pretraining`. The specific commands used to generate each of the four pre-trained models are as follows.

### 2-1. Pre-trained model using NOCAP Core
- Bash command
    
    ```bash
    python3 pretrain.py --data-source CoreOragnoid_atlas.scb --save-dir pretrain_coreOrganoid --batch-size 64 --eval-batch-size 128 --epochs 10 --trunc-by-sample --no-cls --no-cce --fp16 --training-tasks both
    ```
    
- The resulting pretrained model is at `Models/Pretrained/NOCAP_pretrained`
### 2-2. Pre-trained model using human brain tissue
- The model **“scGPT_brain”** was downloaded from the official scGPT GitHub repository ([https://github.com/bowang-lab/scGPT](https://github.com/bowang-lab/scGPT)).
- This model corresponds to `Models/Pretrained/BrainTissue_pretrained`.
### 2-3. Pre-trained model using an endoderm organoid atlas
- The dataset in Xu et al. (2025) ([https://www.nature.com/articles/s41588-025-02182-6](https://www.nature.com/articles/s41588-025-02182-6)) was obtained and preprocessed prior to pre-training.
- Bash command
    
    ```bash
    python3 pretrain.py --data-source HEOCA.scb --save-dir pretrain_HEOCA --batch-size 64 --eval-batch-size 128 --epochs 10 --trunc-by-sample --no-cls --no-cce --fp16 --training-tasks both
    ```
    
- The resulting pretrained model is at `Models/Pretrained/EndodermOrganoid_pretrained`
### 2-4. Pre-trained model using the mouse brain organoid atlas
- Bash command
    
    ```bash
    python3 pretrain.py --data-source MouseAtlas_500000Normal.scb --save-dir pretrain_mouse --batch-size 64 --eval-batch-size 128 --epochs 10 --trunc-by-sample --no-cls --no-cce --fp16 --training-tasks both
    ```
    
- The resulting pretrained model is at `Models/Pretrained/Mouse_pretrained`
## [3] Fine-tuning with perturbation data

For perturbation response prediction task, we fine-tuned the foundation models. Model fine-tuning can be conducted using the code `01_train.py` located in `Codes/Finetuning`. This code generates a total of ten models by varying the train/validation/test splits. An example command for running this script is provided below.

```bash
# Activate the conda environment
conda_env_name="scgpt"
conda activate $conda_env_name

# Set path
pert_name="oragnoid_telencephalicneuron_pcgenes"
pretrain_model="data/pretrain_core_share"
adata_path="data/Output_telenNeuron_pcgenes_normalized.h5ad"
save_dir="model/coreOrganoid_telencephalicneuron"
model_dir="model/coreOrganoid_telencephalicneuron"
output_dir="output/coreOrganoid_telencephalicneuron"

# Run
echo "Step 1: Training models with seeds 1 to 10"
python3 Codes/01_train.py --adata_path $adata_path --pert_name $pert_name --pretrain_model $pretrain_model --save_dir $save_dir --preprocess
```

## [4] Evaluating predictive performances

After generating ten models, we evaluate each model using Pearson correlation, Spearman correlation, RMSE, and MAE, by comparing the predicted values with the ground truth. The evaluation is carried out on all genes, on the top 20 differentially expressed (DE) genes, on expression delta across all genes (*perturbation − control*), and on expression delta for the top 20 DE genes. In total, each model produces 16 performance values (4 metrics × 4 settings) after running the code `02_evaluation.py` at `Codes/Finetuning`. An example command for running this script is provided below.

```bash
echo "Step 2: Evaluating model of each seed"
python3 Codes/02_evaluation.py --pert_name $pert_name --pretrain_model $pretrain_model --save_dir $save_dir
```

## [5] Predicting perturbation responses

After model evaluation, we select the best model (based on the Pearson correlation of delta values for DE genes). Using this model, we generate predictions for gene expression under each perturbation condition. The predictions include expression values for all genes that the model can output.

For each prediction, 1,000 control cells are randomly sampled as the background. This random sampling is introduced to improve robustness, ensuring that the predictions are not biased by a particular set of control cells. For each perturbation condition, the model generates expression profiles for 1,000 different cells, resulting in a matrix of size `1000 cells × (num of genes)` per perturbation, which is saved as an `.h5ad` file. You can run this step using `03_prediction.py` at `Codes/Finetuning`. 

```bash
echo "Step 3: Predicting all genes in the best model"
python3 Codes/03_prediction.py --pert_name $pert_name --pretrain_model $pretrain_model --model_dir $model_dir --output_dir $output_dir
```
