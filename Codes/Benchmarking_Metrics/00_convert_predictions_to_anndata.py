# Convert model prediction matrices to AnnData/H5AD format for downstream benchmarking.
# Notebook outputs and markdown cells were omitted.

# %% [cell 1]
# Load necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
import scipy.stats as st
import libpysal
import warnings 
import argparse

# %% [cell 2]
model_name = 'Telen-GeneCompass'
pred_file = "<DATA_ROOT>/07.Benchmarking_Downstream_Analysis/data/Telen-GeneCompass_seed2.pickle"
res_dir = f"<PROJECT_ROOT>/results/{model_name}"
fig_dir = f"{res_dir}/figures"

os.makedirs(res_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# %% [cell 3]
gnomad_data_file = '<DATA_ROOT>/resources/gnomad.v4.1.constraint_metrics_by_canonical_transcript_gene.tsv'
asd_gene_file = '<DATA_ROOT>/resources/ASD_risk_genes_ASD185_EAGLE_ASD_NDD664.250901.xlsx'

gnomad_data = pd.read_csv(gnomad_data_file, sep='\t')
gnomad_cols = ['gene','transcript_type','cds_length','oe_lof_upper','oe_lof_upper_bin','mis_z','constraint_flag','pLI']
gnomad_data = gnomad_data.loc[:, gnomad_cols]
asd_gene = pd.read_excel(asd_gene_file)

# %% [cell 5]
pred_res = pd.read_pickle(pred_file).T
pred_res

# %% [cell 6]
adata = ad.AnnData(pred_res)
adata.obs.index = pred_res.index
adata.var.index = pred_res.columns

# %% [cell 7]
adata.var

# %% [cell 9]
adata_obs = adata.obs.copy()
adata_obs['perturbed_gene'] = adata_obs.index
adata_obs['gene'] = adata_obs.index.str.replace('_perturbed', '', regex=False)
adata_obs = pd.merge(adata_obs, gnomad_data, on='gene', how='left')
adata_obs = pd.merge(adata_obs, asd_gene, on='gene', how='left')
adata_obs.loc[:, asd_gene.columns] = adata_obs.loc[:, asd_gene.columns].fillna(0)
adata_obs.set_index('perturbed_gene', inplace=True)
adata_obs.rename_axis('', axis=0, inplace=True)
adata.obs = adata_obs.loc[adata.obs_names, :]

# %% [cell 10]
adata.var.head()

# %% [cell 11]
adata.obs.head()

# %% [cell 13]
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='cell_ranger')
adata_copy = adata[:, adata.var['highly_variable']].copy()
# adata: 이미 로딩된 AnnData
# 1) PCA (scDEED에서 K=40으로 결정했다고 가정)
sc.tl.pca(adata_copy, n_comps=50)

# 2) neighbors (scDEED에서 고른 n_neighbors 사용)
sc.pp.neighbors(
    adata_copy,
    n_pcs=40,
    random_state=0,
)

    # 3) UMAP (scDEED에서 고른 min_dist 사용 – 시각화용)
sc.tl.umap(
    adata_copy,
    random_state=0,
)

# %% [cell 14]
adata.obsm = adata_copy.obsm

# %% [cell 15]
adata

# %% [cell 16]
fig_adata = adata.copy()
fig_adata.obs['ASD185_EAGLE'].fillna(0, inplace=True)
fig_adata.obs['NDD664'].fillna(0, inplace=True)


## ASD185_EAGLE UMAP plot
fig_adata.obs['ASD185_EAGLE'] = fig_adata.obs['ASD185_EAGLE'].map({1: 'TRUE', 0: 'FALSE'})
fig_adata.obs['ASD185_EAGLE'] = pd.Categorical(fig_adata.obs['ASD185_EAGLE'], categories=['TRUE','FALSE'], ordered=True)
asd_colors = {'TRUE': "#7704EA", 'FALSE': "#BFBFBF"}

## NDD664 UMAP plot
fig_adata.obs['NDD664'] = fig_adata.obs['NDD664'].map({1: 'TRUE', 0: 'FALSE'})
fig_adata.obs['NDD664'] = pd.Categorical(fig_adata.obs['NDD664'], categories=['TRUE','FALSE'], ordered=True)

ndd_colors = {'TRUE': "#FFA600", 'FALSE': "#BFBFBF"}

fig, ax = plt.subplots(1, 3, figsize=(17, 5))

sc.pl.umap(fig_adata, color='ASD185_EAGLE', title='ASD185_EAGLE', palette=asd_colors, show=False, size=30, ax=ax[0])
sc.pl.umap(fig_adata, color='NDD664', title='NDD664', palette=ndd_colors, show=False, size=30, ax=ax[1])
sc.pl.umap(fig_adata, color='pLI', title='pLI', cmap='viridis_r', show=False, size=30, ax=ax[2])

plt.tight_layout()
plt.savefig(f"{fig_dir}/{model_name}_UMAP_ASD_pLI.pdf", bbox_inches='tight')

# %% [cell 17]
adata.write_h5ad(f"{res_dir}/{model_name}.h5ad")

