# Compute PCA-space Moran's I for ASD/NDD gene-set coherence.
# Notebook outputs and markdown cells were omitted.

# %% [cell 3]
import warnings

warnings.filterwarnings('ignore')

# %% [cell 4]
# Load necessary libraries
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
import scipy.stats as st
import libpysal
from esda.moran import Moran
import warnings 
from tqdm import tqdm
import random
from libpysal.weights import W
from statsmodels.stats.multitest import fdrcorrection
from pybiomart import Server

# %% [cell 5]
sc.settings.verbosity = 3
sc.set_figure_params(vector_friendly = True)

# %% [cell 6]
np.random.seed(42)
random.seed(42)

# %% [cell 7]
# Load gnomAD constraint metrics and ASD risk gene list for annotating genes
gnomad_data_file = '<DATA_ROOT>/resources/gnomad.v4.1.constraint_metrics_by_canonical_transcript_gene.tsv'
asd_gene_file = '<DATA_ROOT>/resources/ASD_risk_genes_ASD185_EAGLE_ASD_NDD664.250901.xlsx'

gnomad_data = pd.read_csv(gnomad_data_file, sep='\t')
gnomad_cols = ['gene','transcript_type','cds_length','oe_lof_upper','oe_lof_upper_bin','mis_z','constraint_flag','pLI']
gnomad_data = gnomad_data.loc[:, gnomad_cols].drop_duplicates(subset=['gene'])
gnomad_data

# %% [cell 8]
asd_gene = pd.read_excel(asd_gene_file)
asd_gene = asd_gene.drop_duplicates(subset=['gene'])
asd_gene

# %% [cell 9]
date = '260202'
output_dir = f'<PROJECT_ROOT>/model_performances/moran/{date}'
adata_dir = '<PROJECT_ROOT>/results_with_scDEED/latest'

os.makedirs(output_dir, exist_ok=True)

# %% [cell 10]
k = 40
n_pcs = 30
n_perm = 1000

# %% [cell 11]
adata_files = glob.glob(adata_dir + '/*.h5ad')
len(adata_files)

# %% [cell 13]
pca_moran_res = pd.DataFrame()

for i, adata_path in enumerate(adata_files):
    adata = sc.read_h5ad(adata_path)
    
    model_name = os.path.basename(adata_path).split('.h5ad')[0]

    try:
        pca = adata.obsm['X_pca'][:, :n_pcs].copy()
        
        pca_df = pd.DataFrame(
            pca,
            columns=[f"PC{i+1}" for i in range(n_pcs)],
            index=adata.obs_names
        )
        
        if 'ASD185_EAGLE' in adata.obs.columns:
            pca_df['ASD185_EAGLE'] = adata.obs.loc[pca_df.index, 'ASD185_EAGLE']
        else:
            pca_df['gene'] = pca_df.index.str.replace('_perturbed', '', regex=False)
            
            if model_name == 'Telen-Mouse':
                server = Server(host='http://www.ensembl.org')
                mouse_ds = server.marts['ENSEMBL_MART_ENSEMBL'].datasets['mmusculus_gene_ensembl']

                mouse_to_human = mouse_ds.query(attributes=[
                    'external_gene_name',
                    'hsapiens_homolog_associated_gene_name'
                ])

                mouse_to_human_dict = {k: v for k, v in zip(mouse_to_human['Gene name'], mouse_to_human['Human gene name'])}
                pca_df['gene'] = pca_df['gene'].apply(lambda x: mouse_to_human_dict[x] if x in mouse_to_human_dict.keys() else x)
                            
            # Merge with annotation data
            pca_df = pca_df.merge(asd_gene, on='gene', how='left')
        
            
        pca_df['ASD185_EAGLE'] = pca_df['ASD185_EAGLE'].fillna(0).astype(int)
        
        ## Moran's I calculation using final pca for ASD risk genes
        X = pca_df.loc[:, [f'PC{i+1}' for i in range(n_pcs)]].values
        asd_labels = (pca_df['ASD185_EAGLE']==1).astype(int).values

        w = libpysal.weights.KNN.from_array(X, k=k)
        w.transform = "R"
        mi = Moran(
            asd_labels, w, permutations=n_perm, two_tailed=False) # Global Moran's score, one-sided permutation
        print(f"({i+1}/{len(adata_files)}) ({model_name}) Moran's I: {mi.I:.4f}  (p={mi.p_sim:.4f})")

        moran_res = pd.DataFrame({
            'Model': model_name,
            'Moran_I_ASD185_EAGLE': mi.I,
            'p_value': mi.p_sim,
            'z_score': mi.z_sim,
            'EI_sim': mi.EI_sim,
            'permutations': n_perm,
            'k': k
        }, index=[0])

        pca_moran_res = pd.concat([pca_moran_res, moran_res])
        moran_res.to_csv(os.path.join(output_dir, f"{model_name}_moran_score_result.{date}.tsv"), index=False, sep="\t")
    except Exception as e:
        print(f"# Error: {model_name}: {e}")

# %% [cell 14]
pca_moran_res['FDR'] = fdrcorrection(pca_moran_res['p_value'].tolist())[1]
pca_moran_res.loc[pca_moran_res['FDR']<0.05].sort_values('Moran_I_ASD185_EAGLE', ascending=False)

# %% [cell 15]
pca_moran_res

# %% [cell 16]
pca_moran_res.to_csv(os.path.join(output_dir, f"All_models_PCA_moran_score_result.{date}.tsv"), index=False, sep="\t")

# %% [cell 19]
for i, adata_path in enumerate(adata_files):
    adata = sc.read_h5ad(adata_path)
    
    model_name = os.path.basename(adata_path).split('.h5ad')[0]

    try:
        umap = adata.obsm['X_umap'].copy()
        umap_df = pd.DataFrame(umap,
                            columns=['UMAP1', 'UMAP2'],
                            index=adata.obs_names)
        umap_df['gene'] = umap_df.index.str.replace('_perturbed', '', regex=False)
            
        # Merge with annotation data
        umap_df = umap_df.merge(gnomad_data, on='gene', how='left')
        umap_df = umap_df.merge(asd_gene, on='gene', how='left')
        
        umap_df['ASD185_EAGLE'] = umap_df['ASD185_EAGLE'].fillna(0).astype(int)
        umap_df['pLI'] = umap_df['pLI'].fillna(0)
        
        umap_df['high_pLI'] = umap_df['pLI'] > 0.9 #0.995에서 0.9로 변경
        
        ## Moran's I calculation using final UMAP for ASD risk genes
        X = umap_df.loc[:, ['UMAP1','UMAP2']].values
        asd_labels = (umap_df['ASD185_EAGLE']==1).astype(int).values

        w = libpysal.weights.KNN.from_array(X, k=k)
        w.transform = "R"
        mi = Moran(asd_labels, w, permutations=n_perm, two_tailed=False) # Global Moran's score, one-sided permutation
        print(f"({i+1}/{len(adata_files)}) ({model_name}) Moran's I: {mi.I:.4f}  (p={mi.p_sim:.4f})")

        moran_res = pd.DataFrame({
            'Model': model_name,
            'Moran_I_ASD185_EAGLE': mi.I,
            'p_value': mi.p_sim,
            'z_score': mi.z_sim,
            'EI_sim': mi.EI_sim,
            'permutations': n_perm,
            'k': k
        }, index=[0])

        moran_res.to_csv(os.path.join(output_dir, f"{model_name}_moran_score_result.tsv"), index=False, sep="\t")
    except Exception as e:
        print(f"# Error: {e}")

# %% [cell 21]
graph_moran_res = pd.DataFrame()

for n, adata_path in enumerate(adata_files):
    adata = sc.read_h5ad(adata_path)
    
    model_name = os.path.basename(adata_path).split('.h5ad')[0]

    try:
        asd_labels = (adata.obs['ASD185_EAGLE']==1).astype(int).values
        
        if 'connectivities' not in adata.obsp:
            raise ValueError(f"Neighbors graph not found. Run sc.pp.neighbors first. : {model_name}")
        else:
            conn = adata.obsp['connectivities']
            
            neighbors = {}
            weights = {}
            
            for i in range(conn.shape[0]):
                row = conn[i].nonzero()[1]
            
                if len(row) > 0:
                    w = conn[i, row].toarray().ravel()
                    
                    neighbors[i] = row.tolist()
                    weights[i] = w.tolist()
            
            w_graph = W(neighbors, weights)
            w_graph.transform = 'R'
        
            mi = Moran(
                asd_labels, 
                w_graph, 
                permutations=n_perm, 
                two_tailed=False
            ) # Global Moran's score, one-sided permutation
            print(f"({n+1}/{len(adata_files)}) ({model_name}) Moran's I: {mi.I:.4f}  (p={mi.p_sim:.4f})")

            moran_res = pd.DataFrame({
                'Model': model_name,
                'Moran_I_ASD185_EAGLE': mi.I,
                'p_value': mi.p_sim,
                'z_score': mi.z_sim,
                'EI_sim': mi.EI_sim,
                'permutations': n_perm,
                'k': k
            }, index=[0])

            graph_moran_res = pd.concat([graph_moran_res, moran_res])
            #moran_res.to_csv(os.path.join(output_dir, f"{model_name}_moran_score_result.tsv"), index=False, sep="\t")
    except Exception as e:
        print(f"# Error: {e}")

# %% [cell 22]
graph_moran_res.sort_values("Moran_I_ASD185_EAGLE", ascending=False)

# %% [cell 23]
adata = sc.read_h5ad('<PROJECT_ROOT>/results/Telen-C2S/Telen-C2S.h5ad')
adata

