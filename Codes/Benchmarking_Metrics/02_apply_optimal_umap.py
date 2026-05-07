# Aggregate scDEED grid-search results and apply optimal UMAP parameters.
# Notebook outputs and markdown cells were omitted.

# %% [cell 2]
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import silhouette_score
import time
import matplotlib.pyplot as plt
import os, glob
from tqdm import tqdm
import warnings
import matplotlib as mpl
from pybiomart import Server

warnings.filterwarnings("ignore")

# %% [cell 3]
mpl.rcParams['pdf.fonttype'] = 42

sc.settings.n_jobs = 36
sc.set_figure_params(figsize=(4,5), vector_friendly = True)
# NOTE: notebook magic/shell command removed for script syntax: %config InlineBackend.print_figure_kwargs={'facecolor' : "w"}
# NOTE: notebook magic/shell command removed for script syntax: %config InlineBackend.figure_format='retina'

warnings.filterwarnings("ignore")

# %% [cell 5]
def adata_umap(adata, n_neighs=15, min_dist=0.5, prep_skip=False):
    pass
    if prep_skip:
        return adata
    else:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='cell_ranger')
        #adata_copy = adata[:, adata.var['highly_variable']].copy()
        # adata: 이미 로딩된 AnnData
        # 1) PCA (scDEED에서 K=40으로 결정했다고 가정)
        sc.tl.pca(adata, n_comps=50, mask_var='highly_variable')

        # 2) neighbors (scDEED에서 고른 n_neighbors 사용)
        sc.pp.neighbors(
            adata,
            n_pcs=40,
            n_neighbors=n_neighs,
            random_state=0,
        )

        # 3) UMAP (scDEED에서 고른 min_dist 사용 – 시각화용)
        sc.tl.umap(
            adata,
            min_dist=min_dist,
            random_state=0,
        )
    
    return adata

# %% [cell 6]
opt_date = '260131'
date = '260202'
opt_file_dir = "<PROJECT_ROOT>/grid_search"
model_res_dir = "<PROJECT_ROOT>/results"
outdir = '<PROJECT_ROOT>/results_with_scDEED/latest'
fig_dir = f'{outdir}/Figures/'

os.makedirs(outdir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# %% [cell 7]
gnomad_data_file = '<DATA_ROOT>/resources/gnomad.v4.1.constraint_metrics_by_canonical_transcript_gene.tsv'
asd_gene_file = '<DATA_ROOT>/resources/ASD_risk_genes_ASD185_EAGLE_ASD_NDD664.250901.xlsx'

gnomad_data = pd.read_csv(gnomad_data_file, sep='\t')
gnomad_cols = ['gene','transcript_type','cds_length','oe_lof_upper','oe_lof_upper_bin','mis_z','constraint_flag','pLI']
gnomad_data = gnomad_data.loc[:, gnomad_cols]
asd_gene = pd.read_excel(asd_gene_file)

# %% [cell 8]
opt_params = pd.read_excel(f'{opt_file_dir}/All_model_prediction_embedding_optimal_paramters.{opt_date}.xlsx')
opt_params

# %% [cell 9]
res_files = glob.glob(outdir +'/*.h5ad')
res_models = [os.path.basename(f).split('.h5ad')[0] for f in res_files]
len(res_models)

# %% [cell 10]
h5ad_files = glob.glob(f'{model_res_dir}/*/*.h5ad')
len(h5ad_files)

# %% [cell 11]
h5ad_files

# %% [cell 12]
for f in tqdm(h5ad_files):
    f_name = os.path.basename(f)
    model_name = f_name.replace('_AllPrediction_umap_annotated', '').split('.h5ad')[0] if "AllPrediction" in f_name else f_name.replace('.h5ad', '')
    print(f"Loading {model_name} from {f}...")
    
    try:    
        if model_name in res_models:
            print(f"Skipping {model_name}: Already {model_name} was processed\n")
            continue
        elif model_name not in opt_params['Model'].values:
            print(f"Skipping {model_name}: No optimal parameters for {model_name}\n")
            continue
        else:
            adata = sc.read_h5ad(f)

            model_params = opt_params.loc[opt_params['Model']==model_name]
            n = model_params['n_neighs'].values[0]
            m = model_params['min_dist'].values[0]
            
            print(f"Processing {model_name} with n_neighbors={n}, min_dist={m}...")
            umap_adata = adata_umap(adata, n_neighs=n, min_dist=m, prep_skip=False)       
        
            print('Add metadata...')
            umap_adata.obs['cds_length'] = 0
            umap_adata_obs = umap_adata.obs.copy()
            umap_adata_obs.insert(0, 'perturbation', umap_adata_obs.index)
            umap_adata_obs.drop(gnomad_cols, axis=1, inplace=True) if set(gnomad_cols)&set(umap_adata_obs.columns) == set(gnomad_cols) else umap_adata_obs.drop(list(set(gnomad_cols)&set(umap_adata_obs.columns)), axis=1, inplace=True)
            umap_adata_obs['gene'] = umap_adata_obs['perturbation'].str.split('_').str[0]
            
            if model_name == 'Telen-Mouse':
                server = Server(host='http://www.ensembl.org')
                mouse_ds = server.marts['ENSEMBL_MART_ENSEMBL'].datasets['mmusculus_gene_ensembl']

                mouse_to_human = mouse_ds.query(attributes=[
                    'external_gene_name',
                    'hsapiens_homolog_associated_gene_name'
                ])
                mouse_to_human_dict = {k: v for k, v in zip(mouse_to_human['Gene name'], mouse_to_human['Human gene name'])}
                umap_adata_obs['gene'] = umap_adata_obs['gene'].apply(lambda x: mouse_to_human_dict[x] if x in mouse_to_human_dict.keys() else x)
            
            umap_adata_obs = pd.merge(umap_adata_obs, gnomad_data, on='gene', how='left')                           
            
            comm_asd_gene_cols = list(set(asd_gene.columns)&set(umap_adata_obs.columns)-set(['gene']))
            umap_adata_obs.drop(comm_asd_gene_cols, axis=1, inplace=True) if len(comm_asd_gene_cols) > 0 else umap_adata_obs
            umap_adata_obs = pd.merge(umap_adata_obs, asd_gene, on='gene', how='left')
                
            if model_name == 'Telen-Mouse':
                umap_adata_obs.rename({'gene':'human_gene'}, axis=1, inplace=True)
                umap_adata_obs['gene'] = umap_adata_obs['perturbation'].str.split('_').str[0]
            
            umap_adata_obs.set_index('perturbation', drop=True, inplace=True)
            umap_adata.obs = umap_adata_obs.loc[umap_adata.obs_names]
            
            print("Drawing optimal UMAP plots...")
            fig_adata = umap_adata.copy()
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
            plt.savefig(f"{fig_dir}/{model_name}_optimal_UMAP_ASD_pLI.pdf", bbox_inches='tight')
                        
            umap_adata.write_h5ad(f"{outdir}/{model_name}.h5ad")
            print('Done!')
            print()
    except Exception as e:
        print(f"\n# Error: {e}")

