# Gather model performance outputs into benchmarking summary tables.
# Notebook outputs and markdown cells were omitted.

# %% [cell 2]
import pandas as pd
import numpy as np
import os
import re
import warnings

warnings.filterwarnings('ignore')

# %% [cell 3]
outdir = '<PROJECT_ROOT>/model_performances'
date = '260202'
moran_res_file = f'<PROJECT_ROOT>/model_performances/moran/{date}/All_models_PCA_moran_score_result.{date}.tsv'

# %% [cell 4]
performance_res_path = '<PROJECT_ROOT>/model_performances/performances'
performance_res_files = os.listdir(performance_res_path)
performance_res_files = [x for x in performance_res_files if 'txt' in x]
performance_res_files = sorted(performance_res_files)
len(performance_res_files)

# %% [cell 5]
performance_res_files

# %% [cell 6]
def extract_seed(filename):
    m = re.search(r'Seed(\d+)', filename)
    return int(m.group(1)) if m else None

# %% [cell 7]
all_res = pd.DataFrame()

for i, res in enumerate(performance_res_files):
    res_file = os.path.join(performance_res_path, res)
    res_df = pd.read_csv(res_file, sep="\t")
    
    model_name = res.split('_Seed')[0]
    
    if i == 0:
        cols = list(res_df.columns)
    
    res_df = res_df.loc[:, cols]
    
    if res_df.shape[1] > 1:
        best_idx = res_df['pearson_de_delta'].idxmax()
        best_seed = best_idx + 1
        
        best_res = res_df.loc[best_idx]
    else:
        best_res = res_df
        best_seed = extract_seed(res)
        
    best_res['seed'] = int(best_seed)
    best_res['Model'] = model_name
    
    all_res = pd.concat([all_res, pd.DataFrame(best_res).T])

all_res.reset_index(drop=True, inplace=True)
all_res

# %% [cell 8]
len(all_res['Model'].unique())

# %% [cell 9]
all_moran_res = pd.read_csv(moran_res_file, sep="\t")
all_moran_res

# %% [cell 10]
len(all_moran_res['Model'].unique())

# %% [cell 11]
final_res = pd.merge(all_moran_res, all_res, on='Model', how='inner')
final_res

# %% [cell 12]
final_res.shape

# %% [cell 13]
final_res.to_excel(os.path.join(outdir, f'AIVC_model_benchmarking_result.{date}.xlsx'), index=False)

# %% [cell 14]
final_res.sort_values('Moran_I_ASD185_EAGLE', ascending=False)

# %% [cell 15]
final_res.columns

