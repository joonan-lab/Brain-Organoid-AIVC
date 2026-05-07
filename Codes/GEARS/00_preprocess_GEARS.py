"""
GEARS preprocessing for the telencephalic-neuron perturbation prediction task
used in the benchmark (Telen-GEARS row).

Pipeline:
  1. Load the scPoli-integrated organoid atlas h5ad
  2. Normalise perturbation-condition labels
  3. Filter to telencephalic excitatory/inhibitory neurons
  4. Restrict to protein-coding genes (19,424 pcgenes)
  5. normalize_total + log1p
  6. Write the preprocessed h5ad
  7. Run GEARS PertData.new_data_process
  8. Save a custom_gene_set.pkl matching the full pcgene vocabulary

Originally extracted from:
  process_organoid_newTelen.ipynb
  model_training_organoid_telencephalicneuron.ipynb (cells 3-4)
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import scanpy as sc

from gears import PertData


def make_perturb_condition(adata):
    """Normalise the Perturbation column into a GEARS-compatible condition label."""
    adata = adata[~adata.obs['Perturbation'].str.contains(',', na=False)]
    adata = adata[~adata.obs['Perturbation'].str.contains('treatment', case=False, na=False)]
    adata.obs['Perturbation'] = adata.obs['Perturbation'].str.replace(r'\s*\([^)]*\)', '', regex=True)
    adata.obs['condition'] = np.where(
        adata.obs['Perturbation'].isna(),
        'ctrl',
        'ctrl+' + adata.obs['Perturbation'],
    )
    return adata


def main():
    parser = argparse.ArgumentParser(description='GEARS preprocessing for telencephalic-neuron perturbation prediction')
    parser.add_argument('--raw_h5ad', type=str, required=True,
                        help='scPoli-integrated organoid atlas (outer-join) h5ad')
    parser.add_argument('--out_h5ad', type=str, required=True,
                        help='Output path for the preprocessed telen h5ad')
    parser.add_argument('--pert_data_dir', type=str, required=True,
                        help='Directory where GEARS PertData writes its processed dataset')
    parser.add_argument('--dataset_name', type=str, default='organoid_telencephalicneuron_pcgenes',
                        help='GEARS dataset name')
    parser.add_argument('--exclude_datasets', nargs='*', default=[],
                        help='Optional list of Dataset names to exclude (e.g. Li_2023 Fleck_2022)')
    args = parser.parse_args()

    print('Loading raw atlas...')
    adata = sc.read_h5ad(args.raw_h5ad)

    adata.obs['Perturbation'] = adata.obs['Perturbation'].replace('SUV420H1', 'KMT5B')
    adata = adata[adata.obs['Category'].isin(['Normal', 'Perturbed'])]
    adata = adata[adata.obs['Query_Dataset'] != 'Core']
    adata = adata[adata.obs['Cell_Type'].isin([
        'telencephalic_excitatory_neuron',
        'telencephalic_inhibitory_neuron',
    ])]
    adata = make_perturb_condition(adata)
    adata.obs.loc[adata.obs['condition'] == 'ctrl+None', 'condition'] = 'ctrl'

    if args.exclude_datasets:
        print(f'Excluding datasets: {args.exclude_datasets}')
        adata = adata[~adata.obs['Dataset'].isin(args.exclude_datasets)]

    pert_list = [c.replace('ctrl+', '') for c in adata.obs['condition'].unique() if c != 'ctrl']
    pert_in_var = np.intersect1d(pert_list, adata.var.index)
    missing = np.setdiff1d(pert_list, pert_in_var)
    if len(missing):
        print(f'Removing {len(missing)} perturbations whose target gene is absent from the expression matrix: {list(missing)}')
        adata = adata[~adata.obs['condition'].isin([f'ctrl+{g}' for g in missing])]

    adata = adata.copy()
    adata.X = adata.layers['counts'].copy()

    pcgenes = adata.var.loc[adata.var['type_of_gene'] == 'protein-coding'].index.tolist()
    adata = adata[:, pcgenes]
    adata.obs['cell_type'] = 'TelenNeuron'

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    adata.var['gene_name'] = adata.var.index
    print(f'Preprocessed shape: {adata.shape}')
    Path(args.out_h5ad).parent.mkdir(parents=True, exist_ok=True)
    adata.write(args.out_h5ad)

    print('Running GEARS PertData.new_data_process...')
    pert_data = PertData(args.pert_data_dir)
    pert_data.new_data_process(dataset_name=args.dataset_name, adata=adata)

    genes = adata.var['gene_name'].tolist()
    gene_set_path = Path(args.pert_data_dir) / args.dataset_name / 'custom_gene_set.pkl'
    gene_set_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gene_set_path, 'wb') as f:
        pickle.dump(list(set(genes)), f)
    print(f'Saved custom gene set ({len(set(genes))} genes) to {gene_set_path}')


if __name__ == '__main__':
    main()
