"""
Generate the gene-by-gene perturbation prediction matrix from a trained
Telen-GEARS model (the row with the highest pearson_de_delta is typically used).

Originally extracted from:
  model_training_organoid_telencephalicneuron.ipynb (cells 11-17)

Output:
  A pandas DataFrame of shape (N_pert, N_response_genes) saved as pickle.
  Rows   = every gene that is representable as a perturbation (limited to
           the gene–GO interaction graph vocabulary; 18,627 for Telen-GEARS).
  Cols   = response genes (19,424 protein-coding genes in the dataset).
"""
import argparse
from pathlib import Path

import pandas as pd

from gears import PertData, GEARS


def main():
    parser = argparse.ArgumentParser(description='Generate Telen-GEARS perturbation-response prediction matrix')
    parser.add_argument('--pert_data_dir', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, default='organoid_telencephalicneuron_pcgenes')
    parser.add_argument('--gene_set_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved GEARS model (e.g. Seed_{best}_model)')
    parser.add_argument('--out_pickle', type=str, required=True)
    parser.add_argument('--best_seed', type=int, default=1,
                        help='Seed used for split reconstruction at prediction time')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    pert_data = PertData(args.pert_data_dir, gene_set_path=args.gene_set_path)
    pert_data.load(data_name=args.dataset_name,
                   data_path=str(Path(args.pert_data_dir) / args.dataset_name))
    pert_data.prepare_split(split='simulation', seed=args.best_seed)
    pert_data.get_dataloader(batch_size=args.batch_size,
                             test_batch_size=args.test_batch_size)

    gears_model = GEARS(pert_data, device=args.device,
                        weight_bias_track=False,
                        proj_name='gears_telen',
                        exp_name='gears_telen_predict')
    gears_model.load_pretrained(args.model_path)

    pert_genes = gears_model.pert_list
    inputs = [[g] for g in pert_genes]
    print(f'Predicting perturbation response for {len(pert_genes)} genes...')
    predictions = gears_model.predict(inputs)

    response_genes = pert_data.adata.var['gene_name'].tolist()
    df = pd.DataFrame.from_dict(predictions, orient='index')
    df.columns = response_genes
    print(f'Prediction matrix: {df.shape}  (perturbations × response genes)')

    Path(args.out_pickle).parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(args.out_pickle)
    print(f'Wrote {args.out_pickle}')


if __name__ == '__main__':
    main()
