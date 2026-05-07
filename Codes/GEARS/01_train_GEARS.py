"""
Train Telen-GEARS across multiple seeds.

Originally extracted from:
  model_training_organoid_telencephalicneuron.ipynb (cell 5)

Hyperparameters (from the manuscript Methods):
  - hidden_size = 64
  - epochs = 15
  - learning_rate = 1e-4
  - batch_size = 64 (train) / 64 (test)
  - split = 'simulation' (unseen single-gene perturbation)
  - GEARS default GO graph (no custom co-expression graph)
"""
import argparse
import time
from pathlib import Path

from gears import PertData, GEARS


def train_seed(pert_data_dir, dataset_name, gene_set_path, save_dir, seed, args):
    pert_data = PertData(pert_data_dir, gene_set_path=gene_set_path)
    pert_data.load(data_name=dataset_name,
                   data_path=str(Path(pert_data_dir) / dataset_name))
    pert_data.prepare_split(split='simulation', seed=seed)
    pert_data.get_dataloader(batch_size=args.batch_size,
                             test_batch_size=args.test_batch_size)

    gears_model = GEARS(
        pert_data,
        device=args.device,
        weight_bias_track=False,
        proj_name='gears_telen',
        exp_name=f'gears_telen_seed_{seed}',
    )
    gears_model.model_initialize(hidden_size=args.hidden_size)
    gears_model.train(epochs=args.epochs, lr=args.lr)

    out_dir = Path(save_dir) / f'Seed_{seed}_model'
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    gears_model.save_model(str(out_dir))
    return out_dir


def main():
    parser = argparse.ArgumentParser(description='Train Telen-GEARS across seeds')
    parser.add_argument('--pert_data_dir', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, default='organoid_telencephalicneuron_pcgenes')
    parser.add_argument('--gene_set_path', type=str, required=True,
                        help='Path to custom_gene_set.pkl produced by 00_preprocess_GEARS.py')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory where per-seed checkpoints are written')
    parser.add_argument('--seed_start', type=int, default=1)
    parser.add_argument('--seed_end', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    for seed in range(args.seed_start, args.seed_end + 1):
        start = time.time()
        print(f'=== Seed {seed} ===', flush=True)
        out_dir = train_seed(
            pert_data_dir=args.pert_data_dir,
            dataset_name=args.dataset_name,
            gene_set_path=args.gene_set_path,
            save_dir=args.save_dir,
            seed=seed,
            args=args,
        )
        elapsed = time.time() - start
        print(f'Seed {seed} done in {elapsed:.1f}s. Saved to {out_dir}', flush=True)


if __name__ == '__main__':
    main()
