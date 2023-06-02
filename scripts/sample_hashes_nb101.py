import click
import os

import pandas as pd
from sklearn.model_selection import train_test_split


@click.command()
@click.option('--hash_csv', default=None)
@click.option('--save_dir')
@click.option('--prefix', default='')
@click.option('--seed', default=None, type=int)
@click.option('--sample_size', default=0.1, help="Float for size as full size ratio, int for number of sampled nets.")
@click.option('--nasbench', required=True, help='Path to the nasbench dataset (can be downloaded from '
                                                'https://github.com/google-research/nasbench#download-the-dataset ).')
@click.option('--train_val_split/--sample', default=False, help='Save data and checkpoint paths only as a basename '
                                                                'of the file (the files are saved in `save_name`.')
def run(hash_csv, save_dir, prefix, seed, sample_size, nasbench, train_val_split):
    """
    If --train_val_split is passed to the script, nasbench hashes will be split to train and val hashes and saved
    to `save_dir/{prefix}{train, val}_hashes.csv`.

    Otherwise, hashes are sampled from --hash_csv and saved to `save_dir/{prefix}_hashes.csv`
    Train networks specified in HASH_CSV (pandas dataframe with one column named 'hash' - NAS-Bench-101 hashes),
    and save the dataset and network checkpoint & data in SAVE_DIR.
    """
    assert sample_size > 0
    prefix = f"{prefix}_" if len(prefix) else prefix

    if train_val_split:
        from searchspace_train.datasets.nasbench101 import load_nasbench

        nb = load_nasbench(nasbench)
        hashes = pd.DataFrame({'hashes': [h for h in nb.hash_iterator()]})

        train, val = train_test_split(hashes, test_size=sample_size, random_state=seed)
        train.to_csv(os.path.join(save_dir, f"{prefix}train_hashes.csv"))
        val.to_csv(os.path.join(save_dir, f"{prefix}val_hashes.csv"))
    else:
        hashes = pd.read_csv(hash_csv)
        _, sample = train_test_split(hashes, test_size=sample_size, random_state=seed)
        sample.to_csv(os.path.join(save_dir, f"{prefix}hashes.csv"))


if __name__ == "__main__":  # pragma: no cover
    run()
