import click
import os
import pandas as pd

from searchspace_train.datasets.nasbench101 import PretrainedNB101, load_nasbench
from searchspace_train.utils import print_verbose


@click.command()
@click.argument('save_dir')
@click.argument('hash_csv')
@click.option('--iloc', default=None)
@click.option('--dataset_name', default='nb_trained_dataset.csv', help='Name of the dataset .csv to be saved.')
@click.option('--nasbench', required=True, help='Path to the nasbench dataset (can be downloaded from '
                                                'https://github.com/google-research/nasbench#download-the-dataset ).')
@click.option('--device', default=None, type=str, help='Device to train on.')
@click.option('--config', default='../searchspace_train/configs/nb101_cifar.yaml', type=str,
              help='Path to dataset and train config.')
@click.option('--verbose', default=False, is_flag=True, show_default=True, help='If True, print more info.')
@click.option('--as_basename/--as_absolute', default=True, help='Save data and checkpoint paths only as a basename '
                                                                'of the file (the files are saved in `save_name`.')
def run(save_dir, hash_csv, iloc, dataset_name, nasbench, device, config, verbose, as_basename):
    """
    Train networks specified in HASH_CSV (pandas dataframe with one column named 'hash' - NAS-Bench-101 hashes),
    and save the dataset and network checkpoint & data in SAVE_DIR.
    """
    hashes = pd.read_csv(hash_csv)
    if iloc is not None:
        if '-' in iloc:
            iloc = iloc.split('-')
            assert len(iloc) == 2
            hashes = hashes.iloc[int(iloc[0]):int(iloc[1])]
        else:
            iloc = [int(i) for i in iloc.split(',')] if ',' in iloc else int(iloc)
            hashes = hashes.iloc[iloc]

    nb = load_nasbench(nasbench)
    pnb = PretrainedNB101(nb, device=device, config=config, verbose=verbose, as_basename=as_basename)

    for net_hash in hashes['hash']:
        print_verbose(f"Training network {net_hash}.", verbose)
        pnb.train(net_hash, save_dir=save_dir)

    save_path = os.path.join(save_dir, dataset_name)
    print_verbose(f"Saving dataset to {save_path}", verbose)
    pnb.save_dataset(save_path)


if __name__ == "__main__":  # pragma: no cover
    run()
