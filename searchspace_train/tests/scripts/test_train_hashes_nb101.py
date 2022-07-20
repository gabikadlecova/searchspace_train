import os
import sys
from click.testing import CliRunner
from unittest import mock

import pandas as pd

file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_dir, '..', '..', '..', 'scripts'))
from train_hashes_nb101 import run


def test_main(nb_path, data_dir, config_path, small_cifar):
    hash_csv = os.path.join(data_dir, 'hash_list.csv')
    with mock.patch('searchspace_train.datasets.nasbench101.prepare_dataset', lambda *args, **kwargs: small_cifar):
        runner = CliRunner()
        runner.invoke(run, args=[data_dir, hash_csv, "--nasbench", nb_path, "--config", config_path])

    dataset_path = os.path.join(data_dir, 'nb_trained_dataset.csv')

    saved_data = pd.read_csv(dataset_path, index_col=0)
    ref_data = pd.read_csv(os.path.join(data_dir, 'saved_dataset.csv'), index_col=0)

    assert os.path.exists(dataset_path)
    assert saved_data.equals(ref_data)
    assert os.path.exists(os.path.join(data_dir, saved_data.iloc[0]['net_path']))
    assert os.path.exists(os.path.join(data_dir, saved_data.iloc[0]['data_path']))

    idx = saved_data.index[0]
    os.remove(os.path.join(data_dir, saved_data.iloc[0]['net_path']))
    os.remove(os.path.join(data_dir, saved_data.iloc[0]['data_path']))
    os.remove(os.path.join(data_dir, f"{idx}_1_script.pt"))
    os.remove(os.path.join(data_dir, f"{idx}_1_data.pt"))
    os.remove(dataset_path)