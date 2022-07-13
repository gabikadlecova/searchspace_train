import os
import pandas as pd
import pytest

from searchspace_train.utils import print_verbose, concat_datasets, load_config


def test_print_verbose_true(capfd):
    print_verbose("Test", True)
    out, _ = capfd.readouterr()
    assert out.strip() == "Test"


def test_print_verbose_false(capfd):
    print_verbose("Test", False)
    out, _ = capfd.readouterr()
    assert out == ""


def test_concat_datasets(data_dir):
    data = [os.path.join(data_dir, 'nb_1.csv'), os.path.join(data_dir, 'nb_2.csv')]
    data = concat_datasets(data)

    data_ref = pd.read_csv(os.path.join(data_dir, 'nb_res.csv'), index_col=0)
    assert data.equals(data_ref)


def test_load_config(config_path):
    cfg = load_config(config_path)

    assert 'dataset' in cfg
    assert cfg['dataset']['name'] == 'CIFAR-10'
