import os.path

import nasbench.api
import numpy as np
import pandas as pd
import pytest
from unittest import mock
from unittest.mock import mock_open

import torch.jit

from searchspace_train.datasets.nasbench101 import load_nasbench, get_net_from_hash, PretrainedNB101
from searchspace_train.utils import load_config


@pytest.fixture
def net_hash():
    return '00005c142e6f48ac74fdcf73e3439874'


def test_get_net_from_hash(nb_path, net_hash):
    res = load_nasbench(nb_path)
    ops, adj = get_net_from_hash(res, net_hash)

    ops_true = ['input', 'conv3x3-bn-relu', 'maxpool3x3', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv1x1-bn-relu',
                'output']

    adj_true = np.array([
        [0, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0]
    ])

    assert ops == ops_true
    assert (adj == adj_true).all()


def get_mock_nasbench(path):
    return "nb"


def test_pretrainednb101_init_with_dataset(small_cifar):
    pnb = PretrainedNB101("nb", dataset=small_cifar)
    assert pnb.dataset is not None


def test_pretrainednb101_init_with_config(small_cifar, config_path):
    with mock.patch('searchspace_train.datasets.nasbench101.prepare_dataset', lambda *args, **kwargs: small_cifar):
        pnb = PretrainedNB101("nb", config=config_path)
        assert pnb.dataset is not None

        config = load_config(config_path)
        pnb = PretrainedNB101("nb", config=config)
        assert pnb.dataset is not None


def test_pretrainednb101_init_invalid_config(data_dir):
    bad_config = os.path.join(data_dir, 'bad_config.yaml')
    with pytest.raises(ValueError):
        PretrainedNB101("nb", config=bad_config)


def test_pretrainednb101_no_dataset_config(data_dir):
    with pytest.raises(AssertionError):
        PretrainedNB101("nb")


def train_net(nb_path, config_path, dataset, net_hash, data_dir, as_basename=True):
    nb = load_nasbench(nb_path)
    pnb = PretrainedNB101(nb, config=config_path, dataset=dataset, verbose=False, as_basename=as_basename)
    net = pnb.train(net_hash, save_dir=data_dir)
    return pnb, net


def are_nets_same(pnb, net, net_saved):
    train_data = pnb.dataset['train']
    for batch, _ in train_data:
        out = net(batch)
        out_saved = net_saved(batch)
        return torch.equal(out, out_saved)


def cleanup(pnb, net_hash, data_dir):
    data = pnb.net_data.loc[net_hash]
    os.remove(os.path.join(data_dir, data['net_path']))
    os.remove(os.path.join(data_dir, data['data_path']))
    os.remove(os.path.join(data_dir, f"{net_hash}_1_script.pt"))
    os.remove(os.path.join(data_dir, f"{net_hash}_1_data.pt"))


def test_pretrainednb101_train_save(nb_path, config_path, small_cifar, net_hash, data_dir):
    # train
    pnb, net = train_net(nb_path, config_path, small_cifar, net_hash, data_dir, as_basename=False)

    # check if saved correctly
    assert net_hash in pnb.net_data.index
    data = pnb.net_data.loc[net_hash]
    assert 'net_path' in data, "Net path missing in dataset."
    assert 'data_path' in data, "Data path missing in dataset"
    assert data['net_path'] == os.path.join(data_dir, f'{net_hash}_script.pt')
    assert data['data_path'] == os.path.join(data_dir, f'{net_hash}_data.pt')

    assert os.path.exists(os.path.join(data_dir, f'{net_hash}_1_script.pt'))  # periodic checkpoint
    assert os.path.exists(data['net_path']), f"Network checkpoint was not saved: {data['net_path']}"
    assert os.path.exists(data['data_path']), f"Training data was not saved: {data['data_path']}"

    net_saved = torch.jit.load(data['net_path'])

    # check saved checkpoint
    assert are_nets_same(pnb, net, net_saved)

    # check saved metrics
    metrics = torch.load(data['data_path'])

    # check if sizes okay
    keys = ['train', 'val', 'test']
    keys = [f'{k}_loss' for k in keys] + [f'{k}_accuracy' for k in keys]
    for k in keys:
        assert k in metrics

    config = load_config(config_path)
    data_len = config['train']['num_epochs']
    assert all([len(metrics[k]) == data_len for k in keys if 'train' in k or 'val' in k])  # same number of epochs
    assert isinstance(metrics['test_accuracy'], float)  # evaluated once
    assert isinstance(metrics['test_loss'], float)

    cleanup(pnb, net_hash, data_dir)


def test_pretrainednb101_dataset_save(nb_path, config_path, small_cifar, net_hash, data_dir):
    pnb, _ = train_net(nb_path, config_path, small_cifar, net_hash, data_dir)

    save_path = os.path.join(data_dir, 'net_dataset.csv')
    pnb.save_dataset(save_path)

    saved = pd.read_csv(save_path, index_col=0)
    ref = pd.read_csv(os.path.join(data_dir, 'saved_dataset.csv'), index_col=0)
    assert saved.equals(ref)

    assert net_hash in saved.index

    cleanup(pnb, net_hash, data_dir)
    os.remove(save_path)


def test_pretrainednb101_load_dataset_and_net(data_dir, config_path, small_cifar, net_hash):
    saved = pd.read_csv(os.path.join(data_dir, 'saved_dataset.csv'), index_col=0)
    pnb = PretrainedNB101("nb", config=config_path, dataset=small_cifar, verbose=False, as_basename=True,
                          net_data=saved)
    pnb.get_network(net_hash, dir_path=data_dir)



def test_pretrainednb101_get_trained_hashes(data_dir, small_cifar, config_path):
    saved = pd.read_csv(os.path.join(data_dir, 'nb_res.csv'), index_col=0)
    pnb = PretrainedNB101("nb", config=config_path, dataset=small_cifar, verbose=False, as_basename=True,
                          net_data=saved)
    hash_list = pnb.get_trained_hashes()
    assert hash_list == ['hash1', 'hash2']


def test_pretrainednb101_load_net_data(nb_path, config_path, data_dir, small_cifar, net_hash):
    # train
    pnb, net_trained = train_net(nb_path, config_path, small_cifar, net_hash, data_dir)
    net = pnb.get_network(net_hash, dir_path=data_dir)

    data = net.load_data()
    checkpoint = net.load_checkpoint()
    assert isinstance(data, dict)
    assert isinstance(checkpoint, torch.nn.Module)
    assert are_nets_same(pnb, checkpoint, net_trained)

    cleanup(pnb, net_hash, data_dir)


@mock.patch("builtins.open", new_callable=mock_open, read_data="")
@mock.patch('pickle.load', get_mock_nasbench)
def test_load_nasbench_pickle(mock_file):
    pickle_path = 'nb.pickle'
    res = load_nasbench(pickle_path)
    assert res == "nb"
    mock_file.assert_called_with(pickle_path, 'rb')


@mock.patch('nasbench.api.NASBench', get_mock_nasbench)
def test_load_nasbench_tfrecord():
    res = load_nasbench('nb.tfrecord')
    assert res == "nb"


def test_load_nasbench_error():
    with pytest.raises(ValueError):
        load_nasbench('nb.txt')


def test_load_nasbench_pickle_real(nb_path):
    res = load_nasbench(nb_path)

    assert isinstance(res, nasbench.api.NASBench)
