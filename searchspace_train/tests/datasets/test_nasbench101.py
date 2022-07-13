import nasbench.api
import numpy as np
import pytest
from unittest import mock
from unittest.mock import mock_open
from searchspace_train.datasets.nasbench101 import load_nasbench, get_net_from_hash


@pytest.fixture
def net_hash():
    return '00005c142e6f48ac74fdcf73e3439874'


def test_get_net_from_hash(nb_path, net_hash):
    res = load_nasbench(nb_path)
    ops, adj = get_net_from_hash(res, net_hash)

    ops_true = ['input', 'conv3x3-bn-relu', 'maxpool3x3', 'conv3x3-bn-relu','conv3x3-bn-relu', 'conv1x1-bn-relu',
                'output']
    adj_true = np.array(
       [[0, 1, 0, 0, 1, 1, 0],
       [0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 1],
       [0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0]]
    )

    assert ops == ops_true
    assert (adj == adj_true).all()


def get_mock_nasbench(path):
    return "nb"


@mock.patch("builtins.open", new_callable=mock_open, read_data="")
@mock.patch('pickle.load', get_mock_nasbench)
def test_load_nasbench_pickle(mock_file):
    pickle_path = 'nb.pickle'
    res = load_nasbench(pickle_path)
    assert res == "nb"
    mock_file.assert_called_with(pickle_path, 'rb')


@mock.patch('nasbench.api.NASBench', get_mock_nasbench)
def test_load_nasbench_tfrecord():
    #with patch.object(api, 'foo', new_foo):
    res = load_nasbench('nb.tfrecord')
    assert res == "nb"


def test_load_nasbench_error():
    with pytest.raises(ValueError):
        load_nasbench('nb.txt')


def test_load_nasbench_pickle_real(nb_path):
    res = load_nasbench(nb_path)

    assert isinstance(res, nasbench.api.NASBench)
