import os
import pickle

import pytest


@pytest.fixture
def data_dir():
    path = os.path.abspath(__file__)
    path = os.path.dirname(path)
    return os.path.join(path, 'test_data')


@pytest.fixture
def config_path(data_dir):
    return os.path.join(data_dir, 'config.yaml')


@pytest.fixture
def nb_path(data_dir):
    return os.path.join(data_dir, 'nb_short.pickle')


@pytest.fixture
def small_cifar(data_dir):
    with open(os.path.join(data_dir, 'cifar_small.pickle'), 'rb') as f:
        return pickle.load(f)
