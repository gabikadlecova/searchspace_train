import os
import pytest


@pytest.fixture
def data_dir():
    path = os.path.abspath(__file__)
    path = os.path.dirname(path)
    return os.path.join(path, 'test_data')


@pytest.fixture
def nb_path(data_dir):
    return os.path.join(data_dir, 'nb_short.pickle')
