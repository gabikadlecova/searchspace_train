import os

import pickle
import time
from typing import Optional, Union, List, Tuple, Iterable

import numpy as np
import pandas as pd
import torch.jit

from nasbench import api
from nasbench_pytorch.datasets.cifar10 import prepare_dataset
from nasbench_pytorch.trainer import train, test
from nasbench_pytorch.model import Network as NBNetwork

from searchspace_train.base import TrainedNetwork, BaseDataset
from searchspace_train.utils import load_config, print_verbose


class PretrainedNB101(BaseDataset):
    def __init__(self, nasbench: api.NASBench, device: Optional[str] = None, net_data: Optional[pd.DataFrame] = None,
                 dataset: Union[str, torch.utils.data.DataLoader, None] = None, config: Union[str, dict, None] = None,
                 verbose: Optional[bool] = True, as_basename: Optional[bool] = False, training=True):

        self.nasbench = nasbench
        self.device = device

        self.config = config if config is not None else None
        if isinstance(config, str):
            self.config = load_config(config)

        self.training = training

        if training:
            assert dataset is not None or config is not None, "Must provide either dataset or config when training."

            if dataset is not None:
                self.dataset = dataset
                self.data_name = None
            else:
                self.data_name = self.config['dataset']['name'].lower()
                data_args = self.config['dataset'].get('args', {})

                if self.data_name in ['cifar-10', 'cifar_10', 'cifar10', 'cifar']:
                    self.dataset = prepare_dataset(**data_args)
                else:
                    raise ValueError(f"Unknown dataset name: {self.data_name}.")
        else:
            self.dataset = None
            self.data_name = None

        self.net_data = pd.DataFrame(columns=['net_path', 'data_path']) if net_data is None else net_data
        self.verbose = verbose
        self.as_basename = as_basename

    def save_dataset(self, save_path: str):
        print_verbose(f"Saving to {save_path}...", self.verbose)
        self.net_data.to_csv(save_path)
        print_verbose("Saved.", self.verbose)

    def get_trained_hashes(self) -> List[str]:
        return self.net_data.index.tolist()

    def train(self, net_hash: str, save_dir: Optional[str] = None):
        if not self.training:
            raise ValueError("Networks can't be trained since training_off was set to True when this instance was "
                             "initialized.")

        ops, adjacency = get_net_from_hash(self.nasbench, net_hash)

        args = [(adjacency, ops)]
        kwargs = self.config.get("model", {})
        net = NBNetwork(*args, **kwargs)

        train_loader, test_loader = self.dataset['train'], self.dataset['test']
        valid_loader = self.dataset.get('validation')

        data_print = f' on {self.data_name}' if self.data_name is not None else ''
        save_dir = '.' if save_dir is None else save_dir

        time_zero = time.process_time()
        
        def checkpoint_func(n, m, e):
            # checkpoint indexed by epoch num
            return _save_net(save_dir, f"{net_hash}_{e}", n, m, args, kwargs, time_start=time_zero)

        # train
        print_verbose(f"Train network {net_hash}{data_print}.", self.verbose)
        net.to(self.device)
        metrics = train(net, train_loader, validation_loader=valid_loader, device=self.device,
                        checkpoint_func=checkpoint_func, **self.config['train'])

        # evaluate
        print_verbose(f"Test network {net_hash}{data_print}.", self.verbose)
        loss = self.config['train'].get('loss')
        test_metrics = test(net, test_loader, loss=loss, device=self.device)
        metrics.update(test_metrics)

        # save network
        print_verbose(f"Saving trained network to directory {save_dir}.", self.verbose)
        npath, dpath = _save_net(save_dir, net_hash, net, metrics, args, kwargs, as_basename=self.as_basename)
        self.net_data.loc[net_hash] = {'net_path': npath, 'data_path': dpath}

        return net

    def get_network(self, net_hash: str, dir_path: Optional[str] = None) -> TrainedNetwork:
        net_info = self.net_data.loc[net_hash]

        net_path, data_path = net_info['net_path'], net_info['data_path']
        net_path = net_path if dir_path is None else os.path.join(dir_path, net_path)
        data_path = data_path if dir_path is None else os.path.join(dir_path, data_path)

        return TrainedNetwork(net_hash, net_path, data_path, load_net)

    def search_space_iterator(self) -> Iterable[str]:
        return self.nasbench.hash_iterator()


def _get_save_names(save_dir, net_hash):
    net_path = os.path.join(save_dir, f'{net_hash}_script.pt')
    data_path = os.path.join(save_dir, f'{net_hash}_data.pt')
    return net_path, data_path


def load_net(checkpoint_dir, device=None):
    checkpoint = torch.load(checkpoint_dir, map_location=device)
    net = NBNetwork(*checkpoint['args'], **checkpoint['kwargs'])
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    return net


def _save_net(save_dir, net_hash, net, metrics, net_args, net_kwargs, as_basename=False, time_start=None):
    net_path, data_path = _get_save_names(save_dir, net_hash)

    checkpoint_dict = {
        'hash': net_hash,
        'model_state_dict': net.state_dict(),
        'args': net_args,
        'kwargs': net_kwargs
    }
    if time_start is not None:
        checkpoint_dict['time'] = time.process_time() - time_start

    torch.save(checkpoint_dict, net_path)
    torch.save(metrics, data_path)

    if as_basename:
        net_path = os.path.basename(net_path)
        data_path = os.path.basename(data_path)

    return net_path, data_path


def get_net_from_hash(nb: api.NASBench, net_hash: str) -> Tuple[np.ndarray, np.ndarray]:
    m = nb.get_metrics_from_hash(net_hash)
    ops = m[0]['module_operations']
    adjacency = m[0]['module_adjacency']

    return ops, adjacency


def load_nasbench(nb_path: str) -> api.NASBench:
    if nb_path.endswith('.tfrecord'):
        return api.NASBench(nb_path)
    elif nb_path.endswith('pickle'):
        with open(nb_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Invalid path to load, supported are .tfrecord and .pickle: {nb_path}")
