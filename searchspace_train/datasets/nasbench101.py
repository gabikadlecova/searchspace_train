import os

import pandas as pd
import torch.jit

from nasbench_pytorch.datasets.cifar10 import prepare_dataset
from nasbench_pytorch.trainer import train, test
from nasbench_pytorch.model import Network as NBNetwork

from searchspace_train.base import TrainedNetwork
from searchspace_train.utils import load_config, print_verbose


class PretrainedNB101:
    def __init__(self, nasbench, device=None, net_data=None, dataset=None, config=None, verbose=True):
        self.nasbench = nasbench
        self.device = device
        self.config = load_config(config)

        assert dataset is not None or config is not None, "Must provide either dataset or config."

        if dataset is not None:
            self.dataset = dataset
            self.data_name = None
        else:
            self.data_name = config['dataset']['name']
            data_args = config['dataset'].get('args', {})

            if self.data_name in ['cifar-10', 'cifar_10', 'cifar10', 'cifar']:
                self.dataset = prepare_dataset(**data_args)
            else:
                raise ValueError(f"Unknown dataset name: {self.data_name}.")

        self.net_data = pd.DataFrame(columns=['net_path', 'data_path']) if net_data is None else net_data
        self.verbose = verbose

    def save_dataset(self, save_path):
        print_verbose(f"Saving to {save_path}...", self.verbose)
        self.net_data.to_csv(save_path, index=False)
        print_verbose("Saved.", self.verbose)


    def train(self, net_hash, save_dir=None):
        ops, adjacency = get_net_from_hash(self.nasbench, net_hash)
        net = NBNetwork((adjacency, ops))

        train_loader, test_loader = self.dataset['train'], self.dataset['test']
        valid_loader = self.dataset.get(['validation'])

        data_print = f' on {self.data_name}' if self.data_name is not None else ''
        save_dir = '.' if save_dir is None else save_dir
        checkpoint_func = lambda n, m, e: _save_net(save_dir, f"{net_hash}_e", n, m)  # checkpoint indexed by epoch num

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
        npath, dpath = _save_net(save_dir, net_hash, net, metrics)
        self.net_data.iloc[net_hash] = {'net_path': npath, 'data_path': dpath}

    def get_network(self, net_hash, data_mode='torch'):
        net_info = self.net_data.iloc[net_hash]
        return TrainedNetwork(net_hash, net_info['net_path'], net_info['data_path'], data_mode=data_mode)


def _get_save_names(save_dir, net_hash):
    net_path = os.path.join(save_dir, f'{net_hash}_script.pt')
    data_path = os.path.join(save_dir, f'{net_hash}_data.pt')
    return net_path, data_path


def _save_net(save_dir, net_hash, net, metrics):
    net_path, data_path = _get_save_names(save_dir, net_hash)
    net = torch.jit.script(net)
    net.save(net_path)
    torch.save(metrics, data_path)
    return net_path, data_path


def get_net_from_hash(nb, net_hash):
    m = nb.get_metrics_from_hash(net_hash)
    ops = m[0]['module_operations']
    adjacency = m[0]['module_adjacency']

    return ops, adjacency
