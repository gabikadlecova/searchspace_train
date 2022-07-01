import pandas as pd

from nasbench_pytorch.datasets.cifar10 import prepare_dataset
from nasbench_pytorch.trainer import train, test
from nasbench_pytorch.model import Network as NBNetwork

from searchspace_train.base import TrainedNetwork
from searchspace_train.utils import load_config, print_verbose


#TODO torchscript ... vyhoda je ze v infonasu uz mi staci jen dataset & torchscript

#TODO jakoby processing architektur do arch2vec formatu (tady mam sit, rovnou z toho dataset siti)
#    - ze proste bude list siti (hash, ops/adj, checkpoint_optional)
#    - pak bude data.to_arch2vec(), kde se udela preprocessing ops/adj
#    - split se udela pak na sitich/img na hoto datech


class PretrainedNB101:
    def __init__(self, nasbench, device=None, net_data=None, dataset=None, config=None, verbose=true):
        self.nasbench = nasbench
        self.device = device
        self.config = load_config(config)

        assert dataset is not None or config is not None, "Must provide either dataset or config."

        if dataset is not None:
            self.dataset = dataset
        else:
            data_name = config['dataset']['name']
            data_args = config['dataset'].get('args', {})

            if data_name in ['cifar-10', 'cifar_10', 'cifar10', 'cifar']:
                self.dataset = prepare_dataset(**data_args)
            else:
                raise ValueError(f"Unknown dataset name: {data_name}.")

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

        # TODO prints
        # TODO checkpoint
        net.to(self.device)
        metrics = train(net, train_loader, validation_loader=valid_loader, device=self.device, **self.config['train'])

        loss = self.config['train'].get('loss')
        test_metrics = test(net, test_loader, loss=loss, device=self.device)
        metrics.update(test_metrics)

        #TODO save + include into self.net_data

    def get_network(self, net_hash, data_mode='pickle'):
        net_info = self.net_data.iloc[net_hash]
        return TrainedNetwork(net_hash, net_info['net_path'], net_info['data_path'], data_mode=data_mode)


def get_net_from_hash(nb, net_hash):
    m = nb.get_metrics_from_hash(net_hash)
    ops = m[0]['module_operations']
    adjacency = m[0]['module_adjacency']

    return ops, adjacency
